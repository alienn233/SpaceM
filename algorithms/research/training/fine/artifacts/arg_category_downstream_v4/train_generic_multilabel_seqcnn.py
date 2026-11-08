from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, label_ranking_average_precision_score

from train_vf_l2_seqcnn import (
    WINDOW, STRIDE, choose_thresholds, encode, evaluate, fit, serialize,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return list(dict.fromkeys(x.strip() for x in str(value).split(";") if x.strip()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--nested", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--labels-key", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--id-col", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--architecture", choices=["SeqCNN", "SeqCNN-Attn"], default="SeqCNN-Attn")
    parser.add_argument("--max-windows", type=int, default=384)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = now()
    model_name = args.architecture
    pred_path = args.out / "predictions" / model_name / f"seed{args.seed}" / f"fold{args.fold}.parquet"
    metric_path = args.out / "metrics" / f"{model_name}_seed{args.seed}_fold{args.fold}.json"
    if pred_path.exists() and metric_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "prediction": str(pred_path)}))
        return

    summary = json.loads(args.summary.read_text())
    labels = list(summary[args.labels_key])
    data = pd.read_parquet(args.data)
    nested = pd.read_parquet(args.nested)
    split = nested[(nested.seed.astype(int) == args.seed) &
                   (nested.outer_fold.astype(int) == args.fold)][
        [args.id_col, "partition"]
    ]
    data = split.merge(data, on=args.id_col, validate="one_to_one").sort_values(args.id_col).reset_index(drop=True)
    if len(data) != len(split):
        raise RuntimeError("data/nested row mismatch")
    label_sets = data[args.label_col].map(split_labels)
    y = np.asarray([[int(label in values) for label in labels] for values in label_sets], dtype=np.int8)
    token_arrays = [encode(sequence) for sequence in data.sequence.astype(str)]
    role = data.partition.to_numpy()
    inner_train = np.flatnonzero(role == "inner_train")
    inner_val = np.flatnonzero(role == "inner_val")
    outer_train = np.flatnonzero(role != "outer_test")
    outer_test = np.flatnonzero(role == "outer_test")
    positive = y[outer_train].sum(axis=0).astype(float)
    pos_weight_np = np.clip((len(outer_train) - positive) / np.maximum(positive, 1.0), 1.0, 20.0)
    device = torch.device("cuda")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

    inner_model, best_epoch, history = fit(
        inner_train, inner_val, token_arrays, y, args.seed, device, args.max_windows,
        len(labels), model_name, pos_weight, args.max_epochs, args.patience,
    )
    val_prob, _ = evaluate(inner_model, inner_val, token_arrays, y, device, args.max_windows, pos_weight)
    locked = choose_thresholds(y[inner_val], val_prob)
    del inner_model
    torch.cuda.empty_cache()
    final_model, _, final_history = fit(
        outer_train, None, token_arrays, y, args.seed, device, args.max_windows,
        len(labels), model_name, pos_weight, args.max_epochs, args.patience,
        fixed_epochs=best_epoch,
    )
    test_prob, _ = evaluate(final_model, outer_test, token_arrays, y, device, args.max_windows, pos_weight)
    y_test = y[outer_test]
    y_pred = (test_prob >= locked).astype(np.int8)
    per_ap = [average_precision_score(y_test[:, j], test_prob[:, j]) for j in range(len(labels))]
    metrics = {
        "dataset": args.dataset, "output_semantics": f"{len(labels)}_independent_sigmoid_logits",
        "loss": "BCEWithLogitsLoss", "weighted_bce": True, "model": model_name,
        "seed": args.seed, "outer_fold": args.fold, "best_epoch": best_epoch,
        "n_inner_train": len(inner_train), "n_inner_val": len(inner_val),
        "n_outer_train": len(outer_train), "n_outer_test": len(outer_test),
        "micro_auprc": float(average_precision_score(y_test, test_prob, average="micro")),
        "macro_auprc": float(np.mean(per_ap)),
        "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "sample_f1": float(f1_score(y_test, y_pred, average="samples", zero_division=0)),
        "lrap": float(label_ranking_average_precision_score(y_test, test_prob)),
        "thresholds": locked.tolist(), "pos_weight": pos_weight_np.tolist(),
        "window": WINDOW, "stride": STRIDE, "protein_level_loss": True,
        "outer_test_used_for_selection": False, "start_time": started, "end_time": now(),
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        args.id_col: data.iloc[outer_test][args.id_col].to_numpy(),
        "outer_fold": args.fold, "seed": args.seed, "model": model_name,
        "true_labels": [";".join(labels[j] for j in np.flatnonzero(row)) for row in y_test],
        "prob_labels": [serialize(labels, row) for row in test_prob],
        "pred_labels": [";".join(labels[j] for j in np.flatnonzero(row)) for row in y_pred],
        "threshold_labels": [serialize(labels, locked, 6)] * len(y_test),
    }).to_parquet(pred_path, index=False)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n")
    history_path = args.out / "training_history" / model_name / f"seed{args.seed}_fold{args.fold}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({"inner": history, "final": final_history}, indent=2) + "\n")
    checkpoint = args.out / "checkpoints" / model_name / f"seed{args.seed}" / f"fold{args.fold}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
                "labels": labels, "metrics": metrics, "architecture": model_name}, checkpoint)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
