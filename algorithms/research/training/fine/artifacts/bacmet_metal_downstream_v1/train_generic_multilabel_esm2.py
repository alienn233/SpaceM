from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, label_ranking_average_precision_score

from train_vf_l2_esm2 import (
    MODEL_NAMES, STRIDE, WINDOW, evaluate, fit_model, serialize, thresholds,
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
    parser.add_argument("--model-dir", default="/root/models/esm2_t6_8M_UR50D")
    parser.add_argument("--mode", choices=list(MODEL_NAMES), default="ft-weighted")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-windows", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    summary = json.loads(args.summary.read_text())
    labels = list(summary[args.labels_key])
    canonical = pd.read_parquet(args.data)
    nested = pd.read_parquet(args.nested)
    split = nested[(nested.seed.astype(int) == args.seed) &
                   (nested.outer_fold.astype(int) == args.fold)][[args.id_col, "partition"]]
    data = split.merge(canonical, on=args.id_col, validate="one_to_one").sort_values(args.id_col).reset_index(drop=True)
    label_sets = data[args.label_col].map(split_labels)
    y = np.asarray([[int(label in values) for label in labels] for values in label_sets], dtype=np.int8)
    sequences = data.sequence.astype(str).str.upper().tolist()
    role = data.partition.to_numpy()
    index = {
        "inner_train": np.flatnonzero(role == "inner_train"),
        "inner_val": np.flatnonzero(role == "inner_val"),
        "outer_train": np.flatnonzero(role != "outer_test"),
        "outer_test": np.flatnonzero(role == "outer_test"),
    }
    positive = y[index["outer_train"]].sum(axis=0).astype(float)
    weighted = np.clip(
        (len(index["outer_train"]) - positive) / np.maximum(positive, 1.0), 1.0, 20.0
    )
    weight_np = weighted if args.mode != "ft-bce" else np.ones_like(weighted)
    device = torch.device("cuda")
    pos_weight = torch.tensor(weight_np, dtype=torch.float32, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)

    if args.pilot:
        if args.seed != 11 or args.fold != 0 or args.mode != "ft-weighted":
            raise ValueError("pilot is locked to seed11/fold0/ft-weighted")
        rows = []
        for lr in (1e-5, 5e-5, 2e-4):
            model, epoch, history, score = fit_model(
                index["inner_train"], index["inner_val"], sequences, y, tokenizer,
                args.seed, device, args.model_dir, args.mode, lr, args.max_windows,
                5, 5, pos_weight,
            )
            rows.append({"lr": lr, "best_epoch": epoch,
                         "best_inner_macro_auprc": score, "history": history})
            del model
            torch.cuda.empty_cache()
        chosen = max(rows, key=lambda row: row["best_inner_macro_auprc"])["lr"]
        result = {"dataset": args.dataset, "pilot_scope": "outer_fold0_inner_only",
                  "seed": 11, "candidates": rows, "chosen_lr": chosen,
                  "outer_test_used": False}
        path = args.out / "pilot" / "esm2_lr_pilot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    model_name = MODEL_NAMES[args.mode]
    pred_path = args.out / "predictions" / model_name / f"seed{args.seed}" / f"fold{args.fold}.parquet"
    metric_path = args.out / "metrics" / f"{model_name}_seed{args.seed}_fold{args.fold}.json"
    if pred_path.exists() and metric_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "prediction": str(pred_path)}))
        return
    started = now()
    inner_model, best_epoch, history, _ = fit_model(
        index["inner_train"], index["inner_val"], sequences, y, tokenizer,
        args.seed, device, args.model_dir, args.mode, args.lr, args.max_windows,
        args.max_epochs, args.patience, pos_weight,
    )
    val_prob, _ = evaluate(
        inner_model, index["inner_val"], sequences, y, tokenizer,
        device, args.max_windows, pos_weight,
    )
    locked = thresholds(y[index["inner_val"]], val_prob)
    del inner_model
    torch.cuda.empty_cache()
    final_model, _, final_history, _ = fit_model(
        index["outer_train"], None, sequences, y, tokenizer,
        args.seed, device, args.model_dir, args.mode, args.lr, args.max_windows,
        args.max_epochs, args.patience, pos_weight, fixed_epochs=best_epoch,
    )
    test_prob, _ = evaluate(
        final_model, index["outer_test"], sequences, y, tokenizer,
        device, args.max_windows, pos_weight,
    )
    y_test = y[index["outer_test"]]
    y_pred = (test_prob >= locked).astype(np.int8)
    per_ap = [average_precision_score(y_test[:, j], test_prob[:, j]) for j in range(len(labels))]
    metrics = {
        "dataset": args.dataset, "output_semantics": f"{len(labels)}_independent_sigmoid_logits",
        "model": model_name, "mode": args.mode, "seed": args.seed, "outer_fold": args.fold,
        "lr": args.lr, "best_epoch": best_epoch, "weighted_bce": args.mode != "ft-bce",
        "encoder_frozen": args.mode == "frozen-weighted",
        "n_inner_train": len(index["inner_train"]), "n_inner_val": len(index["inner_val"]),
        "n_outer_train": len(index["outer_train"]), "n_outer_test": len(index["outer_test"]),
        "micro_auprc": float(average_precision_score(y_test, test_prob, average="micro")),
        "macro_auprc": float(np.mean(per_ap)),
        "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "sample_f1": float(f1_score(y_test, y_pred, average="samples", zero_division=0)),
        "lrap": float(label_ranking_average_precision_score(y_test, test_prob)),
        "thresholds": locked.tolist(), "pos_weight": weight_np.tolist(),
        "window": WINDOW, "stride": STRIDE, "all_window_evaluation": True,
        "outer_test_used_for_selection": False, "start_time": started, "end_time": now(),
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        args.id_col: data.iloc[index["outer_test"]][args.id_col].to_numpy(),
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
                "labels": labels, "metrics": metrics}, checkpoint)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
