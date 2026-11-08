from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

from train_vf_l2_seqcnn import (
    STRIDE, WINDOW, collate, encode, iter_batches, make_model, seed_all, serialize,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def forward(model, batch, device, class_weight):
    import torch
    encoded, mask, owner, weight, onehot = batch
    encoded, mask = encoded.to(device), mask.to(device)
    owner, weight, onehot = owner.to(device), weight.to(device), onehot.to(device)
    target = onehot.argmax(dim=1).long()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(encoded, mask, owner, weight, len(target))
        loss = torch.nn.functional.cross_entropy(logits.float(), target, weight=class_weight)
    return logits, target, loss


def evaluate(model, indices, token_arrays, y_onehot, device, max_windows, class_weight):
    import torch
    model.eval()
    probabilities, losses, sizes = [], [], []
    with torch.no_grad():
        for idx in iter_batches(indices, token_arrays, max_windows, 256, None):
            logits, _, loss = forward(
                model, collate(idx, token_arrays, y_onehot), device, class_weight
            )
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            losses.append(float(loss.item()))
            sizes.append(len(idx))
    return np.concatenate(probabilities), float(np.average(losses, weights=sizes))


def macro_ap(y_onehot: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean([
        average_precision_score(y_onehot[:, j], probability[:, j])
        for j in range(y_onehot.shape[1])
    ]))


def fit(train_idx, val_idx, token_arrays, y_onehot, seed, device, max_windows,
        n_labels, architecture, class_weight, max_epochs, patience, fixed_epochs=None):
    import torch
    seed_all(seed)
    model = make_model(n_labels, architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target_epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    for epoch in range(1, target_epochs + 1):
        model.train()
        losses = []
        for idx in iter_batches(train_idx, token_arrays, max_windows, 256, seed * 10000 + epoch):
            optimizer.zero_grad(set_to_none=True)
            _, _, loss = forward(
                model, collate(idx, token_arrays, y_onehot), device, class_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_idx is not None:
            probability, val_loss = evaluate(
                model, val_idx, token_arrays, y_onehot, device, max_windows, class_weight
            )
            score = macro_ap(y_onehot[val_idx], probability)
            row.update({"val_loss": val_loss, "val_macro_auprc": score})
            if score > best_score + 1e-7:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if val_idx is not None and stale >= patience:
            break
    if val_idx is not None:
        if best_state is None:
            raise RuntimeError("no validation checkpoint")
        model.load_state_dict(best_state)
    return model, (best_epoch if val_idx is not None else target_epochs), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--nested", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
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
    summary = json.loads(args.summary.read_text())
    labels = list(summary["atomic_labels"])
    pred_path = args.out / "predictions" / args.architecture / f"seed{args.seed}" / f"fold{args.fold}.parquet"
    metric_path = args.out / "metrics" / f"{args.architecture}_seed{args.seed}_fold{args.fold}.json"
    if pred_path.exists() and metric_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "prediction": str(pred_path)}))
        return

    canonical = pd.read_parquet(args.data)
    nested = pd.read_parquet(args.nested)
    split = nested[(nested.seed.astype(int) == args.seed) &
                   (nested.outer_fold.astype(int) == args.fold)][[args.id_col, "partition"]]
    data = split.merge(canonical, on=args.id_col, validate="one_to_one").sort_values(args.id_col).reset_index(drop=True)
    label_to_int = {label: index for index, label in enumerate(labels)}
    if not data[args.label_col].astype(str).isin(labels).all():
        raise RuntimeError("MGE contains unknown or non-single labels")
    y = data[args.label_col].astype(str).map(label_to_int).to_numpy(int)
    y_onehot = np.eye(len(labels), dtype=np.int8)[y]
    token_arrays = [encode(sequence) for sequence in data.sequence.astype(str)]
    role = data.partition.to_numpy()
    inner_train = np.flatnonzero(role == "inner_train")
    inner_val = np.flatnonzero(role == "inner_val")
    outer_train = np.flatnonzero(role != "outer_test")
    outer_test = np.flatnonzero(role == "outer_test")
    counts = np.bincount(y[outer_train], minlength=len(labels)).astype(float)
    weight_np = np.clip(len(outer_train) / (len(labels) * np.maximum(counts, 1.0)), 0.25, 20.0)
    class_weight = torch.tensor(weight_np, dtype=torch.float32, device="cuda")
    device = torch.device("cuda")

    started = now()
    inner_model, best_epoch, history = fit(
        inner_train, inner_val, token_arrays, y_onehot, args.seed, device,
        args.max_windows, len(labels), args.architecture, class_weight,
        args.max_epochs, args.patience,
    )
    del inner_model
    torch.cuda.empty_cache()
    final_model, _, final_history = fit(
        outer_train, None, token_arrays, y_onehot, args.seed, device,
        args.max_windows, len(labels), args.architecture, class_weight,
        args.max_epochs, args.patience, fixed_epochs=best_epoch,
    )
    probability, _ = evaluate(
        final_model, outer_test, token_arrays, y_onehot, device, args.max_windows, class_weight
    )
    truth = y[outer_test]
    pred = probability.argmax(axis=1)
    per_ap = [average_precision_score(y_onehot[outer_test, j], probability[:, j])
              for j in range(len(labels))]
    metrics = {
        "dataset": summary["dataset"], "output_semantics": "7_class_softmax",
        "loss": "class_weighted_cross_entropy", "model": args.architecture,
        "seed": args.seed, "outer_fold": args.fold, "best_epoch": best_epoch,
        "n_outer_test": len(outer_test),
        "micro_auprc": float(average_precision_score(y_onehot[outer_test], probability, average="micro")),
        "macro_auprc": float(np.mean(per_ap)),
        "micro_f1": float(f1_score(truth, pred, average="micro")),
        "macro_f1": float(f1_score(truth, pred, average="macro")),
        "mcc": float(matthews_corrcoef(truth, pred)),
        "class_weight": weight_np.tolist(), "window": WINDOW, "stride": STRIDE,
        "outer_test_used_for_selection": False, "start_time": started, "end_time": now(),
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        args.id_col: data.iloc[outer_test][args.id_col].to_numpy(),
        "seed": args.seed, "outer_fold": args.fold, "model": args.architecture,
        "true_label": [labels[index] for index in truth],
        "pred_label": [labels[index] for index in pred],
        "probabilities": [serialize(labels, row) for row in probability],
    }).to_parquet(pred_path, index=False)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n")
    history_path = args.out / "training_history" / args.architecture / f"seed{args.seed}_fold{args.fold}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({"inner": history, "final": final_history}, indent=2) + "\n")
    checkpoint = args.out / "checkpoints" / args.architecture / f"seed{args.seed}" / f"fold{args.fold}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
                "labels": labels, "metrics": metrics}, checkpoint)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
