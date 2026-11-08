from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

from train_vf_l2_esm2 import (
    STRIDE, WINDOW, collate, iter_batches, make_model, seed_all, serialize,
)


MODEL_NAMES = {
    "ft-weighted": "ESM2-8M-FT-WCE",
    "ft-ce": "ESM2-8M-FT-CE",
    "frozen-weighted": "ESM2-8M-Frozen-WCE",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def forward(model, batch, device, class_weight):
    import torch
    input_ids, attention_mask, owner, weight, onehot = batch
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    owner, weight, onehot = owner.to(device), weight.to(device), onehot.to(device)
    target = onehot.argmax(dim=1).long()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids, attention_mask, owner, weight, len(target))
        loss = torch.nn.functional.cross_entropy(logits.float(), target, weight=class_weight)
    return logits, target, loss


def evaluate(model, indices, sequences, y_onehot, tokenizer, device, max_windows, class_weight):
    import torch
    model.eval()
    probabilities, losses, sizes = [], [], []
    with torch.no_grad():
        for idx in iter_batches(indices, sequences, max_windows, max_windows, None):
            logits, _, loss = forward(
                model, collate(idx, sequences, y_onehot, tokenizer), device, class_weight
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


def fit_model(train_idx, val_idx, sequences, y_onehot, tokenizer, seed, device,
              model_dir, mode, lr, max_windows, max_epochs, patience,
              class_weight, fixed_epochs=None):
    import torch
    from transformers import get_linear_schedule_with_warmup
    seed_all(seed)
    frozen = mode == "frozen-weighted"
    model = make_model(model_dir, y_onehot.shape[1], frozen).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    target_epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    batches = sum(1 for _ in iter_batches(train_idx, sequences, max_windows, max_windows, None))
    total_steps = max(1, batches * target_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(total_steps * 0.05)),
        num_training_steps=total_steps,
    )
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    for epoch in range(1, target_epochs + 1):
        model.train()
        losses = []
        for idx in iter_batches(train_idx, sequences, max_windows, max_windows, seed * 10000 + epoch):
            optimizer.zero_grad(set_to_none=True)
            _, _, loss = forward(
                model, collate(idx, sequences, y_onehot, tokenizer), device, class_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "lr": float(scheduler.get_last_lr()[0])}
        if val_idx is not None:
            probability, val_loss = evaluate(
                model, val_idx, sequences, y_onehot, tokenizer, device,
                max_windows, class_weight,
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
    return model, (best_epoch if val_idx is not None else target_epochs), history, best_score


def load_data(args):
    summary = json.loads(args.summary.read_text())
    labels = list(summary["atomic_labels"])
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
    sequences = data.sequence.astype(str).str.upper().tolist()
    role = data.partition.to_numpy()
    index = {
        "inner_train": np.flatnonzero(role == "inner_train"),
        "inner_val": np.flatnonzero(role == "inner_val"),
        "outer_train": np.flatnonzero(role != "outer_test"),
        "outer_test": np.flatnonzero(role == "outer_test"),
    }
    return summary, labels, data, y, y_onehot, sequences, index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--nested", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
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
    summary, labels, data, y, y_onehot, sequences, index = load_data(args)
    counts = np.bincount(y[index["outer_train"]], minlength=len(labels)).astype(float)
    weighted_np = np.clip(
        len(index["outer_train"]) / (len(labels) * np.maximum(counts, 1.0)), 0.25, 20.0
    )
    weight_np = np.ones_like(weighted_np) if args.mode == "ft-ce" else weighted_np
    class_weight = torch.tensor(weight_np, dtype=torch.float32, device="cuda")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)

    if args.pilot:
        if args.seed != 11 or args.fold != 0 or args.mode != "ft-weighted":
            raise ValueError("pilot is locked to seed11/fold0/ft-weighted")
        rows = []
        for lr in (1e-5, 5e-5, 2e-4):
            model, epoch, history, score = fit_model(
                index["inner_train"], index["inner_val"], sequences, y_onehot,
                tokenizer, args.seed, device, args.model_dir, args.mode, lr,
                args.max_windows, 5, 5, class_weight,
            )
            rows.append({"lr": lr, "best_epoch": epoch,
                         "best_inner_macro_auprc": score, "history": history})
            del model
            torch.cuda.empty_cache()
        chosen = max(rows, key=lambda row: row["best_inner_macro_auprc"])["lr"]
        result = {"dataset": summary["dataset"], "seed": 11,
                  "pilot_scope": "outer_fold0_inner_only", "candidates": rows,
                  "chosen_lr": chosen, "outer_test_used": False}
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
        index["inner_train"], index["inner_val"], sequences, y_onehot, tokenizer,
        args.seed, device, args.model_dir, args.mode, args.lr, args.max_windows,
        args.max_epochs, args.patience, class_weight,
    )
    del inner_model
    torch.cuda.empty_cache()
    final_model, _, final_history, _ = fit_model(
        index["outer_train"], None, sequences, y_onehot, tokenizer,
        args.seed, device, args.model_dir, args.mode, args.lr, args.max_windows,
        args.max_epochs, args.patience, class_weight, fixed_epochs=best_epoch,
    )
    probability, _ = evaluate(
        final_model, index["outer_test"], sequences, y_onehot, tokenizer,
        device, args.max_windows, class_weight,
    )
    truth = y[index["outer_test"]]
    pred = probability.argmax(axis=1)
    per_ap = [average_precision_score(y_onehot[index["outer_test"], j], probability[:, j])
              for j in range(len(labels))]
    metrics = {
        "dataset": summary["dataset"], "output_semantics": "7_class_softmax",
        "loss": "cross_entropy" if args.mode == "ft-ce" else "class_weighted_cross_entropy",
        "model": model_name, "mode": args.mode, "seed": args.seed,
        "outer_fold": args.fold, "lr": args.lr, "best_epoch": best_epoch,
        "encoder_frozen": args.mode == "frozen-weighted",
        "micro_auprc": float(average_precision_score(y_onehot[index["outer_test"]], probability, average="micro")),
        "macro_auprc": float(np.mean(per_ap)),
        "micro_f1": float(f1_score(truth, pred, average="micro")),
        "macro_f1": float(f1_score(truth, pred, average="macro")),
        "mcc": float(matthews_corrcoef(truth, pred)),
        "class_weight": weight_np.tolist(), "window": WINDOW, "stride": STRIDE,
        "outer_test_used_for_selection": False, "start_time": started, "end_time": now(),
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        args.id_col: data.iloc[index["outer_test"]][args.id_col].to_numpy(),
        "seed": args.seed, "outer_fold": args.fold, "model": model_name,
        "true_label": [labels[value] for value in truth],
        "pred_label": [labels[value] for value in pred],
        "probabilities": [serialize(labels, row) for row in probability],
    }).to_parquet(pred_path, index=False)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n")
    history_path = args.out / "training_history" / model_name / f"seed{args.seed}_fold{args.fold}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({"inner": history, "final": final_history}, indent=2) + "\n")
    checkpoint = args.out / "checkpoints" / model_name / f"seed{args.seed}" / f"fold{args.fold}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
                "labels": labels, "metrics": metrics}, checkpoint)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
