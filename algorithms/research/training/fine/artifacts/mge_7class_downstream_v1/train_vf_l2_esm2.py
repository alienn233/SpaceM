from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, label_ranking_average_precision_score


WINDOW = 1022
STRIDE = 512
MODEL_NAMES = {
    "frozen-weighted": "ESM2-8M-Frozen-WBCE",
    "ft-bce": "ESM2-8M-FT-BCE",
    "ft-weighted": "ESM2-8M-FT-WBCE",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return list(dict.fromkeys(x.strip() for x in str(value).split(";") if x.strip()))


def seed_all(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def starts(length: int) -> list[int]:
    if length <= WINDOW:
        return [0]
    values = list(range(0, length - WINDOW + 1, STRIDE))
    last = length - WINDOW
    if values[-1] != last:
        values.append(last)
    return values


def iter_batches(indices: np.ndarray, sequences: list[str], max_windows: int,
                 max_proteins: int, shuffle_seed: int | None):
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(order)
    batch: list[int] = []
    count = 0
    for raw in order:
        idx = int(raw)
        n = len(starts(len(sequences[idx])))
        if batch and (count + n > max_windows or len(batch) >= max_proteins):
            yield batch
            batch, count = [], 0
        batch.append(idx)
        count += n
    if batch:
        yield batch


def collate(indices: list[int], sequences: list[str], labels: np.ndarray, tokenizer):
    import torch
    fragments: list[str] = []
    owners: list[int] = []
    weights: list[float] = []
    for owner, idx in enumerate(indices):
        sequence = sequences[idx]
        if not sequence:
            raise ValueError(f"empty protein sequence at row {idx}")
        for start in starts(len(sequence)):
            fragment = sequence[start:start + WINDOW]
            fragments.append(fragment)
            owners.append(owner)
            weights.append(float(len(fragment)))
    encoded = tokenizer(fragments, return_tensors="pt", padding=True, add_special_tokens=True)
    return (
        encoded["input_ids"], encoded["attention_mask"],
        torch.tensor(owners, dtype=torch.long), torch.tensor(weights, dtype=torch.float32),
        torch.from_numpy(labels[np.asarray(indices)].astype(np.float32, copy=False)),
    )


def make_model(model_dir: str, n_labels: int, frozen: bool):
    import torch
    from torch import nn
    from transformers import EsmModel

    class ProteinESM(nn.Module):
        def __init__(self):
            super().__init__()
            self.esm = EsmModel.from_pretrained(
                model_dir, local_files_only=True, add_pooling_layer=False
            )
            self.esm.config.use_cache = False
            if not frozen:
                self.esm.gradient_checkpointing_enable()
            for parameter in self.esm.parameters():
                parameter.requires_grad = not frozen
            self.head = nn.Sequential(
                nn.LayerNorm(self.esm.config.hidden_size),
                nn.Dropout(0.20),
                nn.Linear(self.esm.config.hidden_size, n_labels),
            )

        def forward(self, input_ids, attention_mask, owner, window_weight, n_proteins):
            hidden = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            residue_mask = attention_mask.bool().clone()
            residue_mask[:, 0] = False
            last = attention_mask.sum(dim=1).long() - 1
            residue_mask[torch.arange(len(last), device=last.device), last] = False
            valid = residue_mask.unsqueeze(-1).to(hidden.dtype)
            representation = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            window_logits = self.head(representation)
            window_weight = window_weight.to(window_logits.dtype)
            protein_logits = torch.zeros(
                (n_proteins, n_labels), device=hidden.device, dtype=window_logits.dtype
            )
            denominator = torch.zeros(n_proteins, device=hidden.device, dtype=window_logits.dtype)
            protein_logits.index_add_(0, owner, window_logits * window_weight[:, None])
            denominator.index_add_(0, owner, window_weight)
            return protein_logits / denominator.clamp_min(1.0)[:, None]

    return ProteinESM()


def forward(model, batch, device, pos_weight):
    import torch
    input_ids, attention_mask, owner, weight, target = batch
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    owner, weight, target = owner.to(device), weight.to(device), target.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids, attention_mask, owner, weight, len(target))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), target, pos_weight=pos_weight
        )
    return logits, target, loss


def evaluate(model, indices, sequences, labels, tokenizer, device, max_windows, pos_weight):
    import torch
    model.eval()
    probabilities: list[np.ndarray] = []
    losses: list[float] = []
    sizes: list[int] = []
    with torch.no_grad():
        for idx in iter_batches(indices, sequences, max_windows, max_windows, None):
            logits, _, loss = forward(
                model, collate(idx, sequences, labels, tokenizer), device, pos_weight
            )
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            losses.append(float(loss.item()))
            sizes.append(len(idx))
    return np.concatenate(probabilities), float(np.average(losses, weights=sizes))


def macro_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def fit_model(train_idx, val_idx, sequences, labels, tokenizer, seed, device,
              model_dir, mode, lr, max_windows, max_epochs, patience,
              pos_weight, fixed_epochs=None):
    import torch
    from transformers import get_linear_schedule_with_warmup
    seed_all(seed)
    frozen = mode == "frozen-weighted"
    model = make_model(model_dir, labels.shape[1], frozen).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    target_epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    batches_per_epoch = sum(1 for _ in iter_batches(train_idx, sequences, max_windows, max_windows, None))
    total_steps = max(1, batches_per_epoch * target_epochs)
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
                model, collate(idx, sequences, labels, tokenizer), device, pos_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "lr": float(scheduler.get_last_lr()[0])}
        if val_idx is not None:
            p_val, val_loss = evaluate(
                model, val_idx, sequences, labels, tokenizer, device, max_windows, pos_weight
            )
            score = macro_ap(labels[val_idx], p_val)
            row.update({"val_loss": val_loss, "val_macro_auprc": score})
            if score > best_score + 1e-7:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if val_idx is not None and stale >= patience:
            break
    if val_idx is not None:
        if best_state is None:
            raise RuntimeError("no ESM2 validation checkpoint")
        model.load_state_dict(best_state)
    return model, (best_epoch if val_idx is not None else target_epochs), history, best_score


def thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    result = np.full(y.shape[1], 0.5, dtype=float)
    for j in range(y.shape[1]):
        candidates = []
        for threshold in np.linspace(0.05, 0.95, 19):
            score = f1_score(y[:, j], p[:, j] >= threshold, zero_division=0)
            candidates.append((float(score), -float(threshold), float(threshold)))
        result[j] = max(candidates)[2]
    return result


def serialize(label_names: list[str], values: np.ndarray, digits=8) -> str:
    return ";".join(f"{name}={float(values[i]):.{digits}f}" for i, name in enumerate(label_names))


def load_data(args):
    summary = json.loads((args.out / "qc" / "phase0_summary.json").read_text())
    label_names = list(summary["atomic_l2_labels"])
    nested = pd.read_csv(args.out / "splits" / "vf_l2_nested_splits.tsv", sep="\t", dtype=str)
    split = nested[(nested.seed.astype(int) == args.seed) &
                   (nested.outer_fold.astype(int) == args.fold)].copy()
    source = pd.read_csv(args.label_tsv, sep="\t", dtype=str, low_memory=False)[
        ["record_id", "sequence", "level2_vf_type"]
    ]
    data = split.merge(source, on=["record_id", "level2_vf_type"], validate="one_to_one")
    data = data.sort_values("record_id").reset_index(drop=True)
    label_sets = data.level2_vf_type.map(tokens)
    y = np.asarray([[int(name in values) for name in label_names] for values in label_sets], dtype=np.int8)
    sequences = data.sequence.astype(str).str.upper().tolist()
    role = data.partition.to_numpy()
    indices = {
        "inner_train": np.flatnonzero(role == "inner_train"),
        "inner_val": np.flatnonzero(role == "inner_val"),
        "outer_train": np.flatnonzero(role != "outer_test"),
        "outer_test": np.flatnonzero(role == "outer_test"),
    }
    return summary, label_names, data, y, sequences, indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label-tsv", required=True, type=Path)
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
    summary, label_names, data, y, sequences, idx = load_data(args)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    positive = y[idx["outer_train"]].sum(axis=0).astype(float)
    weighted = np.clip(
        (len(idx["outer_train"]) - positive) / np.maximum(positive, 1.0), 1.0, 20.0
    )
    weight_np = weighted if args.mode != "ft-bce" else np.ones_like(weighted)
    pos_weight = torch.tensor(weight_np, dtype=torch.float32, device=device)

    if args.pilot:
        if args.seed != 11 or args.fold != 0 or args.mode != "ft-weighted":
            raise ValueError("the locked LR pilot is only seed11/fold0/ft-weighted")
        rows = []
        for lr in (1e-5, 5e-5, 2e-4):
            model, epoch, history, score = fit_model(
                idx["inner_train"], idx["inner_val"], sequences, y, tokenizer,
                args.seed, device, args.model_dir, args.mode, lr, args.max_windows,
                5, 5, pos_weight,
            )
            rows.append({"lr": lr, "best_epoch": epoch, "best_inner_macro_auprc": score,
                         "history": history})
            del model
            torch.cuda.empty_cache()
        chosen = max(rows, key=lambda row: row["best_inner_macro_auprc"])["lr"]
        result = {"pilot_scope": "outer_fold0_inner_only", "seed": 11,
                  "candidates": rows, "chosen_lr": chosen, "outer_test_used": False}
        path = args.out / "pilot" / "esm2_lr_pilot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    model_name = MODEL_NAMES[args.mode]
    pred_path = args.out / "predictions" / model_name / f"seed{args.seed}" / f"fold{args.fold}.parquet"
    metric_path = args.out / "metrics" / f"vf_l2_{model_name}_seed{args.seed}_fold{args.fold}.json"
    if pred_path.exists() and metric_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "prediction": str(pred_path)}))
        return
    started = utc_now()
    inner_model, best_epoch, history, _ = fit_model(
        idx["inner_train"], idx["inner_val"], sequences, y, tokenizer, args.seed,
        device, args.model_dir, args.mode, args.lr, args.max_windows,
        args.max_epochs, args.patience, pos_weight,
    )
    val_prob, _ = evaluate(
        inner_model, idx["inner_val"], sequences, y, tokenizer, device,
        args.max_windows, pos_weight,
    )
    locked_thresholds = thresholds(y[idx["inner_val"]], val_prob)
    del inner_model
    torch.cuda.empty_cache()
    final_model, _, final_history, _ = fit_model(
        idx["outer_train"], None, sequences, y, tokenizer, args.seed,
        device, args.model_dir, args.mode, args.lr, args.max_windows,
        args.max_epochs, args.patience, pos_weight, fixed_epochs=best_epoch,
    )
    test_prob, _ = evaluate(
        final_model, idx["outer_test"], sequences, y, tokenizer, device,
        args.max_windows, pos_weight,
    )
    y_test = y[idx["outer_test"]]
    y_pred = (test_prob >= locked_thresholds).astype(np.int8)
    per_ap = [average_precision_score(y_test[:, j], test_prob[:, j]) for j in range(len(label_names))]
    metrics = {
        "head": "VF_L2", "output_semantics": "14_independent_sigmoid_logits",
        "model": model_name, "mode": args.mode, "seed": args.seed, "outer_fold": args.fold,
        "lr": args.lr, "best_epoch": best_epoch, "weighted_bce": args.mode != "ft-bce",
        "encoder_frozen": args.mode == "frozen-weighted",
        "n_inner_train": len(idx["inner_train"]), "n_inner_val": len(idx["inner_val"]),
        "n_outer_train": len(idx["outer_train"]), "n_outer_test": len(idx["outer_test"]),
        "micro_auprc": float(average_precision_score(y_test, test_prob, average="micro")),
        "macro_auprc": float(np.mean(per_ap)),
        "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "sample_f1": float(f1_score(y_test, y_pred, average="samples", zero_division=0)),
        "lrap": float(label_ranking_average_precision_score(y_test, test_prob)),
        "thresholds": locked_thresholds.tolist(), "pos_weight": weight_np.tolist(),
        "window": WINDOW, "stride": STRIDE, "all_window_evaluation": True,
        "outer_test_used_for_selection": False, "start_time": started, "end_time": utc_now(),
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "record_id": data.iloc[idx["outer_test"]].record_id.to_numpy(),
        "outer_fold": args.fold, "seed": args.seed, "model": model_name,
        "true_l2_labels": [";".join(label_names[j] for j in np.flatnonzero(row)) for row in y_test],
        "prob_l2_labels": [serialize(label_names, row) for row in test_prob],
        "pred_l2_labels": [";".join(label_names[j] for j in np.flatnonzero(row)) for row in y_pred],
        "threshold_l2_labels": [serialize(label_names, locked_thresholds, 6)] * len(y_test),
    }).to_parquet(pred_path, index=False)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n")
    hist = args.out / "training_history" / model_name / f"seed{args.seed}_fold{args.fold}.json"
    hist.parent.mkdir(parents=True, exist_ok=True)
    hist.write_text(json.dumps({"inner": history, "final": final_history}, indent=2) + "\n")
    checkpoint = args.out / "checkpoints" / model_name / f"seed{args.seed}" / f"fold{args.fold}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
                "labels": label_names, "metrics": metrics}, checkpoint)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
