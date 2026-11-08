from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, label_ranking_average_precision_score


WINDOW = 1022
STRIDE = 512
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i + 1 for i, aa in enumerate(AA)}
UNKNOWN = len(AA) + 1


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


def encode(sequence: str) -> np.ndarray:
    return np.fromiter((AA_TO_INT.get(aa, UNKNOWN) for aa in sequence.upper()), dtype=np.int16)


def window_starts(length: int) -> list[int]:
    if length <= WINDOW:
        return [0]
    starts = list(range(0, length - WINDOW + 1, STRIDE))
    last = length - WINDOW
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_batches(indices: np.ndarray, token_arrays: list[np.ndarray], max_windows: int,
                 max_proteins: int, shuffle_seed: int | None):
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(order)
    batch: list[int] = []
    windows = 0
    for raw_idx in order:
        idx = int(raw_idx)
        count = len(window_starts(len(token_arrays[idx])))
        if batch and (windows + count > max_windows or len(batch) >= max_proteins):
            yield batch
            batch, windows = [], 0
        batch.append(idx)
        windows += count
    if batch:
        yield batch


def collate(indices: list[int], token_arrays: list[np.ndarray], labels: np.ndarray):
    import torch
    slices: list[np.ndarray] = []
    owners: list[int] = []
    weights: list[float] = []
    for owner, idx in enumerate(indices):
        seq = token_arrays[idx]
        if len(seq) == 0:
            raise ValueError(f"empty sequence at row {idx}")
        for start in window_starts(len(seq)):
            part = seq[start:start + WINDOW]
            slices.append(part)
            owners.append(owner)
            weights.append(float(len(part)))
    max_len = max(map(len, slices))
    encoded = np.zeros((len(slices), max_len), dtype=np.int64)
    mask = np.zeros((len(slices), max_len), dtype=np.bool_)
    for row, part in enumerate(slices):
        encoded[row, :len(part)] = part
        mask[row, :len(part)] = True
    return (
        torch.from_numpy(encoded), torch.from_numpy(mask),
        torch.tensor(owners, dtype=torch.long), torch.tensor(weights, dtype=torch.float32),
        torch.from_numpy(labels[np.asarray(indices)].astype(np.float32, copy=False)),
    )


def make_model(n_labels: int, architecture: str):
    import torch
    from torch import nn

    class ProteinCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(UNKNOWN + 1, 64, padding_idx=0)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.norm3 = nn.LayerNorm(128)
            self.conv5 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
            self.norm5 = nn.LayerNorm(128)
            self.gelu = nn.GELU()
            self.attention = nn.Linear(128, 1) if architecture == "SeqCNN-Attn" else None
            self.head = nn.Sequential(
                nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.30), nn.Linear(128, n_labels)
            )

        def forward(self, encoded, mask, owner, window_weight, n_proteins):
            x = self.embedding(encoded).transpose(1, 2)
            x = self.conv3(x).transpose(1, 2)
            x = self.gelu(self.norm3(x))
            x = self.conv5(x.transpose(1, 2)).transpose(1, 2)
            x = self.gelu(self.norm5(x))
            if self.attention is not None:
                score = self.attention(x).squeeze(-1).masked_fill(~mask, -1e4)
                alpha = torch.softmax(score, dim=1)
                window_repr = torch.sum(x * alpha.unsqueeze(-1), dim=1)
            else:
                valid = mask.unsqueeze(-1).to(x.dtype)
                window_repr = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            window_logits = self.head(window_repr)
            window_weight = window_weight.to(window_logits.dtype)
            protein_logits = torch.zeros(
                (n_proteins, n_labels), device=x.device, dtype=window_logits.dtype
            )
            denominator = torch.zeros(n_proteins, device=x.device, dtype=window_logits.dtype)
            protein_logits.index_add_(0, owner, window_logits * window_weight[:, None])
            denominator.index_add_(0, owner, window_weight)
            return protein_logits / denominator.clamp_min(1.0)[:, None]

    return ProteinCNN()


def forward(model, batch, device, pos_weight):
    import torch
    encoded, mask, owner, weight, target = batch
    encoded, mask = encoded.to(device), mask.to(device)
    owner, weight, target = owner.to(device), weight.to(device), target.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(encoded, mask, owner, weight, len(target))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), target, pos_weight=pos_weight
        )
    return logits, target, loss


def evaluate(model, indices, token_arrays, labels, device, max_windows, pos_weight):
    import torch
    model.eval()
    probability: list[np.ndarray] = []
    losses: list[float] = []
    sizes: list[int] = []
    with torch.no_grad():
        for idx in iter_batches(indices, token_arrays, max_windows, 256, None):
            logits, _, loss = forward(model, collate(idx, token_arrays, labels), device, pos_weight)
            probability.append(torch.sigmoid(logits.float()).cpu().numpy())
            losses.append(float(loss.item()))
            sizes.append(len(idx))
    return np.concatenate(probability), float(np.average(losses, weights=sizes))


def macro_ap(y: np.ndarray, p: np.ndarray) -> float:
    values = [average_precision_score(y[:, j], p[:, j]) for j in range(y.shape[1])]
    return float(np.mean(values))


def fit(train_idx, val_idx, token_arrays, labels, seed, device, max_windows,
        n_labels, architecture, pos_weight, max_epochs, patience, fixed_epochs=None):
    import torch
    seed_all(seed)
    model = make_model(n_labels, architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target_epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    for epoch in range(1, target_epochs + 1):
        model.train()
        epoch_losses = []
        for idx in iter_batches(train_idx, token_arrays, max_windows, 256, seed * 10000 + epoch):
            optimizer.zero_grad(set_to_none=True)
            _, _, loss = forward(model, collate(idx, token_arrays, labels), device, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        if val_idx is None:
            history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses))})
            continue
        p_val, val_loss = evaluate(model, val_idx, token_arrays, labels, device, max_windows, pos_weight)
        score = macro_ap(labels[val_idx], p_val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)),
                        "val_loss": val_loss, "val_macro_auprc": score})
        if score > best_score + 1e-7:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if val_idx is not None:
        if best_state is None:
            raise RuntimeError("no validation checkpoint")
        model.load_state_dict(best_state)
    return model, (best_epoch if val_idx is not None else target_epochs), history


def choose_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    thresholds = np.full(y.shape[1], 0.5, dtype=float)
    for j in range(y.shape[1]):
        best = (-1.0, 0.5)
        for threshold in np.linspace(0.05, 0.95, 19):
            score = f1_score(y[:, j], p[:, j] >= threshold, zero_division=0)
            if score > best[0]:
                best = (float(score), float(threshold))
        thresholds[j] = best[1]
    return thresholds


def serialize(labels: list[str], values: np.ndarray, digits: int = 8) -> str:
    return ";".join(f"{label}={float(values[i]):.{digits}f}" for i, label in enumerate(labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label-tsv", required=True, type=Path)
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
    started = utc_now()
    model_name = args.architecture
    pred_path = args.out / "predictions" / model_name / f"seed{args.seed}" / f"fold{args.fold}.parquet"
    metric_path = args.out / "metrics" / f"vf_l2_{model_name}_seed{args.seed}_fold{args.fold}.json"
    if pred_path.exists() and metric_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "prediction": str(pred_path)}))
        return

    summary = json.loads((args.out / "qc" / "phase0_summary.json").read_text())
    labels_list = list(summary["atomic_l2_labels"])
    nested = pd.read_csv(args.out / "splits" / "vf_l2_nested_splits.tsv", sep="\t", dtype=str)
    split = nested[(nested.seed.astype(int) == args.seed) &
                   (nested.outer_fold.astype(int) == args.fold)].copy()
    source = pd.read_csv(args.label_tsv, sep="\t", dtype=str, low_memory=False)[
        ["record_id", "sequence", "level2_vf_type"]
    ]
    data = split.merge(source, on=["record_id", "level2_vf_type"], validate="one_to_one")
    data = data.sort_values("record_id").reset_index(drop=True)
    if len(data) != summary["l2_eligible"]:
        raise RuntimeError(f"split/source mismatch: {len(data)}")
    label_sets = data.level2_vf_type.map(tokens)
    y = np.asarray([[int(label in values) for label in labels_list] for values in label_sets], dtype=np.int8)
    token_arrays = [encode(x) for x in data.sequence.astype(str)]
    role = data.partition.to_numpy()
    inner_train = np.flatnonzero(role == "inner_train")
    inner_val = np.flatnonzero(role == "inner_val")
    outer_train = np.flatnonzero(role != "outer_test")
    outer_test = np.flatnonzero(role == "outer_test")
    positive = y[outer_train].sum(axis=0).astype(float)
    pos_weight_np = np.clip((len(outer_train) - positive) / np.maximum(positive, 1.0), 1.0, 20.0)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device="cuda")
    device = torch.device("cuda")

    inner_model, best_epoch, history = fit(
        inner_train, inner_val, token_arrays, y, args.seed, device, args.max_windows,
        len(labels_list), model_name, pos_weight, args.max_epochs, args.patience,
    )
    val_prob, _ = evaluate(inner_model, inner_val, token_arrays, y, device, args.max_windows, pos_weight)
    thresholds = choose_thresholds(y[inner_val], val_prob)
    del inner_model
    torch.cuda.empty_cache()
    final_model, _, final_history = fit(
        outer_train, None, token_arrays, y, args.seed, device, args.max_windows,
        len(labels_list), model_name, pos_weight, args.max_epochs, args.patience,
        fixed_epochs=best_epoch,
    )
    test_prob, _ = evaluate(final_model, outer_test, token_arrays, y, device, args.max_windows, pos_weight)
    y_test = y[outer_test]
    y_pred = (test_prob >= thresholds).astype(np.int8)
    per_ap = [average_precision_score(y_test[:, j], test_prob[:, j]) for j in range(len(labels_list))]
    metrics = {
        "head": "VF_L2", "output_semantics": "14_independent_sigmoid_logits",
        "loss": "BCEWithLogitsLoss", "weighted_bce": True,
        "model": model_name, "seed": args.seed, "outer_fold": args.fold,
        "n_inner_train": len(inner_train), "n_inner_val": len(inner_val),
        "n_outer_train": len(outer_train), "n_outer_test": len(outer_test),
        "best_epoch": best_epoch, "micro_auprc": float(average_precision_score(y_test, test_prob, average="micro")),
        "macro_auprc": float(np.mean(per_ap)),
        "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "sample_f1": float(f1_score(y_test, y_pred, average="samples", zero_division=0)),
        "lrap": float(label_ranking_average_precision_score(y_test, test_prob)),
        "thresholds": thresholds.tolist(), "pos_weight": pos_weight_np.tolist(),
        "window": WINDOW, "stride": STRIDE, "protein_level_loss": True,
        "start_time": started, "end_time": utc_now(),
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "record_id": data.iloc[outer_test].record_id.to_numpy(),
        "outer_fold": args.fold, "seed": args.seed, "model": model_name,
        "true_l2_labels": [";".join(labels_list[j] for j in np.flatnonzero(row)) for row in y_test],
        "prob_l2_labels": [serialize(labels_list, row) for row in test_prob],
        "pred_l2_labels": [";".join(labels_list[j] for j in np.flatnonzero(row)) for row in y_pred],
        "threshold_l2_labels": [serialize(labels_list, thresholds, 6)] * len(outer_test),
    }).to_parquet(pred_path, index=False)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n")
    history_path = args.out / "training_history" / model_name / f"seed{args.seed}_fold{args.fold}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({"inner": history, "final": final_history}, indent=2) + "\n")
    checkpoint = args.out / "checkpoints" / model_name / f"seed{args.seed}" / f"fold{args.fold}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
                "labels": labels_list, "metrics": metrics, "architecture": model_name}, checkpoint)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
