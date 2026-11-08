from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, EsmModel


WINDOW = 1022
STRIDE = 512
ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "arg": ("arg_category_downstream_v4", "ESM2-8M-FT-WBCE", "multilabel"),
    "mge": ("mge_7class_downstream_v1", "ESM2-8M-FT-WCE", "multiclass"),
    "bacmet": ("bacmet_metal_downstream_v1", "ESM2-8M-FT-WBCE", "multilabel"),
}


def read_fasta(path: Path):
    name, chunks = None, []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks).upper()
                name, chunks = line[1:].split()[0], []
            else:
                if name is None:
                    raise ValueError("FASTA sequence found before first header")
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks).upper()


def fragments(sequence: str):
    if not sequence:
        raise ValueError("empty sequence")
    if len(sequence) <= WINDOW:
        return [sequence]
    starts = list(range(0, len(sequence) - WINDOW + 1, STRIDE))
    last = len(sequence) - WINDOW
    if starts[-1] != last:
        starts.append(last)
    return [sequence[start:start + WINDOW] for start in starts]


class ProteinESM(nn.Module):
    def __init__(self, model_dir: Path, n_labels: int):
        super().__init__()
        self.esm = EsmModel.from_pretrained(
            str(model_dir), local_files_only=True, add_pooling_layer=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.esm.config.hidden_size),
            nn.Dropout(0.20),
            nn.Linear(self.esm.config.hidden_size, n_labels),
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        residue_mask = attention_mask.bool().clone()
        residue_mask[:, 0] = False
        last = attention_mask.sum(dim=1).long() - 1
        residue_mask[torch.arange(len(last)), last] = False
        valid = residue_mask.unsqueeze(-1).to(hidden.dtype)
        return self.head((hidden * valid).sum(1) / valid.sum(1).clamp_min(1.0))


def load_models(task: str, seeds: list[int], folds: list[int], threads: int):
    rel, model_name, semantics = TASKS[task]
    paths = [
        ROOT / "artifacts" / rel / "checkpoints" / model_name / f"seed{seed}" / f"fold{fold}.pt"
        for seed in seeds for fold in folds
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    torch.set_num_threads(threads)
    model_dir = ROOT / "models" / "esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    models, thresholds, labels = [], [], None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        current_labels = list(checkpoint["labels"])
        if labels is None:
            labels = current_labels
        elif labels != current_labels:
            raise RuntimeError(f"Label order mismatch in {path}")
        model = ProteinESM(model_dir, len(labels))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        models.append(model)
        if semantics == "multilabel":
            thresholds.append(np.asarray(checkpoint["metrics"]["thresholds"], dtype=np.float32))
    return models, tokenizer, labels, semantics, thresholds


def predict_one(sequence: str, models, tokenizer, semantics: str, window_batch: int):
    pieces = fragments(sequence)
    weights = torch.tensor([len(piece) for piece in pieces], dtype=torch.float32)
    all_probabilities = []
    with torch.inference_mode():
        for model in models:
            logits = []
            for start in range(0, len(pieces), window_batch):
                encoded = tokenizer(
                    pieces[start:start + window_batch], return_tensors="pt",
                    padding=True, add_special_tokens=True,
                )
                logits.append(model(encoded["input_ids"], encoded["attention_mask"]))
            window_logits = torch.cat(logits)
            protein_logits = (window_logits * weights[:, None]).sum(0) / weights.sum()
            probability = (
                torch.softmax(protein_logits, dim=0)
                if semantics == "multiclass"
                else torch.sigmoid(protein_logits)
            )
            all_probabilities.append(probability.numpy())
    return np.mean(all_probabilities, axis=0), len(pieces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--window-batch", type=int, default=8)
    args = parser.parse_args()

    models, tokenizer, labels, semantics, thresholds = load_models(
        args.task, args.seeds, args.folds, args.threads
    )
    threshold = np.mean(thresholds, axis=0) if thresholds else None
    rows = []
    for sequence_id, sequence in read_fasta(args.fasta):
        probability, n_windows = predict_one(
            sequence, models, tokenizer, semantics, args.window_batch
        )
        top = int(np.argmax(probability))
        if semantics == "multiclass":
            predicted = labels[top]
        else:
            predicted = ";".join(
                labels[index] for index in np.flatnonzero(probability >= threshold)
            )
        rows.append({
            "sequence_id": sequence_id,
            "task": args.task,
            "semantics": semantics,
            "sequence_length": len(sequence),
            "n_windows": n_windows,
            "n_models": len(models),
            "predicted_labels": predicted,
            "top1_label": labels[top],
            "top1_probability": float(probability[top]),
            "probabilities": ";".join(
                f"{label}={float(probability[index]):.8f}" for index, label in enumerate(labels)
            ),
            "thresholds": "" if threshold is None else ";".join(
                f"{label}={float(threshold[index]):.6f}" for index, label in enumerate(labels)
            ),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    print(f"wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
