from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from common import atomic_json, read_fasta, read_json


ROUTING = {"ARG": "arg", "MRG": "bacmet", "MGE": "mge"}
FIELDS = [
    "sequence_id", "task", "semantics", "sequence_length", "n_windows", "n_models",
    "predicted_labels", "top1_label", "top1_probability", "probabilities", "thresholds",
]


def write_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n").writeheader()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run only the ARG, MRG/BACMET and MGE frozen ESM2-8M fine-tuned models.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--window-batch", type=int, default=8)
    args = parser.parse_args()
    config = read_json(args.config)
    fine = config["fine_models"]
    if fine["routing"] != ROUTING or set(fine["routing"]) != {"ARG", "MRG", "MGE"}:
        raise RuntimeError("fine-model routing violates no-VF contract")
    python = Path(fine["python"])
    predictor = Path(fine["predictor"])
    args.outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for klass, task in ROUTING.items():
        fasta = args.fasta_dir / f"{klass}.faa"
        sequences = read_fasta(fasta) if fasta.exists() and fasta.stat().st_size else {}
        destination = args.outdir / f"{klass}_8M_FINE.tsv"
        if not sequences:
            write_empty(destination)
            status = "EMPTY_CLASS"
        else:
            command = [
                str(python), str(predictor), "--task", task, "--fasta", str(fasta),
                "--out", str(destination), "--threads", str(args.threads),
                "--window-batch", str(args.window_batch), "--seeds",
                *[str(value) for value in fine["seeds"]], "--folds",
                *[str(value) for value in fine["folds"]],
            ]
            subprocess.run(command, check=True)
            status = "COMPLETE"
        records.append({
            "class": klass,
            "task": task,
            "sequences": len(sequences),
            "models": len(fine["seeds"]) * len(fine["folds"]),
            "status": status,
            "output": str(destination),
        })
    manifest = {
        "state": "FINE_MODELS_COMPLETE",
        "routing": ROUTING,
        "virulence_factor_branch_enabled": False,
        "records": records,
    }
    atomic_json(args.outdir / "FINE_MODELS_MANIFEST.json", manifest)
    (args.outdir / "FINE_MODELS_COMPLETE").touch()
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
