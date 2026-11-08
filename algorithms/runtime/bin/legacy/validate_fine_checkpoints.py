from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import atomic_json, read_json


TASKS = {
    "ARG": ("arg_category_downstream_v4", "ESM2-8M-FT-WBCE", 15, "15_independent_sigmoid_logits"),
    "MRG": ("bacmet_metal_downstream_v1", "ESM2-8M-FT-WBCE", 23, "23_independent_sigmoid_logits"),
    "MGE": ("mge_7class_downstream_v1", "ESM2-8M-FT-WCE", 7, "7_class_softmax"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and audit all 45 frozen ESM2-8M fine-tuned checkpoints.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    fine = config["fine_models"]
    vault = Path(fine["vault_root"])
    records = []
    labels_by_class: dict[str, list[str]] = {}
    for klass, (artifact, model_name, n_labels, semantics) in TASKS.items():
        for seed in fine["seeds"]:
            for fold in fine["folds"]:
                path = vault / "artifacts" / artifact / "checkpoints" / model_name / f"seed{seed}" / f"fold{fold}.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                labels = list(checkpoint["labels"])
                metrics = checkpoint["metrics"]
                if len(labels) != n_labels or metrics.get("output_semantics") != semantics:
                    raise RuntimeError(f"label/semantics contract failed: {path}")
                if metrics.get("encoder_frozen") is not False or metrics.get("outer_test_used_for_selection") is not False:
                    raise RuntimeError(f"fine-tuning or selection boundary changed: {path}")
                if metrics.get("model") != model_name:
                    raise RuntimeError(f"checkpoint model name changed: {path}")
                if klass in labels_by_class and labels_by_class[klass] != labels:
                    raise RuntimeError(f"label order differs within {klass}: {path}")
                labels_by_class[klass] = labels
                if klass in {"ARG", "MRG"} and len(metrics.get("thresholds", [])) != n_labels:
                    raise RuntimeError(f"multilabel threshold count changed: {path}")
                records.append({"class": klass, "seed": seed, "fold": fold, "labels": n_labels, "semantics": semantics})
    result = {
        "state": "FINE_CHECKPOINTS_VALIDATED",
        "checkpoint_count": len(records),
        "models_per_class": len(fine["seeds"]) * len(fine["folds"]),
        "encoder_frozen": False,
        "outer_test_used_for_selection": False,
        "labels": labels_by_class,
        "records": records,
    }
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ["state", "checkpoint_count", "models_per_class", "encoder_frozen"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
