from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


PFAM_NUMERIC = [
    "w10_pfam_hit_fraction", "w10_pfam_domain_count",
    "w10_pfam_diversity", "w10_pfam_missing",
]
EGGNOG_NUMERIC = [
    "w10_eggnog_neighbor_count", "w10_eggnog_hit_fraction",
    "w10_cog_hit_fraction", "w10_go_hit_fraction", "w10_ec_hit_fraction",
    "w10_kegg_hit_fraction", "w10_cazy_hit_fraction",
]
CANDIDATES = {
    "PFAM_ONLY_W10": {"eggnog": False, "mag": False},
    "PFAM_EGGNOG_W10": {"eggnog": True, "mag": False},
    "PFAM_EGGNOG_MAG_ABC141": {"eggnog": True, "mag": True},
}
MAG_KEY_COLUMNS = {
    "row_index", "protein_id", "protein_key", "dataset", "genome_id",
    "contig_id", "start0", "end0", "strand",
}
DEFAULT_MAG_ROOT = Path(
    "/ifs1/User/yanyueyang/yyy/gene-finder/protein_negative_pipeline_runs/"
    "MAG_DNA_CONTEXT_V1_20261018"
)
ALLOWED_COHORTS = [
    "ORIGINAL_SUPERVISED", "MGNIFY_SAFE_W5_PSEUDO_NEGATIVE",
    "PRIMARY2000_POSITIVE", "PRIMARY2000_SAFE_W5_PSEUDO_NEGATIVE",
]


def import_library(path: Path):
    sys.path.insert(0, str(path.parent))
    return __import__(path.stem)


def load_all(core, root: Path, pfam_root: Path, eggnog_path: Path):
    manifest, pfam, esm8, esm650, context = core.load_common(root, pfam_root)
    eggnog = pd.read_parquet(eggnog_path)
    if len(eggnog) != len(manifest):
        raise RuntimeError("eggNOG and manifest row counts differ")
    if not np.array_equal(eggnog.row_index.to_numpy(), np.arange(len(manifest))):
        raise RuntimeError("eggNOG row_index is not canonical")
    if not np.array_equal(eggnog.protein_id.astype(str), manifest.protein_id.astype(str)):
        raise RuntimeError("eggNOG and manifest protein IDs differ")
    if not np.isfinite(eggnog[EGGNOG_NUMERIC].to_numpy(np.float32)).all():
        raise RuntimeError("eggNOG numerical features are not finite")
    return manifest, pfam, eggnog, esm8, esm650, context


def load_mag(path: Path, manifest: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if not (path.parent.parent / "FEATURE_EXTRACTION_COMPLETE").exists():
        raise RuntimeError("MAG DNA FEATURE_EXTRACTION_COMPLETE marker is absent")
    frame = pd.read_parquet(path)
    columns = [column for column in frame.columns if column not in MAG_KEY_COLUMNS]
    if len(frame) != len(manifest) or len(columns) != 141:
        raise RuntimeError("MAG DNA row/dimension contract failed")
    if not np.array_equal(frame.row_index.to_numpy(), np.arange(len(manifest))):
        raise RuntimeError("MAG DNA row_index is not canonical")
    if not np.array_equal(frame.protein_id.astype(str), manifest.protein_id.astype(str)):
        raise RuntimeError("MAG DNA protein IDs differ from ROW_MANIFEST")
    if frame.protein_key.duplicated().any():
        raise RuntimeError("MAG DNA protein_key is not unique")
    values = frame[columns].to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise RuntimeError("MAG DNA contains infinite values")
    return values, columns


def make_matrix(
    core,
    esm8: np.ndarray,
    esm650: np.ndarray,
    context: np.ndarray,
    pfam: pd.DataFrame,
    eggnog: pd.DataFrame,
    indices: np.ndarray,
    pfam_vocab: list[str],
    eggnog_vocab: list[str],
    mag_values: np.ndarray | None,
) -> np.ndarray:
    dimension = 1600 + context.shape[1] + len(PFAM_NUMERIC) + len(pfam_vocab)
    dimension += (len(EGGNOG_NUMERIC) + len(eggnog_vocab)) if eggnog_vocab else 0
    dimension += mag_values.shape[1] if mag_values is not None else 0
    matrix = np.empty((len(indices), dimension), dtype=np.float32)
    cursor = 0
    matrix[:, cursor:cursor + 320] = esm8[indices]
    cursor += 320
    matrix[:, cursor:cursor + 1280] = esm650[indices]
    cursor += 1280
    matrix[:, cursor:cursor + context.shape[1]] = context[indices]
    cursor += context.shape[1]
    matrix[:, cursor:cursor + len(PFAM_NUMERIC)] = pfam.iloc[indices][PFAM_NUMERIC].to_numpy(np.float32)
    cursor += len(PFAM_NUMERIC)
    matrix[:, cursor:cursor + len(pfam_vocab)] = 0.0
    core.fill_binary(
        matrix, cursor,
        pfam.iloc[indices].w10_neighbor_pfam_tokens.reset_index(drop=True),
        pfam_vocab,
    )
    cursor += len(pfam_vocab)
    if eggnog_vocab:
        matrix[:, cursor:cursor + len(EGGNOG_NUMERIC)] = eggnog.iloc[indices][EGGNOG_NUMERIC].to_numpy(np.float32)
        cursor += len(EGGNOG_NUMERIC)
        matrix[:, cursor:cursor + len(eggnog_vocab)] = 0.0
        core.fill_binary(
            matrix, cursor,
            eggnog.iloc[indices].w10_neighbor_eggnog_tokens.reset_index(drop=True),
            eggnog_vocab,
        )
        cursor += len(eggnog_vocab)
    if mag_values is not None:
        matrix[:, cursor:cursor + mag_values.shape[1]] = mag_values[indices]
        cursor += mag_values.shape[1]
    if cursor != dimension or np.isinf(matrix).any():
        raise RuntimeError("assembled formal matrix violates dimension/infinity contract")
    if mag_values is None and not np.isfinite(matrix).all():
        raise RuntimeError("non-MAG formal matrix contains missing values")
    return matrix


def reference(core, root: Path, output_root: Path, candidate: str, labels: np.ndarray):
    if candidate == "PFAM_ONLY_W10":
        directory = root / "experiments" / "P1_8M650M_1884"
        result = json.loads((directory / "RESULT.json").read_text())
        scores = np.load(directory / "ORIGINAL_OOF_SCORES.f64.npy")
        name = "P1_8M650M_1884"
    elif candidate == "PFAM_EGGNOG_W10":
        directory = output_root / "experiments" / "PFAM_ONLY_W10"
        result = json.loads((directory / "RESULT.json").read_text())
        scores = np.load(directory / "ORIGINAL_OOF_SCORES.f64.npy")
        name = "PFAM_ONLY_W10"
    else:
        directory = output_root / "experiments" / "PFAM_EGGNOG_W10"
        result = json.loads((directory / "RESULT.json").read_text())
        scores = np.load(directory / "ORIGINAL_OOF_SCORES.f64.npy")
        name = "PFAM_EGGNOG_W10"
    metric = result["original_grouped_oof"]
    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise RuntimeError(f"invalid comparison OOF for {name}")
    return name, metric, scores


def train(args: argparse.Namespace) -> None:
    core = import_library(Path(args.library))
    root = Path(args.root)
    pfam_root = Path(args.pfam_root)
    output_root = Path(args.output)
    output = output_root / "experiments" / args.candidate
    output.mkdir(parents=True, exist_ok=True)



    train_lock = (output / "TRAIN.lock").open("w")
    fcntl.flock(train_lock.fileno(), fcntl.LOCK_EX)
    if (output / "COMPLETE").exists():
        print((output / "RESULT.json").read_text(), end="")
        return
    manifest, pfam, eggnog, esm8, esm650, context = load_all(
        core, root, pfam_root, Path(args.eggnog_features)
    )
    spec = CANDIDATES[args.candidate]
    include_eggnog = spec["eggnog"]
    include_mag = spec["mag"]
    if args.candidate != "PFAM_ONLY_W10" and not (
        DEFAULT_MAG_ROOT / "FEATURE_EXTRACTION_COMPLETE"
    ).exists():
        raise RuntimeError("MAG DNA must complete before augmented models start")
    if include_mag:
        if not args.mag_features:
            raise RuntimeError("--mag-features is required for the MAG candidate")
        mag_values, mag_columns = load_mag(Path(args.mag_features), manifest)
    else:
        mag_values, mag_columns = None, []
    cohorts = manifest.cohort.astype(str)
    allowed = cohorts.isin(ALLOWED_COHORTS).to_numpy()
    folds = manifest.new_mag_fold.to_numpy(np.int8)
    labels = manifest.label.to_numpy(np.int8)
    original_indices = np.flatnonzero(cohorts.eq("ORIGINAL_SUPERVISED"))
    primary_indices = np.flatnonzero(allowed & cohorts.str.startswith("PRIMARY2000").to_numpy())
    original_scores = np.full(len(original_indices), np.nan, dtype=np.float64)
    primary_scores = np.full(len(manifest), np.nan, dtype=np.float64)
    checkpoints = output / "fold_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    params = dict(core.LGB_PARAMS)
    params["n_jobs"] = int(args.threads)
    fold_meta = []
    for fold in range(5):
        paths = {
            "scores": checkpoints / f"fold{fold}.original_scores.f64.npy",
            "positions": checkpoints / f"fold{fold}.original_positions.i64.npy",
            "primary_scores": checkpoints / f"fold{fold}.primary_scores.f64.npy",
            "primary_indices": checkpoints / f"fold{fold}.primary_indices.i64.npy",
            "meta": checkpoints / f"fold{fold}.json",
            "model": checkpoints / f"fold{fold}.model.txt",
            "vocab": checkpoints / f"fold{fold}.vocab.json",
        }
        if all(path.exists() for path in paths.values()):
            positions = np.load(paths["positions"])
            original_scores[positions] = np.load(paths["scores"])
            heldout_primary = np.load(paths["primary_indices"])
            primary_scores[heldout_primary] = np.load(paths["primary_scores"])
            fold_meta.append(json.loads(paths["meta"].read_text()))
            continue
        train_index = np.flatnonzero(allowed & (folds != fold))
        valid_positions = np.flatnonzero(folds[original_indices] == fold)
        valid_index = original_indices[valid_positions]
        heldout_primary = primary_indices[folds[primary_indices] == fold]
        weights, weight_audit = core.fold_weights(manifest, train_index)
        pfam_vocab = core.top_vocab(pfam.iloc[train_index].w10_neighbor_pfam_tokens, 256)
        eggnog_vocab = core.top_vocab(
            eggnog.iloc[train_index].w10_neighbor_eggnog_tokens, 512
        ) if include_eggnog else []
        x_train = make_matrix(
            core, esm8, esm650, context, pfam, eggnog, train_index,
            pfam_vocab, eggnog_vocab, mag_values,
        )
        x_valid = make_matrix(
            core, esm8, esm650, context, pfam, eggnog, valid_index,
            pfam_vocab, eggnog_vocab, mag_values,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_train, labels[train_index], sample_weight=weights,
            eval_set=[(x_valid, labels[valid_index])], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        best_iteration = int(model.best_iteration_ or params["n_estimators"])
        fold_scores = model.predict_proba(x_valid, num_iteration=best_iteration)[:, 1].astype(np.float64)
        del x_train, x_valid
        gc.collect()
        if len(heldout_primary):
            x_primary = make_matrix(
                core, esm8, esm650, context, pfam, eggnog, heldout_primary,
                pfam_vocab, eggnog_vocab, mag_values,
            )
            heldout_scores = model.predict_proba(
                x_primary, num_iteration=best_iteration
            )[:, 1].astype(np.float64)
            del x_primary
        else:
            heldout_scores = np.empty(0, dtype=np.float64)
        model.booster_.save_model(str(paths["model"]), num_iteration=best_iteration)
        feature_dimension = 1600 + 44 + len(PFAM_NUMERIC) + len(pfam_vocab)
        if include_eggnog:
            feature_dimension += len(EGGNOG_NUMERIC) + len(eggnog_vocab)
        if include_mag:
            feature_dimension += len(mag_columns)
        core.atomic_json(paths["vocab"], {
            "pfam_w10_vocab": pfam_vocab,
            "eggnog_w10_vocab": eggnog_vocab,
            "pfam_numeric_columns": PFAM_NUMERIC,
            "eggnog_numeric_columns": EGGNOG_NUMERIC if include_eggnog else [],
            "mag_columns": mag_columns,
            "feature_dimension": feature_dimension,
        })
        cohort_counts = manifest.iloc[train_index].cohort.astype(str).value_counts().sort_index()
        meta = {
            "fold": fold, "best_iteration": best_iteration,
            "training_rows": int(len(train_index)),
            "validation_rows": int(len(valid_index)),
            "primary_diagnostic_rows": int(len(heldout_primary)),
            "pfam_vocab_size": len(pfam_vocab),
            "eggnog_vocab_size": len(eggnog_vocab),
            "mag_feature_count": len(mag_columns),
            "feature_dimension": feature_dimension,
            "cohort_counts": {str(k): int(v) for k, v in cohort_counts.items()},
            "weight_audit": weight_audit,
            "old_fold0_used": False,
        }
        core.atomic_npy(paths["scores"], fold_scores)
        core.atomic_npy(paths["positions"], valid_positions.astype(np.int64))
        core.atomic_npy(paths["primary_scores"], heldout_scores)
        core.atomic_npy(paths["primary_indices"], heldout_primary.astype(np.int64))
        core.atomic_json(paths["meta"], meta)
        original_scores[valid_positions] = fold_scores
        primary_scores[heldout_primary] = heldout_scores
        fold_meta.append(meta)
        print(json.dumps({"event": "fold_complete", "candidate": args.candidate, **meta}, sort_keys=True), flush=True)
        del model
        gc.collect()
    if not np.isfinite(original_scores).all():
        raise RuntimeError("original grouped OOF is incomplete")
    if not np.isfinite(primary_scores[primary_indices]).all():
        raise RuntimeError("PRIMARY grouped diagnostics are incomplete")
    original_labels = labels[original_indices]
    threshold = core.threshold_max_mcc(original_labels, original_scores)
    own = core.metrics(original_labels, original_scores, threshold)
    per_fold = {}
    for fold in range(5):
        mask = folds[original_indices] == fold
        per_fold[str(fold)] = core.metrics(original_labels[mask], original_scores[mask], threshold)
    ref_name, ref_metric, ref_scores = reference(core, root, output_root, args.candidate, original_labels)
    matched_reference = core.metrics(
        original_labels, original_scores,
        core.threshold_at_recall(original_labels, original_scores, ref_metric["recall"]),
    )
    matched_legacy = core.metrics(
        original_labels, original_scores,
        core.threshold_at_recall(original_labels, original_scores, 0.8945674983),
    )
    ref_matched_legacy = core.metrics(
        original_labels, ref_scores,
        core.threshold_at_recall(original_labels, ref_scores, 0.8945674983),
    )
    passed = (
        own["auprc"] - ref_metric["auprc"] >= -0.002
        and matched_reference["fpr"] - ref_metric["fpr"] <= 0.001
    )
    primary_labels = labels[primary_indices]
    primary_metric = core.metrics(primary_labels, primary_scores[primary_indices], threshold)
    core.atomic_npy(output / "ORIGINAL_OOF_SCORES.f64.npy", original_scores)
    core.atomic_npy(output / "PRIMARY_GROUPED_DIAGNOSTIC_SCORES.f64.npy", primary_scores)
    result = {
        "state": "FORMAL_EXPERIMENT_COMPLETE",
        "candidate": args.candidate,
        "feature_contract": {
            "center_esm2": "8M_layer6_mean_320_plus_650M_layer33_mean_1280",
            "portable_context": "44D coordinate-safe protein-length and relative-order features",
            "pfam": "W10 numeric plus fold-training-only 256-token identifier vocabulary",
            "eggnog": (
                "W10 numeric plus fold-training-only 512-token normalized identifier vocabulary"
                if include_eggnog else "not used"
            ),
            "mag_dna": (
                "ABC141 missing-aware coordinate-audited features; NaN retained as missing"
                if include_mag else "not used"
            ),
            "free_text_used": False,
            "coordinates_used": bool(include_mag),
            "pseudo_coordinates_used": False,
            "neighbor_esm2_used": False,
        },
        "folds": fold_meta,
        "original_grouped_oof": own,
        "original_per_fold": per_fold,
        "primary_grouped_diagnostic": {
            "rows": int(len(primary_indices)),
            "positive": int(primary_labels.sum()),
            "negative": int(len(primary_labels) - primary_labels.sum()),
            "interpretation": "grouped training-distribution diagnostic only; not external truth",
            **primary_metric,
        },
        "reference": ref_name,
        "reference_metric": ref_metric,
        "delta_auprc_vs_reference": float(own["auprc"] - ref_metric["auprc"]),
        "delta_mcc_vs_reference": float(own["mcc"] - ref_metric["mcc"]),
        "matched_reference_recall": matched_reference,
        "delta_matched_reference_fpr": float(matched_reference["fpr"] - ref_metric["fpr"]),
        "matched_legacy_m4_recall": matched_legacy,
        "reference_matched_legacy_m4_recall": ref_matched_legacy,
        "predeclared_no_worse": {
            "auprc_delta_minimum": -0.002,
            "matched_reference_recall_fpr_delta_maximum": 0.001,
            "pass": bool(passed),
        },
        "rows_allowed": int(allowed.sum()),
        "excluded_primary_w10_only_rows": int((cohorts == "PRIMARY2000_SAFE_W10_ONLY_PSEUDO_NEGATIVE").sum()),
        "old_fold0_used": False,
        "external_test_used_for_selection": False,
    }
    core.atomic_json(output / "RESULT.json", result)
    (output / "COMPLETE").touch()
    print(json.dumps(result, sort_keys=True), flush=True)


def select(args: argparse.Namespace) -> None:
    core = import_library(Path(args.library))
    output = Path(args.output)
    control = json.loads((output / "experiments/PFAM_ONLY_W10/RESULT.json").read_text())
    augmented = json.loads((output / "experiments/PFAM_EGGNOG_W10/RESULT.json").read_text())
    mag = json.loads((output / "experiments/PFAM_EGGNOG_MAG_ABC141/RESULT.json").read_text())
    if mag["predeclared_no_worse"]["pass"]:
        winner = "PFAM_EGGNOG_MAG_ABC141"
        reason = "MAG DNA augmentation passed the predeclared no-worse comparison against Pfam+eggNOG"
    elif augmented["predeclared_no_worse"]["pass"]:
        winner = "PFAM_EGGNOG_W10"
        reason = "MAG DNA failed no-worse; eggNOG augmentation passed against the locked Pfam-only control"
    elif control["predeclared_no_worse"]["pass"]:
        winner = "PFAM_ONLY_W10"
        reason = "eggNOG augmentation failed no-worse; locked Pfam-only control passed versus the 8M+650M baseline"
    else:
        winner = "P1_8M650M_1884_BASELINE"
        reason = "neither formal extension passed its predeclared no-worse comparison"
    summary = {
        "state": "FORMAL_SELECTION_COMPLETE",
        "winner": winner,
        "reason": reason,
        "selection_uses_old_fold0": False,
        "selection_uses_external_test": False,
        "comparisons": {
            name: {
                "reference": result["reference"],
                "pass": result["predeclared_no_worse"]["pass"],
                "original_grouped_oof": result["original_grouped_oof"],
                "matched_reference_recall": result["matched_reference_recall"],
                "delta_auprc_vs_reference": result["delta_auprc_vs_reference"],
                "delta_mcc_vs_reference": result["delta_mcc_vs_reference"],
            }
            for name, result in [
                ("PFAM_ONLY_W10", control),
                ("PFAM_EGGNOG_W10", augmented),
                ("PFAM_EGGNOG_MAG_ABC141", mag),
            ]
        },
    }
    core.atomic_json(output / "FINAL_RESULT.json", summary)
    (output / "FORMAL_MODEL_COMPLETE").touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--root", required=True)
    train_parser.add_argument("--pfam-root", required=True)
    train_parser.add_argument("--eggnog-features", required=True)
    train_parser.add_argument("--mag-features")
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--library", required=True)
    train_parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    train_parser.add_argument("--threads", type=int, default=30)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--output", required=True)
    select_parser.add_argument("--library", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        select(args)


if __name__ == "__main__":
    main()
