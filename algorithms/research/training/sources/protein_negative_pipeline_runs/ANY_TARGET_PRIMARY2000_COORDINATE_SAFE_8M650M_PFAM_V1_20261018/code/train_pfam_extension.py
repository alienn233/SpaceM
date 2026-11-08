from __future__ import annotations

import argparse
import gc
import json
import os
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20261017
LGB_PARAMS = {
    "objective": "binary",
    "device_type": "cpu",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "n_estimators": 2500,
    "min_child_samples": 100,
    "max_bin": 127,
    "random_state": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}
CANDIDATES = {
    "PFAM_W10": {
        "center_tokens": False,
        "w10_tokens": True,
        "numeric": [
            "w10_pfam_hit_fraction", "w10_pfam_domain_count",
            "w10_pfam_diversity", "w10_pfam_missing",
        ],
    },
    "PFAM_CENTER_W10": {
        "center_tokens": True,
        "w10_tokens": True,
        "numeric": [
            "center_pfam_hit", "center_pfam_match_count", "center_pfam_diversity",
            "w10_pfam_hit_fraction", "w10_pfam_domain_count",
            "w10_pfam_diversity", "w10_pfam_missing",
        ],
    },
}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    denominator = np.sqrt(
        np.float64(tp + fp) * np.float64(tp + fn)
        * np.float64(tn + fp) * np.float64(tn + fn)
    )
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": float(recall), "precision": float(precision),
        "specificity": float(specificity), "fpr": float(1.0 - specificity),
        "accuracy": float((tp + tn) / len(labels)), "mcc": float(mcc),
    }


def threshold_max_mcc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order].astype(np.int8)
    tp = np.cumsum(ordered_labels, dtype=np.int64)
    fp = np.cumsum(1 - ordered_labels, dtype=np.int64)
    ends = np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
    tp = tp[ends].astype(np.float64)
    fp = fp[ends].astype(np.float64)
    fn = np.float64(labels.sum()) - tp
    tn = np.float64(len(labels) - labels.sum()) - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn, denominator,
        out=np.full_like(denominator, -np.inf), where=denominator > 0,
    )
    return float(ordered_scores[ends][int(np.argmax(mcc))])


def threshold_at_recall(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    positives = np.sort(scores[labels == 1])[::-1]
    required = max(1, int(np.ceil(target * len(positives))))
    return float(positives[required - 1])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    value = confusion(labels, scores, threshold)
    value["auprc"] = float(average_precision_score(labels, scores))
    value["auroc"] = float(roc_auc_score(labels, scores))
    return value


def top_vocab(texts: pd.Series, limit: int) -> list[str]:
    counter: Counter = Counter()
    for text in texts.fillna(""):
        if text:
            counter.update(set(str(text).split("|")))
    return [token for token, _count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def fold_weights(manifest: pd.DataFrame, train: np.ndarray) -> tuple[np.ndarray, dict]:
    frame = manifest.iloc[train]
    weights = frame.training_weight.to_numpy(dtype=np.float32).copy()
    cohorts = frame.cohort.astype(str).to_numpy()
    weights[cohorts == "PRIMARY2000_SAFE_W5_PSEUDO_NEGATIVE"] = np.float32(0.5)
    primary_positive = cohorts == "PRIMARY2000_POSITIVE"
    original_positive = (
        (cohorts == "ORIGINAL_SUPERVISED")
        & frame.label.to_numpy(dtype=np.int8).astype(bool)
    )
    if primary_positive.any():
        local = frame.loc[
            primary_positive, ["family_id", "assembly_accession", "source_multiplier"]
        ].copy()
        family_count = local.groupby("family_id").family_id.transform("size").to_numpy(dtype=np.float64)
        genome_count = local.groupby("assembly_accession").assembly_accession.transform("size").to_numpy(dtype=np.float64)
        raw = local.source_multiplier.to_numpy(dtype=np.float64) / np.sqrt(family_count) / np.sqrt(genome_count)
        target_mass = 0.25 * float(weights[original_positive].sum())
        balanced = raw * (target_mass / raw.sum())
        weights[primary_positive] = balanced.astype(np.float32)
    else:
        target_mass = 0.0
    return weights, {
        "original_positive_weight": float(weights[original_positive].sum()),
        "primary_positive_weight": float(weights[primary_positive].sum()),
        "primary_positive_target_weight": float(target_mass),
        "primary_positive_rows": int(primary_positive.sum()),
        "primary_w5_rows": int(np.sum(cohorts == "PRIMARY2000_SAFE_W5_PSEUDO_NEGATIVE")),
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
    }


def fill_binary(
    matrix: np.ndarray,
    start: int,
    texts: pd.Series,
    vocab: list[str],
) -> None:
    if not vocab:
        return
    lookup = {token: index for index, token in enumerate(vocab)}
    for row, text in enumerate(texts.fillna("")):
        if not text:
            continue
        columns = {lookup[token] for token in str(text).split("|") if token in lookup}
        if columns:
            matrix[row, start + np.fromiter(columns, dtype=np.int64)] = 1.0


def make_matrix(
    esm8: np.ndarray,
    esm650: np.ndarray,
    context: np.ndarray,
    pfam: pd.DataFrame,
    indices: np.ndarray,
    spec: dict,
    center_vocab: list[str],
    w10_vocab: list[str],
) -> np.ndarray:
    numeric = spec["numeric"]
    dimension = 1600 + context.shape[1] + len(numeric) + len(center_vocab) + len(w10_vocab)
    matrix = np.empty((len(indices), dimension), dtype=np.float32)
    cursor = 0
    matrix[:, cursor:cursor + 320] = esm8[indices]
    cursor += 320
    matrix[:, cursor:cursor + 1280] = esm650[indices]
    cursor += 1280
    matrix[:, cursor:cursor + context.shape[1]] = context[indices]
    cursor += context.shape[1]
    matrix[:, cursor:cursor + len(numeric)] = pfam.iloc[indices][numeric].to_numpy(dtype=np.float32)
    cursor += len(numeric)
    if center_vocab:
        matrix[:, cursor:cursor + len(center_vocab)] = 0.0
        fill_binary(
            matrix, cursor,
            pfam.iloc[indices].center_pfam_tokens.reset_index(drop=True),
            center_vocab,
        )
        cursor += len(center_vocab)
    if w10_vocab:
        matrix[:, cursor:cursor + len(w10_vocab)] = 0.0
        fill_binary(
            matrix, cursor,
            pfam.iloc[indices].w10_neighbor_pfam_tokens.reset_index(drop=True),
            w10_vocab,
        )
        cursor += len(w10_vocab)
    if cursor != dimension or not np.isfinite(matrix).all():
        raise RuntimeError("assembled candidate matrix violates shape/finite contract")
    return matrix


def load_common(root: Path, pfam_root: Path):
    manifest = pd.read_parquet(root / "ROW_MANIFEST.parquet")
    pfam = pd.read_parquet(pfam_root / "PFAM_FEATURES.parquet")
    if len(pfam) != len(manifest):
        raise RuntimeError("Pfam and manifest row counts differ")
    if not np.array_equal(pfam.row_index.to_numpy(), np.arange(len(manifest))):
        raise RuntimeError("Pfam row_index is not canonical")
    if not np.array_equal(pfam.protein_id.astype(str), manifest.protein_id.astype(str)):
        raise RuntimeError("Pfam and manifest protein IDs differ")
    esm8 = np.load(root / "CENTER_ESM2_8M.f16.npy", mmap_mode="r")
    esm650 = np.load(root / "CENTER_ESM2_650M.f16.npy", mmap_mode="r")
    context = np.load(root / "PORTABLE_CONTEXT.f32.npy", mmap_mode="r")
    if esm8.shape != (len(manifest), 320) or esm8.dtype != np.float16:
        raise RuntimeError("8M matrix contract failed")
    if esm650.shape != (len(manifest), 1280) or esm650.dtype != np.float16:
        raise RuntimeError("650M matrix contract failed")
    if context.shape != (len(manifest), 44) or context.dtype != np.float32:
        raise RuntimeError("portable context contract failed")
    return manifest, pfam, esm8, esm650, context


def train(args: argparse.Namespace) -> None:
    root = Path(args.root)
    pfam_root = Path(args.pfam_root)
    output_root = Path(args.output)
    name = args.candidate
    spec = CANDIDATES[name]
    output = output_root / "experiments" / name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETE").exists():
        print((output / "RESULT.json").read_text(), end="")
        return
    manifest, pfam, esm8, esm650, context = load_common(root, pfam_root)
    cohorts = manifest.cohort.astype(str)
    allowed = cohorts.isin([
        "ORIGINAL_SUPERVISED", "MGNIFY_SAFE_W5_PSEUDO_NEGATIVE",
        "PRIMARY2000_POSITIVE", "PRIMARY2000_SAFE_W5_PSEUDO_NEGATIVE",
    ]).to_numpy()
    folds = manifest.new_mag_fold.to_numpy(dtype=np.int8)
    labels = manifest.label.to_numpy(dtype=np.int8)
    original_indices = np.flatnonzero(cohorts.eq("ORIGINAL_SUPERVISED"))
    primary_indices = np.flatnonzero(
        allowed & cohorts.str.startswith("PRIMARY2000").to_numpy()
    )
    original_scores = np.full(len(original_indices), np.nan, dtype=np.float64)
    primary_scores = np.full(len(manifest), np.nan, dtype=np.float64)
    checkpoints = output / "fold_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    fold_meta = []
    params = dict(LGB_PARAMS)
    params["n_jobs"] = int(args.threads)
    for fold in range(5):
        score_path = checkpoints / f"fold{fold}.original_scores.f64.npy"
        position_path = checkpoints / f"fold{fold}.original_positions.i64.npy"
        primary_score_path = checkpoints / f"fold{fold}.primary_scores.f64.npy"
        primary_index_path = checkpoints / f"fold{fold}.primary_indices.i64.npy"
        meta_path = checkpoints / f"fold{fold}.json"
        model_path = checkpoints / f"fold{fold}.model.txt"
        vocab_path = checkpoints / f"fold{fold}.vocab.json"
        required = [
            score_path, position_path, primary_score_path, primary_index_path,
            meta_path, model_path, vocab_path,
        ]
        if all(path.exists() for path in required):
            positions = np.load(position_path)
            original_scores[positions] = np.load(score_path)
            heldout_primary = np.load(primary_index_path)
            primary_scores[heldout_primary] = np.load(primary_score_path)
            fold_meta.append(json.loads(meta_path.read_text()))
            continue
        train_index = np.flatnonzero(allowed & (folds != fold))
        valid_positions = np.flatnonzero(folds[original_indices] == fold)
        valid_index = original_indices[valid_positions]
        heldout_primary = primary_indices[folds[primary_indices] == fold]
        weights, weight_audit = fold_weights(manifest, train_index)
        center_vocab = top_vocab(
            pfam.iloc[train_index].center_pfam_tokens, 256
        ) if spec["center_tokens"] else []
        w10_vocab = top_vocab(
            pfam.iloc[train_index].w10_neighbor_pfam_tokens, 256
        ) if spec["w10_tokens"] else []
        x_train = make_matrix(
            esm8, esm650, context, pfam, train_index, spec, center_vocab, w10_vocab
        )
        x_valid = make_matrix(
            esm8, esm650, context, pfam, valid_index, spec, center_vocab, w10_vocab
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_train, labels[train_index], sample_weight=weights,
            eval_set=[(x_valid, labels[valid_index])], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        best_iteration = int(model.best_iteration_ or params["n_estimators"])
        fold_scores = model.predict_proba(
            x_valid, num_iteration=best_iteration
        )[:, 1].astype(np.float64)
        del x_train, x_valid
        gc.collect()
        if len(heldout_primary):
            x_primary = make_matrix(
                esm8, esm650, context, pfam, heldout_primary,
                spec, center_vocab, w10_vocab,
            )
            heldout_scores = model.predict_proba(
                x_primary, num_iteration=best_iteration
            )[:, 1].astype(np.float64)
            del x_primary
        else:
            heldout_scores = np.empty(0, dtype=np.float64)
        model.booster_.save_model(str(model_path), num_iteration=best_iteration)
        atomic_json(vocab_path, {
            "center_vocab": center_vocab, "w10_vocab": w10_vocab,
            "numeric_columns": spec["numeric"],
            "feature_dimension": 1600 + context.shape[1] + len(spec["numeric"])
            + len(center_vocab) + len(w10_vocab),
        })
        meta = {
            "fold": fold, "best_iteration": best_iteration,
            "training_rows": int(len(train_index)),
            "validation_rows": int(len(valid_index)),
            "primary_diagnostic_rows": int(len(heldout_primary)),
            "center_vocab_size": len(center_vocab),
            "w10_vocab_size": len(w10_vocab),
            "weight_audit": weight_audit,
            "old_fold0_used": False,
        }
        atomic_npy(score_path, fold_scores)
        atomic_npy(position_path, valid_positions.astype(np.int64))
        atomic_npy(primary_score_path, heldout_scores)
        atomic_npy(primary_index_path, heldout_primary.astype(np.int64))
        atomic_json(meta_path, meta)
        original_scores[valid_positions] = fold_scores
        primary_scores[heldout_primary] = heldout_scores
        fold_meta.append(meta)
        print(json.dumps({"event": "fold_complete", "candidate": name, **meta}, sort_keys=True), flush=True)
        del model
        gc.collect()
    if not np.isfinite(original_scores).all():
        raise RuntimeError("original OOF scores are incomplete")
    if not np.isfinite(primary_scores[primary_indices]).all():
        raise RuntimeError("PRIMARY grouped diagnostic scores are incomplete")
    original_labels = labels[original_indices]
    threshold = threshold_max_mcc(original_labels, original_scores)
    own = metrics(original_labels, original_scores, threshold)
    per_fold = {}
    for fold in range(5):
        mask = folds[original_indices] == fold
        per_fold[str(fold)] = metrics(original_labels[mask], original_scores[mask], threshold)
    baseline_dir = root / "experiments" / "P1_8M650M_1884"
    baseline_result = json.loads((baseline_dir / "RESULT.json").read_text())
    baseline = baseline_result["original_grouped_oof"]
    baseline_scores = np.load(baseline_dir / "ORIGINAL_OOF_SCORES.f64.npy")
    if len(baseline_scores) != len(original_scores) or not np.isfinite(baseline_scores).all():
        raise RuntimeError("baseline OOF scores are invalid")
    matched_baseline = metrics(
        original_labels, original_scores,
        threshold_at_recall(original_labels, original_scores, baseline["recall"]),
    )
    matched_legacy_m4 = metrics(
        original_labels, original_scores,
        threshold_at_recall(original_labels, original_scores, 0.8945674983),
    )
    baseline_matched_legacy_m4 = metrics(
        original_labels, baseline_scores,
        threshold_at_recall(original_labels, baseline_scores, 0.8945674983),
    )
    primary_labels = labels[primary_indices]
    primary_result = {
        "rows": int(len(primary_indices)),
        "positive": int(primary_labels.sum()),
        "negative": int(len(primary_labels) - primary_labels.sum()),
        "interpretation": "grouped training-distribution diagnostic only; not independent external truth",
        **metrics(
            primary_labels, primary_scores[primary_indices],
            threshold,
        ),
    }
    passed = (
        own["auprc"] - baseline["auprc"] >= -0.002
        and matched_baseline["fpr"] - baseline["fpr"] <= 0.001
    )
    atomic_npy(output / "ORIGINAL_OOF_SCORES.f64.npy", original_scores)
    atomic_npy(output / "PRIMARY_GROUPED_DIAGNOSTIC_SCORES.f64.npy", primary_scores)
    result = {
        "state": "PFAM_EXPERIMENT_COMPLETE",
        "candidate": name,
        "feature_contract": {
            "center_esm2": "8M_layer6_mean_320_plus_650M_layer33_mean_1280",
            "portable_context": "44D protein-length and relative gene-order features",
            "pfam": spec,
            "fold_internal_vocab": True,
            "vocab_limit_per_namespace": 256,
            "eggnog_used": False,
            "coordinate_features_used": False,
            "description_text_used": False,
            "neighbor_esm2_used": False,
        },
        "folds": fold_meta,
        "original_grouped_oof": own,
        "original_per_fold": per_fold,
        "primary_grouped_diagnostic": primary_result,
        "baseline": baseline,
        "delta_auprc": float(own["auprc"] - baseline["auprc"]),
        "delta_mcc": float(own["mcc"] - baseline["mcc"]),
        "matched_baseline_recall": matched_baseline,
        "delta_matched_baseline_fpr": float(matched_baseline["fpr"] - baseline["fpr"]),
        "matched_legacy_m4_recall": matched_legacy_m4,
        "baseline_matched_legacy_m4_recall": baseline_matched_legacy_m4,
        "predeclared_no_worse_pass": bool(passed),
        "rows_allowed": int(allowed.sum()),
        "old_fold0_used": False,
    }
    atomic_json(output / "RESULT.json", result)
    (output / "COMPLETE").touch()
    print(json.dumps(result, sort_keys=True), flush=True)


def select(args: argparse.Namespace) -> None:
    output = Path(args.output)
    results = {
        name: json.loads((output / "experiments" / name / "RESULT.json").read_text())
        for name in CANDIDATES
    }
    passing = [name for name, result in results.items() if result["predeclared_no_worse_pass"]]
    if passing:
        winner = max(
            passing,
            key=lambda name: (
                results[name]["original_grouped_oof"]["auprc"],
                -results[name]["matched_baseline_recall"]["fpr"],
            ),
        )
        reason = "passed predeclared no-worse constraints; highest original grouped OOF AUPRC"
    else:
        winner = "P1_8M650M_1884_BASELINE"
        reason = "neither Pfam candidate passed the predeclared no-worse constraints"
    summary = {
        "state": "PFAM_SELECTION_COMPLETE",
        "winner": winner,
        "reason": reason,
        "candidates": {
            name: {
                "pass": result["predeclared_no_worse_pass"],
                "original_grouped_oof": result["original_grouped_oof"],
                "matched_baseline_recall": result["matched_baseline_recall"],
                "matched_legacy_m4_recall": result["matched_legacy_m4_recall"],
                "delta_auprc": result["delta_auprc"],
                "delta_mcc": result["delta_mcc"],
            }
            for name, result in results.items()
        },
        "selection_uses_old_fold0": False,
    }
    atomic_json(output / "FINAL_RESULT.json", summary)
    (output / "COMPLETE").touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--root", required=True)
    train_parser.add_argument("--pfam-root", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    train_parser.add_argument("--threads", type=int, default=16)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        select(args)


if __name__ == "__main__":
    main()
