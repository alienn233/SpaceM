from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.util
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


FINAL_CLASSES = ("OTHER", "ARG", "MRG", "MGE")
TARGET_CLASSES = ("ARG", "MRG", "MGE")
ALLOWED_COHORTS = (
    "ORIGINAL_SUPERVISED",
    "MGNIFY_SAFE_W5_PSEUDO_NEGATIVE",
    "PRIMARY2000_POSITIVE",
    "PRIMARY2000_SAFE_W5_PSEUDO_NEGATIVE",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def main_weights(
    manifest: pd.DataFrame,
    train_index: np.ndarray,
    y_gate: np.ndarray,
    family_ids: np.ndarray,
) -> tuple[np.ndarray, dict]:
    frame = manifest.iloc[train_index]
    cohorts = frame.cohort.astype(str).to_numpy()
    local_y_gate = y_gate[train_index]
    weights = frame.training_weight.to_numpy(np.float64).copy()
    weights[cohorts == "PRIMARY2000_SAFE_W5_PSEUDO_NEGATIVE"] = 0.5
    primary_old_positive = cohorts == "PRIMARY2000_POSITIVE"
    primary_target = primary_old_positive & local_y_gate.astype(bool)
    primary_reclassified_negative = primary_old_positive & ~primary_target
    weights[primary_reclassified_negative] = 1.0
    original_target = (
        (cohorts == "ORIGINAL_SUPERVISED")
        & local_y_gate.astype(bool)
    )
    if primary_target.any():
        selected = pd.DataFrame({
            "family_id": family_ids[train_index][primary_target],
            "assembly_accession": frame.assembly_accession.astype(str).to_numpy()[primary_target],
            "source_multiplier": frame.source_multiplier.to_numpy(np.float64)[primary_target],
        })
        family_count = selected.groupby("family_id").family_id.transform("size").to_numpy(np.float64)
        genome_count = selected.groupby("assembly_accession").assembly_accession.transform("size").to_numpy(np.float64)
        raw = (
            selected.source_multiplier.to_numpy(np.float64)
            / np.sqrt(family_count)
            / np.sqrt(genome_count)
        )
        target_mass = 0.25 * float(weights[original_target].sum())
        weights[primary_target] = raw * (target_mass / raw.sum())
    else:
        target_mass = 0.0
    return weights.astype(np.float32), {
        "rows": int(len(train_index)),
        "target_positive_rows": int(local_y_gate.sum()),
        "target_negative_rows": int(len(train_index) - local_y_gate.sum()),
        "original_target_positive_weight": float(weights[original_target].sum()),
        "primary_target_positive_rows": int(primary_target.sum()),
        "primary_target_positive_weight": float(weights[primary_target].sum()),
        "primary_target_weight_contract": float(target_mass),
        "primary_vf_reclassified_negative_rows": int(primary_reclassified_negative.sum()),
        "primary_vf_reclassified_negative_weight": float(weights[primary_reclassified_negative].sum()),
        "primary_w5_rows": int(np.sum(cohorts == "PRIMARY2000_SAFE_W5_PSEUDO_NEGATIVE")),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }


def balance_binary(weights: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    output = weights.astype(np.float64, copy=True)
    positive = labels == 1
    negative = labels == 0
    positive_mass = float(output[positive].sum())
    negative_mass = float(output[negative].sum())
    if positive_mass <= 0 or negative_mass <= 0:
        raise RuntimeError("binary task lacks one class")
    target_mass = 0.5 * float(output.sum())
    output[positive] *= target_mass / positive_mass
    output[negative] *= target_mass / negative_mass
    output *= len(output) / output.sum()
    return output.astype(np.float32), {
        "positive_rows": int(positive.sum()),
        "negative_rows": int(negative.sum()),
        "positive_mass_before": positive_mass,
        "negative_mass_before": negative_mass,
        "positive_mass_after": float(output[positive].sum()),
        "negative_mass_after": float(output[negative].sum()),
    }


def balance_multiclass(weights: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    output = weights.astype(np.float64, copy=True)
    totals = {value: float(output[labels == value].sum()) for value in range(3)}
    if any(value <= 0 for value in totals.values()):
        raise RuntimeError("target subtype task lacks a class")
    target = float(sum(totals.values()) / 3.0)
    for value in range(3):
        output[labels == value] *= target / totals[value]
    output *= len(output) / output.sum()
    return output.astype(np.float32), {
        "rows_by_class": {TARGET_CLASSES[value]: int(np.sum(labels == value)) for value in range(3)},
        "mass_before": {TARGET_CLASSES[value]: totals[value] for value in range(3)},
        "mass_after": {
            TARGET_CLASSES[value]: float(output[labels == value].sum())
            for value in range(3)
        },
    }


def per_class_metrics(truth: np.ndarray, predicted: np.ndarray, names: tuple[str, ...]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=np.arange(len(names)), zero_division=0,
    )
    return {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(names)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pfam-root", required=True)
    parser.add_argument("--eggnog-features", required=True)
    parser.add_argument("--mag-features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--foldsafe-primary-labels", required=True)
    parser.add_argument("--feature-code", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--reference-oof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=40)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "TRAIN.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if (output / "HIERARCHICAL_SINGLE_LABEL_COMPLETE").exists():
        print((output / "RESULT.json").read_text(), end="")
        return

    formal = load_module("single_label_feature_contract", Path(args.feature_code))
    core = load_module("single_label_core", Path(args.library))
    data_root = Path(args.data_root)
    manifest, pfam, eggnog, esm8, esm650, context = formal.load_all(
        core, data_root, Path(args.pfam_root), Path(args.eggnog_features)
    )
    mag_values, mag_columns = formal.load_mag(Path(args.mag_features), manifest)
    labels = pd.read_parquet(args.labels)
    if len(labels) != len(manifest):
        raise RuntimeError("label and manifest row counts differ")
    if not np.array_equal(labels.row_index.to_numpy(), np.arange(len(manifest))):
        raise RuntimeError("label row_index is not canonical")
    if not np.array_equal(labels.protein_id.astype(str), manifest.protein_id.astype(str)):
        raise RuntimeError("label and manifest protein IDs differ")
    eligible_label_mask = labels.is_training_eligible.to_numpy(np.int8).astype(bool)
    dynamic_primary_mask = labels.is_fold_dynamic_primary_seed_label.to_numpy(np.uint8).astype(bool)
    if not np.all(labels.loc[eligible_label_mask, "final_class"].astype(str).isin(FINAL_CLASSES)):
        raise RuntimeError("label table contains an unsupported class")
    if not np.all(labels.loc[~eligible_label_mask, "final_class"].eq("EXCLUDED_TRANSPORT")):
        raise RuntimeError("ineligible rows are not exclusively unresolved TRANSPORT")

    foldsafe = pd.read_parquet(args.foldsafe_primary_labels)
    required_foldsafe = {
        "row_index", "protein_id", "fold", "fold_safe_class",
        "is_training_eligible", "fold_safe_seed_id",
        "fold_safe_seed_outer_fold", "fold_safe_family_id",
    }
    if required_foldsafe - set(foldsafe.columns):
        raise RuntimeError(
            f"fold-safe label table missing columns: {sorted(required_foldsafe-set(foldsafe.columns))}"
        )
    dynamic_indices = np.flatnonzero(dynamic_primary_mask)
    if len(foldsafe) != 5 * len(dynamic_indices) or foldsafe.duplicated(["row_index", "fold"]).any():
        raise RuntimeError("fold-safe label table shape/key contract failed")
    if set(foldsafe.row_index.astype(int)) != set(dynamic_indices.tolist()):
        raise RuntimeError("fold-safe label rows differ from dynamic PRIMARY rows")
    foldsafe_row_ids = foldsafe.drop_duplicates("row_index").sort_values("row_index")
    if not np.array_equal(
        foldsafe_row_ids.protein_id.astype(str).to_numpy(),
        manifest.iloc[dynamic_indices].protein_id.astype(str).to_numpy(),
    ):
        raise RuntimeError("fold-safe protein IDs differ from the canonical manifest")
    if not foldsafe.fold.isin(range(5)).all():
        raise RuntimeError("fold-safe label table contains an invalid fold")
    if not foldsafe.loc[foldsafe.is_training_eligible.eq(1), "fold_safe_seed_outer_fold"].isin([1, 2, 3, 4]).all():
        raise RuntimeError("fold-safe labels selected old fold0 or unknown seed evidence")

    cohorts = manifest.cohort.astype(str)
    allowed_cohort = cohorts.isin(ALLOWED_COHORTS).to_numpy()
    allowed = allowed_cohort & (eligible_label_mask | dynamic_primary_mask)
    folds = manifest.new_mag_fold.to_numpy(np.int8)
    original_all_indices = np.flatnonzero(cohorts.eq("ORIGINAL_SUPERVISED"))
    original_indices = np.flatnonzero(
        cohorts.eq("ORIGINAL_SUPERVISED").to_numpy() & eligible_label_mask
    )
    primary_indices = np.flatnonzero(
        allowed_cohort & cohorts.str.startswith("PRIMARY2000").to_numpy()
    )
    y_gate = labels.y_target_no_vf.to_numpy(np.int8)
    class_name = labels.final_class.astype(str).to_numpy()
    base_family_ids = manifest.family_id.fillna("").astype(str).to_numpy()
    target_id_lookup = {name: index for index, name in enumerate(TARGET_CLASSES)}
    y_target_subtype = np.full(len(labels), -1, dtype=np.int8)
    for name, value in target_id_lookup.items():
        y_target_subtype[class_name == name] = np.int8(value)

    checkpoints = output / "fold_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    gate_oof = np.full(len(original_indices), np.nan, dtype=np.float64)
    target_oof = np.full((len(original_indices), 3), np.nan, dtype=np.float64)
    primary_gate = np.full(len(manifest), np.nan, dtype=np.float64)
    primary_target = np.full((len(manifest), 3), np.nan, dtype=np.float64)
    params = dict(core.LGB_PARAMS)
    params["n_jobs"] = int(args.threads)
    fold_meta: list[dict] = []

    for fold in range(5):
        fold_dir = checkpoints / f"fold{fold}"
        fold_dir.mkdir(exist_ok=True)
        paths = {
            "positions": fold_dir / "original_positions.i64.npy",
            "gate_scores": fold_dir / "gate_scores.f64.npy",
            "target_scores": fold_dir / "target_scores.f64.npy",
            "primary_indices": fold_dir / "primary_indices.i64.npy",
            "primary_gate": fold_dir / "primary_gate_scores.f64.npy",
            "primary_target": fold_dir / "primary_target_scores.f64.npy",
            "gate_model": fold_dir / "gate.model.txt",
            "target_model": fold_dir / "target_subclass.model.txt",
            "vocab": fold_dir / "vocab.json",
            "meta": fold_dir / "meta.json",
            "complete": fold_dir / "COMPLETE",
        }
        required = [value for key, value in paths.items() if key != "complete"]
        if paths["complete"].exists() and all(path.exists() for path in required):
            positions = np.load(paths["positions"])
            heldout_primary = np.load(paths["primary_indices"])
            gate_oof[positions] = np.load(paths["gate_scores"])
            target_oof[positions] = np.load(paths["target_scores"])
            primary_gate[heldout_primary] = np.load(paths["primary_gate"])
            primary_target[heldout_primary] = np.load(paths["primary_target"])
            fold_meta.append(json.loads(paths["meta"].read_text()))
            continue

        fold_rows = foldsafe[foldsafe.fold.eq(fold)].sort_values("row_index")
        fold_dynamic_index = fold_rows.row_index.to_numpy(np.int64)
        if not np.array_equal(fold_dynamic_index, dynamic_indices):
            raise RuntimeError(f"fold {fold}: dynamic row order differs from label table")
        fold_class_name = class_name.copy()
        fold_eligible = eligible_label_mask.copy()
        fold_family_ids = base_family_ids.copy()
        fold_class_name[fold_dynamic_index] = fold_rows.fold_safe_class.astype(str).to_numpy()
        fold_eligible[fold_dynamic_index] = (
            fold_rows.is_training_eligible.to_numpy(np.uint8).astype(bool)
        )
        fold_family_ids[fold_dynamic_index] = fold_rows.fold_safe_family_id.astype(str).to_numpy()
        y_gate_fold = np.where(
            fold_eligible,
            np.isin(fold_class_name, TARGET_CLASSES).astype(np.int8),
            np.int8(-1),
        )
        y_target_subtype_fold = np.full(len(labels), -1, dtype=np.int8)
        for name, value in target_id_lookup.items():
            y_target_subtype_fold[(fold_class_name == name) & fold_eligible] = np.int8(value)

        train_index = np.flatnonzero(allowed_cohort & fold_eligible & (folds != fold))
        valid_positions = np.flatnonzero(folds[original_indices] == fold)
        valid_index = original_indices[valid_positions]
        heldout_primary = primary_indices[folds[primary_indices] == fold]
        pfam_vocab = core.top_vocab(pfam.iloc[train_index].w10_neighbor_pfam_tokens, 256)
        eggnog_vocab = core.top_vocab(eggnog.iloc[train_index].w10_neighbor_eggnog_tokens, 512)
        x_train = formal.make_matrix(
            core, esm8, esm650, context, pfam, eggnog, train_index,
            pfam_vocab, eggnog_vocab, mag_values,
        )
        x_valid = formal.make_matrix(
            core, esm8, esm650, context, pfam, eggnog, valid_index,
            pfam_vocab, eggnog_vocab, mag_values,
        )
        base_weights, base_audit = main_weights(
            manifest, train_index, y_gate_fold, fold_family_ids,
        )
        gate_weights, gate_weight_audit = balance_binary(
            base_weights, y_gate_fold[train_index],
        )

        gate_model = lgb.LGBMClassifier(**params)
        gate_model.fit(
            x_train, y_gate_fold[train_index], sample_weight=gate_weights,
            eval_set=[(x_valid, y_gate[valid_index])], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        gate_iteration = int(gate_model.best_iteration_ or params["n_estimators"])
        fold_gate = gate_model.predict_proba(x_valid, num_iteration=gate_iteration)[:, 1].astype(np.float64)

        target_train_mask = y_target_subtype_fold[train_index] >= 0
        target_valid_mask = y_target_subtype[valid_index] >= 0
        target_weights, target_weight_audit = balance_multiclass(
            base_weights[target_train_mask],
            y_target_subtype_fold[train_index[target_train_mask]],
        )
        target_params = dict(params)
        target_params.update({"objective": "multiclass", "num_class": 3})
        target_model = lgb.LGBMClassifier(**target_params)
        target_model.fit(
            x_train[target_train_mask],
            y_target_subtype_fold[train_index[target_train_mask]],
            sample_weight=target_weights,
            eval_set=[(x_valid[target_valid_mask], y_target_subtype[valid_index[target_valid_mask]])],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        target_iteration = int(target_model.best_iteration_ or params["n_estimators"])
        fold_target = target_model.predict_proba(x_valid, num_iteration=target_iteration).astype(np.float64)

        if not (
            np.isfinite(fold_gate).all()
            and np.isfinite(fold_target).all()
        ):
            raise RuntimeError(f"fold {fold} generated non-finite predictions")

        del x_train, x_valid
        gc.collect()
        if len(heldout_primary):
            x_primary = formal.make_matrix(
                core, esm8, esm650, context, pfam, eggnog, heldout_primary,
                pfam_vocab, eggnog_vocab, mag_values,
            )
            heldout_gate = gate_model.predict_proba(
                x_primary, num_iteration=gate_iteration
            )[:, 1].astype(np.float64)
            heldout_target = target_model.predict_proba(
                x_primary, num_iteration=target_iteration
            ).astype(np.float64)
            del x_primary
        else:
            heldout_gate = np.empty(0, dtype=np.float64)
            heldout_target = np.empty((0, 3), dtype=np.float64)

        gate_model.booster_.save_model(str(paths["gate_model"]), num_iteration=gate_iteration)
        target_model.booster_.save_model(str(paths["target_model"]), num_iteration=target_iteration)
        feature_dimension = (
            1600 + 44 + len(formal.PFAM_NUMERIC) + len(pfam_vocab)
            + len(formal.EGGNOG_NUMERIC) + len(eggnog_vocab) + len(mag_columns)
        )
        atomic_json(paths["vocab"], {
            "pfam_w10_vocab": pfam_vocab,
            "eggnog_w10_vocab": eggnog_vocab,
            "pfam_numeric_columns": formal.PFAM_NUMERIC,
            "eggnog_numeric_columns": formal.EGGNOG_NUMERIC,
            "mag_columns": mag_columns,
            "feature_dimension": feature_dimension,
        })
        meta = {
            "fold": fold,
            "threads": int(args.threads),
            "training_rows": int(len(train_index)),
            "validation_rows": int(len(valid_index)),
            "primary_diagnostic_rows": int(len(heldout_primary)),
            "feature_dimension": feature_dimension,
            "gate_best_iteration": gate_iteration,
            "target_subclass_best_iteration": target_iteration,
            "base_weight_audit": base_audit,
            "gate_weight_audit": gate_weight_audit,
            "target_weight_audit": target_weight_audit,
            "fold_safe_primary_audit": {
                "dynamic_rows": int(len(fold_rows)),
                "eligible_rows": int(fold_rows.is_training_eligible.sum()),
                "excluded_rows": int((fold_rows.is_training_eligible == 0).sum()),
                "class_counts": {
                    str(key): int(value)
                    for key, value in fold_rows.fold_safe_class.value_counts().sort_index().items()
                },
                "old_fold0_seed_rows_used": 0,
            },
            "old_fold0_used": False,
            "external_test_used": False,
        }
        atomic_npy(paths["positions"], valid_positions.astype(np.int64))
        atomic_npy(paths["gate_scores"], fold_gate)
        atomic_npy(paths["target_scores"], fold_target)
        atomic_npy(paths["primary_indices"], heldout_primary.astype(np.int64))
        atomic_npy(paths["primary_gate"], heldout_gate)
        atomic_npy(paths["primary_target"], heldout_target)
        atomic_json(paths["meta"], meta)
        paths["complete"].touch()
        gate_oof[valid_positions] = fold_gate
        target_oof[valid_positions] = fold_target
        primary_gate[heldout_primary] = heldout_gate
        primary_target[heldout_primary] = heldout_target
        fold_meta.append(meta)
        print(json.dumps({"event": "fold_complete", **meta}, sort_keys=True), flush=True)
        del gate_model, target_model
        gc.collect()

    if not (
        np.isfinite(gate_oof).all()
        and np.isfinite(target_oof).all()
    ):
        raise RuntimeError("original grouped OOF arrays are incomplete")
    if not (
        np.isfinite(primary_gate[primary_indices]).all()
        and np.isfinite(primary_target[primary_indices]).all()
    ):
        raise RuntimeError("PRIMARY grouped diagnostic arrays are incomplete")

    original_gate_truth = y_gate[original_indices]
    gate_threshold = core.threshold_max_mcc(original_gate_truth, gate_oof)
    gate_metric = core.metrics(original_gate_truth, gate_oof, gate_threshold)
    original_target = original_gate_truth == 1
    target_truth = y_target_subtype[original_indices][original_target]
    target_prediction = np.argmax(target_oof[original_target], axis=1).astype(np.int8)
    target_metric = {
        "rows": int(original_target.sum()),
        "accuracy": float(accuracy_score(target_truth, target_prediction)),
        "macro_f1": float(f1_score(target_truth, target_prediction, average="macro")),
        "weighted_f1": float(f1_score(target_truth, target_prediction, average="weighted")),
        "confusion_matrix": confusion_matrix(target_truth, target_prediction, labels=np.arange(3)).tolist(),
        "per_class": per_class_metrics(target_truth, target_prediction, TARGET_CLASSES),
    }

    final_prediction = np.zeros(len(original_indices), dtype=np.int8)
    predicted_target = gate_oof >= gate_threshold
    final_prediction[predicted_target] = 1 + np.argmax(target_oof[predicted_target], axis=1).astype(np.int8)
    final_truth = labels.final_class_id.to_numpy(np.int8)[original_indices]
    final_metric = {
        "rows": int(len(original_indices)),
        "accuracy": float(accuracy_score(final_truth, final_prediction)),
        "macro_f1": float(f1_score(final_truth, final_prediction, average="macro")),
        "weighted_f1": float(f1_score(final_truth, final_prediction, average="weighted")),
        "confusion_matrix": confusion_matrix(final_truth, final_prediction, labels=np.arange(4)).tolist(),
        "per_class": per_class_metrics(final_truth, final_prediction, FINAL_CLASSES),
    }
    vf_mask = labels.is_excluded_vf.to_numpy(np.int8)[original_indices].astype(bool)
    subgroup = {
        "original_vf_rows": int(vf_mask.sum()),
        "original_vf_predicted_target": int(np.sum(predicted_target & vf_mask)),
        "original_vf_target_false_positive_rate": float(np.mean(predicted_target[vf_mask])),
        "original_unresolved_transport_rows_not_scored": int(
            np.sum(cohorts.eq("ORIGINAL_SUPERVISED").to_numpy() & ~eligible_label_mask)
        ),
    }

    reference_scores = np.load(args.reference_oof)
    if len(reference_scores) != len(original_all_indices) or not np.isfinite(reference_scores).all():
        raise RuntimeError("reference OOF violates shape or finite contract")
    original_eligible_positions = np.flatnonzero(eligible_label_mask[original_all_indices])
    reference_scores = reference_scores[original_eligible_positions]
    reference_threshold = core.threshold_max_mcc(original_gate_truth, reference_scores)
    reference_metric = core.metrics(original_gate_truth, reference_scores, reference_threshold)
    matched_reference = core.metrics(
        original_gate_truth,
        gate_oof,
        core.threshold_at_recall(original_gate_truth, gate_oof, reference_metric["recall"]),
    )
    no_worse = {
        "predeclared_before_training": True,
        "auprc_tolerance_absolute": 0.002,
        "matched_recall_fpr_tolerance_absolute": 0.001,
        "auprc_pass": bool(gate_metric["auprc"] >= reference_metric["auprc"] - 0.002),
        "matched_recall_fpr_pass": bool(
            matched_reference["fpr"] <= reference_metric["fpr"] + 0.001
        ),
    }
    no_worse["pass"] = bool(no_worse["auprc_pass"] and no_worse["matched_recall_fpr_pass"])

    atomic_npy(output / "ORIGINAL_GATE_OOF.f64.npy", gate_oof)
    atomic_npy(output / "ORIGINAL_TARGET_SUBCLASS_OOF.f64.npy", target_oof)
    atomic_npy(output / "ORIGINAL_FINAL_CLASS_PREDICTION.i8.npy", final_prediction)
    atomic_npy(output / "PRIMARY_GATE_GROUPED_DIAGNOSTIC.f64.npy", primary_gate)
    atomic_npy(output / "PRIMARY_TARGET_GROUPED_DIAGNOSTIC.f64.npy", primary_target)

    result = {
        "state": "HIERARCHICAL_SINGLE_LABEL_COMPLETE",
        "architecture": {
            "stage1": "ANY_TARGET_NO_VF binary gate",
            "stage2_target": "exclusive ARG/MRG/MGE multiclass",
            "final_output": list(FINAL_CLASSES),
            "exactly_one_label_per_protein": True,
            "transport_output_class": False,
            "unresolved_transport": "alignment-only, unresolved and conflict rows are NOT_SCORED and excluded from formal OOF; never negative",
        },
        "feature_contract": {
            "center_esm2": "8M_layer6_mean_320_plus_650M_layer33_mean_1280",
            "portable_context": "44D coordinate-safe",
            "pfam": "W10 numeric plus fold-training-only 256-token vocabulary",
            "eggnog": "W10 numeric plus fold-training-only 512-token normalized vocabulary",
            "mag_dna": "ABC141 V2 ID-safe missing-aware coordinate-audited features",
            "feature_dimension": fold_meta[0]["feature_dimension"],
            "free_text_used": False,
            "neighbor_esm2_used": False,
        },
        "threads": int(args.threads),
        "folds": fold_meta,
        "thresholds": {
            "target_gate": float(gate_threshold),
        },
        "target_gate_grouped_oof": gate_metric,
        "target_subclass_grouped_oof": target_metric,
        "final_single_label_grouped_oof": final_metric,
        "critical_subgroups": subgroup,
        "rescored_previous_winner_on_new_target": reference_metric,
        "new_gate_at_previous_winner_recall": matched_reference,
        "predeclared_no_worse_decision": no_worse,
        "delta_gate_auprc_vs_rescored_previous_winner": float(gate_metric["auprc"] - reference_metric["auprc"]),
        "delta_gate_mcc_vs_rescored_previous_winner": float(gate_metric["mcc"] - reference_metric["mcc"]),
        "old_fold0_used": False,
        "fold_safe_primary_labels_used": True,
        "gate_class_mass_balanced": True,
        "transport_alignment_only_used": False,
        "external_test_used": False,
    }
    atomic_json(output / "RESULT.json", result)
    (output / "HIERARCHICAL_SINGLE_LABEL_COMPLETE").touch()
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
