from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extend_esm2 import make_matrix, numeric_columns
from train_context import (
    LGB_PARAMS,
    NB_NUMERIC,
    SEED,
    balanced_train_indices,
    bootstrap_ci,
    comparison_metrics,
    metrics,
    rel_name,
    threshold_max_mcc,
    top_pfam_vocab,
)


WINDOWS = (15, 20)
BASELINE = "M4_ESM_EGGNOG_W10"


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_long_wide(root: Path, experiment: Path) -> pd.DataFrame:
    target = experiment / "CONTEXT_FEATURES_LONG.parquet"
    if target.exists():
        return pq.read_table(target).to_pandas()
    centers = pq.read_table(root / "CENTER_BASE_FEATURES.parquet").to_pandas()
    neighbors = pq.read_table(experiment / "NEIGHBOR_INDEX.parquet").to_pandas()
    expected = len(centers) * 40
    if len(neighbors) != expected:
        raise ValueError(f"W20 neighbor rows differ: {len(neighbors)} != {expected}")
    neighbors.sort_values(["protein_id", "relative_position"], inplace=True)
    wide = centers.set_index("protein_id", drop=False)
    for column in NB_NUMERIC:
        pivot = neighbors.pivot(
            index="protein_id", columns="relative_position", values=column
        )
        pivot.columns = [
            f"nb_{rel_name(int(position))}_{column}" for position in pivot.columns
        ]
        wide = wide.join(pivot, how="left", validate="one_to_one")

    for window in WINDOWS:
        part = neighbors[neighbors.relative_position.abs().le(window)].copy()
        valid = part[~part.is_pad.astype(bool)].copy()
        grouped = valid.groupby("protein_id", sort=False)
        aggregate = grouped.agg(
            **{
                f"w{window}_neighbor_count": ("neighbor_id", "size"),
                f"w{window}_mean_protein_length": ("protein_length", "mean"),
                f"w{window}_std_protein_length": ("protein_length", "std"),
                f"w{window}_mean_abs_gc_delta": (
                    "gc_delta_from_center",
                    lambda values: np.nanmean(np.abs(values)),
                ),
                f"w{window}_same_strand_fraction": (
                    "same_strand_as_center",
                    "mean",
                ),
                f"w{window}_overlap_count": (
                    "gap_or_overlap_to_adjacent",
                    lambda values: int((values < 0).sum()),
                ),
                f"w{window}_mean_gap": ("gap_or_overlap_to_adjacent", "mean"),
                f"w{window}_std_gap": ("gap_or_overlap_to_adjacent", "std"),
                f"w{window}_median_gap": ("gap_or_overlap_to_adjacent", "median"),
                f"w{window}_mean_bp_distance": ("bp_distance_to_center", "mean"),
                f"w{window}_min_bp_distance": ("bp_distance_to_center", "min"),
                f"w{window}_max_bp_distance": ("bp_distance_to_center", "max"),
            }
        )
        pfam_rows = []
        for protein_id, group in part.groupby("protein_id", sort=False):
            ordered = []
            hit_genes = 0
            domain_count = 0
            domain_set = set()
            valid_genes = 0
            for row in group.itertuples(index=False):
                if bool(row.is_pad):
                    continue
                valid_genes += 1
                values = [
                    value
                    for value in str(row.pfams).split(",")
                    if value and value != "nan"
                ]
                if values:
                    hit_genes += 1
                domain_count += len(values)
                domain_set.update(values)
                ordered.extend(values)
            pfam_rows.append(
                {
                    "protein_id": protein_id,
                    f"w{window}_neighbor_pfams": "|".join(ordered),
                    f"w{window}_pfam_hit_fraction": (
                        hit_genes / valid_genes if valid_genes else 0.0
                    ),
                    f"w{window}_pfam_domain_count": domain_count,
                    f"w{window}_pfam_diversity": len(domain_set),
                    f"w{window}_pfam_missing": float(domain_count == 0),
                }
            )
        pfam = pd.DataFrame(pfam_rows).set_index("protein_id")
        wide = wide.join(aggregate, how="left").join(pfam, how="left")
    wide.reset_index(drop=True, inplace=True)
    wide.to_parquet(target, index=False, compression="zstd")
    return wide


def build_positive_neighbor_audit(
    experiment: Path, frame: pd.DataFrame
) -> pd.DataFrame:

    target = experiment / "SUPERVISED_POSITIVE_NEIGHBOR_AUDIT.parquet"
    if target.exists():
        audit = pq.read_table(target).to_pandas()
    else:
        neighbors = pq.read_table(
            experiment / "NEIGHBOR_INDEX.parquet",
            columns=["protein_id", "relative_position", "neighbor_id", "is_pad"],
        ).to_pandas()
        positive_ids = set(
            frame.loc[frame.label.eq(1), "protein_id"].astype(str).tolist()
        )
        positive_neighbor = (
            ~neighbors.is_pad.astype(bool)
            & neighbors.neighbor_id.astype(str).isin(positive_ids)
        )
        audit = frame[["protein_id"]].copy().set_index("protein_id")
        for window in WINDOWS:
            selected = neighbors.loc[
                positive_neighbor & neighbors.relative_position.abs().le(window),
                "protein_id",
            ]
            counts = selected.value_counts(sort=False)
            column = f"w{window}_supervised_positive_neighbor_count"
            audit[column] = counts.reindex(audit.index).fillna(0).astype(np.int16)
        audit.reset_index(inplace=True)
        audit.to_parquet(target, index=False, compression="zstd")
    if not np.array_equal(
        audit.protein_id.astype(str).to_numpy(),
        frame.protein_id.astype(str).to_numpy(),
    ):
        raise ValueError("positive-neighbor audit row order differs")
    return audit


def train_long_oof(
    frame: pd.DataFrame, esm: np.ndarray, window: int
) -> tuple[np.ndarray, list[int]]:
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    iterations: list[int] = []
    columns = numeric_columns(window, with_context=True, with_eggnog=True)
    folds = frame.new_mag_fold.to_numpy(dtype=np.int8)
    for fold in range(5):
        valid = np.flatnonzero(folds == fold)
        train_all = np.flatnonzero(folds != fold)

        train = balanced_train_indices(frame, train_all, SEED + 1000 * fold + 10 + 71)
        pfam_vocab = top_pfam_vocab(
            frame.iloc[train][f"w{window}_neighbor_pfams"], 256
        )
        eggnog_vocab = top_pfam_vocab(
            frame.iloc[train][f"w{window}_neighbor_eggnog_tokens"], 512
        )
        x_train = make_matrix(
            frame,
            esm,
            None,
            train,
            columns,
            window,
            True,
            pfam_vocab,
            True,
            eggnog_vocab,
            False,
            False,
            SEED + fold * 17 + 701,
        )
        x_valid = make_matrix(
            frame,
            esm,
            None,
            valid,
            columns,
            window,
            True,
            pfam_vocab,
            True,
            eggnog_vocab,
            False,
            False,
            SEED + fold * 17 + 702,
        )
        classifier = lgb.LGBMClassifier(**LGB_PARAMS)
        classifier.fit(
            x_train,
            frame.iloc[train].label.to_numpy(dtype=np.int8),
            eval_set=[(x_valid, frame.iloc[valid].label.to_numpy(dtype=np.int8))],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        scores[valid] = classifier.predict_proba(
            x_valid, num_iteration=classifier.best_iteration_
        )[:, 1]
        iterations.append(int(classifier.best_iteration_ or LGB_PARAMS["n_estimators"]))
        print(
            json.dumps(
                {
                    "event": "fold_complete",
                    "fold": fold,
                    "window": window,
                    "best_iteration": iterations[-1],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite W{window} scores")
    return scores, iterations


def score_or_fit(
    experiment: Path, frame: pd.DataFrame, esm: np.ndarray, window: int
) -> tuple[np.ndarray, list[int], bool]:
    name = f"M4_ESM_EGGNOG_W{window}"
    score_path = experiment / f"{name}.oof.f64.npy"
    iteration_path = experiment / f"{name}.iterations.json"
    if score_path.exists() and iteration_path.exists():
        scores = np.load(score_path)
        iterations = json.loads(iteration_path.read_text())
        if (
            scores.shape != (len(frame),)
            or scores.dtype != np.float64
            or not np.isfinite(scores).all()
            or len(iterations) != 5
        ):
            raise ValueError(f"invalid W{window} checkpoint")
        return scores, [int(value) for value in iterations], True
    scores, iterations = train_long_oof(frame, esm, window)
    atomic_save_npy(score_path, scores)
    atomic_write_json(iteration_path, iterations)
    return scores, iterations, False


def run(args: argparse.Namespace) -> None:
    root = Path(args.output)
    experiment = root / "LONG_CONTEXT_W20"
    if not (experiment / "LONG_EGGNOG_COMPLETE").exists():
        raise RuntimeError("long-context eggNOG extraction is incomplete")
    frame = build_long_wide(root, experiment)
    eggnog = pq.read_table(
        experiment / "EGGNOG_CONTEXT_FEATURES_LONG.parquet"
    ).to_pandas()
    frame = frame.merge(eggnog, on="protein_id", how="left", validate="one_to_one")
    split = pd.read_csv(
        root / "SPLIT_MANIFEST.tsv",
        sep="\t",
        usecols=["protein_id", "new_mag_fold"],
    )
    frame = frame.merge(split, on="protein_id", how="left", validate="one_to_one")
    if frame.new_mag_fold.isna().any() or int((frame.old_homology_fold == 0).sum()) != 0:
        raise ValueError("long-context split contract failed")
    if frame.groupby("mag_id").new_mag_fold.nunique().max() != 1:
        raise ValueError("MAG crosses long-context development folds")
    positive_neighbor_audit = build_positive_neighbor_audit(experiment, frame)

    center_index = pd.read_csv(root / "CENTER_ESM2_INDEX.tsv.gz", sep="\t")
    if not np.array_equal(
        frame.protein_id.astype(str).to_numpy(),
        center_index.protein_id.astype(str).to_numpy(),
    ):
        raise ValueError("long-context and center ESM2 row order differ")
    esm = np.load(root / "CENTER_ESM2_T6_8M_LAYER6_MEAN.f16.npy", mmap_mode="r")
    if esm.shape != (len(frame), 320) or esm.dtype != np.float16:
        raise ValueError("center ESM2 contract failed")

    baseline_scores = np.load(
        root / "M4_EGGNOG_EVALUATION" / f"{BASELINE}.oof.f64.npy"
    )
    if baseline_scores.shape != (len(frame),) or not np.isfinite(baseline_scores).all():
        raise ValueError("strict W10 baseline is invalid")
    model_scores = {BASELINE: baseline_scores}
    iteration_map = {}
    reuse_map = {}
    for window in WINDOWS:
        name = f"M4_ESM_EGGNOG_W{window}"
        scores, iterations, reused = score_or_fit(experiment, frame, esm, window)
        model_scores[name] = scores
        iteration_map[name] = iterations
        reuse_map[name] = reused

    y = frame.label.to_numpy(dtype=np.int8)
    baseline_threshold = threshold_max_mcc(y, baseline_scores)
    baseline_first = comparison_metrics(y, baseline_scores, baseline_threshold, 0.5)
    reference_recall = float(baseline_first["recall"])
    rows = []
    row_by_name = {}
    for name, scores in model_scores.items():
        threshold = threshold_max_mcc(y, scores)
        row = comparison_metrics(y, scores, threshold, reference_recall)
        window = int(name.rsplit("W", 1)[1])
        row.update(
            {
                "model": name,
                "window": window,
                "feature_contract": f"center_ESM2_320+W{window}_context+fold_train_only_Pfam+eggNOG",
                "threshold_source": "MAG-grouped development OOF max MCC",
                "matched_recall_reference": BASELINE,
                "old_fold0_rows_used": 0,
            }
        )
        rows.append(row)
        row_by_name[name] = row
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        experiment / "W10_W15_W20_MODEL_COMPARISON.tsv", sep="\t", index=False
    )
    subgroup_rows = []
    for audit_window in WINDOWS:
        count_column = f"w{audit_window}_supervised_positive_neighbor_count"
        has_positive_neighbor = positive_neighbor_audit[count_column].to_numpy() > 0
        for state, mask in (
            ("present", has_positive_neighbor),
            ("absent", ~has_positive_neighbor),
        ):
            subset_y = y[mask]
            for name, scores in model_scores.items():
                threshold = float(
                    row_by_name[name]["matched_baseline_recall_threshold"]
                )
                subgroup = metrics(subset_y, scores[mask], threshold)
                subgroup.update(
                    {
                        "audit_window": audit_window,
                        "supervised_positive_neighbor": state,
                        "model": name,
                        "global_threshold_source": "matched W10 development recall",
                        "audit_flag_used_as_feature": False,
                    }
                )
                subgroup_rows.append(subgroup)
    pd.DataFrame(subgroup_rows).to_csv(
        experiment / "SUPERVISED_POSITIVE_NEIGHBOR_SUBGROUP_METRICS.tsv",
        sep="\t",
        index=False,
    )
    candidates = comparison.sort_values(
        ["fpr_at_baseline_recall", "mcc"], ascending=[True, False]
    )
    selected = str(candidates.iloc[0].model)

    oof = frame[
        [
            "protein_id",
            "label",
            "environment",
            "mag_id",
            "contig_id",
            "new_mag_fold",
            "old_homology_fold",
        ]
    ].copy()
    for name, scores in model_scores.items():
        oof[name] = scores
    oof.to_csv(
        experiment / "W10_W15_W20_OOF_PREDICTIONS.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "compresslevel": 1},
    )
    bootstrap_ci(
        oof,
        [
            (name, float(row_by_name[name]["matched_baseline_recall_threshold"]))
            for name in model_scores
        ],
        experiment,
        reps=args.bootstrap_reps,
    )
    summary = {
        "state": "LONG_CONTEXT_W20_COMPLETE",
        "evaluation_scope": "strict MAG-grouped development OOF only; no independent test",
        "rows": int(len(frame)),
        "old_fold0_rows_used": 0,
        "windows_evaluated": [10, 15, 20],
        "matched_recall": reference_recall,
        "selected_development_window": int(row_by_name[selected]["window"]),
        "selected_development_model": selected,
        "models": row_by_name,
        "iterations": iteration_map,
        "checkpoint_reused": reuse_map,
        "neighbor_esm2_used": False,
        "supervised_positive_neighbor_audit": {
            f"w{window}_centers_with_positive_neighbor": int(
                (
                    positive_neighbor_audit[
                        f"w{window}_supervised_positive_neighbor_count"
                    ]
                    > 0
                ).sum()
            )
            for window in WINDOWS
        },
        "supervised_positive_neighbor_flag_used_as_feature": False,
        "note": "This experiment changes only gene-context span; W10 raw-center ESM2/Pfam/eggNOG is the locked comparator.",
    }
    atomic_write_json(experiment / "LONG_CONTEXT_RESULT_SUMMARY.json", summary)
    (experiment / "COMPLETE").touch()
    (root / "LONG_CONTEXT_W20_COMPLETE").touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
