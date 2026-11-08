from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_CLASSES = {"OTHER", "ARG", "MRG", "MGE"}


def normalize_seed_label(value: str, direct_transport: dict[str, str], seed_id: str) -> tuple[str, str]:
    label = str(value).strip().upper()
    if label == "ARG":
        return "ARG", "SEED_ARG"
    if label in {"MRG", "METAL", "BACMET"}:
        return "MRG", "SEED_MRG"
    if label in {"MGE", "MOBILOME"}:
        return "MGE", "SEED_MGE"
    if label in {"VF", "VIRULENCE"}:
        return "OTHER", "SEED_VF_TO_OTHER"
    if label in {"TRANSPORT", "TRANSPORT_EFFLUX"}:
        direct = direct_transport.get(seed_id, "NEEDS_ALIGNMENT")
        if direct in {"ARG", "MRG"}:
            return direct, "SEED_TRANSPORT_DIRECT_PROVENANCE"
        return "EXCLUDED_NO_DIRECT_TRANSPORT_LABEL", "SEED_TRANSPORT_NON_DIRECT_EXCLUDED"
    raise RuntimeError(f"unsupported seed primary label {value!r} for {seed_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--original-context", required=True)
    parser.add_argument("--primary-map", required=True)
    parser.add_argument("--fold-hit-dir", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--transport-assignments", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(
        args.manifest,
        columns=["row_index", "protein_id", "cohort", "label", "new_mag_fold", "target_class"],
    )
    original_mask = manifest.cohort.eq("ORIGINAL_SUPERVISED").to_numpy()
    original_indices = np.flatnonzero(original_mask)
    context = pd.read_parquet(
        args.original_context,
        columns=["protein_id", "label", "reference_seed", "old_homology_fold"],
    )
    if len(context) != len(original_indices):
        raise RuntimeError("original context row count differs from manifest")
    if not np.array_equal(
        manifest.iloc[original_indices].protein_id.astype(str).to_numpy(),
        context.protein_id.astype(str).to_numpy(),
    ):
        raise RuntimeError("original context order differs from manifest")
    if int(context.old_homology_fold.eq(0).sum()) != 0:
        raise RuntimeError("old sealed fold0 entered the original center cohort")

    primary_map = pd.read_parquet(args.primary_map)
    seed_rows = manifest[
        manifest.cohort.eq("PRIMARY2000_POSITIVE")
        & manifest.target_class.eq("SEED_DERIVED")
    ][["row_index", "protein_id", "new_mag_fold"]].copy()
    seed_rows = seed_rows.merge(
        primary_map[["protein_id", "best_seed_id", "from_seed_alignment"]],
        on="protein_id", how="left", validate="one_to_one",
    )
    if len(seed_rows) != 88_481 or seed_rows.best_seed_id.isna().any():
        raise RuntimeError(
            f"unexpected seed-derived PRIMARY map: rows={len(seed_rows)} "
            f"missing={int(seed_rows.best_seed_id.isna().sum())}"
        )
    if not seed_rows.from_seed_alignment.eq(1).all():
        raise RuntimeError("a seed-derived PRIMARY row lacks alignment provenance")

    seeds = pd.read_csv(
        args.seed_manifest, sep="\t",
        usecols=["seed_id", "outer_fold", "v3_primary_label", "cluster_rep_locked"],
    )
    if seeds.seed_id.duplicated().any():
        raise RuntimeError("seed manifest IDs are not unique")

    transport = pd.read_csv(args.transport_assignments, sep="\t")
    transport_direct = transport[["canonical_seed_id", "direct_assignment"]].drop_duplicates()
    conflicts = transport_direct.groupby("canonical_seed_id").direct_assignment.nunique()
    if (conflicts > 1).any():
        raise RuntimeError("a TRANSPORT seed has conflicting direct assignments")
    direct_transport = transport_direct.drop_duplicates("canonical_seed_id").set_index(
        "canonical_seed_id"
    ).direct_assignment.astype(str).to_dict()

    original_fold = manifest.iloc[original_indices].new_mag_fold.to_numpy(np.int8)
    original_positive = context.label.to_numpy(np.int8) == 1
    seed_by_id = seeds.set_index("seed_id")
    query_ids = set(seed_rows.protein_id.astype(str))
    hit_columns = [
        "query_id", "subject_id", "percent_identity", "alignment_length",
        "mismatches", "gap_opens", "query_start", "query_end", "subject_start",
        "subject_end", "evalue", "bit_score", "query_length", "subject_length",
        "query_coverage", "subject_coverage",
    ]
    outputs: list[pd.DataFrame] = []
    fold_audits: list[dict] = []

    for fold in range(5):
        validation_seeds = set(
            context.loc[original_positive & (original_fold == fold), "reference_seed"]
            .dropna().astype(str)
        )
        validation_seeds.discard("")
        validation_seed_ids_in_manifest = validation_seeds & set(seed_by_id.index)
        validation_clusters = set(
            seed_by_id.loc[
                sorted(validation_seed_ids_in_manifest), "cluster_rep_locked"
            ]
            .dropna().astype(str)
        ) if validation_seed_ids_in_manifest else set()
        hit_path = Path(args.fold_hit_dir) / f"fold_{fold}" / "hits.tsv"
        allowed_id_path = Path(args.fold_hit_dir) / f"fold_{fold}_ALLOWED_SEED_IDS.tsv"
        if not hit_path.is_file():
            raise RuntimeError(f"fold {fold}: missing DIAMOND hit file {hit_path}")
        if not allowed_id_path.is_file():
            raise RuntimeError(f"fold {fold}: missing allowed-seed ID audit {allowed_id_path}")
        allowed_subject_ids = set(
            pd.read_csv(allowed_id_path, sep="\t", dtype=str).seed_id.astype(str)
        )
        if hit_path.stat().st_size:
            allowed_hits = pd.read_csv(
                hit_path, sep="\t", header=None, names=hit_columns,
            )
        else:
            allowed_hits = pd.DataFrame(columns=hit_columns)
        numeric_columns = [
            "percent_identity", "evalue", "bit_score", "query_coverage",
            "subject_coverage",
        ]
        for column in numeric_columns:
            allowed_hits[column] = pd.to_numeric(allowed_hits[column], errors="raise")
        unknown_queries = set(allowed_hits.query_id.astype(str)) - query_ids
        if unknown_queries:
            raise RuntimeError(f"fold {fold}: hit table has unknown queries")
        allowed_hits = allowed_hits.merge(
            seeds,
            left_on="subject_id",
            right_on="seed_id",
            how="left",
            validate="many_to_one",
        )
        if allowed_hits.seed_id.isna().any():
            raise RuntimeError(f"fold {fold}: hit table has unknown seeds")
        disallowed_subjects = set(allowed_hits.subject_id.astype(str)) - allowed_subject_ids
        if disallowed_subjects:
            raise RuntimeError(
                f"fold {fold}: {len(disallowed_subjects)} subjects absent from prepared allowed bank"
            )
        strict = (
            allowed_hits.percent_identity.gt(90.0)
            & allowed_hits.query_coverage.gt(90.0)
            & allowed_hits.subject_coverage.gt(90.0)
        )
        allowed_hits = allowed_hits.loc[strict].copy()
        forbidden = (
            ~allowed_hits.outer_fold.isin([1, 2, 3, 4])
            | allowed_hits.subject_id.astype(str).isin(validation_seeds)
        )
        if forbidden.any():
            raise RuntimeError(
                f"fold {fold}: {int(forbidden.sum())} forbidden DIAMOND subjects present"
            )
        allowed_hits = allowed_hits.sort_values(
            [
                "query_id", "evalue", "bit_score", "percent_identity",
                "query_coverage", "subject_coverage", "subject_id",
            ],
            ascending=[True, True, False, False, False, False, True],
        ).drop_duplicates("query_id")
        best = seed_rows.merge(
            allowed_hits[[
                "query_id", "subject_id", "outer_fold", "v3_primary_label",
                "cluster_rep_locked", "percent_identity", "query_coverage",
                "subject_coverage", "evalue", "bit_score",
            ]],
            left_on="protein_id", right_on="query_id", how="left", validate="one_to_one",
        )
        fold_class: list[str] = []
        provenance: list[str] = []
        for row in best.itertuples(index=False):
            if pd.isna(row.subject_id):
                fold_class.append("EXCLUDED_NO_FOLDSAFE_SEED")
                provenance.append("NO_ALLOWED_TRAINING_SEED_HIT")
            else:
                cls, source = normalize_seed_label(
                    str(row.v3_primary_label), direct_transport, str(row.subject_id)
                )
                fold_class.append(cls)
                provenance.append(source)
        best["fold"] = np.int8(fold)
        best["fold_safe_class"] = fold_class
        best["label_provenance"] = provenance
        best["is_training_eligible"] = best.fold_safe_class.isin(FINAL_CLASSES).astype(np.uint8)
        best["fold_safe_seed_id"] = best.subject_id.fillna("").astype(str)
        best["fold_safe_seed_outer_fold"] = best.outer_fold.fillna(-1).astype(np.int8)
        best["fold_safe_family_id"] = np.where(
            best.is_training_eligible.eq(1),
            "SEED::" + best.fold_safe_seed_id,
            "",
        )

        eligible = best.is_training_eligible.eq(1)
        if best.loc[eligible, "fold_safe_seed_outer_fold"].eq(0).any():
            raise RuntimeError(f"fold {fold}: old fold0 seed selected")
        if best.loc[eligible, "fold_safe_seed_id"].isin(validation_seeds).any():
            raise RuntimeError(f"fold {fold}: validation seed selected")
        if not best.loc[eligible, "fold_safe_class"].isin(FINAL_CLASSES).all():
            raise RuntimeError(f"fold {fold}: invalid output class")

        fold_audits.append({
            "fold": fold,
            "validation_seed_count": len(validation_seeds),
            "validation_seed_in_manifest_count": len(validation_seed_ids_in_manifest),
            "validation_cluster_count": len(validation_clusters),
            "prepared_allowed_seed_count": len(allowed_subject_ids),
            "raw_diamond_hit_rows": int(len(allowed_hits)),
            "unique_diamond_hit_queries": int(allowed_hits.query_id.nunique()),
            "rows": int(len(best)),
            "eligible_rows": int(eligible.sum()),
            "excluded_rows": int((~eligible).sum()),
            "reselected_seed_rows": int(
                (eligible & best.fold_safe_seed_id.ne(best.best_seed_id.astype(str))).sum()
            ),
            "old_fold0_selected": 0,
            "validation_seed_selected": 0,
            "class_counts": {
                str(key): int(value)
                for key, value in best.fold_safe_class.value_counts().sort_index().items()
            },
            "selected_seed_outer_fold_counts": {
                str(int(key)): int(value)
                for key, value in best.loc[eligible, "fold_safe_seed_outer_fold"]
                .value_counts().sort_index().items()
            },
        })
        outputs.append(best[[
            "row_index", "protein_id", "fold", "fold_safe_class",
            "is_training_eligible", "fold_safe_seed_id",
            "fold_safe_seed_outer_fold", "fold_safe_family_id",
            "label_provenance", "best_seed_id", "percent_identity",
            "query_coverage", "subject_coverage", "evalue", "bit_score",
        ]])

    result = pd.concat(outputs, ignore_index=True)
    if len(result) != 5 * len(seed_rows) or result.duplicated(["row_index", "fold"]).any():
        raise RuntimeError("fold-safe label table shape/key contract failed")
    temporary = output / "FOLD_SAFE_PRIMARY_LABELS.parquet.tmp"
    result.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output / "FOLD_SAFE_PRIMARY_LABELS.parquet")
    audit = {
        "state": "FOLD_SAFE_PRIMARY_LABELS_COMPLETE",
        "seed_derived_primary_rows": int(len(seed_rows)),
        "rows": int(len(result)),
        "strict_hit_contract": "independent per-fold DIAMOND; pident>90,qcov>90,scov>90",
        "seed_reference_contract": (
            "old folds 1-4 only; current validation canonical seeds and "
            "exact-sequence aliases excluded before DIAMOND; independent homologs retained"
        ),
        "transport_contract": "direct ARG/MRG provenance only; alignment-only excluded",
        "vf_policy": "VF maps to OTHER",
        "old_fold0_used": False,
        "external_test_used": False,
        "free_text_used": False,
        "folds": fold_audits,
    }
    (output / "FOLD_SAFE_PRIMARY_LABELS_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    (output / "FOLD_SAFE_PRIMARY_LABELS_COMPLETE").touch()
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
