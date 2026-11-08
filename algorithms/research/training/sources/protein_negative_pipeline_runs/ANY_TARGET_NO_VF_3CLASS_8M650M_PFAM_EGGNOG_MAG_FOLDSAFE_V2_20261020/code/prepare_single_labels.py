from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_CLASSES = ("OTHER", "ARG", "MRG", "MGE")
TARGET_CLASSES = {"ARG", "MRG", "MGE"}
ALLOWED_COHORTS = {
    "ORIGINAL_SUPERVISED",
    "MGNIFY_SAFE_W5_PSEUDO_NEGATIVE",
    "PRIMARY2000_POSITIVE",
    "PRIMARY2000_SAFE_W5_PSEUDO_NEGATIVE",
}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def normalize_primary(value: object) -> str:

    text = "" if pd.isna(value) else str(value).strip().upper()
    tokens = {
        token.strip()
        for token in text.replace(";", "+").split("+")
        if token.strip()
    }
    if "ARG" in tokens:
        return "ARG"
    if tokens.intersection({"MRG", "METAL", "BACMET"}):
        return "MRG"
    if tokens.intersection({"MGE", "MOBILOME"}):
        return "MGE"
    if "TRANSPORT_EFFLUX" in tokens or "TRANSPORT" in tokens:
        return "TRANSPORT"
    if tokens.intersection({"VF", "VIRULENCE"}):
        return "VF"
    if not tokens:
        return "OTHER"
    raise RuntimeError(f"unsupported source primary label: {text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--original-context", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--transport-assignments", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.manifest)
    required = {
        "row_index", "cohort", "protein_id", "label", "new_mag_fold",
        "target_class", "family_id", "training_weight",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise RuntimeError(f"manifest missing columns: {missing}")
    if len(manifest) != 1_134_996:
        raise RuntimeError(f"unexpected manifest rows: {len(manifest)}")
    if not np.array_equal(manifest.row_index.to_numpy(), np.arange(len(manifest))):
        raise RuntimeError("manifest row_index is not canonical")
    if manifest.protein_id.astype(str).duplicated().any():
        raise RuntimeError("manifest protein_id is not unique")

    original = pd.read_parquet(
        args.original_context,
        columns=[
            "protein_id", "label", "old_homology_fold", "target_class",
            "seed_primary_label", "seed_source_database", "reference_seed",
        ],
    )
    original_mask = manifest.cohort.astype(str).eq("ORIGINAL_SUPERVISED").to_numpy()
    original_indices = np.flatnonzero(original_mask)
    if len(original_indices) != len(original):
        raise RuntimeError("original context row count differs from manifest cohort")
    if not np.array_equal(
        manifest.iloc[original_indices].protein_id.astype(str).to_numpy(),
        original.protein_id.astype(str).to_numpy(),
    ):
        raise RuntimeError("original context protein order differs from manifest")
    if int((original.old_homology_fold == 0).sum()) != 0:
        raise RuntimeError("sealed old fold0 entered the current manifest")

    seeds = pd.read_csv(
        args.seed_manifest,
        sep="\t",
        usecols=[
            "seed_id", "v3_primary_label", "source_database",
            "original_class", "outer_fold",
        ],
    )
    if seeds.seed_id.astype(str).duplicated().any():
        raise RuntimeError("seed_id is not unique")
    seed_lookup = seeds.set_index(seeds.seed_id.astype(str), drop=False)

    transport = pd.read_csv(args.transport_assignments, sep="\t")
    transport_required = {
        "cohort", "protein_id", "canonical_seed_id", "direct_assignment",
        "strict_assignment", "resolved_assignment",
    }
    if transport_required - set(transport.columns):
        raise RuntimeError(
            f"transport assignments missing columns: {sorted(transport_required-set(transport.columns))}"
        )
    if len(transport) != 14_235:
        raise RuntimeError(f"unexpected selected TRANSPORT rows: {len(transport)}")
    if transport.duplicated(["cohort", "protein_id"]).any():
        raise RuntimeError("transport assignment cohort/protein key is not unique")
    if not transport.resolved_assignment.isin(
        ["ARG", "MRG", "UNRESOLVED", "CONFLICT_BOTH"]
    ).all():
        raise RuntimeError("transport audit contains unsupported resolved assignment")
    transport_lookup = transport.set_index(["cohort", "protein_id"])

    n = len(manifest)
    source_primary = np.full(n, "OTHER", dtype=object)
    final_class = np.full(n, "OTHER", dtype=object)
    eligible = np.ones(n, dtype=bool)
    provenance = np.full(n, "BACKGROUND_OR_PSEUDO_NEGATIVE", dtype=object)
    seed_outer_fold = np.full(n, -1, dtype=np.int8)
    canonical_seed_id = np.full(n, "", dtype=object)
    is_fold_dynamic = np.zeros(n, dtype=np.uint8)

    original_positive_local = np.flatnonzero(original.label.to_numpy(np.int8) == 1)
    for local in original_positive_local:
        row = int(original_indices[local])
        restored = normalize_primary(original.seed_primary_label.iat[local])
        canonical_seed_id[row] = str(original.reference_seed.iat[local])
        source_primary[row] = restored
        if restored == "TRANSPORT":
            key = ("ORIGINAL_SUPERVISED", str(manifest.protein_id.iat[row]))
            if key not in transport_lookup.index:
                raise RuntimeError(f"missing original TRANSPORT audit row: {key}")
            audit = transport_lookup.loc[key]
            if str(audit.canonical_seed_id) != str(original.reference_seed.iat[local]):
                raise RuntimeError(f"original TRANSPORT seed mismatch: {key}")
            direct = str(audit.direct_assignment)
            if direct in {"ARG", "MRG"}:
                final_class[row] = direct
                provenance[row] = "TRANSPORT_DIRECT_PROVENANCE"
            else:
                final_class[row] = "EXCLUDED_TRANSPORT"
                eligible[row] = False
                provenance[row] = "TRANSPORT_NON_DIRECT_EXCLUDED_FORMAL_OOF"
        else:
            final_class[row] = "OTHER" if restored == "VF" else restored
            provenance[row] = "ORIGINAL_CONTEXT_SEED_PRIMARY_LABEL"

    primary_positive = manifest.cohort.astype(str).eq("PRIMARY2000_POSITIVE").to_numpy()
    for row in np.flatnonzero(primary_positive):
        target = str(manifest.target_class.iat[row])
        if target == "SEED_DERIVED":
            family = str(manifest.family_id.iat[row])
            if not family.startswith("SEED::"):
                raise RuntimeError(f"invalid SEED_DERIVED family at row {row}: {family}")
            seed_id = family.removeprefix("SEED::")
            if seed_id not in seed_lookup.index:
                raise RuntimeError(f"seed ID absent from canonical manifest: {seed_id}")
            seed = seed_lookup.loc[seed_id]
            restored = normalize_primary(seed.v3_primary_label)
            seed_outer_fold[row] = np.int8(seed.outer_fold)
            canonical_seed_id[row] = seed_id
            is_fold_dynamic[row] = 1
            provenance[row] = "STATIC_DIAGNOSTIC_ONLY_REPLACED_BY_FOLD_SAFE_SEED_LABEL"
        else:
            restored = normalize_primary(target)
            provenance[row] = "PRIMARY2000_DIRECT_TARGET_CLASS"
        source_primary[row] = restored
        if restored == "TRANSPORT":
            key = ("PRIMARY2000_POSITIVE", str(manifest.protein_id.iat[row]))
            if key not in transport_lookup.index:
                raise RuntimeError(f"missing PRIMARY2000 TRANSPORT audit row: {key}")
            audit = transport_lookup.loc[key]
            if str(audit.canonical_seed_id) != str(seed_id):
                raise RuntimeError(f"PRIMARY2000 TRANSPORT seed mismatch: {key}")
            direct = str(audit.direct_assignment)
            if direct in {"ARG", "MRG"}:
                final_class[row] = direct
                provenance[row] = "TRANSPORT_DIRECT_PROVENANCE"
            else:
                final_class[row] = "EXCLUDED_TRANSPORT"
                eligible[row] = False
                provenance[row] = "TRANSPORT_NON_DIRECT_EXCLUDED_FORMAL_OOF"
        else:
            final_class[row] = "OTHER" if restored == "VF" else restored

    if np.any(~np.isin(final_class[eligible], FINAL_CLASSES)):
        raise RuntimeError("an unsupported final class was produced")
    if np.any(final_class[~eligible] != "EXCLUDED_TRANSPORT"):
        raise RuntimeError("an ineligible row is not an excluded TRANSPORT row")
    y_target = np.where(
        eligible, np.isin(final_class, sorted(TARGET_CLASSES)).astype(np.int8), -1
    ).astype(np.int8)
    class_id_lookup = {name: index for index, name in enumerate(FINAL_CLASSES)}
    class_id = np.asarray([class_id_lookup.get(value, -1) for value in final_class], dtype=np.int8)
    excluded_vf = source_primary == "VF"
    is_transport_source = source_primary == "TRANSPORT"

    old_positive = manifest.label.to_numpy(np.int8) == 1
    explicit = original_mask | primary_positive
    if np.any((source_primary != "OTHER") & ~old_positive):
        raise RuntimeError("a previous negative received a positive provenance class")
    if np.any(old_positive & explicit & (source_primary == "OTHER")):
        raise RuntimeError("an old positive lacks a restored primary class")

    result = pd.DataFrame({
        "row_index": manifest.row_index.to_numpy(np.int64),
        "protein_id": manifest.protein_id.astype(str).to_numpy(),
        "cohort": manifest.cohort.astype(str).to_numpy(),
        "new_mag_fold": manifest.new_mag_fold.to_numpy(np.int8),
        "old_binary_label": manifest.label.to_numpy(np.int8),
        "source_primary_label": source_primary,
        "final_class": final_class,
        "final_class_id": class_id,
        "y_target_no_vf": y_target,
        "is_training_eligible": eligible.astype(np.int8),
        "is_transport_source": is_transport_source.astype(np.int8),
        "is_excluded_vf": excluded_vf.astype(np.int8),
        "label_provenance": provenance,
        "seed_outer_fold_provenance_only": seed_outer_fold,
        "canonical_seed_id_static_provenance_only": canonical_seed_id,
        "is_fold_dynamic_primary_seed_label": is_fold_dynamic,
    })
    if result.final_class.isna().any() or result.final_class_id.isna().any():
        raise RuntimeError("single-label table contains missing labels")
    if not np.all(result.groupby("row_index").size().to_numpy() == 1):
        raise RuntimeError("a row received more than one output label")

    allowed = manifest.cohort.astype(str).isin(ALLOWED_COHORTS).to_numpy()
    by_cohort = (
        result.loc[allowed]
        .groupby(["cohort", "source_primary_label", "final_class"], dropna=False)
        .size().rename("rows").reset_index()
    )
    by_fold = (
        result.loc[original_mask]
        .groupby(["new_mag_fold", "final_class"], dropna=False)
        .size().rename("rows").reset_index()
    )
    temporary = output / "SINGLE_LABELS.parquet.tmp"
    result.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output / "SINGLE_LABELS.parquet")
    by_cohort.to_csv(output / "LABEL_COUNTS_BY_COHORT.tsv", sep="\t", index=False)
    by_fold.to_csv(output / "ORIGINAL_COUNTS_BY_FOLD.tsv", sep="\t", index=False)

    audit = {
        "state": "SINGLE_LABEL_AUDIT_COMPLETE",
        "rows": n,
        "allowed_training_rows": int(allowed.sum()),
        "output_classes": list(FINAL_CLASSES),
        "target_classes": sorted(TARGET_CLASSES),
        "exactly_one_output_class_per_eligible_row": True,
        "vf_policy": "VF and VIRULENCE map to OTHER; VF is not a model output class",
        "transport_policy": "formal OOF accepts direct old ARG-only/MRG-only provenance only; alignment-only/unresolved/conflict excluded, never negative",
        "composite_priority_for_non_transport": ["ARG", "MRG_or_METAL", "MGE_or_MOBILOME", "VF"],
        "allowed_final_class_counts": {
            str(key): int(value)
            for key, value in result.loc[allowed, "final_class"].value_counts().sort_index().items()
        },
        "allowed_source_primary_counts": {
            str(key): int(value)
            for key, value in result.loc[allowed, "source_primary_label"].value_counts().sort_index().items()
        },
        "original_final_class_counts": {
            str(key): int(value)
            for key, value in result.loc[original_mask, "final_class"].value_counts().sort_index().items()
        },
        "primary2000_final_class_counts": {
            str(key): int(value)
            for key, value in result.loc[primary_positive, "final_class"].value_counts().sort_index().items()
        },
        "eligible_training_rows": int((allowed & eligible).sum()),
        "target_positive_rows_allowed": int((result.loc[allowed & eligible, "y_target_no_vf"] == 1).sum()),
        "excluded_vf_rows_allowed": int(result.loc[allowed, "is_excluded_vf"].sum()),
        "transport_source_rows_allowed": int(result.loc[allowed, "is_transport_source"].sum()),
        "transport_rows_resolved_arg": int(((source_primary == "TRANSPORT") & (final_class == "ARG") & allowed).sum()),
        "transport_rows_resolved_mrg": int(((source_primary == "TRANSPORT") & (final_class == "MRG") & allowed).sum()),
        "transport_rows_excluded": int(((source_primary == "TRANSPORT") & ~eligible & allowed).sum()),
        "transport_non_direct_rows_excluded": int(
            ((source_primary == "TRANSPORT") & ~eligible & allowed).sum()
        ),
        "fold_dynamic_primary_seed_rows": int(is_fold_dynamic[allowed].sum()),
        "static_diagnostic_old_fold0_seed_rows": int(
            ((is_fold_dynamic == 1) & (seed_outer_fold == 0) & allowed).sum()
        ),
        "old_fold0_used_for_training": False,
        "excluded_transport_rows_are_negative": False,
        "old_fold0_used": False,
        "external_test_used": False,
        "free_text_used": False,
    }
    atomic_json(output / "LABEL_AUDIT.json", audit)
    (output / "LABEL_AUDIT_COMPLETE").touch()
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
