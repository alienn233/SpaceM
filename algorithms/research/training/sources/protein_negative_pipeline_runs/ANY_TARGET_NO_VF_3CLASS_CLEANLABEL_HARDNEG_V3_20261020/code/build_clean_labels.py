from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TARGETS = {"ARG", "MRG", "MGE"}

GENERIC_TRANSPORT = re.compile(
    r"\b(transporter|permease|symporter|antiporter|importer|uptake system|"
    r"ion channel|porin|solute carrier|abc[- ]type transport)\b", re.I,
)
HOUSEKEEPING = re.compile(
    r"\b(ribosomal protein|elongation factor|translation factor|"
    r"dna gyrase|topoisomerase|rna polymerase|replicative dna helicase|"
    r"dna repair|mismatch repair|transcription[- ]repair coupling|"
    r"phosphotransferase system|phosphate abc transporter|"
    r"amino acid abc transporter|sugar transporter|atp synthase|"
    r"chaperone|housekeeping|glycolysis|tricarboxylic acid cycle)\b", re.I,
)
MUTATION_TARGET = re.compile(
    r"\b(rpsl|rpsj|rpob|rpoc|gyra|gyrb|parc|pare|fola|folp|fabi|inha|"
    r"embb|katg|pnca|mfd|tuf|ptsi)\b", re.I,
)
ARG_SUPPORT = re.compile(
    r"\b(resistance|resistant|antibiotic|antimicrobial|multidrug|drug efflux|"
    r"efflux pump|beta[- ]lactamase|carbapenemase|biocide|disinfectant|"
    r"antiseptic|chloramphenicol|macrolide|tetracycline|aminoglycoside|"
    r"quinolone|rifamp|glycopeptide|vancomycin|bacitracin|fosfomycin|"
    r"sulfonamide|trimethoprim|methicillin)\b", re.I,
)
METAL_WORD = (
    r"copper|cobalt|nickel|zinc|cadmium|mercury|arsenic|arsenate|arsenite|"
    r"chromate|tellurite|silver|gold|lead|metal"
)
MRG_SUPPORT = re.compile(
    rf"(?:\b(?:{METAL_WORD})\b.{{0,80}}\b(?:resistance|detoxification|efflux|export)\b|"
    rf"\b(?:resistance|detoxification|efflux|export)\b.{{0,80}}\b(?:{METAL_WORD})\b)",
    re.I,
)
MGE_SUPPORT = re.compile(
    r"\b(transposase|integrase|recombinase|relaxase|resolvase|conjugat|"
    r"mobiliz|plasmid|insertion sequence|integron|prophage|bacteriophage|"
    r"phage|capsid|terminase|type iv secretion|replication initiator|tnp[ab]?|"
    r"inti1|mob[a-z0-9]*|virb[0-9]*)\b", re.I,
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_matching_parquet(path: Path, wanted: set[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["protein_id", "Description", "Preferred_name"], batch_size=250_000
    ):
        frame = batch.to_pandas()
        frame["protein_id"] = frame.protein_id.astype(str)
        keep = frame.protein_id.isin(wanted)
        if keep.any():
            pieces.append(frame.loc[keep].copy())
    if not pieces:
        return pd.DataFrame(columns=["protein_id", "Description", "Preferred_name"])
    result = pd.concat(pieces, ignore_index=True)
    return result.drop_duplicates("protein_id", keep="first")


def read_primary_annotations(path: Path, wanted: set[str]) -> pd.DataFrame:
    columns = [
        "canonical_protein_id", "original_description", "Description", "Preferred_name"
    ]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path, sep="\t", usecols=columns, dtype=str, chunksize=250_000,
        compression="gzip", keep_default_na=False,
    ):
        keep = chunk.canonical_protein_id.astype(str).isin(wanted)
        if keep.any():
            selected = chunk.loc[keep].copy()
            selected = selected.rename(columns={"canonical_protein_id": "protein_id"})
            pieces.append(selected)
    if not pieces:
        return pd.DataFrame(columns=["protein_id", *columns[1:]])
    result = pd.concat(pieces, ignore_index=True)
    return result.drop_duplicates("protein_id", keep="first")


def normalized_text(frame: pd.DataFrame) -> pd.Series:
    fields = []
    for column in ["original_description", "Description", "Preferred_name"]:
        if column in frame:
            fields.append(frame[column].fillna("").astype(str).replace("-", ""))
        else:
            fields.append(pd.Series("", index=frame.index, dtype=str))
    return (fields[0] + " | " + fields[1] + " | " + fields[2]).str.lower()


def support_for_class(text: str, cls: str) -> bool:
    if cls == "ARG":
        return bool(ARG_SUPPORT.search(text))
    if cls == "MRG":
        return bool(MRG_SUPPORT.search(text))
    if cls == "MGE":
        return bool(MGE_SUPPORT.search(text))
    return False


def risk_for_class(text: str, cls: str) -> tuple[bool, str]:
    risks: list[str] = []
    if MUTATION_TARGET.search(text):
        risks.append("MUTATION_TARGET_OR_CORE_GENE")
    if HOUSEKEEPING.search(text):
        risks.append("HOUSEKEEPING_FUNCTION")
    if GENERIC_TRANSPORT.search(text):
        risks.append("GENERIC_TRANSPORT")
    if cls == "MGE" and not MGE_SUPPORT.search(text) and risks:
        return True, "+".join(risks)
    if cls in {"ARG", "MRG"} and not support_for_class(text, cls) and risks:
        return True, "+".join(risks)
    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-labels", required=True)
    parser.add_argument("--base-foldsafe", required=True)
    parser.add_argument("--original-eggnog", required=True)
    parser.add_argument("--mgnify-background-eggnog", required=True)
    parser.add_argument("--primary-eggnog", required=True)
    parser.add_argument("--primary-amrfinder", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.manifest)
    base = pd.read_parquet(args.base_labels)
    if len(manifest) != 1_134_996 or len(base) != len(manifest):
        raise RuntimeError("canonical row count failed")
    if not np.array_equal(base.row_index.to_numpy(), np.arange(len(base))):
        raise RuntimeError("base label order is not canonical")
    if not np.array_equal(base.protein_id.astype(str), manifest.protein_id.astype(str)):
        raise RuntimeError("base label IDs differ from manifest")

    allowed = manifest.cohort.astype(str).isin({
        "ORIGINAL_SUPERVISED", "MGNIFY_SAFE_W5_PSEUDO_NEGATIVE",
        "PRIMARY2000_POSITIVE", "PRIMARY2000_SAFE_W5_PSEUDO_NEGATIVE",
        "PRIMARY2000_SAFE_W10_ONLY_PSEUDO_NEGATIVE",
    })
    wanted = set(manifest.loc[allowed, "protein_id"].astype(str))
    original_ids = set(manifest.loc[
        allowed & manifest.cohort.astype(str).isin({
            "ORIGINAL_SUPERVISED", "MGNIFY_SAFE_W5_PSEUDO_NEGATIVE"
        }), "protein_id"
    ].astype(str))
    primary_ids = wanted - original_ids

    annotations = [
        read_matching_parquet(Path(args.original_eggnog), original_ids),
        read_matching_parquet(Path(args.mgnify_background_eggnog), original_ids),
        read_primary_annotations(Path(args.primary_eggnog), primary_ids),
    ]
    annotation = pd.concat(annotations, ignore_index=True, sort=False)
    annotation = annotation.drop_duplicates("protein_id", keep="first")
    annotation["annotation_text"] = normalized_text(annotation)
    text_lookup = annotation.set_index("protein_id").annotation_text.to_dict()
    annotation_text = manifest.protein_id.astype(str).map(text_lookup).fillna("").astype(str)

    amr = pd.read_csv(args.primary_amrfinder, sep="\t", dtype=str)
    amr_ids = set(
        amr.assembly_accession.astype(str) + "|" + amr["Protein id"].astype(str)
    )
    protected_direct = (
        manifest.source_tier.fillna("").astype(str).eq("AMRFINDER_SUPPORTED")
        | manifest.protein_id.astype(str).isin(amr_ids)
    ).to_numpy()

    clean = base.copy()
    clean["base_is_training_eligible"] = base.is_training_eligible.to_numpy(np.int8)
    clean["annotation_available"] = annotation_text.ne("").to_numpy(np.int8)
    clean["direct_evidence_protected"] = protected_direct.astype(np.int8)
    clean["is_high_risk_annotation"] = np.zeros(len(clean), dtype=np.int8)
    clean["cleaning_action"] = "KEEP"
    clean["cleaning_reason"] = ""
    clean["hard_negative_multiplier"] = np.ones(len(clean), dtype=np.float32)

    static_excluded = 0
    for row in np.flatnonzero(
        base.is_training_eligible.to_numpy(np.int8).astype(bool)
        & base.final_class.astype(str).isin(TARGETS).to_numpy()
    ):
        cls = str(base.final_class.iat[row])
        risky, reason = risk_for_class(annotation_text.iat[row], cls)
        if risky:
            clean.at[row, "is_high_risk_annotation"] = 1
        if risky and not protected_direct[row]:
            clean.at[row, "final_class"] = "EXCLUDED_AMBIGUOUS_POSITIVE"
            clean.at[row, "final_class_id"] = -1
            clean.at[row, "y_target_no_vf"] = -1
            clean.at[row, "is_training_eligible"] = 0
            clean.at[row, "cleaning_action"] = "EXCLUDE_AMBIGUOUS_POSITIVE"
            clean.at[row, "cleaning_reason"] = reason
            static_excluded += 1

    negative = clean.y_target_no_vf.to_numpy(np.int8) == 0
    for row in np.flatnonzero(negative):
        cls_risk = bool(
            GENERIC_TRANSPORT.search(annotation_text.iat[row])
            or HOUSEKEEPING.search(annotation_text.iat[row])
            or MUTATION_TARGET.search(annotation_text.iat[row])
        )
        if cls_risk:
            clean.at[row, "is_high_risk_annotation"] = 1
            clean.at[row, "hard_negative_multiplier"] = np.float32(2.0)
            clean.at[row, "cleaning_action"] = "HIGH_RISK_HARD_NEGATIVE"
            clean.at[row, "cleaning_reason"] = "GENERIC_TRANSPORT_OR_HOUSEKEEPING"

    fold = pd.read_parquet(args.base_foldsafe)
    required = {"row_index", "fold", "fold_safe_class", "is_training_eligible"}
    if required - set(fold.columns):
        raise RuntimeError("base fold-safe table is incomplete")
    annotation_by_row = annotation_text.to_numpy()
    protected_by_row = protected_direct
    fold["base_is_training_eligible"] = fold.is_training_eligible.to_numpy(np.uint8)
    fold["cleaning_action"] = "KEEP"
    fold["cleaning_reason"] = ""
    dynamic_excluded = 0
    for index in np.flatnonzero(
        fold.is_training_eligible.to_numpy(np.uint8).astype(bool)
        & fold.fold_safe_class.astype(str).isin(TARGETS).to_numpy()
    ):
        row = int(fold.row_index.iat[index])
        cls = str(fold.fold_safe_class.iat[index])
        risky, reason = risk_for_class(annotation_by_row[row], cls)
        if risky and not protected_by_row[row]:
            fold.at[index, "fold_safe_class"] = "EXCLUDED_AMBIGUOUS_POSITIVE"
            fold.at[index, "is_training_eligible"] = np.uint8(0)
            fold.at[index, "fold_safe_family_id"] = ""
            fold.at[index, "cleaning_action"] = "EXCLUDE_AMBIGUOUS_POSITIVE"
            fold.at[index, "cleaning_reason"] = reason
            dynamic_excluded += 1

    label_tmp = output / "CLEAN_SINGLE_LABELS.parquet.tmp"
    clean.to_parquet(label_tmp, index=False, compression="zstd")
    os.replace(label_tmp, output / "CLEAN_SINGLE_LABELS.parquet")
    fold_tmp = output / "CLEAN_FOLD_SAFE_PRIMARY_LABELS.parquet.tmp"
    fold.to_parquet(fold_tmp, index=False, compression="zstd")
    os.replace(fold_tmp, output / "CLEAN_FOLD_SAFE_PRIMARY_LABELS.parquet")

    review = pd.DataFrame({
        "row_index": manifest.row_index,
        "protein_id": manifest.protein_id.astype(str),
        "cohort": manifest.cohort.astype(str),
        "base_class": base.final_class.astype(str),
        "clean_class": clean.final_class.astype(str),
        "cleaning_action": clean.cleaning_action.astype(str),
        "cleaning_reason": clean.cleaning_reason.astype(str),
        "direct_evidence_protected": protected_direct.astype(np.int8),
        "annotation_text": annotation_text,
    })
    review = review[review.cleaning_action.ne("KEEP")]
    review.to_csv(
        output / "LABEL_CURATION_REVIEW.tsv.gz", sep="\t", index=False,
        compression="gzip",
    )

    audit = {
        "state": "CLEAN_LABEL_AUDIT_COMPLETE",
        "rows": int(len(clean)),
        "annotations_matched": int(annotation_text.ne("").sum()),
        "direct_evidence_protected_rows": int(protected_direct.sum()),
        "static_ambiguous_positive_rows_excluded": int(static_excluded),
        "fold_dynamic_ambiguous_positive_rows_excluded": int(dynamic_excluded),
        "high_risk_hard_negative_rows": int(
            ((clean.y_target_no_vf == 0) & (clean.hard_negative_multiplier > 1)).sum()
        ),
        "ambiguous_positive_rows_are_negative": False,
        "primary_w10_only_rows_available_for_expanded_candidate": int(
            manifest.cohort.astype(str).eq("PRIMARY2000_SAFE_W10_ONLY_PSEUDO_NEGATIVE").sum()
        ),
        "free_text_used_for_label_curation": True,
        "free_text_used_for_model_features": False,
        "external_test_used": False,
        "old_fold0_used": False,
        "rules": {
            "direct_primary_amrfinder": "always protected",
            "generic_transport_or_housekeeping_positive": "exclude unless class-specific support",
            "uncertain_positive": "excluded, never converted to negative",
            "generic_transport_or_housekeeping_negative": "2x hard-negative multiplier in expanded arm",
        },
        "clean_final_class_counts": {
            str(k): int(v) for k, v in clean.final_class.value_counts().sort_index().items()
        },
    }
    atomic_json(output / "CLEAN_LABEL_AUDIT.json", audit)
    (output / "CLEAN_LABEL_AUDIT_COMPLETE").touch()
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
