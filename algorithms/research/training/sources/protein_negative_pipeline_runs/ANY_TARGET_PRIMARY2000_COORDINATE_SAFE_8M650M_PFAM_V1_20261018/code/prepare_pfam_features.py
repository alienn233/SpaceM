from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PFAM_RE = re.compile(r"PF\d{5}", re.IGNORECASE)
OLD_COLUMNS = [
    "protein_id", "center_pfams", "w10_neighbor_pfams",
    "w10_pfam_hit_fraction", "w10_pfam_domain_count",
    "w10_pfam_diversity", "w10_pfam_missing",
]
OUTPUT_NUMERIC = [
    "center_pfam_hit", "center_pfam_match_count", "center_pfam_diversity",
    "w10_pfam_hit_fraction", "w10_pfam_domain_count",
    "w10_pfam_diversity", "w10_pfam_missing",
]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def normalize_tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [token.upper() for token in PFAM_RE.findall(str(value))]


def normalized_text(value: object) -> str:
    return "|".join(normalize_tokens(value))


def load_old_block(path: Path, wanted: pd.DataFrame, require_w5: bool) -> pd.DataFrame:
    columns = list(OLD_COLUMNS)
    if require_w5:
        columns.append("within_positive_w5")
    frame = pd.read_parquet(path, columns=columns)
    if require_w5:
        frame = frame[frame.within_positive_w5.astype(bool)].drop(columns="within_positive_w5")
    if frame.protein_id.duplicated().any():
        raise RuntimeError(f"duplicate old Pfam protein IDs in {path}")
    expected = set(wanted.protein_id.astype(str))
    observed = set(frame.protein_id.astype(str))
    if observed != expected:
        raise RuntimeError(
            f"old Pfam ID mismatch in {path}: missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    frame = wanted[["row_index", "protein_id"]].merge(
        frame, on="protein_id", how="left", validate="one_to_one"
    ).sort_values("row_index")
    frame["center_pfam_tokens"] = frame.center_pfams.map(normalized_text)
    frame["w10_neighbor_pfam_tokens"] = frame.w10_neighbor_pfams.map(normalized_text)
    frame["center_pfam_hit"] = frame.center_pfam_tokens.str.len().gt(0).astype(np.float32)
    frame["center_pfam_match_count"] = frame.center_pfams.map(
        lambda value: len(normalize_tokens(value))
    ).astype(np.float32)
    frame["center_pfam_diversity"] = frame.center_pfams.map(
        lambda value: len(set(normalize_tokens(value)))
    ).astype(np.float32)
    for column in OUTPUT_NUMERIC[3:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(np.float32)
    return frame[[
        "row_index", "protein_id", "center_pfam_tokens",
        "w10_neighbor_pfam_tokens", *OUTPUT_NUMERIC,
    ]]


def primary_rows(
    manifest: pd.DataFrame,
    gene_order_path: Path,
    per_genome: Path,
    provenance: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    wanted_frame = manifest[manifest.cohort.astype(str).str.startswith("PRIMARY2000")].copy()
    wanted_by_assembly = {
        assembly: dict(zip(part.protein_id.astype(str), part.row_index.astype(int)))
        for assembly, part in wanted_frame.groupby("assembly_accession", sort=False)
    }
    order_source = provenance.set_index("assembly_accession").order_source.astype(str).to_dict()
    pf = pq.ParquetFile(gene_order_path)
    output: list[dict] = []
    found: set[str] = set()
    annotation_missing = 0
    per_genome_extra = 0
    no_gff_rows = 0
    for group_index in range(pf.num_row_groups):
        frame = pf.read_row_group(
            group_index,
            columns=["assembly_accession", "contig_id", "protein_id", "gene_order"],
        ).to_pandas()
        assemblies = frame.assembly_accession.astype(str).unique()
        if len(assemblies) != 1:
            raise RuntimeError("GENE_ORDER row group is not assembly-pure")
        assembly = str(assemblies[0])
        wanted = wanted_by_assembly.get(assembly)
        if not wanted:
            continue
        annotation_path = per_genome / f"{assembly}.pfam.tsv.gz"
        if not annotation_path.exists():
            raise RuntimeError(f"missing per-genome Pfam file: {annotation_path}")
        annotation = pd.read_csv(
            annotation_path,
            sep="\t",
            usecols=[
                "canonical_protein_id", "assembly_accession", "pfam_hit",
                "pfam_match_count", "pfam_accessions",
            ],
            low_memory=False,
        )
        if annotation.canonical_protein_id.duplicated().any():
            raise RuntimeError(f"duplicate Pfam IDs in {annotation_path}")
        if not annotation.assembly_accession.astype(str).eq(assembly).all():
            raise RuntimeError(f"mixed assembly IDs in {annotation_path}")
        ann_ids = set(annotation.canonical_protein_id.astype(str))
        order_ids = set(frame.protein_id.astype(str))
        missing = order_ids - ann_ids
        if missing:
            annotation_missing += len(missing)
            raise RuntimeError(f"{assembly}: Pfam annotations miss {len(missing)} GENE_ORDER proteins")
        per_genome_extra += len(ann_ids - order_ids)
        annotation["tokens"] = annotation.pfam_accessions.map(normalize_tokens)
        annotation["match_count"] = pd.to_numeric(
            annotation.pfam_match_count, errors="coerce"
        ).fillna(0).astype(int)
        token_map = annotation.set_index("canonical_protein_id").tokens.to_dict()
        match_map = annotation.set_index("canonical_protein_id").match_count.to_dict()
        is_no_gff = order_source.get(assembly) == "PREDICTED_FAA_IDENTIFIER_ORDER"
        for _contig, contig in frame.groupby("contig_id", sort=False):
            contig = contig.sort_values(["gene_order", "protein_id"])
            identifiers = contig.protein_id.astype(str).tolist()
            positions = {protein_id: index for index, protein_id in enumerate(identifiers)}
            for protein_id in wanted.keys() & positions.keys():
                center = positions[protein_id]
                center_tokens = token_map[protein_id]
                ordered: list[str] = []
                domain_set: set[str] = set()
                hit_genes = 0
                valid_genes = 0
                domain_count = 0
                for relative in list(range(-10, 0)) + list(range(1, 11)):
                    position = center + relative
                    if position < 0 or position >= len(identifiers):
                        continue
                    neighbor = identifiers[position]
                    tokens = token_map[neighbor]
                    valid_genes += 1
                    if tokens:
                        hit_genes += 1
                    ordered.extend(tokens)
                    domain_set.update(tokens)
                    domain_count += int(match_map[neighbor])
                output.append({
                    "row_index": int(wanted[protein_id]),
                    "protein_id": protein_id,
                    "center_pfam_tokens": "|".join(center_tokens),
                    "w10_neighbor_pfam_tokens": "|".join(ordered),
                    "center_pfam_hit": np.float32(bool(center_tokens)),
                    "center_pfam_match_count": np.float32(match_map[protein_id]),
                    "center_pfam_diversity": np.float32(len(set(center_tokens))),
                    "w10_pfam_hit_fraction": np.float32(hit_genes / valid_genes if valid_genes else 0.0),
                    "w10_pfam_domain_count": np.float32(domain_count),
                    "w10_pfam_diversity": np.float32(len(domain_set)),
                    "w10_pfam_missing": np.float32(domain_count == 0),
                })
                found.add(protein_id)
                no_gff_rows += int(is_no_gff)
        if (group_index + 1) % 25 == 0:
            print(json.dumps({
                "event": "primary_pfam_progress", "row_groups": group_index + 1,
                "rows": len(output),
            }), flush=True)
    expected = set(wanted_frame.protein_id.astype(str))
    if found != expected:
        raise RuntimeError(
            f"PRIMARY Pfam feature assembly mismatch: missing={len(expected-found)} extra={len(found-expected)}"
        )
    result = pd.DataFrame(output).sort_values("row_index").reset_index(drop=True)
    return result, {
        "primary_rows": len(result),
        "primary_assemblies": len(wanted_by_assembly),
        "no_gff_ordinal_only_model_rows": no_gff_rows,
        "annotation_missing_from_gene_order": annotation_missing,
        "annotation_rows_not_in_gene_order": per_genome_extra,
    }


def token_universe(texts: pd.Series) -> Counter:
    counter: Counter = Counter()
    for text in texts.fillna(""):
        counter.update(set(str(text).split("|") if text else []))
    counter.pop("", None)
    return counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--pfam-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_root = Path(args.model_root)
    context_root = Path(args.context_root)
    primary_root = Path(args.primary_root)
    pfam_root = Path(args.pfam_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "PFAM_FEATURES_COMPLETE"
    if marker.exists():
        print((output / "PFAM_AUDIT.json").read_text(), end="")
        return
    if not (pfam_root / "COMPLETE").exists():
        raise RuntimeError("PRIMARY2000 Pfam COMPLETE marker is absent")

    manifest = pd.read_parquet(model_root / "ROW_MANIFEST.parquet")
    if not np.array_equal(manifest.row_index.to_numpy(), np.arange(len(manifest))):
        raise RuntimeError("ROW_MANIFEST row_index is not canonical")
    if manifest.protein_id.duplicated().any():
        raise RuntimeError("ROW_MANIFEST protein IDs are not unique")

    original_wanted = manifest[manifest.cohort.eq("ORIGINAL_SUPERVISED")]
    background_wanted = manifest[manifest.cohort.eq("MGNIFY_SAFE_W5_PSEUDO_NEGATIVE")]
    original = load_old_block(
        context_root / "CONTEXT_FEATURES.parquet", original_wanted, False
    )
    background = load_old_block(
        context_root / "ADDITIONAL_W10_BACKGROUND_NEGATIVES_V1" / "CONTEXT_FEATURES.parquet",
        background_wanted,
        True,
    )
    provenance = pd.read_csv(model_root / "PRIMARY2000_COORDINATE_PROVENANCE.tsv", sep="\t")
    if int(provenance.order_source.eq("PREDICTED_FAA_IDENTIFIER_ORDER").sum()) != 320:
        raise RuntimeError("expected exactly 320 no-GFF ordinal-only assemblies")
    primary, primary_audit = primary_rows(
        manifest,
        primary_root / "02_context" / "GENE_ORDER.parquet",
        pfam_root / "03_all_genes" / "per_genome",
        provenance,
    )
    combined = pd.concat([original, background, primary], ignore_index=True).sort_values("row_index")
    if len(combined) != len(manifest):
        raise RuntimeError(f"Pfam row count mismatch: {len(combined)} != {len(manifest)}")
    if not np.array_equal(combined.row_index.to_numpy(), np.arange(len(manifest))):
        raise RuntimeError("Pfam row order is incomplete")
    if not np.array_equal(combined.protein_id.astype(str), manifest.protein_id.astype(str)):
        raise RuntimeError("Pfam protein IDs do not align to ROW_MANIFEST")
    if not np.isfinite(combined[OUTPUT_NUMERIC].to_numpy(dtype=np.float32)).all():
        raise RuntimeError("non-finite Pfam numeric features")
    malformed = []
    for column in ("center_pfam_tokens", "w10_neighbor_pfam_tokens"):
        for text in combined[column]:
            if text and any(not re.fullmatch(r"PF\d{5}", token) for token in str(text).split("|")):
                malformed.append((column, text))
                break
    if malformed:
        raise RuntimeError(f"malformed normalized Pfam tokens: {malformed[:3]}")

    temporary = output / "PFAM_FEATURES.parquet.tmp"
    combined.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output / "PFAM_FEATURES.parquet")

    old_rows = pd.concat([original, background], ignore_index=True)
    old_tokens = token_universe(
        old_rows.center_pfam_tokens + "|" + old_rows.w10_neighbor_pfam_tokens
    )
    new_tokens = token_universe(
        primary.center_pfam_tokens + "|" + primary.w10_neighbor_pfam_tokens
    )
    cohort_stats = {}
    aligned = combined.merge(manifest[["row_index", "cohort"]], on="row_index", validate="one_to_one")
    for cohort, part in aligned.groupby("cohort", sort=True):
        cohort_stats[str(cohort)] = {
            "rows": int(len(part)),
            "center_pfam_hit_rows": int(part.center_pfam_hit.astype(bool).sum()),
            "center_pfam_hit_fraction": float(part.center_pfam_hit.mean()),
            "w10_all_missing_rows": int(part.w10_pfam_missing.astype(bool).sum()),
            "mean_w10_hit_fraction": float(part.w10_pfam_hit_fraction.mean()),
            "mean_w10_match_count": float(part.w10_pfam_domain_count.mean()),
        }
    no_gff_expected = int(json.loads(
        (model_root / "NO_PSEUDO_COORDINATE_AUDIT.json").read_text()
    )["pseudo_assembly_model_rows"])
    if primary_audit["no_gff_ordinal_only_model_rows"] != no_gff_expected:
        raise RuntimeError(
            "no-GFF model-row count changed: "
            f"{primary_audit['no_gff_ordinal_only_model_rows']} != {no_gff_expected}"
        )
    audit = {
        "state": "PFAM_FEATURES_AUDITED_COMPLETE",
        "rows": int(len(combined)),
        "row_order_exact": True,
        "protein_ids_unique": True,
        "old_source": {
            "format": "comma/pipe separated Pfam accessions in existing context parquet",
            "database_version": "not encoded in the context parquet",
            "rows": int(len(old_rows)),
            "unique_tokens": int(len(old_tokens)),
        },
        "primary_source": {
            "format": "Pfam38 semicolon separated pfam_accessions plus pfam_match_count",
            "database_version": "Pfam 38 via completed all-gene InterProScan workflow",
            "rows": int(len(primary)),
            "unique_tokens": int(len(new_tokens)),
        },
        "normalization": "regex PF followed by exactly five digits; upper-case; binary model tokens are version-free",
        "shared_old_primary_tokens": int(len(set(old_tokens) & set(new_tokens))),
        "old_only_tokens": int(len(set(old_tokens) - set(new_tokens))),
        "primary_only_tokens": int(len(set(new_tokens) - set(old_tokens))),
        "coordinate_features_used": False,
        "gene_order_usage": "relative W10 membership only",
        "no_gff_ordinal_only_assemblies": 320,
        "no_gff_ordinal_only_model_rows": no_gff_expected,
        "pseudo_bp_coordinates_used": False,
        "description_or_interpro_text_used": False,
        "eggnog_used": False,
        "old_fold0_used": False,
        "numeric_columns": OUTPUT_NUMERIC,
        "cohort_stats": cohort_stats,
        **primary_audit,
    }
    atomic_json(output / "PFAM_AUDIT.json", audit)
    marker.touch()
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
