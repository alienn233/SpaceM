from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


NUMERIC = [
    "w10_eggnog_neighbor_count", "w10_eggnog_hit_fraction",
    "w10_cog_hit_fraction", "w10_go_hit_fraction", "w10_ec_hit_fraction",
    "w10_kegg_hit_fraction", "w10_cazy_hit_fraction",
]
PREFIXES = ("OG", "COG", "GO", "EC", "KO", "KPATH", "KMOD", "KRXN", "BRITE", "KTC", "CAZY", "PFAM")
FIELD_PREFIX = {
    "eggNOG_OGs": "OG", "COG_category": "COG", "GOs": "GO", "EC": "EC",
    "KEGG_ko": "KO", "KEGG_Pathway": "KPATH", "KEGG_Module": "KMOD",
    "KEGG_Reaction": "KRXN", "BRITE": "BRITE", "KEGG_TC": "KTC",
    "CAZy": "CAZY", "PFAMs": "PFAM",
}
ANNOTATION_COLUMNS = ["canonical_protein_id", "eggnog_hit", *FIELD_PREFIX]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def canonical_token(prefix: str, value: str) -> str:
    value = value.strip()
    if prefix == "OG":
        value = value.split("|", 1)[0]
    elif prefix == "GO" and value.startswith("GO:"):
        value = value[3:]
    elif prefix == "KO" and value.startswith("ko:"):
        value = value[3:]
    return f"{prefix}:{value}"


def normalize_identifier_fields(row: object) -> str:
    tokens = set()
    for field, prefix in FIELD_PREFIX.items():
        value = getattr(row, field)
        if value is None or str(value) in {"", "-", "NA", "nan"}:
            continue
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item and item not in {"-", "NA"}:
                tokens.add(canonical_token(prefix, item))
    return "|".join(sorted(tokens))


def normalize_legacy_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    tokens = set()
    for raw in str(value).split("|"):
        if ":" not in raw:
            continue
        prefix, item = raw.split(":", 1)
        if prefix not in PREFIXES or not item:
            continue
        tokens.add(canonical_token(prefix, item))
    return "|".join(sorted(tokens))


def load_old(path: Path, wanted: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["protein_id", *NUMERIC, "w10_neighbor_eggnog_tokens"])
    if frame.protein_id.duplicated().any():
        raise RuntimeError(f"duplicate protein IDs in {path}")
    frame = wanted[["row_index", "protein_id"]].merge(frame, on="protein_id", how="left", validate="one_to_one")
    if frame[NUMERIC].isna().any().any() or frame.w10_neighbor_eggnog_tokens.isna().any():
        raise RuntimeError(f"missing old eggNOG features after alignment: {path}")
    frame["w10_neighbor_eggnog_tokens"] = frame.w10_neighbor_eggnog_tokens.map(normalize_legacy_text)
    for column in NUMERIC:
        frame[column] = frame[column].astype(np.float32)
    return frame[["row_index", "protein_id", *NUMERIC, "w10_neighbor_eggnog_tokens"]]


def primary_features(manifest: pd.DataFrame, gene_order: Path, per_genome: Path) -> tuple[pd.DataFrame, dict]:
    wanted_frame = manifest[manifest.cohort.astype(str).str.startswith("PRIMARY2000")]
    wanted_by_assembly = {
        assembly: dict(zip(part.protein_id.astype(str), part.row_index.astype(int)))
        for assembly, part in wanted_frame.groupby("assembly_accession", sort=False)
    }
    output = []
    found = set()
    annotation_rows = annotation_hits = 0
    pf = pq.ParquetFile(gene_order)
    for group_index in range(pf.num_row_groups):
        order = pf.read_row_group(
            group_index, columns=["assembly_accession", "contig_id", "protein_id", "gene_order"]
        ).to_pandas()
        assemblies = order.assembly_accession.astype(str).unique()
        if len(assemblies) != 1:
            raise RuntimeError("GENE_ORDER row group is not assembly-pure")
        assembly = str(assemblies[0])
        wanted = wanted_by_assembly.get(assembly)
        if not wanted:
            continue
        annotation_path = per_genome / f"{assembly}.eggnog.tsv.gz"
        annotation = pd.read_csv(annotation_path, sep="\t", usecols=ANNOTATION_COLUMNS, low_memory=False)
        if annotation.canonical_protein_id.duplicated().any():
            raise RuntimeError(f"duplicate eggNOG IDs in {annotation_path}")
        if set(annotation.canonical_protein_id.astype(str)) != set(order.protein_id.astype(str)):
            raise RuntimeError(f"eggNOG/GENE_ORDER ID mismatch for {assembly}")
        annotation_rows += len(annotation)
        annotation_hits += int(annotation.eggnog_hit.astype(int).sum())
        annotation["tokens"] = [normalize_identifier_fields(row) for row in annotation.itertuples(index=False)]
        annotation["has_eggnog"] = annotation.eggnog_hit.astype(np.int8)
        def present(column: str) -> pd.Series:
            values = annotation[column].fillna("-").astype(str)
            return (~values.isin(["", "-", "NA", "nan"])).astype(np.int8)

        annotation["has_cog"] = present("COG_category")
        annotation["has_go"] = present("GOs")
        annotation["has_ec"] = present("EC")
        annotation["has_kegg"] = present("KEGG_ko")
        annotation["has_cazy"] = present("CAZy")
        ann = annotation.set_index("canonical_protein_id")
        for _contig, contig in order.groupby("contig_id", sort=False):
            identifiers = contig.sort_values(["gene_order", "protein_id"]).protein_id.astype(str).tolist()
            positions = {protein_id: index for index, protein_id in enumerate(identifiers)}
            for protein_id in wanted.keys() & positions.keys():
                center = positions[protein_id]
                neighbor_ids = [
                    identifiers[position]
                    for offset in [*range(-10, 0), *range(1, 11)]
                    if 0 <= (position := center + offset) < len(identifiers)
                ]
                neighbors = ann.loc[neighbor_ids]
                valid = len(neighbors)
                tokens = "|".join(text for text in neighbors.tokens if isinstance(text, str) and text)
                output.append({
                    "row_index": wanted[protein_id], "protein_id": protein_id,
                    "w10_eggnog_neighbor_count": np.float32(neighbors.has_eggnog.sum()),
                    "w10_eggnog_hit_fraction": np.float32(neighbors.has_eggnog.mean() if valid else 0),
                    "w10_cog_hit_fraction": np.float32(neighbors.has_cog.mean() if valid else 0),
                    "w10_go_hit_fraction": np.float32(neighbors.has_go.mean() if valid else 0),
                    "w10_ec_hit_fraction": np.float32(neighbors.has_ec.mean() if valid else 0),
                    "w10_kegg_hit_fraction": np.float32(neighbors.has_kegg.mean() if valid else 0),
                    "w10_cazy_hit_fraction": np.float32(neighbors.has_cazy.mean() if valid else 0),
                    "w10_neighbor_eggnog_tokens": normalize_legacy_text(tokens),
                })
                found.add(protein_id)
        if (group_index + 1) % 25 == 0:
            print(json.dumps({"event": "primary_eggnog_features", "row_groups": group_index + 1, "rows": len(output)}), flush=True)
    expected = set(wanted_frame.protein_id.astype(str))
    if found != expected:
        raise RuntimeError(f"PRIMARY feature mismatch missing={len(expected-found)} extra={len(found-expected)}")
    return pd.DataFrame(output).sort_values("row_index"), {
        "primary_model_rows": len(output), "primary_model_assemblies": len(wanted_by_assembly),
        "primary_all_gene_rows_read": annotation_rows, "primary_all_gene_hits_read": annotation_hits,
    }


def token_counter(texts: pd.Series) -> Counter:
    result = Counter()
    for text in texts:
        if text:
            result.update(set(str(text).split("|")))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--eggnog-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model_root, context_root = Path(args.model_root), Path(args.context_root)
    primary_root, eggnog_root, output = Path(args.primary_root), Path(args.eggnog_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "EGGNOG_FEATURES_COMPLETE"
    if marker.exists():
        print((output / "EGGNOG_FEATURE_AUDIT.json").read_text(), end="")
        return
    if not (eggnog_root / "STRICT_AUDIT_COMPLETE").exists():
        raise RuntimeError("strict all-gene eggNOG audit is incomplete")
    manifest = pd.read_parquet(model_root / "ROW_MANIFEST.parquet")
    if len(manifest) != 1_134_996 or not np.array_equal(manifest.row_index, np.arange(len(manifest))):
        raise RuntimeError("ROW_MANIFEST contract failed")
    original = load_old(
        context_root / "EGGNOG_CONTEXT_FEATURES.parquet",
        manifest[manifest.cohort.eq("ORIGINAL_SUPERVISED")],
    )
    background = load_old(
        context_root / "ADDITIONAL_W10_BACKGROUND_NEGATIVES_V1" / "EGGNOG_CONTEXT_FEATURES.parquet",
        manifest[manifest.cohort.eq("MGNIFY_SAFE_W5_PSEUDO_NEGATIVE")],
    )
    primary, primary_audit = primary_features(
        manifest, primary_root / "02_context" / "GENE_ORDER.parquet",
        eggnog_root / "03_all_genes" / "per_genome",
    )
    combined = pd.concat([original, background, primary], ignore_index=True).sort_values("row_index").reset_index(drop=True)
    if len(combined) != len(manifest) or not np.array_equal(combined.row_index, np.arange(len(manifest))):
        raise RuntimeError("combined eggNOG row alignment failed")
    if not np.array_equal(combined.protein_id.astype(str), manifest.protein_id.astype(str)):
        raise RuntimeError("combined eggNOG protein IDs do not match ROW_MANIFEST")
    if combined.protein_id.duplicated().any() or not np.isfinite(combined[NUMERIC].to_numpy(np.float32)).all():
        raise RuntimeError("combined eggNOG uniqueness/finite contract failed")
    malformed = [
        token for text in combined.w10_neighbor_eggnog_tokens for token in str(text).split("|")
        if token and token.split(":", 1)[0] not in PREFIXES
    ]
    if malformed:
        raise RuntimeError(f"malformed normalized tokens: {malformed[:10]}")
    temporary = output / "EGGNOG_FEATURES.parquet.tmp"
    combined.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output / "EGGNOG_FEATURES.parquet")
    old_tokens = token_counter(pd.concat([original.w10_neighbor_eggnog_tokens, background.w10_neighbor_eggnog_tokens]))
    new_tokens = token_counter(primary.w10_neighbor_eggnog_tokens)
    cohort_stats = {}
    aligned = combined.merge(manifest[["row_index", "cohort"]], on="row_index", validate="one_to_one")
    for cohort, part in aligned.groupby("cohort", sort=True):
        cohort_stats[str(cohort)] = {
            "rows": len(part), "mean_neighbor_hits": float(part.w10_eggnog_neighbor_count.mean()),
            "mean_hit_fraction": float(part.w10_eggnog_hit_fraction.mean()),
            "token_nonempty_rows": int(part.w10_neighbor_eggnog_tokens.astype(bool).sum()),
        }
    audit = {
        "state": "EGGNOG_FEATURES_AUDITED_COMPLETE", "rows": len(combined),
        "row_order_exact": True, "protein_ids_unique": True, "numeric_finite": True,
        "numeric_columns": NUMERIC, "token_namespaces": list(PREFIXES),
        "normalization": "namespace identifiers only; OG taxonomy display names removed; GO/KO prefixes canonicalized",
        "old_unique_tokens": len(old_tokens), "primary_unique_tokens": len(new_tokens),
        "shared_old_primary_tokens": len(set(old_tokens) & set(new_tokens)),
        "free_text_columns_read": [], "description_used": False, "preferred_name_used": False,
        "seed_ortholog_used": False, "old_fold0_used": False,
        "gene_order_usage": "relative W10 membership only; no coordinates",
        "cohort_stats": cohort_stats, **primary_audit,
    }
    atomic_json(output / "EGGNOG_FEATURE_AUDIT.json", audit)
    marker.touch()
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
