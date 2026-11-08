from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_string_dtype


BASE = Path("/ifs1/User/yanyueyang/yyy/gene-finder")
DL = BASE / "MGnify_DOWNLOAD_20261010"
RUNS = BASE / "protein_negative_pipeline_runs"
OLD = RUNS / "MGNIFY_ESM2_8M_BINARY_92664_POS_926640_NEG_20261014"
STRICT = RUNS / "MGNIFY_TWO_PASS_STRICT_HITS_20261013"
FINAL_POS = RUNS / "MGNIFY_FINAL_ORIGINAL_FIRST90_SECOND6718_20261014"
SPECIALIZED = RUNS / "MGNIFY_AMRFINDER_MOBILOME_UNION_DIAMOND_20261013"
POS_DEDUP = RUNS / "MGNIFY_MAG_FILTERED_ONLY_DEDUP_100_95_90_20261014"
NEG_SAMPLE = RUNS / "MGNIFY_MMSEQS30_FILTERED_ESM2_SAMPLE_926640_NO_POS_EXACT_20261015"
RESCUE = RUNS / "MGNIFY_ANNOTATION_RESCUE_20261014_V3"
SEEDS = RUNS / "ANY_TARGET_FULL_SELF_PUBLIC_HMM_V1_20261012"

ENV = {
    "HG": ("human_gut", "human-gut/v3.0", "v4.0.0"),
    "MAR": ("marine", "marine/v2.0", "v2.3.0"),
    "SOIL": ("soil", "soil/v1.0", "v2.5.0"),
}
ENV_FROM_NAME = {v[0]: k for k, v in ENV.items()}
ID_RE = re.compile(r"^(HG|MAR|SOIL)__(MGYG\d{9})__(\d+)$")
MAG_RE = re.compile(r"/(MGYG\d{9})/genome/")
RESCUED = {"SIGNIFICANT_GENE_RESCUED", "ENRICHED_SAME_GENE_3OF4_RESCUED"}
_RESCUED_LOOKUP: dict[str, tuple[str, str]] | None = None


def open_text(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="replace")
    return path.open(mode, encoding="utf-8", errors="replace")


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def resolve_genome_file(env_code: str, mag: str, suffix: str) -> Path | None:
    rel = ENV[env_code][1]
    stem = DL / "03_raw" / rel / mag[:11] / mag / "genome" / f"{mag}{suffix}"
    for candidate in (stem, Path(str(stem) + ".gz")):
        if candidate.is_file():
            return candidate
    return None


def read_downloaded_mags() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    file_names = {
        "HG": "human-gut_v3.0.txt",
        "MAR": "marine_v2.0.txt",
        "SOIL": "soil_v1.0.txt",
    }
    for code, name in file_names.items():
        mags: set[str] = set()
        path = DL / "00_config" / "aspera_filelists" / name
        with path.open() as handle:
            for line in handle:
                match = MAG_RE.search(line)
                if match:
                    mags.add(match.group(1))
        result[code] = mags
    return result


def load_mag_qc() -> pd.DataFrame:
    downloaded = read_downloaded_mags()
    rows = []
    for code, (environment, rel, pipeline_version) in ENV.items():
        path = DL / "01_metadata" / rel / "genomes-all_metadata.tsv"
        use = [
            "Genome", "Genome_type", "Length", "N_contigs", "N50", "GC_content",
            "Completeness", "Contamination", "Species_rep", "Lineage",
            "Sample_accession", "Study_accession", "Country", "Continent",
        ]
        frame = pd.read_csv(path, sep="\t", usecols=use, low_memory=False)
        frame = frame[frame["Genome"].astype(str).isin(downloaded[code])].copy()
        frame.rename(columns={
            "Genome": "mag_id", "Genome_type": "genome_type", "Length": "genome_length",
            "N_contigs": "n_contigs", "N50": "n50", "GC_content": "metadata_gc_percent",
            "Completeness": "completeness", "Contamination": "contamination",
            "Species_rep": "species_rep", "Lineage": "lineage",
            "Sample_accession": "sample_accession", "Study_accession": "study_accession",
            "Country": "country", "Continent": "continent",
        }, inplace=True)
        frame.insert(0, "environment", environment)
        frame.insert(0, "env_code", code)
        frame["catalogue_release"] = rel.split("/")[-1]
        frame["mgnify_pipeline_version"] = pipeline_version
        frame["quality_source"] = "MGnify genomes-all_metadata.tsv"
        frame["quality_tool_version"] = "not_stated_in_local_release_README"
        frame["qc_pass"] = (frame["completeness"] >= 80.0) & (frame["contamination"] <= 5.0)
        for suffix, col in [(".fna", "has_fna"), (".faa", "has_faa"), (".gff", "has_gff"), ("_eggNOG.tsv", "has_eggnog")]:
            frame[col] = [resolve_genome_file(code, mag, suffix) is not None for mag in frame["mag_id"]]
        frame["raw_complete"] = frame[["has_fna", "has_faa", "has_gff", "has_eggnog"]].all(axis=1)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def parse_seq_id(seq_id: str) -> tuple[str, str, str, str]:
    match = ID_RE.match(seq_id)
    if not match:
        raise ValueError(f"unexpected sequence ID: {seq_id}")
    code, mag, gene = match.groups()
    return code, mag, gene, f"{mag}_{gene}"


def load_seed_classes() -> dict[str, tuple[str, str, str]]:
    path = SEEDS / "01_input" / "FULL_SEED_MANIFEST.internal.tsv"
    result = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            primary = row["v3_primary_label"]
            target = "BACMET" if primary in {"MRG", "TRANSPORT_EFFLUX"} else primary
            result[row["seed_id"]] = (target, primary, row["source_database"])
    return result


def load_strict_hits() -> dict[str, dict[str, str | float]]:
    seed_classes = load_seed_classes()
    best: dict[str, dict[str, str | float]] = {}
    specialized_subjects: set[str] = set()
    for pass_name, path in [
        ("first90_revalidated", FINAL_POS / "first90_revalidated_strict_hits.tsv.gz"),
        ("second6718_specialized90", FINAL_POS / "second6718_strict_hits.tsv.gz"),
    ]:
        with open_text(path) as handle:
            for line in handle:
                p = line.rstrip("\n").split("\t")
                if len(p) < 10:
                    continue
                q, seed = p[0], p[1]
                record = {
                    "strict_pass": pass_name,
                    "strict_evidence_protein_id": q,
                    "reference_seed": seed,
                    "identity": float(p[2]),
                    "qcov": float(p[6]),
                    "scov": float(p[7]),
                    "evalue": float(p[8]),
                    "bitscore": float(p[9]),
                }
                if seed in seed_classes:
                    record["target_class"], record["seed_primary_label"], record["seed_source_database"] = seed_classes[seed]
                else:
                    specialized_subjects.add(seed)
                previous = best.get(q)
                if previous is None or float(record["bitscore"]) > float(previous["bitscore"]):
                    best[q] = record
    if specialized_subjects:
        with open_text(SPECIALIZED / "protein_manifest.tsv.gz") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            categories = {row["protein_id"]: row["categories"] for row in reader if row["protein_id"] in specialized_subjects}
        for record in best.values():
            seed = str(record["reference_seed"])
            if seed in specialized_subjects:
                raw = categories.get(seed, "")
                mapped = []
                for value in raw.split(";"):
                    target = {"MOBILOME": "MGE", "ARG": "ARG", "METAL": "BACMET", "VIRULENCE": "VF"}.get(value)
                    if target and target not in mapped:
                        mapped.append(target)
                record["target_class"] = ";".join(mapped) or "SPECIALIZED_UNMAPPED"
                record["seed_primary_label"] = raw
                record["seed_source_database"] = "MGnify_AMRFinderPlus_or_explicit_mobilome"




    groups = POS_DEDUP / "02_results" / "identity100_exact" / "duplicate_groups.tsv.gz"
    with open_text(groups) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            members = [x.strip() for x in parts[1].split(",") if x.strip()]
            candidates = [best[x] for x in members if x in best]
            if not candidates:
                continue
            evidence = max(candidates, key=lambda x: float(x["bitscore"]))
            for member in members:
                if member not in best:
                    propagated = dict(evidence)
                    propagated["strict_pass"] = str(evidence["strict_pass"]) + "_exact_duplicate_propagated"
                    best[member] = propagated
    return best


def load_rescued() -> dict[str, tuple[str, str]]:
    path = RESCUE / "all_hit_annotation_classification.tsv.gz"
    result: dict[str, tuple[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["classification"] in RESCUED:
                result[row["protein_id"]] = (row["classification"], row["evidence"])
    return result


def build_dev_centers(out: Path) -> pd.DataFrame:
    row_manifest = pd.read_csv(OLD / "04_models" / "matrix" / "row_manifest.tsv.gz", sep="\t")
    oof = pd.read_csv(OLD / "04_models" / "strict_binary_model" / "development_oof_predictions.tsv.gz", sep="\t")
    merged = row_manifest.merge(oof[["row_index", "label", "fold", "probability"]], on="row_index", suffixes=("", "_oof"), validate="one_to_one")
    if not ((merged["label"] == merged["label_oof"]).all() and (merged["fold"] == merged["fold_oof"]).all()):
        raise ValueError("OOF label/fold mismatch")
    if (merged["fold"] == 0).any():
        raise ValueError("old fold-0 leaked into development OOF")
    if len(merged) != 815_573:
        raise ValueError(f"expected 815573 development OOF rows, found {len(merged)}")

    parsed = [parse_seq_id(x) for x in merged["sequence_id"].astype(str)]
    merged["env_code"] = [x[0] for x in parsed]
    merged["environment"] = [ENV[x[0]][0] for x in parsed]
    merged["mag_id"] = [x[1] for x in parsed]
    merged["gene_suffix"] = [x[2] for x in parsed]
    merged["raw_protein_id"] = [x[3] for x in parsed]
    merged.rename(columns={"sequence_id": "protein_id", "probability": "old_mlp_oof_score", "fold": "old_homology_fold"}, inplace=True)
    merged.drop(columns=["label_oof", "fold_oof"], inplace=True)

    split = pd.read_csv(
        OLD / "03_homology_split" / "sequence_labels_clusters_folds.tsv.gz",
        sep="\t", usecols=["sequence_id", "homology_cluster", "fold"],
    ).rename(columns={"sequence_id": "protein_id", "fold": "split_contract_fold"})
    merged = merged.merge(split, on="protein_id", how="left", validate="one_to_one")
    if merged["homology_cluster"].isna().any() or not (merged["old_homology_fold"] == merged["split_contract_fold"]).all():
        raise ValueError("old homology split provenance mismatch")
    merged.drop(columns=["split_contract_fold"], inplace=True)

    neg = pd.read_csv(NEG_SAMPLE / "sample_manifest.tsv.gz", sep="\t")
    neg.rename(columns={"protein_id": "protein_id", "selection_group": "negative_selection_group", "cluster_size": "negative_cluster_size"}, inplace=True)
    merged = merged.merge(neg, on="protein_id", how="left", validate="one_to_one")

    strict = load_strict_hits()
    for col in ["strict_pass", "strict_evidence_protein_id", "reference_seed", "identity", "qcov", "scov", "evalue", "bitscore", "target_class", "seed_primary_label", "seed_source_database"]:
        merged[col] = [strict.get(x, {}).get(col, np.nan) for x in merged["protein_id"]]
    if merged.loc[merged["label"] == 1, "reference_seed"].isna().any():
        missing = int(merged.loc[merged["label"] == 1, "reference_seed"].isna().sum())
        raise ValueError(f"{missing} positive OOF centres lack strict alignment evidence")
    if merged.loc[merged["label"] == 0, "negative_selection_group"].isna().any():
        missing = int(merged.loc[merged["label"] == 0, "negative_selection_group"].isna().sum())
        raise ValueError(f"{missing} negative OOF centres lack sampling provenance")
    merged["label_state"] = np.where(merged["label"] == 1, "POS_STRICT", "NEG_CANDIDATE")
    merged["old_score_contract"] = "development_OOF_folds_1_to_4_only"
    merged.to_csv(out / "DEV_OOF_CENTERS.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "compresslevel": 1})
    return merged


def build_input_manifest(out: Path) -> pd.DataFrame:
    files = [
        ("old_row_manifest", OLD / "04_models" / "matrix" / "row_manifest.tsv.gz", "1,019,304 rows; ID,label,old homology fold"),
        ("old_development_oof", OLD / "04_models" / "strict_binary_model" / "development_oof_predictions.tsv.gz", "815,573 old development OOF rows only"),
        ("old_metrics", OLD / "04_models" / "strict_binary_model" / "metrics.json", "frozen MLP128 contract and unsealed fold-0 result"),
        ("negative_manifest", NEG_SAMPLE / "sample_manifest.tsv.gz", "926,640 candidate negatives with sampling group"),
        ("positive_exact_dedup", POS_DEDUP / "02_results" / "identity100_exact" / "representative_sequences.faa.gz", "92,664 strict-positive representatives"),
        ("strict_hits_first90_revalidated", FINAL_POS / "first90_revalidated_strict_hits.tsv.gz", "authoritative revalidated DIAMOND evalue<1e-5 identity/qcov/scov >90"),
        ("strict_hits_second6718", FINAL_POS / "second6718_strict_hits.tsv.gz", "authoritative specialized strict second-pass hits"),
        ("specialized_subject_manifest", SPECIALIZED / "protein_manifest.tsv.gz", "AMRFinderPlus and explicit mobilome subject categories"),
        ("strict_seed_manifest", SEEDS / "01_input" / "FULL_SEED_MANIFEST.internal.tsv", "269,190 reference seeds and target class provenance"),
        ("rescue_classification", RESCUE / "all_hit_annotation_classification.tsv.gz", "60,003 conservative rescued proteins kept separate"),
        ("rescue_summary", RESCUE / "summary.tsv", "rescue counts and policy summary"),
    ]
    for code, (_, rel, version) in ENV.items():
        files.extend([
            (f"{code}_metadata", DL / "01_metadata" / rel / "genomes-all_metadata.tsv", f"MGnify metadata; pipeline {version}"),
            (f"{code}_release_readme", next((DL / "01_metadata" / rel).glob("README*")), f"local MGnify release documentation; pipeline {version}"),
        ])
    rows = []
    for logical, path, note in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({
            "logical_input": logical,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "note": note,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "INPUT_MANIFEST.tsv", sep="\t", index=False)
    return frame


def run_audit(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "FIGURES").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "code").mkdir(exist_ok=True)
    centers = build_dev_centers(out)
    qc = load_mag_qc()
    qc.to_csv(out / "MAG_QC.tsv", sep="\t", index=False)
    build_input_manifest(out)
    rescued = load_rescued()
    pd.DataFrame(
        [(protein_id, values[0], values[1]) for protein_id, values in rescued.items()],
        columns=["protein_id", "classification", "evidence"],
    ).to_csv(out / "RESCUED_SEPARATE.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "compresslevel": 1})
    summary = {
        "state": "AUDIT_COMPLETE",
        "old_fold0_rows_used": 0,
        "development_oof_centers": len(centers),
        "development_positive": int((centers.label == 1).sum()),
        "development_negative": int((centers.label == 0).sum()),
        "development_mags": int(centers.mag_id.nunique()),
        "downloaded_mag_metadata_rows": len(qc),
        "qc_pass_mags": int(qc.qc_pass.sum()),
        "raw_complete_mags": int(qc.raw_complete.sum()),
        "conservative_rescued_registered_separately": len(rescued),
        "quality_tool_version_note": "The local MGnify release READMEs name pipeline versions but do not state the CheckM/CheckM2 version; no version was invented.",
    }
    (out / "AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "AUDIT_COMPLETE").touch()
    print(json.dumps(summary, sort_keys=True))


def parse_attributes(text: str) -> dict[str, str]:
    out = {}
    for item in text.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
    return out


def fasta_dict(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    result[name] = "".join(chunks).upper()
                name = line[1:].split(None, 1)[0]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            result[name] = "".join(chunks).upper()
    return result


def gc_percent(seq: str) -> float:
    valid = sum(1 for x in seq if x in "ACGT")
    if not valid:
        return float("nan")
    return 100.0 * sum(1 for x in seq if x in "GC") / valid


def prefixed_id(env_code: str, mag: str, raw_id: str) -> str:
    if raw_id.startswith(mag + "_"):
        return f"{env_code}__{mag}__{raw_id[len(mag)+1:]}"
    return f"{env_code}__{mag}__{raw_id}"


def process_mag(task: tuple[dict, list[dict], bool] | tuple[dict, list[dict], bool, int]) -> dict:
    global _RESCUED_LOOKUP
    if len(task) == 3:
        mag_row, center_rows, include_all_contig_genes = task
        max_gene_window = 10
    else:
        mag_row, center_rows, include_all_contig_genes, max_gene_window = task
    if max_gene_window < 1:
        raise ValueError("max_gene_window must be positive")
    if _RESCUED_LOOKUP is None:
        _RESCUED_LOOKUP = load_rescued()
    rescued = _RESCUED_LOOKUP
    code = mag_row["env_code"]
    mag = mag_row["mag_id"]
    fna = resolve_genome_file(code, mag, ".fna")
    faa = resolve_genome_file(code, mag, ".faa")
    gff = resolve_genome_file(code, mag, ".gff")
    result = {"mag_id": mag, "centers": [], "neighbors": [], "contigs": [], "all_genes": [], "error": "", "warning": "", "unmapped_centers": 0}
    if not bool(mag_row["qc_pass"]):
        result["error"] = "MAG_QC_FAIL"
        return result
    if not all((fna, faa, gff)):
        result["error"] = "RAW_FILE_MISSING"
        return result
    try:
        dna = fasta_dict(fna)
        protein_seqs = fasta_dict(faa)
        genes_by_contig: dict[str, list[dict]] = defaultdict(list)
        with open_text(gff) as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) != 9 or p[2] != "CDS":
                    continue
                contig, start, end, strand = p[0], int(p[3]), int(p[4]), p[6]
                attr = parse_attributes(p[8])
                raw_id = attr.get("ID") or attr.get("locus_tag")
                if not raw_id:
                    continue
                pfam = attr.get("pfam") or attr.get("Pfam") or ""
                protein = protein_seqs.get(raw_id, "")
                contig_seq = dna.get(contig, "")
                cds_seq = contig_seq[start - 1:end] if contig_seq else ""
                genes_by_contig[contig].append({
                    "raw_protein_id": raw_id,
                    "protein_id": prefixed_id(code, mag, raw_id),
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "cds_length": end - start + 1,
                    "protein_length": len(protein) if protein else max(0, (end - start + 1) // 3),
                    "gc_percent": gc_percent(cds_seq) if cds_seq else float("nan"),
                    "pfams": ",".join(sorted(set(x for x in pfam.split(",") if x))),
                    "product": attr.get("product", ""),
                    "eggnog": attr.get("eggNOG", ""),
                })

        center_by_raw = {x["raw_protein_id"]: x for x in center_rows}
        all_gff_ids = {g["raw_protein_id"] for genes in genes_by_contig.values() for g in genes}
        seen_centers: set[str] = set()
        for contig, genes in genes_by_contig.items():
            genes.sort(key=lambda x: (x["start"], x["end"], x["raw_protein_id"]))
            seq = dna.get(contig, "")
            length = len(seq)
            retained = length > 10_000
            contig_gc = gc_percent(seq) if seq else float("nan")
            result["contigs"].append({
                "env_code": code, "environment": ENV[code][0], "mag_id": mag,
                "contig_id": contig, "contig_length": length, "contig_gc_percent": contig_gc,
                "cds_count": len(genes), "retained_gt_10000": retained,
                "has_supervised_center": any(g["raw_protein_id"] in center_by_raw for g in genes),
            })
            if not retained:
                continue
            positions = {g["raw_protein_id"]: i for i, g in enumerate(genes)}
            has_center = any(x in positions for x in center_by_raw)
            if include_all_contig_genes and has_center:
                for g in genes:
                    state = "UNLABELED"
                    evidence = ""
                    if g["raw_protein_id"] in center_by_raw:
                        state = center_by_raw[g["raw_protein_id"]]["label_state"]
                    elif g["protein_id"] in rescued:
                        state = "UNLABELED_RESCUE_SEPARATE"
                        evidence = rescued[g["protein_id"]][0]
                    result["all_genes"].append({
                        "environment": ENV[code][0], "mag_id": mag, "contig_id": contig,
                        "protein_id": g["protein_id"], "raw_protein_id": g["raw_protein_id"],
                        "start": g["start"], "end": g["end"], "strand": g["strand"],
                        "pfams": g["pfams"], "label_state": state,
                        "rescue_evidence": evidence,
                    })
            for raw_id, center_meta in center_by_raw.items():
                if raw_id not in positions:
                    continue
                i = positions[raw_id]
                center = genes[i]
                seen_centers.add(raw_id)
                base = dict(center_meta)
                base.update({
                    "contig_id": contig, "start": center["start"], "end": center["end"],
                    "strand": center["strand"], "cds_length": center["cds_length"],
                    "protein_length": center["protein_length"], "center_gc_percent": center["gc_percent"],
                    "center_pfams": center["pfams"], "center_product": center["product"],
                    "contig_length": length, "contig_gc_percent": contig_gc,
                    "center_contig_gc_delta": center["gc_percent"] - contig_gc,
                    "distance_to_left_end": center["start"] - 1,
                    "distance_to_right_end": length - center["end"],
                    "nearest_contig_end_distance": min(center["start"] - 1, length - center["end"]),
                    "genes_within_5kb": sum(1 for j, g in enumerate(genes) if j != i and g["start"] <= center["end"] + 5000 and g["end"] >= center["start"] - 5000),
                    "genes_within_10kb": sum(1 for j, g in enumerate(genes) if j != i and g["start"] <= center["end"] + 10000 and g["end"] >= center["start"] - 10000),
                })
                result["centers"].append(base)
                for relpos in list(range(-max_gene_window, 0)) + list(range(1, max_gene_window + 1)):
                    j = i + relpos
                    if j < 0 or j >= len(genes):
                        result["neighbors"].append({
                            "protein_id": center["protein_id"], "mag_id": mag, "contig_id": contig,
                            "relative_position": relpos, "neighbor_id": "PAD", "is_pad": True,
                            "neighbor_start": np.nan, "neighbor_end": np.nan, "neighbor_strand": "PAD",
                            "protein_length": np.nan, "cds_length": np.nan, "bp_distance_to_center": np.nan,
                            "gap_or_overlap_to_adjacent": np.nan, "same_strand_as_center": np.nan,
                            "gc_percent": np.nan, "gc_delta_from_center": np.nan, "pfams": "",
                        })
                        continue
                    nb = genes[j]
                    if relpos < 0:
                        distance = center["start"] - nb["end"] - 1
                        adjacent = genes[j + 1]
                        gap = adjacent["start"] - nb["end"] - 1
                    else:
                        distance = nb["start"] - center["end"] - 1
                        adjacent = genes[j - 1]
                        gap = nb["start"] - adjacent["end"] - 1
                    result["neighbors"].append({
                        "protein_id": center["protein_id"], "mag_id": mag, "contig_id": contig,
                        "relative_position": relpos, "neighbor_id": nb["protein_id"], "is_pad": False,
                        "neighbor_start": nb["start"], "neighbor_end": nb["end"], "neighbor_strand": nb["strand"],
                        "protein_length": nb["protein_length"], "cds_length": nb["cds_length"],
                        "bp_distance_to_center": distance, "gap_or_overlap_to_adjacent": gap,
                        "same_strand_as_center": float(nb["strand"] == center["strand"]),
                        "gc_percent": nb["gc_percent"], "gc_delta_from_center": nb["gc_percent"] - center["gc_percent"],
                        "pfams": nb["pfams"],
                    })
        missing = set(center_by_raw) - seen_centers
        if missing:
            short = len(missing & all_gff_ids)
            absent = len(missing - all_gff_ids)
            result["unmapped_centers"] = len(missing)
            result["warning"] = f"CENTERS_EXCLUDED_SHORT_CONTIG:{short};ABSENT_GFF_ID:{absent}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}:{exc}"
    return result


def parquet_writer(path: Path, rows: list[dict], writer: pq.ParquetWriter | None) -> pq.ParquetWriter | None:
    if not rows:
        return writer
    table = pa.Table.from_pylist(rows)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer


def run_extract(args: argparse.Namespace) -> None:
    global _RESCUED_LOOKUP
    out = Path(args.output)
    qc = pd.read_csv(out / "MAG_QC.tsv", sep="\t")
    centers = pd.read_csv(out / "DEV_OOF_CENTERS.tsv.gz", sep="\t", low_memory=False)
    for column in centers.columns:
        if is_string_dtype(centers[column].dtype) or centers[column].dtype == object:
            centers[column] = centers[column].fillna("").astype(str)


    rescue_cache = out / "RESCUED_SEPARATE.tsv.gz"
    if rescue_cache.is_file():
        rescue_frame = pd.read_csv(rescue_cache, sep="\t").fillna("")
        _RESCUED_LOOKUP = {r.protein_id: (r.classification, r.evidence) for r in rescue_frame.itertuples(index=False)}
    else:
        _RESCUED_LOOKUP = load_rescued()
        pd.DataFrame(
            [(protein_id, values[0], values[1]) for protein_id, values in _RESCUED_LOOKUP.items()],
            columns=["protein_id", "classification", "evidence"],
        ).to_csv(rescue_cache, sep="\t", index=False, compression={"method": "gzip", "compresslevel": 1})
    qc_index = {(r.env_code, r.mag_id): r._asdict() for r in qc.itertuples(index=False)}
    by_mag: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in centers.to_dict("records"):
        by_mag[(row["env_code"], row["mag_id"])].append(row)
    keys = sorted(by_mag)
    if args.smoke:
        selected = []
        per_env = Counter()
        for key in keys:
            labels = {x["label"] for x in by_mag[key]}
            if labels == {0, 1} and per_env[key[0]] < args.smoke_mags_per_env:
                mag = qc_index.get(key)
                if mag and mag["qc_pass"] and mag["raw_complete"]:
                    selected.append(key)
                    per_env[key[0]] += 1
            if all(per_env[x] >= args.smoke_mags_per_env for x in ENV):
                break
        keys = selected
    tasks = []
    pre_issues: list[dict] = []
    for key in keys:
        mag = qc_index.get(key)
        if mag is None:
            pre_issues.append({"mag_id": key[1], "severity": "ERROR", "issue": "MAG_METADATA_MISSING", "unmapped_centers": len(by_mag[key])})
            continue
        if not bool(mag["qc_pass"]):
            pre_issues.append({"mag_id": key[1], "severity": "EXPECTED_EXCLUSION", "issue": "MAG_QC_FAIL", "unmapped_centers": len(by_mag[key])})
            continue
        if not bool(mag["raw_complete"]):
            pre_issues.append({"mag_id": key[1], "severity": "ERROR", "issue": "RAW_FILE_MISSING", "unmapped_centers": len(by_mag[key])})
            continue
        tasks.append((mag, by_mag[key], True))
    if not tasks:
        raise ValueError("no MAG tasks selected")

    stem = "SMOKE_" if args.smoke else ""
    center_path = out / f"{stem}CENTER_BASE_FEATURES.parquet"
    neighbor_path = out / f"{stem}NEIGHBOR_INDEX.parquet"
    all_gene_path = out / f"{stem}CONTIG_GENE_STATUS.parquet"
    contig_rows: list[dict] = []
    issue_rows: list[dict] = list(pre_issues)
    center_buffer: list[dict] = []
    neighbor_buffer: list[dict] = []
    all_gene_buffer: list[dict] = []
    center_writer = neighbor_writer = gene_writer = None
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(process_mag, task): task[0]["mag_id"] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            center_buffer.extend(result["centers"])
            neighbor_buffer.extend(result["neighbors"])
            all_gene_buffer.extend(result["all_genes"])
            contig_rows.extend(result["contigs"])
            if result["error"]:
                issue_rows.append({"mag_id": result["mag_id"], "severity": "ERROR", "issue": result["error"], "unmapped_centers": result["unmapped_centers"]})
            elif result["warning"]:
                issue_rows.append({"mag_id": result["mag_id"], "severity": "EXPECTED_EXCLUSION", "issue": result["warning"], "unmapped_centers": result["unmapped_centers"]})
            if len(center_buffer) >= 20_000:
                center_writer = parquet_writer(center_path, center_buffer, center_writer)
            if len(neighbor_buffer) >= 200_000:
                neighbor_writer = parquet_writer(neighbor_path, neighbor_buffer, neighbor_writer)
            if len(all_gene_buffer) >= 200_000:
                gene_writer = parquet_writer(all_gene_path, all_gene_buffer, gene_writer)
            if done % 500 == 0:
                print(json.dumps({"mags_done": done, "mags_total": len(tasks), "issues": len(issue_rows)}), flush=True)
    center_writer = parquet_writer(center_path, center_buffer, center_writer)
    neighbor_writer = parquet_writer(neighbor_path, neighbor_buffer, neighbor_writer)
    gene_writer = parquet_writer(all_gene_path, all_gene_buffer, gene_writer)
    for writer in (center_writer, neighbor_writer, gene_writer):
        if writer is not None:
            writer.close()
    pd.DataFrame(contig_rows).to_csv(out / f"{stem}CONTIG_QC.tsv", sep="\t", index=False)
    pd.DataFrame(issue_rows, columns=["mag_id", "severity", "issue", "unmapped_centers"]).to_csv(out / f"{stem}EXTRACTION_ERRORS.tsv", sep="\t", index=False)
    if center_writer is None or neighbor_writer is None:
        raise ValueError("no eligible centers or neighbors written")

    label_columns = [
        "protein_id", "raw_protein_id", "environment", "mag_id", "contig_id",
        "label_state", "label", "strict_pass", "strict_evidence_protein_id", "reference_seed", "identity", "qcov", "scov",
        "evalue", "bitscore", "target_class", "seed_primary_label", "seed_source_database",
        "negative_selection_group", "negative_cluster_size", "old_score_contract",
    ]
    extracted = pq.read_table(center_path).to_pandas()
    labels = extracted[[x for x in label_columns if x in extracted.columns]].copy()
    labels.to_csv(out / f"{stem}CENTER_LABELS.tsv", sep="\t", index=False)
    summary = {
        "state": "SMOKE_EXTRACT_COMPLETE" if args.smoke else "EXTRACT_COMPLETE",
        "mags_requested": len(tasks),
        "mags_with_errors": sum(x["severity"] == "ERROR" for x in issue_rows),
        "mags_with_expected_exclusions": sum(x["severity"] == "EXPECTED_EXCLUSION" for x in issue_rows),
        "centers_excluded_short_contig_or_absent_gff": sum(int(x["unmapped_centers"]) for x in issue_rows),
        "mapped_supervised_centers": len(extracted),
        "mapped_positive": int((extracted.label_state == "POS_STRICT").sum()),
        "mapped_negative": int((extracted.label_state == "NEG_CANDIDATE").sum()),
        "mapped_mags": int(extracted.mag_id.nunique()),
        "environment_counts": extracted.environment.value_counts().to_dict(),
        "old_fold0_rows_used": 0,
    }
    (out / f"{stem}EXTRACTION_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / ("SMOKE_COMPLETE" if args.smoke else "EXTRACTION_COMPLETE")).touch()
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--output", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--output", required=True)
    extract.add_argument("--workers", type=int, default=24)
    extract.add_argument("--smoke", action="store_true")
    extract.add_argument("--smoke-mags-per-env", type=int, default=5)
    args = parser.parse_args()
    if args.command == "audit":
        run_audit(args)
    else:
        run_extract(args)


if __name__ == "__main__":
    main()
