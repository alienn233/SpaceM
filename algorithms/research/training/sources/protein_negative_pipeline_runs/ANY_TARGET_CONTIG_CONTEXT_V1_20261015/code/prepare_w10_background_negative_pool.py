from __future__ import annotations

import argparse
import gzip
import json
import os
import zlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


DOWNLOAD = Path("/ifs1/User/yanyueyang/yyy/gene-finder/MGnify_DOWNLOAD_20261010")
ENVIRONMENT_PATHS = {
    "HG": "human-gut/v3.0",
    "MAR": "marine/v2.0",
    "SOIL": "soil/v1.0",
}
DEFAULT_DIAMOND_AUDIT = Path(
    "/ifs1/User/yanyueyang/yyy/gene-finder/protein_negative_pipeline_runs/"
    "MGNIFY_ANNOTATION_RESCUE_20261014_V3/all_hit_annotation_classification.tsv.gz"
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def open_fasta(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def iter_fasta(path: Path):
    identifier = None
    sequence: list[str] = []
    with open_fasta(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(sequence)
                identifier = line[1:].split(None, 1)[0]
                sequence = []
            else:
                sequence.append(line)
        if identifier is not None:
            yield identifier, "".join(sequence)


def resolve_faa(env_code: str, mag_id: str) -> Path:
    stem = (
        DOWNLOAD
        / "03_raw"
        / ENVIRONMENT_PATHS[env_code]
        / mag_id[:11]
        / mag_id
        / "genome"
        / f"{mag_id}.faa"
    )
    if stem.is_file():
        return stem
    compressed = Path(str(stem) + ".gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(stem)


def collect_neighbor_statistics(root: Path, positive_ids: set[str]):
    count: dict[str, int] = defaultdict(int)
    minimum: dict[str, int] = {}
    within_w5: set[str] = set()
    slots = 0
    parquet = pq.ParquetFile(root / "NEIGHBOR_INDEX.parquet")
    for batch in parquet.iter_batches(
        batch_size=250_000,
        columns=["protein_id", "neighbor_id", "relative_position", "is_pad"],
    ):
        values = batch.to_pydict()
        for center, neighbor, relative, is_pad in zip(
            values["protein_id"],
            values["neighbor_id"],
            values["relative_position"],
            values["is_pad"],
        ):
            if is_pad or str(center) not in positive_ids:
                continue
            protein_id = str(neighbor)
            distance = abs(int(relative))
            slots += 1
            count[protein_id] += 1
            minimum[protein_id] = min(minimum.get(protein_id, distance), distance)
            if distance <= 5:
                within_w5.add(protein_id)
    return count, minimum, within_w5, slots


def collect_status(root: Path, wanted: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    columns = [
        "environment",
        "mag_id",
        "contig_id",
        "protein_id",
        "raw_protein_id",
        "start",
        "end",
        "strand",
        "pfams",
        "label_state",
        "rescue_evidence",
    ]
    parquet = pq.ParquetFile(root / "CONTIG_GENE_STATUS.parquet")
    for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
        values = batch.to_pydict()
        for index, protein_id in enumerate(values["protein_id"]):
            if str(protein_id) in wanted:
                rows.append({column: values[column][index] for column in columns})
    frame = pd.DataFrame(rows)
    if len(frame) != len(wanted) or frame.protein_id.astype(str).nunique() != len(wanted):
        raise ValueError(
            f"status alignment failed: wanted={len(wanted)} rows={len(frame)} "
            f"unique={frame.protein_id.nunique() if len(frame) else 0}"
        )
    return frame


def historical_diamond_hits(path: Path, wanted: set[str]):
    classification: dict[str, str] = {}
    evidence: dict[str, str] = {}
    for chunk in pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=["protein_id", "classification", "evidence"],
        chunksize=500_000,
        dtype=str,
    ):
        selected = chunk[chunk.protein_id.astype(str).isin(wanted)]
        for row in selected.itertuples(index=False):
            protein_id = str(row.protein_id)
            classification[protein_id] = str(row.classification)
            evidence[protein_id] = str(row.evidence)
    return classification, evidence


def write_gpu_shards(
    root: Path,
    out: Path,
    safe: pd.DataFrame,
    shard_count: int,
) -> dict:
    safe_ids = set(safe.protein_id.astype(str))
    shard_dir = out / "GPU_INPUT_SHARDS"
    shard_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        gzip.open(shard_dir / f"shard_{index + 1:03d}.faa.gz.tmp", "wt", compresslevel=1)
        for index in range(shard_count)
    ]
    found: set[str] = set()
    residues = 0

    def emit(protein_id: str, sequence: str) -> None:
        nonlocal residues
        if protein_id in found or protein_id not in safe_ids:
            return
        shard = zlib.crc32(protein_id.encode("utf-8")) % shard_count
        handles[shard].write(f">{protein_id}\n{sequence}\n")
        found.add(protein_id)
        residues += len(sequence)

    try:
        for path in sorted((root / "NEIGHBOR_ESM2" / "01_shards").glob("shard_*.faa.gz")):
            for protein_id, sequence in iter_fasta(path):
                emit(protein_id, sequence)

        remaining = safe[~safe.protein_id.astype(str).isin(found)].copy()
        for (env_code, mag_id), part in remaining.groupby(["env_code", "mag_id"], sort=False):
            wanted_raw = dict(
                zip(part.raw_protein_id.astype(str), part.protein_id.astype(str))
            )
            for raw_id, sequence in iter_fasta(resolve_faa(str(env_code), str(mag_id))):
                protein_id = wanted_raw.get(raw_id)
                if protein_id is not None:
                    emit(protein_id, sequence)
    finally:
        for handle in handles:
            handle.close()

    for index in range(shard_count):
        temporary = shard_dir / f"shard_{index + 1:03d}.faa.gz.tmp"
        final = shard_dir / f"shard_{index + 1:03d}.faa.gz"
        os.replace(temporary, final)

    missing = sorted(safe_ids - found)
    (out / "SEQUENCE_EXTRACTION_ERRORS.tsv").write_text(
        "protein_id\terror\n"
        + "".join(f"{protein_id}\tNOT_FOUND\n" for protein_id in missing)
    )
    if missing:
        raise ValueError(f"{len(missing)} safe candidates lack sequences")

    safe = safe.copy()
    safe["gpu_shard"] = safe.protein_id.astype(str).map(
        lambda value: zlib.crc32(value.encode("utf-8")) % shard_count + 1
    )
    safe.to_csv(
        out / "SAFE_NEGATIVE_CANDIDATES.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "compresslevel": 1},
    )
    shard_summary = (
        safe.groupby("gpu_shard", as_index=False)
        .agg(proteins=("protein_id", "size"), mags=("mag_id", "nunique"))
        .sort_values("gpu_shard")
    )
    shard_summary.to_csv(out / "GPU_SHARD_SUMMARY.tsv", sep="\t", index=False)
    return {
        "safe_candidate_sequences": len(found),
        "amino_acid_residues": residues,
        "gpu_shards": shard_count,
        "sequence_extraction_missing": 0,
    }


def run(args: argparse.Namespace) -> None:
    root = Path(args.train_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "PREPARE_COMPLETE").exists():
        print((out / "PREPARE_RESULT.json").read_text(), end="")
        return

    centers = pq.read_table(
        root / "CONTEXT_FEATURES.parquet",
        columns=["protein_id", "label"],
    ).to_pandas()
    supervised = set(centers.protein_id.astype(str))
    positive_ids = set(centers.loc[centers.label.eq(1), "protein_id"].astype(str))
    count, minimum, within_w5, slots = collect_neighbor_statistics(root, positive_ids)
    direct_ids = set(count)
    unsupervised = direct_ids - supervised
    status = collect_status(root, unsupervised)
    status["env_code"] = status.protein_id.astype(str).str.split("__", n=1).str[0]
    status["positive_center_neighbor_count"] = status.protein_id.astype(str).map(count)
    status["minimum_gene_distance_to_positive"] = status.protein_id.astype(str).map(minimum)
    status["within_positive_w5"] = status.protein_id.astype(str).isin(within_w5)
    rescue_clean = (
        status.label_state.astype(str).eq("UNLABELED")
        & status.rescue_evidence.fillna("").astype(str).eq("")
    )
    eligible = status[rescue_clean].copy()
    classification, evidence = historical_diamond_hits(
        Path(args.diamond_audit), set(eligible.protein_id.astype(str))
    )
    eligible["historical_broad_diamond_hit"] = eligible.protein_id.astype(str).isin(
        classification
    )
    eligible["historical_diamond_classification"] = (
        eligible.protein_id.astype(str).map(classification).fillna("")
    )
    eligible["historical_diamond_evidence"] = (
        eligible.protein_id.astype(str).map(evidence).fillna("")
    )
    safe = eligible[~eligible.historical_broad_diamond_hit].copy()
    safe["pseudo_label"] = 0
    safe["pseudo_label_weight"] = float(args.pseudo_label_weight)
    safe["selection_contract"] = (
        "unsupervised_W10_neighbor_of_positive; no_rescue_evidence; "
        "absent_from_authoritative_historical_broad_DIAMOND_hit_union"
    )
    safe["old_fold0_used"] = False

    status.to_csv(
        out / "ALL_UNSUPERVISED_W10_NEIGHBORS.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "compresslevel": 1},
    )
    sequence_summary = write_gpu_shards(root, out, safe, args.shards)
    result = {
        "state": "W10_BACKGROUND_NEGATIVE_POOL_PREPARED",
        "positive_centers": len(positive_ids),
        "positive_center_w10_nonpad_slots": slots,
        "unique_w10_neighbors": len(direct_ids),
        "existing_supervised_neighbors": len(direct_ids & supervised),
        "new_unsupervised_neighbors": len(unsupervised),
        "excluded_rescue_or_non_unlabeled": int((~rescue_clean).sum()),
        "eligible_after_rescue_exclusion": len(eligible),
        "excluded_any_historical_broad_diamond_hit": int(
            eligible.historical_broad_diamond_hit.sum()
        ),
        "safe_no_historical_diamond_hit": len(safe),
        "safe_within_positive_w5": int(safe.within_positive_w5.sum()),
        "pseudo_label_weight": float(args.pseudo_label_weight),
        "historical_diamond_audit": str(Path(args.diamond_audit)),
        "old_fold0_used": False,
        "protein_duplication": False,
        **sequence_summary,
    }
    atomic_json(out / "PREPARE_RESULT.json", result)
    (out / "PREPARE_COMPLETE").touch()
    print(json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diamond-audit", default=str(DEFAULT_DIAMOND_AUDIT))
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--pseudo-label-weight", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.pseudo_label_weight <= 1.0:
        raise ValueError("pseudo-label weight must be in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()
