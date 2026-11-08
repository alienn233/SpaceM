from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def run(args: argparse.Namespace) -> None:
    train_root = Path(args.train_root)
    pool_root = Path(args.pool_root)
    if (pool_root / "CONTEXT_COMPLETE").exists():
        print((pool_root / "CONTEXT_RESULT.json").read_text(), end="")
        return

    sys.path.insert(0, str(train_root / "code"))
    import context_data
    import train_context

    safe = pd.read_csv(pool_root / "SAFE_NEGATIVE_CANDIDATES.tsv.gz", sep="\t")




    text_columns = [
        "environment",
        "mag_id",
        "contig_id",
        "protein_id",
        "raw_protein_id",
        "strand",
        "pfams",
        "label_state",
        "rescue_evidence",
        "env_code",
        "historical_diamond_classification",
        "historical_diamond_evidence",
        "selection_contract",
    ]
    for column in text_columns:
        safe[column] = safe[column].fillna("").astype(str)
    safe["label"] = 0
    safe["label_state"] = "PSEUDO_NEGATIVE_W10_NO_BROAD_DIAMOND_HIT"
    safe["negative_selection_group"] = "positive_w10_no_broad_diamond_hit"
    safe["old_score_contract"] = "NOT_AVAILABLE_NOT_USED"
    safe["old_homology_fold"] = -1
    safe["homology_cluster"] = "PSEUDO_" + safe.protein_id.astype(str)

    fold_manifest = pd.read_csv(
        train_root / "SPLIT_MANIFEST.tsv",
        sep="\t",
        usecols=["mag_id", "new_mag_fold"],
    ).drop_duplicates("mag_id")
    safe = safe.merge(fold_manifest, on="mag_id", how="left", validate="many_to_one")
    if safe.new_mag_fold.isna().any():
        raise ValueError("a safe candidate MAG lacks the authoritative grouped fold")
    if not set(safe.new_mag_fold.astype(int)).issubset({0, 1, 2, 3, 4}):
        raise ValueError("invalid grouped fold in pseudo-negative pool")

    rescued = pd.read_csv(train_root / "RESCUED_SEPARATE.tsv.gz", sep="\t").fillna("")
    context_data._RESCUED_LOOKUP = {
        row.protein_id: (row.classification, row.evidence)
        for row in rescued.itertuples(index=False)
    }
    qc = pd.read_csv(train_root / "MAG_QC.tsv", sep="\t")
    qc_index = {
        (row.env_code, row.mag_id): row._asdict() for row in qc.itertuples(index=False)
    }
    by_mag: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in safe.to_dict("records"):
        by_mag[(row["env_code"], row["mag_id"])].append(row)
    tasks = []
    for key, rows in by_mag.items():
        mag = qc_index.get(key)
        if mag is None or not bool(mag["qc_pass"]) or not bool(mag["raw_complete"]):
            raise ValueError(f"candidate MAG is not eligible: {key}")
        tasks.append((mag, rows, False, 10))

    center_path = pool_root / "CENTER_BASE_FEATURES.parquet"
    neighbor_path = pool_root / "NEIGHBOR_INDEX.parquet"
    center_buffer: list[dict] = []
    neighbor_buffer: list[dict] = []
    issue_rows: list[dict] = []
    center_writer = None
    neighbor_writer = None
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("fork"),
    ) as executor:
        futures = {
            executor.submit(context_data.process_mag, task): task[0]["mag_id"]
            for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            done += 1
            center_buffer.extend(result["centers"])
            neighbor_buffer.extend(result["neighbors"])
            if result["error"] or result["warning"] or result["unmapped_centers"]:
                issue_rows.append(
                    {
                        "mag_id": result["mag_id"],
                        "error": result["error"],
                        "warning": result["warning"],
                        "unmapped_centers": result["unmapped_centers"],
                    }
                )
            if len(center_buffer) >= 20_000:
                center_writer = context_data.parquet_writer(
                    center_path, center_buffer, center_writer
                )
            if len(neighbor_buffer) >= 200_000:
                neighbor_writer = context_data.parquet_writer(
                    neighbor_path, neighbor_buffer, neighbor_writer
                )
            if done % 250 == 0:
                print(
                    json.dumps(
                        {"event": "mags_processed", "done": done, "total": len(tasks)}
                    ),
                    flush=True,
                )
    center_writer = context_data.parquet_writer(center_path, center_buffer, center_writer)
    neighbor_writer = context_data.parquet_writer(
        neighbor_path, neighbor_buffer, neighbor_writer
    )
    for writer in (center_writer, neighbor_writer):
        if writer is not None:
            writer.close()

    pd.DataFrame(
        issue_rows,
        columns=["mag_id", "error", "warning", "unmapped_centers"],
    ).to_csv(pool_root / "CONTEXT_EXTRACTION_ERRORS.tsv", sep="\t", index=False)
    centers = pq.read_table(center_path).to_pandas()
    neighbors = pq.read_table(neighbor_path, columns=["protein_id"]).to_pandas()
    if len(centers) != len(safe) or centers.protein_id.nunique() != len(safe):
        raise ValueError(
            f"center count mismatch: expected={len(safe)} rows={len(centers)} "
            f"unique={centers.protein_id.nunique()}"
        )
    if len(neighbors) != 20 * len(safe):
        raise ValueError(
            f"neighbor count mismatch: expected={20 * len(safe)} rows={len(neighbors)}"
        )
    train_context.build_wide(pool_root, False)
    split = safe[
        [
            "protein_id",
            "mag_id",
            "new_mag_fold",
            "pseudo_label_weight",
            "within_positive_w5",
            "minimum_gene_distance_to_positive",
            "positive_center_neighbor_count",
        ]
    ].copy()
    split.to_csv(pool_root / "PSEUDO_SPLIT_MANIFEST.tsv", sep="\t", index=False)
    result = {
        "state": "W10_BACKGROUND_CONTEXT_COMPLETE",
        "safe_pseudo_negative_centers": len(safe),
        "mags": int(safe.mag_id.nunique()),
        "context_rows": len(centers),
        "neighbor_rows": len(neighbors),
        "expected_neighbor_rows": 20 * len(safe),
        "mags_with_issues": len(issue_rows),
        "old_fold0_used": False,
        "neighbor_labels_used_as_features": False,
    }
    (pool_root / "CONTEXT_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (pool_root / "CONTEXT_COMPLETE").touch()
    print(json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--workers", type=int, default=24)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
