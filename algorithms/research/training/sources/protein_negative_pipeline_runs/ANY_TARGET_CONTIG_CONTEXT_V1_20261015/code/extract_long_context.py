from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_string_dtype

import context_data


def write_buffer(
    path: Path, rows: list[dict], writer: pq.ParquetWriter | None
) -> pq.ParquetWriter | None:
    if not rows:
        return writer
    table = pa.Table.from_pylist(rows)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer


def run(args: argparse.Namespace) -> None:
    root = Path(args.output)
    experiment = root / f"LONG_CONTEXT_W{args.max_window}"
    experiment.mkdir(parents=True, exist_ok=True)
    output = experiment / "NEIGHBOR_INDEX.parquet"
    marker = experiment / "LONG_NEIGHBOR_EXTRACTION_COMPLETE"
    if marker.exists():
        print(json.dumps({"state": "ALREADY_COMPLETE", "path": str(output)}))
        return

    qc = pd.read_csv(root / "MAG_QC.tsv", sep="\t")
    centers = pd.read_csv(root / "DEV_OOF_CENTERS.tsv.gz", sep="\t", low_memory=False)
    for column in centers.columns:
        if is_string_dtype(centers[column].dtype) or centers[column].dtype == object:
            centers[column] = centers[column].fillna("").astype(str)

    rescued = pd.read_csv(root / "RESCUED_SEPARATE.tsv.gz", sep="\t").fillna("")
    context_data._RESCUED_LOOKUP = {
        row.protein_id: (row.classification, row.evidence)
        for row in rescued.itertuples(index=False)
    }
    qc_index = {(row.env_code, row.mag_id): row._asdict() for row in qc.itertuples(index=False)}
    by_mag: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in centers.to_dict("records"):
        by_mag[(row["env_code"], row["mag_id"])].append(row)

    tasks = []
    pre_errors = []
    for key in sorted(by_mag):
        mag = qc_index.get(key)
        if mag is None or not bool(mag["qc_pass"]) or not bool(mag["raw_complete"]):
            pre_errors.append(f"{key[0]}\t{key[1]}\tMAG_NOT_ELIGIBLE")
            continue
        tasks.append((mag, by_mag[key], False, args.max_window))
    if not tasks:
        raise ValueError("no MAGs eligible for long-context extraction")

    writer = None
    buffer: list[dict] = []
    errors = list(pre_errors)
    centers_written = 0
    rows_written = 0
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("fork")
    ) as pool:
        futures = [pool.submit(context_data.process_mag, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result["error"]:
                errors.append(f"NA\t{result['mag_id']}\t{result['error']}")
                continue
            rows = result["neighbors"]
            centers_written += len(result["centers"])
            rows_written += len(rows)
            buffer.extend(rows)
            if len(buffer) >= 200_000:
                writer = write_buffer(output, buffer, writer)
            if done % 500 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "mags_done": done,
                            "mags_total": len(tasks),
                            "neighbor_rows": rows_written,
                        }
                    ),
                    flush=True,
                )
    writer = write_buffer(output, buffer, writer)
    if writer is None:
        raise ValueError("no long-context rows written")
    writer.close()

    expected = centers_written * 2 * args.max_window
    if rows_written != expected:
        raise ValueError(f"long-context row count mismatch: {rows_written} != {expected}")
    (experiment / "LONG_CONTEXT_EXTRACTION_ERRORS.tsv").write_text(
        "env_code\tmag_id\terror\n" + "\n".join(errors) + ("\n" if errors else "")
    )
    summary = {
        "state": "LONG_NEIGHBOR_EXTRACTION_COMPLETE",
        "max_window": args.max_window,
        "mags_requested": len(tasks),
        "mags_with_errors": len(errors),
        "centers": centers_written,
        "neighbor_rows": rows_written,
        "expected_neighbor_rows": expected,
        "old_fold0_rows_used": 0,
    }
    (experiment / "LONG_NEIGHBOR_EXTRACTION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    marker.touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-window", type=int, default=20)
    parser.add_argument("--workers", type=int, default=24)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
