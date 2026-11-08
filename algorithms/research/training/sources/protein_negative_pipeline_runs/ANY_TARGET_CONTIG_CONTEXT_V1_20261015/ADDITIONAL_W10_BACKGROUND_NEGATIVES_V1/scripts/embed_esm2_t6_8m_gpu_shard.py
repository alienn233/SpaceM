from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    seqs: list[str] = []
    current_id: str | None = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    ids.append(current_id)
                    seqs.append("".join(chunks).upper())
                current_id = line[1:].split(None, 1)[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        ids.append(current_id)
        seqs.append("".join(chunks).upper())
    if not ids or len(ids) != len(seqs) or len(set(ids)) != len(ids):
        raise RuntimeError(
            f"Invalid FASTA: records={len(ids)} unique_ids={len(set(ids))}"
        )
    return ids, seqs


def sequence_windows(sequence: str, size: int, stride: int) -> list[str]:
    if len(sequence) <= size:
        return [sequence]
    windows: list[str] = []
    start = 0
    while start < len(sequence):
        end = min(start + size, len(sequence))
        windows.append(sequence[start:end])
        if end == len(sequence):
            break
        start += stride
    return windows


def batches_by_padded_tokens(
    tasks: list[tuple[int, str, int]], token_budget: int, max_batch: int
):
    start = 0
    while start < len(tasks):
        end = start
        max_len = 0
        while end < len(tasks) and end - start < max_batch:
            next_len = len(tasks[end][1]) + 2
            candidate_max = max(max_len, next_len)
            if end > start and candidate_max * (end - start + 1) > token_budget:
                break
            max_len = candidate_max
            end += 1
        if end == start:
            end += 1
        yield tasks[start:end]
        start = end


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--window", type=int, default=1022)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if not args.device.startswith("cuda"):
        raise RuntimeError("This GPU handoff worker requires --device cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch installation")
    device = torch.device(args.device)
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = args.out_dir / "COMPLETE"
    if done.exists():
        print(f"already complete: {args.out_dir}", flush=True)
        return 0

    started = time.time()
    ids, seqs = read_fasta(args.input)
    print(
        f"loaded records={len(ids)} residues={sum(map(len, seqs))} "
        f"device={device} dtype={args.dtype}",
        flush=True,
    )

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir),
        local_files_only=args.offline,
    )
    model = EsmModel.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir),
        local_files_only=args.offline,
        torch_dtype=dtype_map[args.dtype],
    ).to(device).eval()
    dimension = int(model.config.hidden_size)
    if dimension != 320:
        raise RuntimeError(f"Expected ESM2-t6 dimension 320, found {dimension}")

    tasks: list[tuple[int, str, int]] = []
    n_windows = np.zeros(len(ids), dtype=np.int32)
    for owner, sequence in enumerate(seqs):
        windows = sequence_windows(sequence, args.window, args.stride)
        n_windows[owner] = len(windows)
        tasks.extend((owner, window, len(window)) for window in windows)
    tasks.sort(key=lambda item: len(item[1]))

    accum = np.zeros((len(ids), dimension), dtype=np.float32)
    total_weight = np.zeros(len(ids), dtype=np.float64)
    total_batches = 0
    total_tasks = 0
    for batch in batches_by_padded_tokens(tasks, args.token_budget, args.max_batch):
        strings = [item[1] for item in batch]
        encoded_cpu = tokenizer(strings, return_tensors="pt", padding=True)
        encoded = {key: value.to(device, non_blocking=True) for key, value in encoded_cpu.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].clone()
        for row in range(mask.shape[0]):
            valid = torch.nonzero(mask[row], as_tuple=False).flatten()
            if valid.numel() >= 2:
                mask[row, valid[0]] = 0
                mask[row, valid[-1]] = 0
        weighted_mask = mask.unsqueeze(-1).to(torch.float32)
        pooled = (
            (hidden * weighted_mask).sum(dim=1)
            / weighted_mask.sum(dim=1).clamp_min(1)
        ).cpu().numpy()
        for item, vector in zip(batch, pooled):
            owner, _, weight = item
            accum[owner] += vector.astype(np.float32, copy=False) * weight
            total_weight[owner] += weight
        total_batches += 1
        total_tasks += len(batch)
        if total_batches % 200 == 0:
            print(
                f"progress tasks={total_tasks}/{len(tasks)} batches={total_batches} "
                f"elapsed_s={time.time() - started:.1f}",
                flush=True,
            )

    if np.any(total_weight <= 0):
        raise RuntimeError("At least one protein has zero pooled weight")
    accum /= total_weight[:, None].astype(np.float32)
    if not np.isfinite(accum).all():
        raise RuntimeError("Non-finite embedding detected")
    embeddings = accum.astype(np.float16)

    atomic_save_npy(args.out_dir / "embeddings.f16.npy", embeddings)
    ids_tmp = args.out_dir / "ids.txt.tmp"
    ids_tmp.write_text("\n".join(ids) + "\n")
    os.replace(ids_tmp, args.out_dir / "ids.txt")
    metadata_tmp = args.out_dir / "metadata.tsv.tmp"
    with metadata_tmp.open("wt") as handle:
        handle.write("protein_id\tlength\tn_windows\n")
        for protein_id, sequence, windows in zip(ids, seqs, n_windows):
            handle.write(f"{protein_id}\t{len(sequence)}\t{int(windows)}\n")
    os.replace(metadata_tmp, args.out_dir / "metadata.tsv")
    manifest_tmp = args.out_dir / "manifest.tsv.tmp"
    elapsed = time.time() - started
    with manifest_tmp.open("wt") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"records\t{len(ids)}\n")
        handle.write(f"dimension\t{dimension}\n")
        handle.write(f"model\t{args.model}\n")
        handle.write("layer\tlast_hidden_state_layer_6\n")
        handle.write("pooling\tmean_residue_then_window_length_weighted_mean\n")
        handle.write(f"window\t{args.window}\n")
        handle.write(f"stride\t{args.stride}\n")
        handle.write(f"device\t{device}\n")
        handle.write(f"compute_dtype\t{args.dtype}\n")
        handle.write(f"elapsed_seconds\t{elapsed:.3f}\n")
    os.replace(manifest_tmp, args.out_dir / "manifest.tsv")
    done.touch()
    print(
        f"complete records={len(ids)} dimension={dimension} tasks={len(tasks)} "
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
