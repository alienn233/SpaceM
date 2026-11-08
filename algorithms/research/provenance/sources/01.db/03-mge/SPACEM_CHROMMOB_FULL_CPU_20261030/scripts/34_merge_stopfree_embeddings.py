from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path('/ifs1/User/yanyueyang/yyy/gene-finder/01.db/03-mge')
ROOT = BASE / 'SPACEM_CHROMMOB_FULL_CPU_20261030'
AUDIT = ROOT / 'cache_audit_v3_stopfree'
GPU = ROOT / 'gpu_results'
OUT = ROOT / 'features_full_host'
P0 = BASE / 'MOBILITY_V3_BOTELHO_V3HN_8M_20261028'
OLD = BASE / 'SPACEM_ARG_MRG_MGE_BINARY_LGBM_20261030'
NEW = BASE / 'SPACEM_ARG_MRG_MGE_BINARY_LGBM_NEW200_20261030'
DEV = BASE / 'SPACEM_CHROMMOB_V2_STAGE3_DEV600_20261030'
TEST = BASE / 'SPACEM_CHROMMOB_V2_STAGE3_TEST200_20261030'
FRAG = BASE / 'MOBILITY_V3_BOTELHO_SPACEM_FRAGMENT_20261029'
FRAG_NEW = BASE / 'MOBILITY_V3_BOTELHO_SPACEM_FRAGMENT_NEW200_20261030'
FRAG_DEV = BASE / 'MOBILITY_V3_BOTELHO_SPACEM_FRAGMENT_DEV600_20261030'

CACHE_MATRICES = {
    'p0_8m': P0 / 'features/esm2_8m/embeddings.f16.npy',
    'old200_8m': FRAG / 'data/mini200/esm8/embeddings.f16.npy',
    'new200_8m': FRAG_NEW / 'data/mini200/esm8/embeddings.f16.npy',
    'dev600_8m': FRAG_DEV / 'data/mini200/esm8/embeddings.f16.npy',
    'test200_8m': FRAG / 'independent_test200/data/mini200/esm8/embeddings.f16.npy',
    'p0_650m': P0 / 'features/esm2_650m/embeddings.f16.npy',
    'old200_650m': OLD / 'features/esm2_650m/embeddings.f16.npy',
    'new200_650m': NEW / 'features/esm2_650m/embeddings.f16.npy',
    'dev600_650m': DEV / 'features/esm2_650m/embeddings.f16.npy',
    'test200_650m': TEST / 'features/esm2_650m/embeddings.f16.npy',
}
NEW_8M = ['esm8_candidates', 'esm8_candidates_delta',
          'esm8_neighbors', 'esm8_neighbors_delta']
NEW_650M = ['esm650_candidates', 'esm650_candidates_delta']


def fasta_ids(path: Path) -> list[str]:
    with path.open('rb') as handle:
        return [line[1:].split()[0].decode() for line in handle if line.startswith(b'>')]


def read_ids(path: Path) -> list[str]:
    return path.read_text().splitlines()


def load_new(names: list[str], width: int) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    matrices, ids = {}, {}
    for name in names:
        folder = GPU / name
        if not (folder / 'TRANSFER_COMPLETE').exists():
            raise RuntimeError(f'GPU result not SHA-verified on B5: {name}')
        rows = read_ids(folder / 'ids.txt')
        matrix = np.load(folder / 'embeddings.f16.npy', mmap_mode='r', allow_pickle=False)
        if matrix.shape != (len(rows), width) or matrix.dtype != np.float16:
            raise RuntimeError(f'GPU matrix contract failed: {name}')
        if len(set(rows)) != len(rows):
            raise RuntimeError(f'GPU IDs duplicated: {name}')
        matrices[name], ids[name] = matrix, rows
    return matrices, ids


def build_lookup(target_ids: list[str], model: str, new_names: list[str],
                 new_matrices: dict[str, np.ndarray],
                 new_ids: dict[str, list[str]]) -> dict[str, list[tuple[int, int]]]:
    index = {name: i for i, name in enumerate(target_ids)}
    if len(index) != len(target_ids):
        raise RuntimeError(f'{model}: duplicate target IDs')
    coverage = np.zeros(len(target_ids), dtype=np.uint8)
    pairs = defaultdict(list)
    reused = pd.read_parquet(AUDIT / f'{model}_QUERY_REUSE.parquet',
                             columns=['query_raw_sha256', 'cache_source', 'matrix_row'])
    for row in reused.itertuples(index=False):
        key = 'H_' + row.query_raw_sha256
        if key not in index:
            if model == '8M':
                raise RuntimeError(f'{model}: cached ID outside all-window target')
            continue
        i = index[key]
        if coverage[i]:
            raise RuntimeError(f'{model}: duplicate cached target {key}')
        coverage[i] = 1
        pairs[row.cache_source].append((i, int(row.matrix_row)))
    for name in new_names:
        if name not in new_matrices:
            raise RuntimeError(f'{model}: missing new matrix {name}')
        for row, key in enumerate(new_ids[name]):
            if key not in index:
                raise RuntimeError(f'{model}: new GPU ID outside target {key}')
            i = index[key]
            if coverage[i]:
                raise RuntimeError(f'{model}: cached/new overlap {key}')
            coverage[i] = 1
            pairs[name].append((i, row))
    if not coverage.all():
        missing = int((coverage == 0).sum())
        raise RuntimeError(f'{model}: {missing} targets lack an embedding')
    print(json.dumps({'model': model, 'targets': len(target_ids),
                      'source_counts': {key: len(value) for key, value in pairs.items()}}),
          flush=True)
    return pairs


def materialize(path: Path, n: int, width: int,
                pairs: dict[str, list[tuple[int, int]]],
                matrices: dict[str, np.ndarray]) -> None:
    part = path.with_suffix(path.suffix + '.part')
    output = np.lib.format.open_memmap(part, mode='w+', dtype=np.float16,
                                       shape=(n, width))
    for source, locations in pairs.items():
        if source not in matrices:
            raise RuntimeError(f'matrix source missing: {source}')
        matrix = matrices[source]
        order = np.argsort(np.fromiter((x[0] for x in locations),
                                       dtype=np.int64, count=len(locations)))
        to_rows = np.fromiter((locations[i][0] for i in order),
                              dtype=np.int64, count=len(locations))
        from_rows = np.fromiter((locations[i][1] for i in order),
                                dtype=np.int64, count=len(locations))
        if (from_rows < 0).any() or (from_rows >= len(matrix)).any():
            raise RuntimeError(f'{source}: source matrix row outside range')
        for start in range(0, len(locations), 4096):
            stop = min(start + 4096, len(locations))
            block = np.asarray(matrix[from_rows[start:stop]], dtype=np.float16)
            if block.shape != (stop - start, width) or not np.isfinite(block).all():
                raise RuntimeError(f'{source}: nonfinite or wrong width block')
            output[to_rows[start:stop]] = block
        print(json.dumps({'source': source, 'written': len(locations)}), flush=True)
    output.flush()
    del output
    os.replace(part, path)


def expand_to_loci(unique8: Path, unique650: Path,
                   all_ids: list[str], candidate_ids: list[str]) -> int:
    manifest = pd.read_parquet(ROOT / 'candidate_manifest/CANDIDATE_POSITIONS.parquet',
                               columns=['host_protein_id', 'aa_sha256'])
    if len(manifest) != 1_131_354:
        raise RuntimeError('candidate locus count differs from frozen manifest')
    locus_ids = manifest.host_protein_id.astype(str).tolist()
    if len(set(locus_ids)) != len(locus_ids):
        raise RuntimeError('candidate locus IDs duplicated')
    all_index = {item: i for i, item in enumerate(all_ids)}
    candidate_index = {item: i for i, item in enumerate(candidate_ids)}
    ids8, ids650 = [], []
    for digest in manifest.aa_sha256:
        key = 'H_' + digest
        if key not in all_index or key not in candidate_index:
            raise RuntimeError(f'candidate locus missing unique source: {key}')
        ids8.append(all_index[key])
        ids650.append(candidate_index[key])
    indices8 = np.asarray(ids8, dtype=np.int64)
    indices650 = np.asarray(ids650, dtype=np.int64)
    sources = [np.load(unique8, mmap_mode='r'), np.load(unique650, mmap_mode='r')]
    for name, source, indices, width in [
        ('ESM8_CANDIDATE_LOCUS.f16.npy', sources[0], indices8, 320),
        ('ESM650_CANDIDATE_LOCUS.f16.npy', sources[1], indices650, 1280)]:
        target = OUT / name
        part = target.with_suffix(target.suffix + '.part')
        output = np.lib.format.open_memmap(part, mode='w+', dtype=np.float16,
                                           shape=(len(locus_ids), width))
        for start in range(0, len(locus_ids), 8192):
            stop = min(start + 8192, len(locus_ids))
            block = np.asarray(source[indices[start:stop]], dtype=np.float16)
            if not np.isfinite(block).all():
                raise RuntimeError(f'nonfinite {name} at loci {start}:{stop}')
            output[start:stop] = block
        output.flush()
        del output
        os.replace(part, target)
        print(json.dumps({'locus_matrix': name, 'rows': len(locus_ids),
                          'columns': width}), flush=True)
    expected_ids = read_ids(OUT / 'ids.txt')
    if locus_ids != expected_ids:
        raise RuntimeError('structural and embedding locus ID order differs')
    return len(locus_ids)


def main() -> None:
    if (OUT / 'STOPFREE_EMBEDDINGS_COMPLETE').exists():
        print((OUT / 'embeddings_audit.json').read_text(), flush=True)
        return
    if not (OUT / 'STRUCTURAL_INPUTS_COMPLETE').exists():
        raise RuntimeError('structural candidate order is not finalized')
    if not (AUDIT / 'CACHE_AUDIT_COMPLETE').exists():
        raise RuntimeError('stop-free cache audit incomplete')
    all_ids = fasta_ids(ROOT / 'broad_screen/HOST_WINDOW_UNIQUE.faa')
    candidate_ids = read_ids(ROOT / 'broad_screen/candidate_ids.txt')
    if len(all_ids) != 1_536_214 or len(candidate_ids) != 582_906:
        raise RuntimeError('host unique/candidate counts differ from audit')
    if not set(candidate_ids).issubset(set(all_ids)):
        raise RuntimeError('candidate unique ID outside host-window FASTA')

    new8, new8_ids = load_new(NEW_8M, 320)
    new650, new650_ids = load_new(NEW_650M, 1280)
    matrices = {name: np.load(path, mmap_mode='r', allow_pickle=False)
                for name, path in CACHE_MATRICES.items()}
    matrices.update(new8)
    matrices.update(new650)
    pairs8 = build_lookup(all_ids, '8M', NEW_8M, new8, new8_ids)
    pairs650 = build_lookup(candidate_ids, '650M', NEW_650M, new650, new650_ids)
    unique8 = OUT / 'ESM8_ALL_UNIQUE.f16.npy'
    unique650 = OUT / 'ESM650_CANDIDATE_UNIQUE.f16.npy'
    materialize(unique8, len(all_ids), 320, pairs8, matrices)
    materialize(unique650, len(candidate_ids), 1280, pairs650, matrices)
    loci = expand_to_loci(unique8, unique650, all_ids, candidate_ids)
    (OUT / 'all_unique_ids.txt').write_text('\n'.join(all_ids) + '\n')
    (OUT / 'candidate_unique_ids.txt').write_text('\n'.join(candidate_ids) + '\n')
    audit = {'state': 'STOPFREE_EMBEDDINGS_COMPLETE',
             'all_window_unique_8m': len(all_ids),
             'candidate_unique_650m': len(candidate_ids),
             'candidate_loci': loci, '8m_new_sequences': sum(len(new8_ids[x]) for x in NEW_8M),
             '650m_new_sequences': sum(len(new650_ids[x]) for x in NEW_650M),
             'stopfree_cache_only': True,
             'gpu_transfer_sha256_verified': True}
    (OUT / 'embeddings_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (OUT / 'STOPFREE_EMBEDDINGS_COMPLETE').touch()
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
