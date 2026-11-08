from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path('/ifs1/User/yanyueyang/yyy/gene-finder/01.db/03-mge')
P0 = BASE / 'MOBILITY_V3_BOTELHO_V3HN_8M_20261028'
OLD = BASE / 'SPACEM_ARG_MRG_MGE_BINARY_LGBM_20261030'
NEW = BASE / 'SPACEM_ARG_MRG_MGE_BINARY_LGBM_NEW200_20261030'
DEV = BASE / 'SPACEM_CHROMMOB_V2_STAGE3_DEV600_20261030'
TEST = BASE / 'SPACEM_CHROMMOB_V2_STAGE3_TEST200_20261030'
FRAG = BASE / 'MOBILITY_V3_BOTELHO_SPACEM_FRAGMENT_20261029'
FRAG_NEW = BASE / 'MOBILITY_V3_BOTELHO_SPACEM_FRAGMENT_NEW200_20261030'
FRAG_DEV = BASE / 'MOBILITY_V3_BOTELHO_SPACEM_FRAGMENT_DEV600_20261030'


def fasta(path):
    name, seq = None, []
    with path.open('rb') as handle:
        for line in handle:
            if line.startswith(b'>'):
                if name is not None:
                    yield name, b''.join(seq).upper()
                name, seq = line[1:].split()[0].decode(), []
            else:
                seq.append(line.strip())
        if name is not None:
            yield name, b''.join(seq).upper()


def source(name, fasta_file, ids_file, matrix_file, dimension):
    if not fasta_file.exists() or not ids_file.exists() or not matrix_file.exists():
        raise RuntimeError(f'cache source missing: {name}')
    ids = ids_file.read_text().splitlines()
    matrix = np.load(matrix_file, mmap_mode='r')
    if matrix.shape != (len(ids), dimension) or len(ids) != len(set(ids)):
        raise RuntimeError(f'cache source shape/ID mismatch: {name}')
    index = {item: i for i, item in enumerate(ids)}
    mapping = {}
    raw_stops = 0
    matched = 0
    for item, sequence in fasta(fasta_file):
        i = index.get(item)
        if i is None:
            continue
        matched += 1
        raw_stops += int(b'*' in sequence)
        digest = hashlib.sha256(sequence).hexdigest()
        mapping.setdefault(digest, (name, i))
    if matched != len(ids):
        raise RuntimeError(f'cache FASTA coverage mismatch: {name} {matched}/{len(ids)}')
    return mapping, {'source': name, 'matrix_rows': len(ids), 'exact_input_hashes': len(mapping),
                     'input_sequences_with_stop': raw_stops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    args = ap.parse_args()
    out = args.root / 'cache_audit_v2'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'CACHE_AUDIT_COMPLETE').exists():
        print((out / 'cache_audit.json').read_text(), flush=True)
        return
    screen = args.root / 'broad_screen'
    if not (screen / 'BROAD_SCREEN_COMPLETE').exists():
        raise RuntimeError('permissive screen incomplete')
    configs = {
        '8M': [
            ('p0_8m', P0 / 'screen/broad_candidates_unique.faa', P0 / 'features/esm2_8m/ids.txt', P0 / 'features/esm2_8m/embeddings.f16.npy', 320),
            ('old200_8m', FRAG / 'data/mini200/esm8_unique.faa', FRAG / 'data/mini200/esm8/ids.txt', FRAG / 'data/mini200/esm8/embeddings.f16.npy', 320),
            ('new200_8m', FRAG_NEW / 'data/mini200/esm8_unique.faa', FRAG_NEW / 'data/mini200/esm8/ids.txt', FRAG_NEW / 'data/mini200/esm8/embeddings.f16.npy', 320),
            ('dev600_8m', FRAG_DEV / 'data/mini200/esm8_unique.faa', FRAG_DEV / 'data/mini200/esm8/ids.txt', FRAG_DEV / 'data/mini200/esm8/embeddings.f16.npy', 320),
            ('test200_8m', FRAG / 'independent_test200/data/mini200/esm8_unique.faa', FRAG / 'independent_test200/data/mini200/esm8/ids.txt', FRAG / 'independent_test200/data/mini200/esm8/embeddings.f16.npy', 320)],
        '650M': [
            ('p0_650m', P0 / 'screen/broad_candidates_unique.faa', P0 / 'features/esm2_650m/ids.txt', P0 / 'features/esm2_650m/embeddings.f16.npy', 1280),
            ('old200_650m', OLD / 'screen/broad_candidates_unique.faa', OLD / 'features/esm2_650m/ids.txt', OLD / 'features/esm2_650m/embeddings.f16.npy', 1280),
            ('new200_650m', NEW / 'screen/broad_candidates_unique.faa', NEW / 'features/esm2_650m/ids.txt', NEW / 'features/esm2_650m/embeddings.f16.npy', 1280),
            ('dev600_650m', DEV / 'screen/broad_candidates_unique.faa', DEV / 'features/esm2_650m/ids.txt', DEV / 'features/esm2_650m/embeddings.f16.npy', 1280),
            ('test200_650m', TEST / 'screen/broad_candidates_unique.faa', TEST / 'features/esm2_650m/ids.txt', TEST / 'features/esm2_650m/embeddings.f16.npy', 1280)]}
    indices = {}
    audit = {'sources': {}, 'scope': 'strict exact stop-free sequence matching to verified matrix IDs'}
    for model, sources in configs.items():
        lookup = {}
        for spec in sources:
            found, info = source(*spec)
            audit['sources'][spec[0]] = info
            for digest, record in found.items():
                lookup.setdefault(digest, record)
            print(json.dumps(info), flush=True)
        indices[model] = lookup
        pd.DataFrame([{'aa_sha256': key, 'cache_source': val[0], 'matrix_row': val[1]}
                      for key, val in lookup.items()]).to_parquet(
            out / f'{model}_EXACT_CACHE_INDEX.parquet', index=False, compression='zstd')
    candidates = set((screen / 'candidate_ids.txt').read_text().splitlines())
    all_unique = 0
    candidate_seen = 0
    reuse8_all = reuse8_cand = reuse650 = 0
    normalized8 = normalized650 = 0
    rows8, rows650 = [], []
    missing8_all = out / 'MISSING_8M_ALL_WINDOWS.faa.part'
    missing650 = out / 'MISSING_650M_CANDIDATES.faa.part'
    with missing8_all.open('wb') as f8, missing650.open('wb') as f650:
        for name, sequence in fasta(screen / 'HOST_WINDOW_UNIQUE.faa'):
            digest = name.removeprefix('H_')
            if hashlib.sha256(sequence).hexdigest() != digest:
                raise RuntimeError(f'unique FASTA sequence digest mismatch: {name}')
            all_unique += 1
            is_cand = name in candidates
            candidate_seen += int(is_cand)
            plain = sequence[:-1] if sequence.endswith(b'*') and b'*' not in sequence[:-1] else sequence
            normalized_digest = hashlib.sha256(plain).hexdigest()
            match8 = indices['8M'].get(digest)
            mode8 = 'raw_exact'
            if match8 is None and plain != sequence:
                match8 = indices['8M'].get(normalized_digest)
                mode8 = 'terminal_stop_removed'
            if match8 is not None:
                reuse8_all += 1
                reuse8_cand += int(is_cand)
                normalized8 += int(mode8 != 'raw_exact')
                rows8.append({'query_raw_sha256': digest, 'cache_source': match8[0],
                              'matrix_row': match8[1], 'normalization': mode8})
            else:
                f8.write(b'>' + name.encode() + b'\n' + plain + b'\n')
            if is_cand:
                match650 = indices['650M'].get(digest)
                mode650 = 'raw_exact'
                if match650 is None and plain != sequence:
                    match650 = indices['650M'].get(normalized_digest)
                    mode650 = 'terminal_stop_removed'
                if match650 is not None:
                    reuse650 += 1
                    normalized650 += int(mode650 != 'raw_exact')
                    rows650.append({'query_raw_sha256': digest, 'cache_source': match650[0],
                                    'matrix_row': match650[1], 'normalization': mode650})
                else:
                    f650.write(b'>' + name.encode() + b'\n' + plain + b'\n')
    if candidate_seen != len(candidates):
        raise RuntimeError('candidate FASTA and IDs disagree')
    os.replace(missing8_all, out / 'MISSING_8M_ALL_WINDOWS.faa')
    os.replace(missing650, out / 'MISSING_650M_CANDIDATES.faa')
    pd.DataFrame(rows8).to_parquet(out / '8M_QUERY_REUSE.parquet', index=False, compression='zstd')
    pd.DataFrame(rows650).to_parquet(out / '650M_QUERY_REUSE.parquet', index=False, compression='zstd')
    audit.update({'all_window_unique_sequences': all_unique,
                  'broad_candidate_unique_sequences': candidate_seen,
                  '8M_all_window_reusable': reuse8_all,
                  '8M_all_window_missing': all_unique - reuse8_all,
                  '8M_candidate_reusable': reuse8_cand,
                  '8M_candidate_missing': candidate_seen - reuse8_cand,
                  '8M_terminal_stop_normalized_reuse': normalized8,
                  '650M_candidate_reusable': reuse650,
                  '650M_candidate_missing': candidate_seen - reuse650,
                  '650M_terminal_stop_normalized_reuse': normalized650,
                  'missing_sequence_format': 'one terminal stop removed; IDs preserve raw host SHA256',
                  'GPU_not_started': True})
    (out / 'cache_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (out / 'CACHE_AUDIT_COMPLETE').touch()
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
