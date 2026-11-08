from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = Path('/ifs1/User/yanyueyang/yyy/gene-finder')
ROOT = BASE / '01.db/03-mge/SPACEM_CHROMMOB_FULL_CPU_20261030'
SOURCE = BASE / '01.db/03-mge/MOBILITY_V3_BOTELHO_V3HN_8M_20261028/features'
CONTRACT = BASE / 'protein_negative_pipeline_runs/ANY_TARGET_PRIMARY2000_COORDINATE_SAFE_LOCALIZATION_V2_20261017/FEATURE_RESULT.json'
PAT = re.compile(r'(?:^|;)ID=\d+_(\d+)')
OFFSETS = list(range(-10, 0)) + list(range(1, 11))
PORTABLE_COLUMNS = ['protein_length'] + [
    f"nb_{'m'+str(abs(k)).zfill(2) if k<0 else 'p'+str(k).zfill(2)}_{suffix}"
    for k in OFFSETS for suffix in ('protein_length', 'is_pad')
] + ['w10_neighbor_count', 'w10_mean_protein_length', 'w10_std_protein_length']
NEIGHBOR_COLUMNS = [f"neighbor_{'m'+str(abs(k)).zfill(2) if k<0 else 'p'+str(k).zfill(2)}"
                    for k in OFFSETS]


def prepare():
    out = ROOT / 'portable_context_full_host'
    out.mkdir(exist_ok=True)
    if (out / 'INPUTS_COMPLETE').exists():
        return
    frozen = json.loads(CONTRACT.read_text())['portable_context_columns']
    if PORTABLE_COLUMNS != frozen or len(PORTABLE_COLUMNS) != 44:
        raise RuntimeError('portable 44-column frozen schema changed')
    manifest = json.loads((SOURCE / 'host_replicons/batch_manifest.json').read_text())
    batch = {acc: i for i, group in enumerate(manifest['batches']) for acc in group}
    candidates = pd.read_parquet(ROOT / 'candidate_manifest/CANDIDATE_POSITIONS.parquet',
                                 columns=['host_protein_id', 'accession', 'aa_length'])
    candidates['batch'] = candidates.accession.map(batch)
    if candidates.batch.isna().any() or candidates.host_protein_id.duplicated().any():
        raise RuntimeError('portable candidate IDs invalid')
    inputs = out / 'inputs'
    inputs.mkdir(exist_ok=True)
    for i in range(len(manifest['batches'])):
        candidates.loc[candidates.batch.eq(i), ['host_protein_id', 'accession', 'aa_length']].to_parquet(
            inputs / f'batch{i:04d}.parquet', index=False, compression='zstd')
    (out / 'input_audit.json').write_text(json.dumps({'candidate_positions': len(candidates),
                                                      'batches': len(manifest['batches'])}, indent=2) + '\n')
    (out / 'INPUTS_COMPLETE').touch()


def protein_lengths(path, needed):
    lengths = {}
    name, n = None, 0
    with path.open('rb') as handle:
        for line in handle:
            if line.startswith(b'>'):
                if name is not None:
                    lengths[name] = n
                name = line[1:].split()[0].decode()
                n = 0
            else:
                n += len(line.strip())
        if name is not None:
            lengths[name] = n
    return lengths


def run_batch(i):
    out = ROOT / 'portable_context_full_host'
    folder = out / f'batch{i:04d}'
    folder.mkdir(exist_ok=True)
    if (folder / 'COMPLETE').exists():
        return i, 'cached', 0
    candidates = pd.read_parquet(out / 'inputs' / f'batch{i:04d}.parquet')
    if candidates.empty:
        pd.DataFrame(columns=['host_protein_id', *PORTABLE_COLUMNS]).to_parquet(folder / 'PORTABLE.parquet', index=False)
        pd.DataFrame(columns=['host_protein_id', *NEIGHBOR_COLUMNS]).to_parquet(folder / 'NEIGHBORS.parquet', index=False)
        (folder / 'audit.json').write_text(json.dumps({'rows': 0, 'center_length_agreement': 0}) + '\n')
        (folder / 'COMPLETE').touch()
        return i, 'empty', 0
    needed = set(candidates.accession)
    lengths = protein_lengths(SOURCE / f'host_prodigal/batch{i:04d}/host_proteins.faa', needed)
    genes = defaultdict(list)
    with (SOURCE / f'host_prodigal/batch{i:04d}/host_genes.gff').open() as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9 or parts[2] != 'CDS' or parts[0] not in needed:
                continue
            match = PAT.search(parts[8])
            if match is None:
                raise RuntimeError(f'portable GFF gene ID missing: {i}')
            pid = f'{parts[0]}_{match.group(1)}'
            genes[parts[0]].append((int(parts[3]), int(parts[4]), pid))
    portable, neighbors, agreements = [], [], 0
    for acc, group in candidates.groupby('accession', sort=False):
        ordered = sorted(genes[acc], key=lambda item: (item[0], item[1], item[2]))
        index = {pid: j for j, (_, _, pid) in enumerate(ordered)}
        for item in group.itertuples(index=False):
            pid = item.host_protein_id
            j = index.get(pid)
            if j is None or pid not in lengths:
                raise RuntimeError(f'portable candidate not in full host: {pid}')
            agreements += int(lengths[pid] == int(item.aa_length))
            vals, neighbor_ids, observed = [float(lengths[pid])], [], []
            for offset in OFFSETS:
                k = j + offset
                neighbor = ordered[k][2] if 0 <= k < len(ordered) else ''
                neighbor_ids.append(neighbor)
                if neighbor:
                    if neighbor not in lengths:
                        raise RuntimeError(f'portable neighbor protein absent: {neighbor}')
                    length = float(lengths[neighbor])
                    vals.extend([length, 0.0])
                    observed.append(length)
                else:
                    vals.extend([-1.0, 1.0])
            vals.extend([float(len(observed)), statistics.mean(observed) if observed else -1.0,
                         statistics.stdev(observed) if len(observed) > 1 else -1.0])
            portable.append([pid, *vals])
            neighbors.append([pid, *neighbor_ids])
    if agreements != len(candidates):
        raise RuntimeError(f'portable center AA length mismatch: {i} {agreements}/{len(candidates)}')
    pf = pd.DataFrame(portable, columns=['host_protein_id', *PORTABLE_COLUMNS])
    nf = pd.DataFrame(neighbors, columns=['host_protein_id', *NEIGHBOR_COLUMNS])
    if len(pf) != len(candidates) or pf.host_protein_id.duplicated().any():
        raise RuntimeError(f'portable row coverage failure: {i}')
    for frame, name in ((pf, 'PORTABLE'), (nf, 'NEIGHBORS')):
        part = folder / f'{name}.parquet.part'
        frame.to_parquet(part, index=False, compression='zstd')
        os.replace(part, folder / f'{name}.parquet')
    (folder / 'audit.json').write_text(json.dumps({'rows': len(pf),
                                                  'center_length_agreement': agreements}) + '\n')
    (folder / 'COMPLETE').touch()
    return i, 'complete', len(pf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    prepare()
    out = ROOT / 'portable_context_full_host'
    if (out / 'PORTABLE_CONTEXT_COMPLETE').exists():
        print((out / 'audit.json').read_text(), flush=True)
        return
    manifest = json.loads((SOURCE / 'host_replicons/batch_manifest.json').read_text())
    n = len(manifest['batches'])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_batch, i): i for i in range(n)}
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    if not all((out / f'batch{i:04d}/COMPLETE').exists() for i in range(n)):
        raise RuntimeError('portable batches incomplete')
    rows = sum(json.loads((out / f'batch{i:04d}/audit.json').read_text())['rows'] for i in range(n))
    expected = json.loads((out / 'input_audit.json').read_text())['candidate_positions']
    if rows != expected:
        raise RuntimeError(f'portable rows {rows} != {expected}')
    audit = {'candidate_positions': rows, 'portable_features': 44,
             'w10_neighbor_ids_per_center': 20, 'full_source_replicon_context': True}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (out / 'PORTABLE_CONTEXT_COMPLETE').touch()
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
