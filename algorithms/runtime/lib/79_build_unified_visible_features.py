from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

KEYS = ['host_protein_id', 'length_kb', 'placement']
LENGTHS = (2, 5, 10, 20, 50, 100)
PLACEMENTS = (.1, .5, .9)
META = ['host_protein_id', 'accession', 'connected_group', 'fold', 'length_kb',
        'placement', 'y', 'baseline_call', 'cargo_class', 'inside_element_type',
        'cohort_weight', 'fit_weight']
ROOT = Path('/ifs1/User/yanyueyang/yyy/gene-finder/01.db/03-mge/SPACEM_CHROMMOB_GENOME_FRAG1KB_CPU_20261031')


def dump(path: Path, value):
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def key_digest(frame):
    h = hashlib.sha256()
    for row in frame[KEYS].itertuples(index=False, name=None):
        h.update(('\t'.join(map(str, row)) + '\n').encode())
    return h.hexdigest()


def eligible_counts(cargo):

    cs = cargo.start1.to_numpy(np.int64)
    ce = cargo.end1.to_numpy(np.int64)
    host = cargo.b_contig_len_bp.to_numpy(np.int64)
    mid = (cs + ce) / 2
    count = np.zeros(len(cargo), np.int16)
    for kb in LENGTHS:
        width = kb * 1000
        for pos in PLACEMENTS:
            start = np.maximum(1, np.minimum(np.rint(mid-pos*width).astype(np.int64), host-width+1))
            count += ((host >= width) & (start <= cs) & (ce <= start+width-1))
    return count


def local_map():

    result = {c:c for c in ('p_ARG', 'p_MRG', 'p_MGE_self', 'gate_score', 'cargo_length_bp')}
    result.update(visible_length_bp='b_contig_len_bp',
                  visible_left_flank_bp='b_left_edge_bp',
                  visible_right_flank_bp='b_right_edge_bp',
                  nearest_mge_gap_bp='nearest_mge_edge_gap_bp',
                  nearest_mge_score='nearest_mge_p_MGE',
                  near_has_mge='mge_count_100kb > 0',
                  near_within_5kb='near_mge_present_5kb')
    for width in (1000, 2000, 5000, 10000, 20000, 50000, 100000):
        result[f'near_count_{width}'] = f'mge_count_{width//1000}kb'
        result[f'near_score_sum_{width}'] = f'mge_score_sum_{width//1000}kb'
    for lam in (1000, 2000, 5000, 10000, 20000, 50000):
        result[f'near_exp_sum_{lam}'] = f'mge_exp_sum_lambda_{lam//1000}kb'
        result[f'nearest_proximity_{lam}'] = f'near_mge_proximity_{lam//1000}kb'
        result[f'nearest_weighted_proximity_{lam}'] = f'near_mge_score_proximity_{lam//1000}kb'
    assert len(result) == 44
    return result


def construct_unified(base, local, old_columns, augment):

    matrix, columns, constraints = augment(base, old_columns)
    unified = pd.DataFrame(matrix, columns=columns)
    if len(columns) != 236:
        raise RuntimeError('full236 feature contract changed')
    mapping = local_map()
    present = local.near_has_mge.to_numpy(bool)
    if not np.array_equal(present, base.mge_count_100kb.to_numpy() > 0):
        raise RuntimeError('visible MGE existence disagrees between cached views')
    added = []
    for old, new in mapping.items():
        values = local[old].to_numpy(np.float64)
        if new == 'mge_count_100kb > 0':
            continue
        if new not in unified:
            unified[new] = values.astype(np.float32)
            constraints[new] = 0 if new == 'cargo_length_bp' else 1
            added.append(new)
            continue
        expected = unified[new].to_numpy(np.float64)
        compare = np.ones(len(values), dtype=bool)


        if old in ('nearest_mge_gap_bp', 'nearest_mge_score'):
            compare = present
        if not np.allclose(values[compare], expected[compare], rtol=2e-4, atol=1e-5, equal_nan=True):
            bad = np.flatnonzero(~np.isclose(values, expected, rtol=2e-4, atol=1e-5, equal_nan=True) & compare)
            raise RuntimeError(f'cached semantic alias mismatch: {old} -> {new}; n={len(bad)}, first={bad[:5].tolist()}')
    if len(unified.columns) != 248 or len(added) != 12:
        raise RuntimeError(f'unified feature union changed: {len(unified.columns)}; added={added}')
    if np.isinf(unified.to_numpy(np.float32)).any():
        raise RuntimeError('infinite unified feature values')
    return unified, list(unified.columns), constraints, added, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--context-dir', type=Path)
    args = ap.parse_args()
    root = args.root
    out = args.out or root/'unified_visible_dev_20261102'
    context = args.context_dir or root/'fragment_full_near_monotone_dev_20261101'
    out.mkdir(parents=True, exist_ok=True)
    context_audit = context/'audit.json'
    if not (context/'FRAGMENT_CONTEXT_COMPLETE').exists() or not context_audit.exists():
        raise RuntimeError('visible fragment229 contexts are not complete; preserve running build; do not copy full-host features into fragments')
    localpath = root/'fragment_near_monotone_dev_20261101/DEVELOPMENT_FEATURES.parquet'
    fullpath = root/'strict_v3hn_spatial/CARGO_EVIDENCE_FULL.parquet'
    columnpath = root/'strict_v3hn_spatial/feature_columns.json'
    augmentpath = root/'scripts/58_train_near_monotone_dev.py'
    protocol = {'purpose':'one visible-context feature schema for full replicons and fragments',
                'version':'unified248_20261102', 'script_digest':sha(Path(__file__)),
                'full_input_digest':sha(fullpath), 'local44_input_digest':sha(localpath),
                'visible_context_audit_digest':sha(context_audit),
                'full236_augment_script_digest':sha(augmentpath),
                'development_only':True, 'test_predictions_or_labels_read':False,
                'old_model_scores_or_thresholds_used_as_features':False,
                'frozen_host_V3HN_annotation_reused':True,
                'end_to_end_fragment_V3HN_annotation_recomputed':False,
                'exact_length_only':True, 'lengths_kb':list(LENGTHS),
                'placements':list(PLACEMENTS),
                'outside_is_unverified_background':True,
                'missing_DNA_or_annotation_values_retained_as_NaN':True,
                'local44_absence_flag_deduplicated_against_100kb_count':True,
                'all_parent_geometry_and_labels_are_metadata_only':True,
                'old_cohort_and_fit_weights_are_provenance_only_trainer_reweights':True}
    pp = out/'BUILD_PROTOCOL.json'
    if pp.exists() and json.loads(pp.read_text()) != protocol:
        raise RuntimeError('existing unified build protocol differs; preserve old directory and use a new output location')
    dump(pp, protocol)
    if (out/'FEATURES_COMPLETE').exists():
        print((out/'audit.json').read_text(), flush=True)
        return
    local = pd.read_parquet(localpath)
    if local.duplicated(KEYS).any() or local.groupby('connected_group').fold.nunique().max() != 1:
        raise RuntimeError('local44 keys or development fold grouping changed')
    keep = local.length_kb.eq(0) | local.visible_length_bp.eq(local.length_kb*1000)
    excluded = int((~keep).sum())
    local = local.loc[keep].copy().reset_index(drop=True)
    full = pd.read_parquet(fullpath, filters=[('partition', '==', 'development')])
    full = full.loc[full.reference_position.isin(('inside','outside'))].copy()
    if full.host_protein_id.duplicated().any() or not full.partition.eq('development').all():
        raise RuntimeError('full development source contract failed')
    full = full.set_index('host_protein_id', drop=False)
    ids = local.host_protein_id
    if set(ids)-set(full.index):
        raise RuntimeError('unified requests outside development cargo source')
    parents = full.loc[ids].reset_index(drop=True)
    if not local.accession.eq(parents.accession).all():
        raise RuntimeError('source accession mismatch')
    if not np.array_equal(local.y.to_numpy(), parents.reference_position.eq('inside').to_numpy()):
        raise RuntimeError('development target label mismatch')
    old_columns = json.loads(columnpath.read_text())
    if len(old_columns) != 229:
        raise RuntimeError('229 source feature contract changed')
    chunks = []
    for folder in sorted(context.glob('batch[0-9][0-9][0-9][0-9]')):
        path = folder/'FRAGMENT_FULL_CONTEXT.parquet'
        if not (folder/'COMPLETE').exists() or not path.exists():
            raise RuntimeError(f'incomplete cached fragment batch: {folder}')
        chunks.append(pd.read_parquet(path))
    fragment = pd.concat(chunks, ignore_index=True)
    if fragment.duplicated(KEYS).any():
        raise RuntimeError('cached fragment feature keys duplicated')
    fragmeta = local.loc[local.length_kb.gt(0), KEYS]
    fragment = fragmeta.merge(fragment, on=KEYS, how='left', validate='one_to_one', indicator=True)
    if not fragment._merge.eq('both').all():
        raise RuntimeError('cached fragment context coverage incomplete')
    if not np.array_equal(fragment.audit_baseline_5kb_call.to_numpy(bool),
                          local.loc[local.length_kb.gt(0), 'baseline_call'].to_numpy(bool)):
        raise RuntimeError('visible 5kb baseline disagreement')





    base = parents[old_columns].astype(np.float64).copy()
    ix = local.length_kb.gt(0).to_numpy()
    base.loc[ix, old_columns] = fragment[old_columns].to_numpy()
    if not np.array_equal(local.loc[~ix, 'baseline_call'].to_numpy(bool),
                          parents.loc[~ix, 'baseline_predicted_mge_within_5kb'].to_numpy(bool)):
        raise RuntimeError('full-host5kb baseline disagreement')
    augment = runpy.run_path(str(augmentpath))['augment']
    unified, features, constraints, added, mapping = construct_unified(base, local, old_columns, augment)
    meta = local[META].copy()
    meta['partition'] = 'development'
    meta['start1'] = parents.start1.to_numpy(np.int64)
    meta['end1'] = parents.end1.to_numpy(np.int64)
    meta['source_replicon_length_bp'] = parents.b_contig_len_bp.to_numpy(np.int64)
    full['eligible_fragment_design_count'] = eligible_counts(full)
    meta['eligible_fragment_design_count'] = ids.map(full.eligible_fragment_design_count).to_numpy(np.int16)
    meta['fragment_start1'] = 1
    meta['fragment_end1'] = meta.source_replicon_length_bp
    meta.loc[ix, 'fragment_start1'] = fragment.fragment_start1.to_numpy(np.int64)
    meta.loc[ix, 'fragment_end1'] = fragment.fragment_end1.to_numpy(np.int64)
    meta['actual_width_bp'] = meta.fragment_end1-meta.fragment_start1+1
    if not np.array_equal(meta.actual_width_bp.to_numpy(), local.visible_length_bp.to_numpy()):
        raise RuntimeError('cached visible DNA width and local44 geometry differ')
    if not ((meta.fragment_start1 <= meta.start1) & (meta.end1 <= meta.fragment_end1)).all():
        raise RuntimeError('non-intact cargo entered unified training requests')
    if not (meta.loc[ix].actual_width_bp.eq(meta.loc[ix].length_kb*1000)).all():
        raise RuntimeError('short-host clamped fragment entered exact training cohort')
    actual = meta.loc[ix].groupby('host_protein_id').size()
    wanted = meta.loc[ix].drop_duplicates('host_protein_id').set_index('host_protein_id').eligible_fragment_design_count
    if not actual.eq(wanted.loc[actual.index]).all():
        raise RuntimeError('retained fragment designs differ from independent eligibility reconstruction')
    if set(features)&set(meta.columns):
        raise RuntimeError('metadata or label columns entered feature whitelist')
    result = pd.concat([meta.reset_index(drop=True), unified.reset_index(drop=True)], axis=1)
    dest = out/'DEVELOPMENT_FEATURES.parquet'
    result.to_parquet(dest.with_suffix('.parquet.part'), index=False, compression='zstd')
    os.replace(dest.with_suffix('.parquet.part'), dest)
    dump(out/'feature_columns.json', features)
    dump(out/'CONSTRAINTS.json', constraints)
    dump(out/'FEATURE_MAP.json', {'local44_to_unified248':mapping,
         'features_added_to_full236':added,
         'different_missing_gap_sentinels_are_equal_absence_states':True,
         'full236_missing_gap_sentinel_bp':101001,
         'local44_missing_gap_sentinel_bp':100001,
         'explicit_MGE_presence_is_encoded_by_positive_100kb_count':True})
    summary = result.groupby('length_kb').agg(rows=('host_protein_id','size'),
                   cargo=('host_protein_id','nunique'),inside_windows=('y','sum')).reset_index()
    summary.to_csv(out/'COHORT_BY_LENGTH.tsv', sep='\t', index=False)
    audit = {**protocol, 'rows':len(result), 'full_rows':int((~ix).sum()),
             'fragment_rows':int(ix.sum()), 'feature_count':len(features),
             'added_to_full236':len(added), 'excluded_clamped_rows':excluded,
             'feature_key_digest':key_digest(result),
             'nan_feature_cells':int(unified.isna().sum().sum()),
             'alias_consistency_checks_passed':True,
             'visible_baseline_consistency_checks_passed':True,
             'independent_exact_design_count_checks_passed':True,
             'models_fitted':0, 'GPU_used':False}
    dump(out/'audit.json', audit)
    (out/'FEATURES_COMPLETE').touch()
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
