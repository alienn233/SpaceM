from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 20261102
LENGTHS = (2, 5, 10, 20, 50, 100)
ROUNDS = 300
PARAMS = dict(objective='binary', learning_rate=.05, num_leaves=15,
              max_depth=5, min_data_in_leaf=40, feature_fraction=.8,
              bagging_fraction=.8, bagging_freq=1, lambda_l2=5.,
              verbosity=-1, monotone_constraints_method='basic',
              deterministic=True, force_col_wise=True, device_type='cpu')
KEYS = ['host_protein_id', 'length_kb', 'placement']
META = ['host_protein_id', 'accession', 'connected_group', 'fold', 'length_kb',
        'placement', 'y', 'baseline_call', 'cargo_class', 'inside_element_type']


def dump(path, obj):
    tmp = path.with_name(path.name + '.part')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def frame_digest(d, columns):
    values = pd.util.hash_pandas_object(d[columns], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def atomic_parquet(d, path):
    tmp = path.with_name(path.name + '.part')
    d.to_parquet(tmp, index=False, compression='zstd')
    os.replace(tmp, path)


def atomic_model(model, path):
    tmp = path.with_name(path.name + '.part')
    model.save_model(str(tmp))
    os.replace(tmp, path)


def atomic_npy(values, path):
    tmp = path.with_name(path.name + '.part')
    with tmp.open('wb') as f:
        np.save(f, values)
    os.replace(tmp, path)


def sampled(pid):
    return int.from_bytes(hashlib.sha256(str(pid).encode()).digest()[:8], 'big') % 16 == 0


def weights(d):

    full = d.length_kb.eq(0)
    fragment = ~full
    original = d.loc[full]
    if original.host_protein_id.duplicated().any():
        raise RuntimeError('more than one full row per cargo')
    counts = original.groupby('connected_group').size()
    scope = np.full(len(d), .5, dtype=float)
    evaluation = np.ones(len(d), dtype=float)
    g = d.loc[fragment, ['host_protein_id', 'length_kb', 'y']]
    placements = g.groupby(['host_protein_id', 'length_kb']).host_protein_id.transform('size').to_numpy(float)
    lengths = g.groupby('host_protein_id').length_kb.transform('nunique').to_numpy(float)
    ipw = np.where(g.y.eq(0), 16., 1.)
    scope[fragment] = .5 * ipw / (placements*lengths)
    evaluation[fragment] = ipw / placements
    fit = scope / np.sqrt(d.connected_group.map(counts).to_numpy(float))
    fit[d.y.eq(0)] *= .5
    if not np.isfinite(fit).all() or not (fit > 0).all():
        raise RuntimeError('nonpositive/nonfinite training weights')
    masses = pd.DataFrame({'pid': d.loc[fragment, 'host_protein_id'],
                           'mass': scope[fragment], 'y': g.y}).groupby('pid').agg(
                               mass=('mass','sum'), y=('y','first'))
    target = np.where(masses.y.eq(0), 8., .5)
    if not np.allclose(masses.mass.to_numpy(), target, rtol=0., atol=1e-12):
        raise RuntimeError('fragment mass not normalized per cargo')
    audit = {'full_scope_mass_per_cargo': .5,
             'fragment_scope_mass_per_eligible_cargo_before_IPW': .5,
             'outside_fragment_sampling_probability': 1/16,
             'inverse_sampling_weight': 16,
             'retained_lengths_equal_within_cargo': True,
             'retained_placements_equal_within_cargo_length': True,
             'full_cargo': len(original), 'fragment_cargo': len(masses),
             'full_cargo_without_eligible_exact_intact_fragment': int(original.eligible_fragment_design_count.eq(0).sum()),
             'scope_mass_full': float(scope[full].sum()),
             'scope_mass_fragments_IPW': float(scope[fragment].sum()),
             'fit_group_downweight': 'divide by sqrt(original full cargo count per connected group)',
             'fit_outside_control_multiplier': .5,
             'fit_normalization': 'each OOF fit divides by its training-row mean only; final fit uses all development rows',
             'per_length_evaluation': 'each cargo equal across retained placements; outside IPW16',
             'no_eligible_fragment_policy': 'retain full .5 mass; no synthetic fragment or zero-fill'}
    return scope, evaluation, fit.astype('float32'), audit


def validate(d, features, root):
    required = set(META + ['partition', 'actual_width_bp', 'eligible_fragment_design_count'])
    missing = required - set(d)
    if missing:
        raise RuntimeError(f'missing metadata: {sorted(missing)}')
    if not d.partition.eq('development').all() or not d.y.isin([0,1]).all():
        raise RuntimeError('development-only binary-label boundary failed')
    if d.duplicated(KEYS).any() or d.groupby('host_protein_id').fold.nunique().max() != 1:
        raise RuntimeError('duplicate design or one cargo across folds')
    if set(d.fold.unique()) != set(range(5)) or d.groupby('connected_group').fold.nunique().max() != 1:
        raise RuntimeError('original connected groups cross folds')
    if not set(d.length_kb.unique()) <= {0, *LENGTHS}:
        raise RuntimeError('unexpected development length')
    fragment = d.length_kb.gt(0)
    if not d.loc[fragment, 'actual_width_bp'].eq(d.loc[fragment, 'length_kb']*1000).all():
        raise RuntimeError('shorter-than-requested fragment must not enter exact-kb fit')
    full = d.loc[~fragment]
    if full.host_protein_id.duplicated().any():
        raise RuntimeError('duplicate full-host cargo')
    if set(d.host_protein_id) != set(full.host_protein_id):
        raise RuntimeError('fragment lacks a full-host development anchor')
    anchor = full.set_index('host_protein_id')
    for column in ['accession','connected_group','fold','cargo_class','y']:
        if not np.array_equal(d[column].to_numpy(),
                              d.host_protein_id.map(anchor[column]).to_numpy()):
            raise RuntimeError(f'fragment {column} differs from its original full-host anchor')
    if d.groupby('host_protein_id').y.nunique().max() != 1:
        raise RuntimeError('one cargo has inconsistent labels across windows')
    if not d.loc[d.y.eq(1), 'inside_element_type'].isin(['ICE', 'IME']).all():
        raise RuntimeError('positive type is not reference ICE/IME')

    old = pd.read_parquet(root/'strict_reference_localization_dev/DEV_OOF_PREDICTIONS.parquet',
                         columns=['host_protein_id','variant','fold','connected_group',
                                  'accession','reference_position'])
    old = old.loc[old.variant.eq('all_evidence')].set_index('host_protein_id')
    if old.index.duplicated().any() or set(old.index) != set(full.host_protein_id):
        raise RuntimeError('full development cargo differs from frozen original cohort')
    old = old.loc[full.host_protein_id]
    for column in ['fold','connected_group','accession']:
        if not np.array_equal(old[column].to_numpy(), full[column].to_numpy()):
            raise RuntimeError(f'frozen development {column} changed')
    if not np.array_equal(old.reference_position.eq('inside').to_numpy(), full.y.to_numpy(bool)):
        raise RuntimeError('frozen development label changed')
    seen = set(d.loc[fragment, 'host_protein_id'])
    expected = set(full.loc[full.eligible_fragment_design_count.gt(0) &
                           (full.y.eq(1) | full.host_protein_id.map(sampled)), 'host_protein_id'])
    if seen != expected:
        raise RuntimeError('sampled/positive eligible fragment cargo coverage mismatch')
    frag_counts = d.loc[fragment].groupby('host_protein_id').size()
    expected_counts = full.set_index('host_protein_id').eligible_fragment_design_count.reindex(sorted(expected))
    if not np.array_equal(frag_counts.reindex(expected_counts.index).to_numpy(), expected_counts.to_numpy()):
        raise RuntimeError('not all retained exact intact designs are represented')
    if len(features) != len(set(features)) or set(features) - set(d):
        raise RuntimeError('invalid feature whitelist')
    forbidden = set(META + ['partition','reference_position','fragment_start1','fragment_end1',
                            'actual_width_bp','eligible_fragment_design_count','source_replicon_length_bp',
                            'start1','end1','cohort_weight','fit_weight','score','old_full_score'])
    if set(features) & forbidden:
        raise RuntimeError('labels, scope bookkeeping, or model scores entered feature whitelist')
    x = d[features].to_numpy('float32')
    if np.isinf(x).any():
        raise RuntimeError('infinite model input; do not silently zero-fill')
    return x


def threshold_at_budget(scores, w, rate):
    if not len(scores) or not np.isfinite(scores).all() or not 0 <= rate <= 1:
        raise RuntimeError('invalid outside threshold inputs')
    order = np.argsort(-scores, kind='stable')
    s, weights_sorted = scores[order], w[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    cumulative = np.cumsum(weights_sorted)[ends]
    allowed = ends[cumulative <= rate*weights_sorted.sum()+1e-10]
    return float(s[allowed[-1]]) if len(allowed) else float(np.nextafter(s.max(), np.inf))


def monotone_audit(model, x, features, constraints):
    rng = np.random.default_rng(SEED)
    ix = rng.choice(len(x), min(256, len(x)), replace=False)
    sample = x[ix].copy()
    total = 0
    constrained = 0
    for j, sign in enumerate(constraints):
        if not sign or np.isnan(x[:,j]).all():
            continue
        finite = x[:,j][np.isfinite(x[:,j])]
        lo, hi = sample.copy(), sample.copy()
        lo[:,j], hi[:,j] = finite.min(), finite.max()
        delta = model.predict(hi,num_threads=1) - model.predict(lo,num_threads=1)
        total += int(np.sum(delta*sign < -1e-10))
        constrained += 1
    return {'individual_feature_violations': total, 'sample_rows': len(sample),
            'constrained_nonconstant_or_observed_features': constrained,
            'other_inputs_held_fixed': True,
            'not_a_physical_deletion_or_causal_transfer_experiment': True}


def grouped_delta_ci(g, pred, baseline, w, y_value, reps=2000):
    ix = g.y.to_numpy() == y_value
    table = pd.DataFrame({'group':g.connected_group.to_numpy()[ix],
                          'w':w[ix], 'delta':w[ix]*(pred[ix].astype(float)-baseline[ix])})
    a = table.groupby('group')[['w','delta']].sum().to_numpy(float)
    if not len(a):
        return [None, None]
    rng = np.random.default_rng(SEED + y_value)
    draws = []
    for start in range(0,reps,100):
        indices = rng.integers(0,len(a),size=(min(100,reps-start),len(a)))
        total = a[indices].sum(axis=1)
        draws.extend((total[:,1]/total[:,0]).tolist())
    return [float(v) for v in np.quantile(draws,[.025,.975])]


def one_metric(g, cutoff, w, label, reps):
    y = g.y.to_numpy(bool)
    p = g.score.to_numpy() >= cutoff
    b = g.baseline_call.to_numpy(bool)
    mean = lambda z, ix: float(np.average(z[ix],weights=w[ix])) if np.any(ix) else None
    two = len(np.unique(y)) == 2
    ci = grouped_delta_ci(g,p,b,w,1,reps)
    outside_ci = grouped_delta_ci(g,p,b,w,0,reps)
    return {'scope':label, 'windows':len(g), 'unique_cargo':int(g.host_protein_id.nunique()),
            'inside_windows':int(y.sum()), 'inside_cargo':int(g.loc[y,'host_protein_id'].nunique()),
            'outside_windows':int((~y).sum()), 'outside_cargo':int(g.loc[~y,'host_protein_id'].nunique()),
            'connected_groups':int(g.connected_group.nunique()),
            'model_inside_recall':mean(p,y), 'baseline_inside_recall':mean(b,y),
            'model_outside_alert':mean(p,~y), 'baseline_outside_alert':mean(b,~y),
            'model_reference_precision':mean(y,p),
            'recall_delta_ci_low':ci[0], 'recall_delta_ci_high':ci[1],
            'outside_alert_delta_ci_low':outside_ci[0], 'outside_alert_delta_ci_high':outside_ci[1],
            'auroc':float(roc_auc_score(y,g.score,sample_weight=w)) if two else None,
            'auprc':float(average_precision_score(y,g.score,sample_weight=w)) if two else None,
            'reference_inside_prevalence':float(np.average(y,weights=w)),
            'threshold':cutoff, 'fixed_threshold_across_all_scopes':True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--input-dir', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--threads', type=int, default=64)
    args = ap.parse_args()
    inp = args.input_dir or args.root/'unified_visible_dev_20261102'
    out = args.out or args.root/'unified_visible_model_20261102'
    out.mkdir(exist_ok=True)
    held_lock = (out/'RUN.lock').open('a+')
    try:
        fcntl.flock(held_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('unified training already running; retain active process',flush=True)
        return
    if not (inp/'FEATURES_COMPLETE').exists():
        raise RuntimeError('unified visible features are not complete')
    featurefile = inp/'DEVELOPMENT_FEATURES.parquet'
    features = json.loads((inp/'feature_columns.json').read_text())
    raw_constraints = json.loads((inp/'CONSTRAINTS.json').read_text())
    if isinstance(raw_constraints,dict):
        if set(raw_constraints) != set(features):
            raise RuntimeError('constraint feature whitelist differs')
        constraints = [int(raw_constraints[f]) for f in features]
    else:
        constraints = list(raw_constraints)
    if len(constraints) != len(features) or not set(constraints) <= {-1,0,1}:
        raise RuntimeError('invalid constraints')
    protocol = {'state':'one_unified_model_development_only', 'created_date':'2026-11-02',
                'task':'reference ICE/IME association conditional on intact frozen V3HN ARG/MRG',
                'features':features, 'constraints':constraints, 'params':PARAMS, 'rounds':ROUNDS,
                'base_seed':SEED, 'lengths_kb':[0,*LENGTHS], 'placements':[.1,.5,.9],
                'full_replicon_code':0, 'same_feature_schema_for_all_scopes':True,
                'same_final_model_for_all_scopes':True, 'length_specific_fits_forbidden':True,
                'threshold':'one pooled development OOF outside budget matched to fixed5kb',
                'scope_mass':'full .5 per cargo; fragment .5 split equally across retained eligible lengths and placements',
                'outside_fragment_probability':1/16, 'inverse_probability_weight':16,
                'no_eligible_fragment_policy':'full .5 only; do not infer eligibility from sampling absence',
                'group_fit_weight':'divide scope/IPW weight by sqrt(original group cargo count)',
                'unverified_outside_fit_multiplier':.5,
                'fit_weight_mean_normalization':'training fold only for OOF; entire development cohort for final fit',
                'OOF_folds':'unchanged original five host/element connected-group folds',
                'model_and_threshold_selection_use_external_or_prior_fresh_test':False,
                'external_already_inspected_can_only_be_descriptive':True,
                'all_bootstrap_groups':'original connected groups; all cargo replicas stay together',
                'bootstrap_reps':2000, 'bootstrap_CI_conditions_on_selected_cutoff':True,
                'input_parquet_sha256':sha(featurefile),
                'input_feature_manifest_sha256':sha(inp/'feature_columns.json'),
                'input_constraint_manifest_sha256':sha(inp/'CONSTRAINTS.json'),
                'training_script_sha256':sha(Path(__file__))}
    protocolpath = out/'TRAIN_PROTOCOL.json'
    if protocolpath.exists():
        if json.loads(protocolpath.read_text()) != protocol:
            raise RuntimeError('immutable training protocol/input changed; use a separate output directory')
    else:
        dump(protocolpath,protocol)
    protocol_sha = sha(protocolpath)
    if (out/'MODEL_COMPLETE').exists():
        print((out/'DEV_AUDIT.json').read_text(),flush=True)
        return
    d = pd.read_parquet(featurefile)
    x = validate(d,features,args.root)
    scope,evaluation,fit,weight_audit = weights(d)
    dump(out/'WEIGHT_AUDIT.json',weight_audit)
    metadata = d[[c for c in d if c not in features and c not in ['cohort_weight','fit_weight','old_full_score']]].copy()
    metadata['cohort_weight'] = scope
    metadata['per_length_eval_weight'] = evaluation
    metadata['fit_weight'] = fit
    y = d.y.to_numpy('int8')
    folds = d.fold.to_numpy('int8')
    scores = np.full(len(d),np.nan)
    checks = []
    print(json.dumps({'stage':'fit_ready','rows':len(d),'features':len(features),
                      'cargo':int(d.host_protein_id.nunique()),'threads':args.threads}),flush=True)
    for fold in range(5):
        started = time.time()
        val = folds == fold
        train = ~val
        modelpath,scorepath = out/f'fold{fold}.txt',out/f'fold{fold}.npy'
        checkpoint = out/f'fold{fold}_COMPLETE.json'
        expected = {'fold':fold,'protocol_sha256':protocol_sha,
                    'validation_rows':int(val.sum()),
                    'validation_key_digest':frame_digest(d.loc[val],KEYS)}
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if any(saved.get(k) != v for k,v in expected.items()):
                raise RuntimeError(f'fold{fold} resume provenance differs')
            if sha(modelpath) != saved['model_sha256'] or sha(scorepath) != saved['scores_sha256']:
                raise RuntimeError(f'fold{fold} checkpoint contents changed')
            model = lgb.Booster(model_file=str(modelpath))
            p = np.load(scorepath)
            check = saved['monotone_audit']
        else:
            train_weight = fit[train] / fit[train].mean()
            model = lgb.train({**PARAMS,'num_threads':args.threads,'seed':SEED+fold,
                               'monotone_constraints':constraints},
                              lgb.Dataset(x[train],label=y[train],weight=train_weight,feature_name=features),
                              num_boost_round=ROUNDS)
            p = model.predict(x[val],num_threads=args.threads)
            check = monotone_audit(model,x[val],features,constraints)
            if check['individual_feature_violations']:
                raise RuntimeError(f'fold{fold} violates declared feature constraints')
            atomic_model(model,modelpath)
            atomic_npy(p,scorepath)
            dump(checkpoint,{**expected,'model_sha256':sha(modelpath),'scores_sha256':sha(scorepath),
                             'monotone_audit':check})
        if model.feature_name() != features or len(p) != val.sum() or not np.isfinite(p).all():
            raise RuntimeError(f'fold{fold} prediction/feature contract differs')
        scores[val] = p
        checks.append(check)
        print(json.dumps({'stage':'OOF','fold':fold,'seconds':round(time.time()-started,2),
                          'validation_rows':int(val.sum()),**check}),flush=True)
    if not np.isfinite(scores).all():
        raise RuntimeError('incomplete OOF')
    metadata['score'] = scores
    outside = y == 0
    baseline = d.baseline_call.to_numpy(bool)
    budget = float(np.average(baseline[outside],weights=scope[outside]))
    cutoff = threshold_at_budget(scores[outside],scope[outside],budget)
    metadata['model_call'] = scores >= cutoff
    atomic_parquet(metadata,out/'DEV_OOF_PREDICTIONS.parquet')
    metrics = [one_metric(metadata,cutoff,scope,'POOLED_STANDARD_MIX',2000)]
    for length,g in metadata.groupby('length_kb',sort=True):
        metrics.append(one_metric(g,cutoff,g.per_length_eval_weight.to_numpy(),
                                  'FULL_REPLICON' if length == 0 else f'{int(length)}kb',2000))
    pd.DataFrame(metrics).to_csv(out/'DEV_BY_SCOPE.tsv',sep='\t',index=False)
    strata = []
    for length,g in metadata.groupby('length_kb',sort=True):
        for grouping in ['cargo_class','inside_element_type']:
            for value,h in g.groupby(grouping,dropna=False):
                if grouping == 'inside_element_type' and value not in ['ICE','IME']:
                    continue
                if len(h):
                    row = one_metric(h,cutoff,h.per_length_eval_weight.to_numpy(),
                                     'FULL_REPLICON' if length == 0 else f'{int(length)}kb',500)
                    row.update(grouping=grouping,value=str(value))
                    strata.append(row)
    pd.DataFrame(strata).to_csv(out/'DEV_STRATA.tsv',sep='\t',index=False)
    finalpath = out/'unified_full_development_model.txt'
    final_checkpoint = out/'FINAL_MODEL_COMPLETE.json'
    if final_checkpoint.exists():
        saved = json.loads(final_checkpoint.read_text())
        if saved['protocol_sha256'] != protocol_sha or saved['model_sha256'] != sha(finalpath):
            raise RuntimeError('final model resume provenance differs')
        final = lgb.Booster(model_file=str(finalpath))
        finalcheck = saved['monotone_audit']
    else:
        final = lgb.train({**PARAMS,'num_threads':args.threads,'seed':SEED,
                           'monotone_constraints':constraints},
                          lgb.Dataset(x,label=y,weight=fit/fit.mean(),feature_name=features),num_boost_round=ROUNDS)
        finalcheck = monotone_audit(final,x,features,constraints)
        if finalcheck['individual_feature_violations']:
            raise RuntimeError('final model violates constraints')
        atomic_model(final,finalpath)
        dump(final_checkpoint,{'protocol_sha256':protocol_sha,'model_sha256':sha(finalpath),
                               'monotone_audit':finalcheck})
    pd.DataFrame({'feature':features,'gain':final.feature_importance(importance_type='gain'),
                  'split':final.feature_importance(importance_type='split'),
                  'monotone_direction':constraints}).to_csv(out/'FEATURE_IMPORTANCE.tsv',sep='\t',index=False)
    lock = {'state':'unified_development_candidate_locked', 'model_file':finalpath.name,
            'model_sha256':sha(finalpath),'threshold':cutoff,'features':features,
            'constraints':constraints, 'one_model_and_one_cutoff_across_all_lengths':True,
            'cutoff_source':'pooled development OOF weighted outside5kb budget',
            'pooled_5kb_outside_budget':budget,
            'pooled_model_outside_rate':float(np.average((scores[outside]>=cutoff),weights=scope[outside])),
            'production_models_replaced':False,
            'new_untouched_independent_test_evaluated':False,
            'previous_fresh_test_and_external_not_read':True,
            'frozen_host_V3HN_annotations_reused':True,
            'not_end_to_end_raw_fragment_annotation':True,
            'outside_is_not_biological_negative':True}
    dump(out/'CANDIDATE_LOCK.json',lock)
    audit = {**lock,'rows':len(d),'full_cargo':int(d.length_kb.eq(0).sum()),
             'feature_count':len(features), 'feature_missing_values':int(np.isnan(x).sum()),
             'OOF_fold_audits':checks,'final_audit':finalcheck, 'weight_audit':weight_audit,
             'same_feature_schema_all_lengths':True,'scope_metrics':metrics}
    dump(out/'DEV_AUDIT.json',audit)
    (out/'MODEL_COMPLETE').touch()
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print(json.dumps({'stage':'complete','rows':len(d),'features':len(features),'threshold':cutoff}),flush=True)


if __name__ == '__main__':
    main()
