from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

LENGTHS = (2, 5, 10, 20, 50, 100)
PLACEMENTS = (.1, .5, .9)
WIDTHS = (1000, 2000, 5000, 10000, 20000, 50000, 100000)
LAMBDAS = (1000, 2000, 5000, 10000, 20000, 50000)
BASE = ['p_ARG', 'p_MRG', 'p_MGE_self', 'gate_score', 'cargo_length_bp',
        'visible_length_bp', 'visible_left_flank_bp', 'visible_right_flank_bp']
PARAMS = dict(objective='binary', learning_rate=.05, num_leaves=15,
              max_depth=5, min_data_in_leaf=40, feature_fraction=1.,
              bagging_fraction=.8, bagging_freq=1, lambda_l2=5., verbosity=-1,
              monotone_constraints_method='basic', deterministic=True,
              force_col_wise=True, device_type='cpu')

def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def spatial(gaps, scores):
    keep = gaps <= 100000
    g, p = gaps[keep], scores[keep]
    j = int(np.argmin(g)) if len(g) else 0
    d = float(g[j]) if len(g) else 100001.
    result = {'nearest_mge_gap_bp': d, 'near_has_mge': float(len(g) > 0),
              'nearest_mge_score': float(p[j]) if len(g) else 0.,
              'near_within_5kb': float(bool(len(g)) and d <= 5000)}
    for w in WIDTHS:
        ix = g <= w
        result[f'near_count_{w}'] = float(ix.sum())
        result[f'near_score_sum_{w}'] = float(p[ix].sum())
    for lam in LAMBDAS:
        result[f'near_exp_sum_{lam}'] = float((p * np.exp(-g/lam)).sum())
        result[f'nearest_proximity_{lam}'] = float(np.exp(-d/lam)) if len(g) else 0.
        result[f'nearest_weighted_proximity_{lam}'] = result[f'nearest_proximity_{lam}'] * result['nearest_mge_score']
    return result

SPATIAL_NAMES = list(spatial(np.array([], float), np.array([], float)))
FEATURES = BASE + SPATIAL_NAMES
CONSTRAINTS = [0] * len(BASE) + [-1 if c == 'nearest_mge_gap_bp' else 1 for c in SPATIAL_NAMES]

def sampled(pid):
    return int.from_bytes(hashlib.sha256(pid.encode()).digest()[:8], 'big') % 16 == 0

def build(root, out):
    cache = out/'DEVELOPMENT_FEATURES.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    data = pd.read_parquet(root/'strict_v3hn_spatial/CARGO_EVIDENCE_FULL.parquet',
                           filters=[('partition', '==', 'development')])
    data = data.loc[data.reference_position.isin(['inside','outside'])].copy()
    old = pd.read_parquet(root/'strict_reference_localization_dev/DEV_OOF_PREDICTIONS.parquet')
    old = old.loc[old.variant.eq('all_evidence'), ['host_protein_id','connected_group','fold','score']]
    data = data.merge(old, on='host_protein_id', how='left', validate='one_to_one')
    assert data.fold.notna().all() and len(data) == 206587
    hosts = set(data.accession)
    mge = pd.concat([pd.read_parquet(root/f'fragment_visible_test/batch{i:04d}/MGE.parquet',
                                    filters=[('accession','in',list(hosts))])
                     for i in range(736)], ignore_index=True)
    by_acc = {a:(g.start1.to_numpy(), g.end1.to_numpy(), g.p_MGE.to_numpy())
              for a,g in mge.groupby('accession',sort=False)}
    empty = (np.array([],float),)*3
    records = []
    for n,c in enumerate(data.itertuples(index=False)):
        ms, me, mp = by_acc.get(c.accession,empty)
        hostlen = int(c.b_contig_len_bp)
        windows = [(0, .5, 1, hostlen)]
        augment = c.reference_position == 'inside' or sampled(c.host_protein_id)
        if augment:
            mid = (int(c.start1)+int(c.end1))/2
            for kb in LENGTHS:
                width = min(kb*1000,hostlen)
                for pos in PLACEMENTS:
                    start = max(1,min(int(round(mid-pos*width)),hostlen-width+1))
                    windows.append((kb,pos,start,start+width-1))

        for kb,pos,start,end in windows:
            if not (start <= c.start1 and c.end1 <= end):
                continue
            visible = (ms >= start) & (me <= end)
            gaps = np.maximum.reduce((ms[visible]-int(c.end1)-1,
                                      int(c.start1)-me[visible]-1,
                                      np.zeros(int(visible.sum()))))
            s = spatial(gaps,mp[visible])
            weight = .5 if kb == 0 else .5/(len(LENGTHS)*len(PLACEMENTS))*(1 if c.reference_position=='inside' else 16)
            row = {k:float(getattr(c,k)) for k in BASE[:4]}
            row.update(cargo_length_bp=int(c.end1-c.start1+1),visible_length_bp=end-start+1,
                       visible_left_flank_bp=int(c.start1-start),visible_right_flank_bp=int(end-c.end1))
            row.update(s)
            row.update(host_protein_id=c.host_protein_id,accession=c.accession,
                       connected_group=c.connected_group,fold=int(c.fold),length_kb=kb,
                       placement=pos,y=int(c.reference_position=='inside'),cargo_class=c.cargo_class,
                       inside_element_type=c.inside_element_type,cohort_weight=weight,
                       baseline_call=bool(s['near_within_5kb']),old_full_score=float(c.score) if kb==0 else np.nan)
            records.append(row)
        if n%20000==0:
            print(json.dumps({'stage':'features','cargo':n,'rows':len(records)}),flush=True)
    result = pd.DataFrame.from_records(records)
    counts = data.groupby('connected_group').size()
    result['fit_weight'] = (result.cohort_weight / np.sqrt(result.connected_group.map(counts))).astype('float32')
    result.loc[result.y.eq(0),'fit_weight'] *= .5
    assert not result[FEATURES].isna().any().any()
    assert result.groupby('connected_group').fold.nunique().max() == 1
    result.to_parquet(cache.with_suffix('.parquet.part'),index=False,compression='zstd')
    os.replace(cache.with_suffix('.parquet.part'),cache)
    return result

def audit(model, x):
    sample = x[:min(512,len(x))].copy()
    violations = 0
    for col,sign in enumerate(CONSTRAINTS):
        if not sign: continue
        lo,hi = sample.copy(),sample.copy()
        lo[:,col],hi[:,col] = np.nanmin(x[:,col]),np.nanmax(x[:,col])
        delta = model.predict(hi,num_threads=1)-model.predict(lo,num_threads=1)
        violations += int(np.sum(delta*sign < -1e-10))

    previous = None; coherent = 0
    for distance in (0,500,1000,2000,5000,10000,20000,50000,100000,110000):
        s = spatial(np.array([distance,distance+1000.]),np.array([.95,.8]))
        z = sample.copy()
        for key,value in s.items():z[:,FEATURES.index(key)] = value
        p = model.predict(z,num_threads=1)
        if previous is not None: coherent += int(np.sum(p > previous+1e-10))
        previous = p
    return {'individual_feature_violations':violations,'coherent_distance_profile_violations':coherent,
            'sample_rows':len(sample),'non_distance_features_held_fixed':True}

def threshold(scores, weights, rate):
    order = np.argsort(-scores,kind='stable'); s,w = scores[order],weights[order]

    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]),len(s)-1]
    cum = np.cumsum(w)[ends]; allowed = ends[cum <= rate*w.sum()+1e-10]
    return float(s[allowed[-1]]) if len(allowed) else float(np.nextafter(s.max(),np.inf))

def summarize(d, cut, reps=1000):
    rows=[]
    rng=np.random.default_rng(20261101)
    for length,g in d.groupby('length_kb',sort=True):
        y=g.y.to_numpy(bool); p=g.score.to_numpy()>=cut; b=g.baseline_call.to_numpy(bool); w=g.cohort_weight.to_numpy()
        mean=lambda z,ix:float(np.average(z[ix],weights=w[ix]))
        pos=g.loc[g.y.eq(1),['connected_group','cohort_weight']].copy()
        pos['a']=w[y]*p[y];pos['b']=w[y]*b[y]
        arr=pos.groupby('connected_group')[['cohort_weight','a','b']].sum().to_numpy()
        draws=rng.integers(0,len(arr),size=(reps,len(arr)));tot=arr[draws].sum(1)
        ci=np.quantile((tot[:,1]-tot[:,2])/tot[:,0],[.025,.975])
        rows.append(dict(length_kb=int(length),visible_windows=len(g),positive_visible_windows=int(y.sum()),
                         positive_connected_groups=len(arr),model_inside_recall=mean(p,y),baseline_inside_recall=mean(b,y),
                         model_outside_alert=mean(p,~y),baseline_outside_alert=mean(b,~y),
                         recall_delta_ci_low=float(ci[0]),recall_delta_ci_high=float(ci[1]),
                         auroc=float(roc_auc_score(y,g.score,sample_weight=w)),
                         auprc=float(average_precision_score(y,g.score,sample_weight=w))))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--threads',type=int,default=64)
    args=ap.parse_args();out=args.root/'fragment_near_monotone_dev_20261101';out.mkdir(exist_ok=True)
    if (out/'DEV_COMPLETE').exists():print((out/'audit.json').read_text());return
    protocol={'target':'Botelho reference ICE/IME localization conditional on intact frozen V3HN ARG/MRG',
              'development_only':True,'previous_fresh_test_not_used':True,'lengths_kb':LENGTHS,'placements':PLACEMENTS,
              'outside_fragment_sampling':'deterministic SHA-prefix 1/16; inverse probability correction',
              'threshold':'one global cutoff matched to fixed5kb weighted outside alerts in development 20kb OOF',
              'features':FEATURES,'constraints':dict(zip(FEATURES,CONSTRAINTS)),
              'params':PARAMS,'rounds':300,'seeds':[20261101+i for i in range(5)],
              'promotion_gate':'20kb recall delta CI low>0; 5/10kb recall>=5kb and outside<=5kb+0.01; full-host recall loss<=0.05 versus old model; independent validation required'}
    if (out/'PROTOCOL.json').exists():assert json.loads((out/'PROTOCOL.json').read_text())==json.loads(json.dumps(protocol))
    else:dump(out/'PROTOCOL.json',protocol)
    d=build(args.root,out); x=d[FEATURES].to_numpy('float32');y=d.y.to_numpy('int8');scores=np.full(len(d),np.nan)
    checks=[]
    for fold in range(5):
        train=d.fold.to_numpy()!=fold;val=~train;modelpath=out/f'fold{fold}.txt';scorepath=out/f'fold{fold}.npy'
        if modelpath.exists() and scorepath.exists():
            model=lgb.Booster(model_file=str(modelpath));p=np.load(scorepath);assert len(p)==val.sum()
        else:
            dataset=lgb.Dataset(x[train],label=y[train],weight=d.fit_weight.to_numpy()[train],feature_name=FEATURES)
            model=lgb.train({**PARAMS,'num_threads':args.threads,'seed':20261101+fold,'monotone_constraints':CONSTRAINTS},dataset,num_boost_round=300)
            p=model.predict(x[val],num_threads=args.threads);model.save_model(str(modelpath));np.save(scorepath,p)
        scores[val]=p;check=audit(model,x[val]);checks.append(check)
        print(json.dumps({'fold':fold,'validation_rows':int(val.sum()),**check}),flush=True)
    assert np.isfinite(scores).all() and not any(c['individual_feature_violations'] or c['coherent_distance_profile_violations'] for c in checks)
    d['score']=scores;g=d.loc[d.length_kb.eq(20)&d.y.eq(0)];rate=float(np.average(g.baseline_call,weights=g.cohort_weight))
    cut=threshold(g.score.to_numpy(),g.cohort_weight.to_numpy(),rate)
    metrics=summarize(d,cut);metrics.to_csv(out/'DEV_BY_LENGTH.tsv',sep='\t',index=False)
    d.drop(columns=FEATURES).to_parquet(out/'DEV_OOF_PREDICTIONS.parquet',index=False,compression='zstd')
    full=metrics.set_index('length_kb').loc[0];old=pd.read_csv(args.root/'strict_reference_localization_dev/DEV_OOF_METRICS.tsv',sep='\t').set_index('variant').loc['all_evidence']
    by=metrics.set_index('length_kb');passed=bool(by.loc[20,'recall_delta_ci_low']>0 and all(by.loc[k,'model_inside_recall']>=by.loc[k,'baseline_inside_recall'] and by.loc[k,'model_outside_alert']<=by.loc[k,'baseline_outside_alert']+.01 for k in (5,10)) and full.model_inside_recall>=old.model_inside_recall_matched_outside_rate-.05)
    final=lgb.train({**PARAMS,'num_threads':args.threads,'seed':20261101,'monotone_constraints':CONSTRAINTS},lgb.Dataset(x,label=y,weight=d.fit_weight.to_numpy(),feature_name=FEATURES),num_boost_round=300)
    final.save_model(str(out/'full_development_model.txt'));finalcheck=audit(final,x)
    assert not finalcheck['individual_feature_violations'] and not finalcheck['coherent_distance_profile_violations']
    lock={'threshold':cut,'features':FEATURES,'monotone_constraints':CONSTRAINTS,'development_promotion_gate_passed':passed,
          'new_independent_validation':'pending genuinely unused external cohort','prior_fresh_test_excluded':True,
          'stage1_v3hn_predictions_reused':True,'reference_outside_not_true_negative':True}
    dump(out/'CANDIDATE_LOCK.json',lock);dump(out/'audit.json',{'rows':len(d),'fold_audits':checks,'final_audit':finalcheck,**lock})
    (out/'DEV_COMPLETE').touch();print(metrics.to_string(index=False),flush=True);print(json.dumps(lock),flush=True)

if __name__=='__main__':main()
