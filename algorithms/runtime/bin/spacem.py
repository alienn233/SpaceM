from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

if __name__=='__main__' and os.environ.get('PYTHONNOUSERSITE')!='1':
    env=os.environ.copy();env['PYTHONNOUSERSITE']='1'
    os.execve(sys.executable,[sys.executable,'-s',str(Path(__file__).resolve()),*sys.argv[1:]],env)

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.part');q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def config():
    d=read(ROOT/'config/runtime.json')
    def resolve(v):
        if isinstance(v,str):return v.replace('${RUNTIME}',str(ROOT))
        if isinstance(v,list):return [resolve(x) for x in v]
        if isinstance(v,dict):return {k:resolve(x) for k,x in v.items()}
        return v
    d=resolve(d);d['hierarchy']['python']=sys.executable
    d['fine_models']['python']=os.environ.get('SPACEM_FINE_PYTHON',sys.executable)
    return d
def table(p):
    import pandas as pd
    return pd.read_parquet(p) if str(p).endswith(('.parquet','.pq')) else pd.read_csv(p,sep='\t')
def output_dir(p):
    p=Path(p)
    if p.exists() and any(p.iterdir()):raise ValueError(f'Output folder must be empty: {p}')
    p.mkdir(parents=True,exist_ok=True);return p
def run(script,args,python=None):
    subprocess.run([python or sys.executable,str(ROOT/'bin/legacy'/script),*map(str,args)],check=True)
def resolved_config(out):
    path=out/'runtime.resolved.json';dump(path,config());return path

def mobility(features,out,threads=8,prediction_context='deployment'):
    import lightgbm as lgb
    import numpy as np
    d=table(features);md=ROOT/'models/unified248';lock=read(md/'CANDIDATE_LOCK.json')
    columns=read(md/'feature_columns.json');modelpath=md/'unified_full_development_model.txt'
    if len(columns)!=248 or len(set(columns))!=248:raise ValueError('Invalid feature whitelist')
    missing=set(columns)-set(d)
    if missing:raise ValueError(f'Missing UNIFIED248 features: {sorted(missing)}')
    if hashlib.sha256(modelpath.read_bytes()).hexdigest()!=lock['model_sha256']:raise ValueError('Model checksum mismatch')
    model=lgb.Booster(model_file=str(modelpath))
    if model.feature_name()!=columns:raise ValueError('Model feature names/order differ from lock')
    x=d[columns].to_numpy(np.float32)
    if len(d)==0 or np.isinf(x).any():raise ValueError('Empty input or infinite feature values')
    keys=[c for c in ['host_protein_id','length_kb','placement'] if c in d]
    if keys and d.duplicated(keys).any():raise ValueError(f'Duplicate prediction keys: {keys}')
    scores=model.predict(x,num_threads=threads)
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():raise ValueError('Invalid model scores')
    result=d[[c for c in d if c not in columns and 'sha256' not in c.lower()]].copy()
    result['unified248_score']=scores;result['unified248_threshold']=float(lock['threshold'])
    result['unified248_call']=scores>=float(lock['threshold'])
    result['prediction_context']=prediction_context
    result['score_semantics']='ICE_IME_reference_interval_association_not_transfer_probability'
    result.to_parquet(out/'MOBILITY_UNIFIED248.parquet',index=False,compression='zstd')
    result.to_csv(out/'MOBILITY_UNIFIED248.tsv.gz',sep='\t',index=False)
    info={'model':'UNIFIED248_20261102','rows':len(d),'features':248,'threshold':float(lock['threshold']),
          'model_sha256':lock['model_sha256'],'prediction_context':prediction_context,
          'model_refitted':False,'old_mobility_v3_called':False,
          'input':str(Path(features).resolve()),'columns_selected_by_locked_whitelist':True}
    dump(out/'MOBILITY_MANIFEST.json',info);return result

def classify(inp,out,threads,fine=False):
    c=resolved_config(out)
    required=['ids','protein_table','protein_fasta','esm8','esm650','portable_context','pfam','eggnog','mag']
    for k in required:
        if k not in inp or not Path(inp[k]).is_file():raise ValueError(f'Missing input {k}')
    args=['--config',c,'--outdir',out/'01_matrices']
    for key in ['ids','esm8','esm650','portable_context','pfam','eggnog','mag']:args.extend(['--'+key.replace('_','-'),inp[key]])
    run('assemble_fold_matrices.py',args)
    args=['--config',c,'--matrix-dir',out/'01_matrices','--outdir',out/'02_hierarchy','--threads',threads]
    for key in ['ids','protein_table','protein_fasta']:args.extend(['--'+key.replace('_','-'),inp[key]])
    run('score_hierarchy.py',args)
    if fine:
        run('run_fine_models.py',['--config',c,'--fasta-dir',out/'02_hierarchy/fasta','--outdir',out/'03_fine','--threads',threads])
    return table(out/'02_hierarchy/HIERARCHY_PREDICTIONS.tsv')

def main():
    ap=argparse.ArgumentParser(description='Portable frozen V3HN/V2/ESM2-8M and UNIFIED248 inference from feature inputs.');sub=ap.add_subparsers(dest='command',required=True)
    for name in ['hierarchy','pipeline']:
        p=sub.add_parser(name);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=8)
    p=sub.add_parser('mobility');p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=8);p.add_argument('--prediction-context',default='deployment',choices=['deployment','development_refit_not_independent_evaluation','external_previously_seen'])
    p=sub.add_parser('fine');p.add_argument('--fasta-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=8)
    p=sub.add_parser('check');p.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();out=output_dir(a.out)
    if a.command=='mobility':mobility(a.features,out,a.threads,a.prediction_context)
    elif a.command=='fine':run('run_fine_models.py',['--config',resolved_config(out),'--fasta-dir',a.fasta_dir,'--outdir',out/'fine','--threads',a.threads])
    elif a.command=='check':
        import lightgbm as lgb
        cfg=config();models=[]
        for family,model in [('v3_gate','gate.model.txt'),('v2_conditional','target_subclass.model.txt')]:
            for fold in range(5):
                p=ROOT/f'models/{family}/fold_checkpoints/fold{fold}/{model}'
                booster=lgb.Booster(model_file=str(p));assert booster.num_feature()==2564;models.append(str(p.relative_to(ROOT)))
        run('validate_fine_checkpoints.py',['--config',resolved_config(out),'--output',out/'FINE_CHECKPOINTS.json'],cfg['fine_models']['python'])
        dump(out/'CORE_CHECK.json',{'hierarchy_models_loaded':models,'fine_checkpoint_count':45})
    else:
        import numpy as np
        inp=read(a.inputs)
        inp={k:str((a.inputs.parent/v).resolve()) if isinstance(v,str) and not Path(v).is_absolute() else v for k,v in inp.items()}
        hierarchy=classify(inp,out,a.threads,fine=a.command=='pipeline')
        if a.command=='pipeline':
            if 'mobility_features' not in inp:raise ValueError('pipeline requires mobility_features (visible-context UNIFIED248 parquet)')
            f=table(inp['mobility_features']);pid='host_protein_id' if 'host_protein_id'in f else 'protein_id'
            h=hierarchy.set_index('protein_id')
            if not set(f[pid])<=set(h.index):raise ValueError('Mobility IDs missing from current hierarchy')
            aligned=h.loc[f[pid]]
            if not aligned.final_class.isin(['ARG','MRG']).all():raise ValueError('Mobility rows must route from ARG/MRG cargo')
            for src,dst in [('p_ARG','p_ARG'),('p_MRG','p_MRG'),('p_MGE','p_MGE_self'),('gate_score','gate_score')]:
                if not np.allclose(aligned[src].to_numpy(),f[dst].to_numpy(),rtol=1e-5,atol=1e-7):raise ValueError(f'Mobility feature annotation mismatch: {src}')
            mo=out/'04_unified248';mo.mkdir();mobility(inp['mobility_features'],mo,a.threads)
        dump(out/'PIPELINE_MANIFEST.json',{'state':'COMPLETE','mode':a.command,'mobility_version':'UNIFIED248' if a.command=='pipeline' else None,'raw_feature_generation_included_in_this_command':False})
if __name__=='__main__':main()
