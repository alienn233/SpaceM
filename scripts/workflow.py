import argparse,importlib.util,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd

def run(inputs,runtime,out,threads):
    spec=importlib.util.spec_from_file_location('spacem_runtime',runtime/'bin/spacem.py')
    rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
    raw=json.loads(inputs.read_text());inp={k:str((inputs.parent/v).resolve()) if k!='prediction_context' and isinstance(v,str) else v for k,v in raw.items()}
    rt.output_dir(out)
    h=rt.classify(inp,out,threads,fine=True)
    full=h.copy()
    for klass in ['ARG','MRG','MGE']:
        path=out/f'03_fine/{klass}_8M_FINE.tsv'
        if path.exists():
            f=pd.read_csv(path,sep='\t').rename(columns={'sequence_id':'protein_id'});f=f.rename(columns={c:'fine_'+c for c in f if c!='protein_id'})
            for c in f:
                if c!='protein_id' and c not in full:full[c]=None
            f=f.set_index('protein_id');mask=full.protein_id.isin(f.index)
            for c in f:full.loc[mask,c]=full.loc[mask,'protein_id'].map(f[c]).to_numpy()
    if 'mobility_base229' in inp:
        features=out/'FEATURES248.parquet'
        subprocess.run([sys.executable,str(runtime/'bin/build_unified_features.py'),'--base229',inp['mobility_base229'],'--local44',inp['mobility_local44'],'--out',str(features)],check=True)
    else:features=Path(inp['mobility_features'])
    f=rt.table(features);lookup=h.set_index('protein_id');pid='host_protein_id'
    if not set(f[pid])<=set(lookup.index):raise ValueError('Mobility IDs absent from hierarchy')
    aligned=lookup.loc[f[pid]]
    if not aligned.final_class.isin(['ARG','MRG']).all():raise ValueError('Only ARG/MRG can enter cargo scoring')
    for src,dst in [('p_ARG','p_ARG'),('p_MRG','p_MRG_self' if 'p_MRG_self' in f else 'p_MRG'),('p_MGE','p_MGE_self'),('gate_score','gate_score')]:
        if not np.allclose(aligned[src].to_numpy(),f[dst].to_numpy(),rtol=1e-5,atol=1e-7):raise ValueError('Upstream score mismatch: '+src)
    md=out/'04_mobility';md.mkdir()
    context=inp.get('prediction_context','deployment')
    m=rt.mobility(features,md,threads,context)
    joined=full.merge(m,left_on='protein_id',right_on='host_protein_id',how='left',validate='one_to_many',suffixes=('','_mobility'))
    joined['mobility_status']=np.where(joined.unified248_score.notna(),'SCORED',np.where(joined.final_class.isin(['ARG','MRG']),'CONTEXT_NOT_PROVIDED','NOT_APPLICABLE'))
    joined.to_csv(out/'ALL_STAGES.tsv',sep='\t',index=False)
    full.to_csv(out/'GENE_CLASSIFICATION.tsv',sep='\t',index=False)
    if 'positive_references' in inp:
        diamond=os.environ.get('SPACEM_DIAMOND','diamond')
        subprocess.run([sys.executable,str(Path(__file__).with_name('align_positives.py')),'--query',inp['protein_fasta'],'--predictions',str(out/'GENE_CLASSIFICATION.tsv'),'--references',inp['positive_references'],'--diamond',diamond,'--out',str(out/'05_alignment'),'--threads',str(threads)],check=True)
    subprocess.run([os.environ.get('SPACEM_PLOT_PYTHON',os.environ.get('SPACEM_FINE_PYTHON',sys.executable)),str(Path(__file__).with_name('report.py')),'--out',str(out),'--inputs',str(inputs)],check=True)
    manifest={'state':'COMPLETE','proteins':len(h),'class_counts':h.final_class.value_counts().to_dict(),'mobility_contexts':len(m),'fine_models_per_task':15,'prediction_context':context,'entry_point':'prepared_sequence_and_context_features','raw_annotation_recomputed':False}
    rt.dump(out/'RUN_MANIFEST.json',manifest)
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description='SpaceM: linked hierarchy, fine labels, mobile association and reports.');p.add_argument('--inputs',type=Path,required=True);p.add_argument('--runtime',type=Path,default=os.environ.get('SPACEM_RUNTIME'));p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.runtime is None:p.error('--runtime or SPACEM_RUNTIME is required')
    run(a.inputs.resolve(),a.runtime.resolve(),a.out.resolve(),a.threads)
