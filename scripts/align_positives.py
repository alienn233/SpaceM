import argparse,csv,json,subprocess,hashlib
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--query',type=Path,required=True);p.add_argument('--predictions',type=Path);p.add_argument('--references',type=Path,required=True);p.add_argument('--diamond',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);refs=json.loads(a.references.read_text());records=[];audit=[]
fields=['protein_id','subject_raw','identity_pct','aligned_length','query_length','subject_length','query_start','query_end','subject_start','subject_end','evalue','bitscore','query_coverage_pct','subject_coverage_pct']
query=a.query
if a.predictions:
    with a.predictions.open() as f:
        selected={r['protein_id'] for r in csv.DictReader(f,delimiter='\t') if r['final_class'] in {'ARG','MRG','MGE'}}
    query=a.out/'PREDICTED_TARGETS.faa'
    with a.query.open() as src, query.open('w') as dst:
        keep=False
        for line in src:
            if line.startswith('>'):keep=line[1:].split()[0] in selected
            if keep:dst.write(line)
version=subprocess.check_output([a.diamond,'version'],text=True).strip()
for klass,entry in refs.items():
    faa=(a.references.parent/entry['fasta']).resolve();meta=(a.references.parent/entry['metadata']).resolve();db=a.out/klass;raw=a.out/(klass+'_hits.tsv')
    subprocess.run([a.diamond,'makedb','--in',str(faa),'--db',str(db),'--threads',str(a.threads)],check=True)
    subprocess.run([a.diamond,'blastp','--query',str(query),'--db',str(db),'--out',str(raw),'--evalue','1e-5','--sensitive','--max-target-seqs','5','--max-hsps','1','--threads',str(a.threads),'--outfmt','6','qseqid','sseqid','pident','length','qlen','slen','qstart','qend','sstart','send','evalue','bitscore','qcovhsp','scovhsp'],check=True)
    hits=list(csv.DictReader(raw.open(),fieldnames=fields,delimiter='\t'))
    ids={x['subject_raw'].split('|')[0] for x in hits};annotations={}
    with meta.open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            key=r.get('sequence_id') or r.get('record_id')
            if key in ids:
                annotations[key]={
                    'reference_name':r.get('original_gene_names') or r.get('gene_name') or r.get('protein_name') or r.get('aro_name') or key,
                    'reference_function':r.get('aro_name') or r.get('protein_name') or '',
                    'reference_labels':r.get('drug_class') or r.get('metal_normalized') or r.get('mobility_type') or r.get('mobility_tag') or '',
                    'reference_sources':r.get('source_databases') or r.get('source_database') or r.get('duplicate_sources') or r.get('source_db') or '',
                    'reference_original_ids':r.get('source_gene_ids') or r.get('source_accession') or r.get('protein_id') or '',
                    'reference_evidence':r.get('source_class') or r.get('label_evidence_type') or r.get('argnorm_status') or ''}
    for r in hits:
        r['reference_id']=r['subject_raw'].split('|')[0];r['reference_class']=klass
        if r['reference_id'] not in annotations:raise ValueError('Unresolved reference metadata: '+r['reference_id'])
        r.update(annotations[r['reference_id']])
        for k in fields[2:]:r[k]=float(r[k])
        r['strict_support']=r['identity_pct']>90 and r['query_coverage_pct']>90 and r['subject_coverage_pct']>90 and r['evalue']<1e-5
        records.append(r)
    digest=hashlib.sha256()
    with faa.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    audit.append({'class':klass,'fasta':str(faa),'metadata':str(meta),'fasta_sha256':digest.hexdigest(),'hits':len(hits)})
records.sort(key=lambda r:(r['protein_id'],-r['bitscore'],r['evalue'],r['reference_class'],r['reference_id']))
columns=[*fields,'reference_id','reference_class','reference_name','reference_function','reference_labels','reference_sources','reference_original_ids','reference_evidence','strict_support']
with (a.out/'POSITIVE_REFERENCE_HITS.tsv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=columns,delimiter='\t');writer.writeheader();writer.writerows(records)
(a.out/'ALIGNMENT_MANIFEST.json').write_text(json.dumps({'tool':version,'query':str(query),'query_scope':'predicted ARG/MRG/MGE only' if a.predictions else 'all supplied queries','references':audit,'retrieval':'up to five targets per reference class; sensitive; evalue <= 1e-5','strict_support':'identity >90%, query coverage >90%, subject coverage >90%, evalue <1e-5','purpose':'descriptive reference annotation; full positive libraries, not fold-safe benchmarking; predictions unchanged'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'state':'COMPLETE','hits':len(records),'query_ids_with_hits':len({r['protein_id'] for r in records})}),flush=True)
