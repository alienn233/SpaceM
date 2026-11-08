import argparse,json,html,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow,Patch,FancyBboxPatch

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args()
out=a.out;figdir=out/'figures';figdir.mkdir(exist_ok=True)
cfg=json.loads(a.inputs.read_text());base=a.inputs.parent
if 'archived_mobility_oof' in cfg:shutil.copy2(base/cfg['archived_mobility_oof'],out/'ARCHIVED_MOBILITY_OOF.tsv')
def input_table(k):return pd.read_csv(base/cfg[k],sep='\t')
h=pd.read_csv(out/'GENE_CLASSIFICATION.tsv',sep='\t')
m=pd.read_parquet(out/'04_mobility/MOBILITY_UNIFIED248.parquet')
m=m[m.length_kb.eq(0)].copy()
colors={'ARG':'#CB4662','MRG':'#8664AD','MGE':'#26A18D','OTHER':'#8996A5','UNASSESSED':'#DBE0E6'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(figdir/(name+'.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def label(pid):return str(pid).replace('NZ_','')
fig,ax=plt.subplots(figsize=(13,4));ax.set(xlim=(0,13),ylim=(0,4));ax.axis('off')
nodes=[(.2,2,'Protein + genomic context',f'{len(h)} selected proteins'),(3.4,2,'V3HN binary gate',f'{sum(h.final_class!="OTHER")} targets / {sum(h.final_class=="OTHER")} OTHER'),(6.6,2,'V2 conditional classes','ARG / MRG / MGE'),(9.8,2,'ESM2-8M fine functions','ARG15 / MRG23 / MGE7'),(6.6,.2,'Genomic context','Full available host context'),(9.8,.2,'UNIFIED248 association',f'{len(m)} resistance loci')]
for x,y,title,sub in nodes:
    ax.add_patch(FancyBboxPatch((x,y),2.8,1.25,boxstyle='round,pad=0.10',fc='#EEF4F8',ec='#6B8EA8',lw=1))
    ax.text(x+1.4,y+.83,title,ha='center',fontsize=10,fontweight='bold');ax.text(x+1.4,y+.35,sub,ha='center',fontsize=9)
for x1,y1,x2,y2 in [(3,2.6,3.3,2.6),(6.2,2.6,6.5,2.6),(9.4,2.6,9.7,2.6),(8,1.9,8,1.5),(9.4,.8,9.7,.8)]:ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','lw':1.3,'color':'#526A80'})
ax.text(7.8,1.64,'ARG / MRG cargo',ha='right',fontsize=8)
save(fig,'00_workflow')
fig,axs=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.5,1]})
y=np.arange(len(h));axs[0].scatter(h.gate_score,y,c=[colors[x] for x in h.final_class],s=65,zorder=3)
axs[0].axvline(h.gate_threshold.iloc[0],ls='--',color='#536477',lw=1)
axs[0].set(yticks=y,yticklabels=[label(x) for x in h.protein_id],xlabel='Target probability',xlim=(-.03,1.04),title='A  Target versus background')
axs[0].invert_yaxis();axs[0].grid(axis='x',alpha=.15)
target=h[h.final_class!='OTHER'].copy()
im=axs[1].imshow(target[['conditional_p_ARG','conditional_p_MRG','conditional_p_MGE']],vmin=0,vmax=1,cmap='Blues',aspect='auto')
axs[1].set(xticks=range(3),xticklabels=['ARG','MRG','MGE'],yticks=range(len(target)),yticklabels=[label(x) for x in target.protein_id],title='B  Conditional three-class prediction')
for i,row in enumerate(target.itertuples()):
    for j,v in enumerate([row.conditional_p_ARG,row.conditional_p_MRG,row.conditional_p_MGE]):axs[1].text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>.65 else '#28374A',fontsize=8)
fig.colorbar(im,ax=axs[1],label='Conditional probability',shrink=.65);fig.tight_layout();save(fig,'01_hierarchy')
fig,axs=plt.subplots(3,1,figsize=(12,8),layout='constrained')
for ax,klass in zip(axs,['ARG','MRG','MGE']):
    f=pd.read_csv(out/f'03_fine/{klass}_8M_FINE.tsv',sep='\t')
    parsed=[{k:float(v) for k,v in (part.rsplit('=',1) for part in s.split(';'))} for s in f.probabilities]
    mat=pd.DataFrame(parsed);keep=mat.max().sort_values(ascending=False).head(9).index;mat=mat[keep]
    ax.imshow(mat,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
    ax.set(xticks=range(len(keep)),xticklabels=keep,yticks=range(len(f)),yticklabels=[label(x) for x in f.sequence_id],title=f'{klass}: '+('multilabel resistance profile' if klass!='MGE' else 'single-label function'))
    ax.tick_params(axis='x',rotation=20,labelsize=8)
    for i in range(len(mat)):
        predicted=str(f.iloc[i].predicted_labels).split(';')
        for j,k in enumerate(keep):
            v=mat.iloc[i,j];ax.text(j,i,f'{v:.2f}'+('*' if k in predicted else ''),ha='center',va='center',fontsize=8,color='white' if v>.65 else '#243449')
save(fig,'02_fine_functions')
fig,ax=plt.subplots(figsize=(9,4.2));cargo=list(dict.fromkeys(m.host_protein_id))
for j,(length,marker,col) in enumerate([(0,'o','#2677B3')]):
    part=m[m.length_kb==length].set_index('host_protein_id').loc[cargo]
    ax.scatter(np.arange(len(cargo)),part.unified248_score,s=75,marker=marker,color=col,label='Full genomic context')
ax.axhline(m.unified248_threshold.iloc[0],ls='--',color='#596775',lw=1,label=f'Model decision threshold ({m.unified248_threshold.iloc[0]:.6f})')
ax.set(xticks=range(len(cargo)),xticklabels=[label(x) for x in cargo],ylabel='Mobility association score',ylim=(-.03,1.07),title='Mobile genetic association in full genomic context')
ax.tick_params(axis='x',rotation=12);ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.24));fig.tight_layout();save(fig,'03_mobility_context')
case_text=[]
if 'case_table' in cfg and 'case_genes' in cfg:
    cases=input_table('case_table');genes=input_table('case_genes')
    fig,axs=plt.subplots(len(cases),1,figsize=(14,2.5*len(cases)),layout='constrained')
    for ax,c in zip(np.atleast_1d(axs),cases.itertuples()):
        g=genes[genes.case_id==c.case_id].sort_values('start1').reset_index(drop=True)
        ix=int(g.index[g.gene_id==c.cargo_gene_id][0]);g=g.iloc[max(0,ix-10):ix+11].copy()
        centre=(c.cargo_start1+c.cargo_end1)/2
        lo=(g.start1.min()-centre)/1000;hi=(g.end1.max()-centre)/1000
        ax.axvspan(max(lo,(c.reference_start1-centre)/1000),min(hi,(c.reference_end1-centre)/1000),color='#26A18D',alpha=.08)
        ax.axvspan(-5,5,color='#8D98A4',alpha=.10)
        for row in g.itertuples():
            start=(row.start1-centre)/1000;end=(row.end1-centre)/1000;is_target=row.gene_id==c.cargo_gene_id
            width=.24 if is_target else .12;direction=1 if row.strand=='+' else -1
            color=colors.get(row.v3hn_class,colors['UNASSESSED']);length=end-start
            ax.add_patch(FancyArrow(start if direction==1 else end,0,direction*length,0,width=width,head_width=width*1.65,head_length=min(.22,length*.4),length_includes_head=True,facecolor=color,edgecolor='#526273',lw=.45))
            if is_target:ax.text((start+end)/2,.3,label(row.gene_id),ha='center',fontsize=9,fontweight='bold')
        f=h.set_index('protein_id').loc[c.cargo_gene_id]
        scores=m[m.host_protein_id==c.cargo_gene_id].set_index('length_kb').unified248_score
        ax.set(xlim=(lo-.8,hi+.8),ylim=(-.6,.65),yticks=[],xlabel='Distance from cargo centre (kb)',title=f'{c.case_id} | {c.reference_element_type} reference | nearest predicted MGE {c.nearest_mge_gap_bp/1000:.3f} kb away')
        ax.text(.99,.90,f'Gate {f.gate_score:.3f} | {f.final_class}\nMobility association {scores.loc[0]:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.spines['left'].set_visible(False)
        case_text.append(f'{c.case_id} ({c.cargo_gene_id}): target probability {f.gate_score:.4f}; class {f.final_class}; functional labels {f.fine_predicted_labels}. Full-context mobility score {scores.loc[0]:.4f}; nearest predicted MGE {c.nearest_mge_gap_bp/1000:.3f} kb away.')
    fig.legend(handles=[Patch(facecolor=v,label=k) for k,v in colors.items()],loc='outside lower center',ncol=5,frameon=False)
    save(fig,'04_genomic_neighborhoods')
counts=h.final_class.value_counts().to_dict()
joined=pd.read_csv(out/'ALL_STAGES.tsv',sep='\t')
joined=joined[joined.length_kb.eq(0)|joined.length_kb.isna()].copy()
summary_cols=['protein_id','sample_id','start','end','strand','gate_score','binary_prediction','final_class','conditional_p_ARG','conditional_p_MRG','conditional_p_MGE','fine_predicted_labels','fine_top1_label','fine_top1_probability','length_kb','placement','unified248_score','unified248_threshold','unified248_call','mobility_status']
joined[[c for c in summary_cols if c in joined]].to_csv(out/'EXAMPLE_SUMMARY.tsv',sep='\t',index=False)
text=f'SpaceM workflow example\n\n{len(h)} proteins were processed through target recognition, conditional classification and functional profiling. Class counts: {counts}.\n\n'+'\n\n'.join(case_text)
(out/'RESULTS.txt').write_text(text)
images=''.join(f'<h2>{html.escape(f.stem)}</h2><img src="figures/{f.name}" style="width:100%;max-width:1300px">' for f in sorted(figdir.glob('*.png')))
(out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>SpaceM example report</title><style>body{max-width:1400px;margin:40px auto;font:16px/1.7 sans-serif;color:#243449;padding:20px}pre{white-space:pre-wrap}h1{color:#2677B3}</style><h1>SpaceM · continuous workflow</h1><pre>'+html.escape(text)+'</pre>'+images)
subprocess.run([sys.executable,str(Path(__file__).with_name('interactive_report.py')),'--inputs',str(a.inputs),'--out',str(out)],check=True)
