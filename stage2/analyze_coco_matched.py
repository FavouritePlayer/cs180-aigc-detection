"""Analyze the frozen 50-group / 300-image caption-matched cosine pilot."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from detection_metrics import auroc
from matched_metrics import matched_summary
p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--manifest',default='stage2/coco_matched_manifest.json');a=p.parse_args();root=Path(a.results)
rows=[json.loads(s) for s in (root/'per_image.jsonl').read_text().splitlines()];expected={r['id']:r for r in json.loads(Path(a.manifest).read_text())};assert len(rows)==300 and len({r['id'] for r in rows})==300 and set(expected)=={r['id'] for r in rows}
generators=['sd21','sdxl','sd3','dalle3','midjourney'];groups={}
for r in rows:
 for key in ['file','label','generator','sha256','group_id','archive_sha256']:assert r[key]==expected[r['id']][key]
 groups.setdefault(r['group_id'],{})[r['generator']]=r
 assert r['score']==r['scores']['outdoor']['mean_cosine_residual']
 for v in ['outdoor','indoor']:
  q=r['scores'][v];assert all(np.isfinite(x) for x in q.values());assert 0<=q['mean_cosine_residual']<=2 and 0<q['interior_coverage']<=1
assert len(groups)==50 and all(set(g)==set(['real']+generators) for g in groups.values());keys=sorted(groups)
summary={'images':300,'real':50,'ai':250,'caption_groups':50,'ai_per_generator':50,'primary':'outdoor_large_forward1_mean_cosine_residual','direction':'higher residual = generated, fixed, not flipped','score_range':[0,2],'generator_order':generators,'variants':{},'failed_images':0}
matrices={}
for variant in ['outdoor','indoor']:
 real=np.array([groups[k]['real']['scores'][variant]['mean_cosine_residual'] for k in keys]);fake=np.array([[groups[k][g]['scores'][variant]['mean_cosine_residual'] for g in generators] for k in keys]);matrices[variant]=(real,fake)
 stat=matched_summary(real,fake);stat['per_generator']={g:{'auroc':stat['per_generator_auroc'][j],'auroc_ci95':stat['per_generator_auroc_ci95'][j],'mean_paired_difference':stat['per_generator_paired_mean_difference'][j],'fraction_above_matched_real':stat['per_generator_fraction_above_matched_real'][j]} for j,g in enumerate(generators)}
 stat['classes']={}
 for g in ['real','ai']+generators:
  rr=[r for r in rows if (r['label']=='ai' if g=='ai' else r['generator']==g)];vv=[r['scores'][variant] for r in rr]
  stat['classes'][g]={'n':len(rr),'mean_cosine_residual':float(np.mean([q['mean_cosine_residual'] for q in vv])),'median_image_cosine_residual':float(np.median([q['mean_cosine_residual'] for q in vv])),'mean_angle_deg':float(np.mean([q['mean_angle_deg'] for q in vv])),'mean_interior_coverage':float(np.mean([q['interior_coverage'] for q in vv]))}
 labels=[r['label']=='ai' for r in rows];stat['angular_auroc_secondary']=auroc(labels,[r['scores'][variant]['mean_angle_deg'] for r in rows]);summary['variants'][variant]=stat
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'scores.csv').open('w') as f:
 fields=['id','group_id','generator','label','outdoor_mean_cosine_residual','indoor_mean_cosine_residual','outdoor_mean_angle_deg','indoor_mean_angle_deg'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
 for r in rows:w.writerow({**{k:r[k] for k in ['id','group_id','generator','label']},**{v+'_'+metric:r['scores'][v][metric] for v in ['outdoor','indoor'] for metric in ['mean_cosine_residual','mean_angle_deg']}})
real,fake=matrices['outdoor'];difference=fake.mean(axis=1)-real
with (root/'group_scores.csv').open('w') as f:
 fields=['group_id','real']+generators+['mean_ai_minus_real'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
 for i,k in enumerate(keys):w.writerow({'group_id':k,'real':real[i],**{g:fake[i,j] for j,g in enumerate(generators)},'mean_ai_minus_real':difference[i]})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
stat=summary['variants']['outdoor'];fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');values=np.r_[real,fake.ravel()];bins=np.linspace(max(0,values.min()-.01),values.max()+.01,16)
for label,arr,color in [('real (n=50)',real,'#2478b4'),('generated (n=250)',fake.ravel(),'#d56b27')]:axes[0].hist(arr,bins=bins,weights=np.ones(len(arr))/len(arr),alpha=.55,label=label,color=color)
axes[0].set(xlabel='Mean cosine residual (0–2)',ylabel='Fraction of class',title='Primary score distributions');axes[0].legend()
thresholds=np.r_[np.inf,np.unique(values)[::-1],-np.inf];fpr=[np.mean(real>=t) for t in thresholds];tpr=[np.mean(fake>=t) for t in thresholds]
axes[1].plot(fpr,tpr,color='#7b3294',label=f'AUROC {stat["auroc"]:.3f}');axes[1].plot([0,1],[0,1],'--',color='gray',label='Chance');axes[1].set(xlabel='Real images flagged (FPR)',ylabel='Generated images detected (TPR)',title='Higher residual = generated',xlim=(0,1),ylim=(0,1));axes[1].legend(loc='lower right')
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('COCO indoor matched pilot · 50 caption groups · 300 images');fig.savefig(root/'detection.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');axes[0].scatter(real,fake.mean(axis=1),s=25,alpha=.8,color='#2878ad');limits=[min(real.min(),fake.mean(axis=1).min())-.02,max(real.max(),fake.mean(axis=1).max())+.02];axes[0].plot(limits,limits,'--',color='gray');axes[0].set(xlabel='Real image residual',ylabel='Mean residual of five generated counterparts',title='Each point is one caption group',xlim=limits,ylim=limits);axes[1].hist(difference,bins=14,color='#7b3294',alpha=.8);axes[1].axvline(0,color='gray',linestyle='--');axes[1].set(xlabel='Mean generated residual − matched real residual',ylabel='Caption groups',title='Paired differences: positive supports hypothesis')
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'paired_differences.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4),layout='constrained');point=np.array(stat['per_generator_auroc']);ci=np.array(stat['per_generator_auroc_ci95']);ax.hlines(range(5),ci[:,0],ci[:,1],color='#2878ad');ax.scatter(point,range(5),color='#2878ad');ax.axvline(.5,color='gray',linestyle='--');ax.set(yticks=range(5),yticklabels=['SD2.1','SDXL','SD3','DALL·E 3','Midjourney'],xlabel='AUROC (95% caption-group bootstrap interval)',xlim=(0,1),title='Each generator: 50 AI vs the same 50 real images');ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);fig.savefig(root/'per_generator.png',dpi=180);plt.close(fig)
print(json.dumps({'pooled_auroc':stat['auroc'],'ci95':stat['auroc_95pct_caption_cluster_bootstrap'],'paired_mean_difference':stat['paired_mean_difference'],'paired_ci95':stat['paired_difference_95pct_caption_cluster_bootstrap'],'per_generator':stat['per_generator'],'secondary_indoor_auroc':summary['variants']['indoor']['auroc']},indent=2))
