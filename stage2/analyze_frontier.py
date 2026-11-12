"""Validate the frozen pilot and summarize the predeclared disagreement detector."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from detection_metrics import auroc,bootstrap_auc
p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--manifest',default='stage2/frontier_manifest.json');a=p.parse_args();root=Path(a.results)
rows=[json.loads(s) for s in (root/'per_image.jsonl').read_text().splitlines()];expected={r['id']:r for r in json.loads(Path(a.manifest).read_text())}
assert len(rows)==100 and len({r['id'] for r in rows})==100 and {r['id'] for r in rows}==set(expected)
for r in rows:
 for key in ['sha256','label','file','revision']:assert r[key]==expected[r['id']][key]
 assert np.isfinite(r['score']) and r['score']==r['scores']['4']['mean_angle_deg']
 for s in ['1','4']:
  assert 0<r['scores'][s]['interior_coverage']<=1
  assert 0<=r['scores'][s]['mean_angle_deg']<=180
labels=np.array([r['label']=='ai' for r in rows],dtype=int);assert labels.sum()==50
summary={'n':100,'real':50,'ai':50,'primary_score':'mean angular disagreement, radius 4','direction':'higher = AI (not flipped)','auroc':auroc(labels,[r['score'] for r in rows]),'auroc_95pct_stratified_image_bootstrap':bootstrap_auc(labels,[r['score'] for r in rows]),'bootstrap_replicates':2000,'bootstrap_seed':180,'classes':{},'secondary':{},'failures':0}
for label in ['real','ai']:
 rr=[r for r in rows if r['label']==label];v=np.array([r['score'] for r in rr]);summary['classes'][label]={'n':len(rr),'mean_disagreement_deg':float(v.mean()),'median_disagreement_deg':float(np.median(v)),'q25_deg':float(np.percentile(v,25)),'q75_deg':float(np.percentile(v,75)),'mean_interior_coverage':float(np.mean([r['scores']['4']['interior_coverage'] for r in rr])),'mean_moge_mask_fraction':float(np.mean([r['predicted_mask_fraction'] for r in rr]))}
for stride,field in [('1','mean_angle_deg'),('4','median_angle_deg'),('4','mean_abs_facing_component_difference')]:
 summary['secondary'][f'radius{stride}_{field}']={'auroc':auroc(labels,[r['scores'][stride][field] for r in rows])}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fields=['id','label','generator','score_deg','radius1_score_deg','interior_coverage','moge_mask_fraction','sha256']
with (root/'scores.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
 for r in rows:w.writerow(dict(id=r['id'],label=r['label'],generator=r['generator'],score_deg=r['score'],radius1_score_deg=r['scores']['1']['mean_angle_deg'],interior_coverage=r['scores']['4']['interior_coverage'],moge_mask_fraction=r['predicted_mask_fraction'],sha256=r['sha256']))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
all_scores=np.array([r['score'] for r in rows]);bins=np.linspace(all_scores.min()-.5,all_scores.max()+.5,16)
for label,color in [('real','#2478b4'),('ai','#d56b27')]:
 values=[r['score'] for r in rows if r['label']==label];axes[0].hist(values,bins=bins,alpha=.55,label=label+' (n=50)',color=color)
axes[0].set(xlabel='Mean angular disagreement (degrees)',ylabel='Images',title='Primary score distribution');axes[0].legend()
thresholds=np.r_[np.inf,np.sort(np.unique(all_scores))[::-1],-np.inf];tpr=[np.mean(all_scores[labels==1]>=t) for t in thresholds];fpr=[np.mean(all_scores[labels==0]>=t) for t in thresholds]
axes[1].plot(fpr,tpr,color='#7b3294',label=f'AUROC {summary["auroc"]:.3f}');axes[1].plot([0,1],[0,1],'--',color='gray',label='Chance');axes[1].set(xlabel='Real images flagged (FPR)',ylabel='Generated images detected (TPR)',title='Higher disagreement = generated',xlim=(0,1),ylim=(0,1));axes[1].legend(loc='lower right')
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('MoGe-2 + Marigold · Frontier original-image pilot');fig.savefig(root/'detection.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2))
