"""Summarize the fixed outdoor-primary cosine follow-up and indoor sensitivity check."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from detection_metrics import auroc,bootstrap_auc
p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--manifest',default='stage2/frontier_manifest.json');a=p.parse_args();root=Path(a.results);rows=[json.loads(s) for s in (root/'per_image.jsonl').read_text().splitlines()];expected={r['id']:r for r in json.loads(Path(a.manifest).read_text())}
assert len(rows)==100 and len({r['id'] for r in rows})==100 and {r['id'] for r in rows}==set(expected)
labels=np.array([r['label']=='ai' for r in rows],dtype=int);assert labels.sum()==50
for r in rows:
 for key in ['label','sha256','revision','file']:assert r[key]==expected[r['id']][key]
 assert r['score']==r['scores']['outdoor']['mean_cosine_residual']
 for v in ['outdoor','indoor']:
  q=r['scores'][v];assert all(np.isfinite(x) for x in q.values());assert 0<=q['mean_cosine_residual']<=2 and 0<q['interior_coverage']<=1
s={'n':100,'real':50,'ai':50,'primary':'outdoor_large_forward1_mean_cosine_residual','direction':'higher = generated, fixed, not flipped','auroc':auroc(labels,[r['score'] for r in rows]),'auroc_95pct_stratified_image_bootstrap':bootstrap_auc(labels,[r['score'] for r in rows]),'bootstrap_replicates':2000,'bootstrap_seed':180,'variants':{},'failed_images':0}
for v in ['outdoor','indoor']:
 d={'cosine_auroc':auroc(labels,[r['scores'][v]['mean_cosine_residual'] for r in rows]),'angular_auroc':auroc(labels,[r['scores'][v]['mean_angle_deg'] for r in rows]),'classes':{}}
 for label in ['real','ai']:
  rr=[r for r in rows if r['label']==label];scores=[r['scores'][v] for r in rr]
  d['classes'][label]={'n':50,'mean_cosine_residual':float(np.mean([x['mean_cosine_residual'] for x in scores])),'median_image_cosine_residual':float(np.median([x['mean_cosine_residual'] for x in scores])),'mean_angular_disagreement_deg':float(np.mean([x['mean_angle_deg'] for x in scores])),'mean_interior_coverage':float(np.mean([x['interior_coverage'] for x in scores]))}
 s['variants'][v]=d
(root/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
with (root/'scores.csv').open('w') as f:
 fields=['id','label','generator','outdoor_mean_cosine_residual','indoor_mean_cosine_residual','outdoor_mean_angle_deg','indoor_mean_angle_deg'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
 for r in rows:w.writerow({**{k:r[k] for k in ['id','label','generator']},**{v+'_'+metric:r['scores'][v][metric] for v in ['outdoor','indoor'] for metric in ['mean_cosine_residual','mean_angle_deg']}})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');values=np.array([r['score'] for r in rows]);bins=np.linspace(max(0,values.min()-.01),values.max()+.01,16)
for label,color in [('real','#2478b4'),('ai','#d56b27')]:axes[0].hist([r['score'] for r in rows if r['label']==label],bins=bins,alpha=.55,label=label+' (n=50)',color=color)
axes[0].set(xlabel='Mean cosine residual (0–2)',ylabel='Images',title='Outdoor Large: primary score');axes[0].legend()
thresholds=np.r_[np.inf,np.unique(values)[::-1],-np.inf];fpr=[np.mean(values[labels==0]>=t) for t in thresholds];tpr=[np.mean(values[labels==1]>=t) for t in thresholds]
axes[1].plot(fpr,tpr,color='#7b3294',label=f'AUROC {s["auroc"]:.3f}');axes[1].plot([0,1],[0,1],'--',color='gray',label='Chance');axes[1].set(xlabel='Real images flagged (FPR)',ylabel='Generated images detected (TPR)',title='Higher residual = generated',xlim=(0,1),ylim=(0,1));axes[1].legend(loc='lower right')
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('DAV2 Metric + GeoCalib + Marigold · supplied cosine method');fig.savefig(root/'detection.png',dpi=180);plt.close(fig)
print(json.dumps(s,indent=2))
