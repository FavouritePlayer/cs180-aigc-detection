"""Summarize the additional models' normal accuracy and plot all raw depth scores."""
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--previous',default='results/stage1');a=p.parse_args();root=Path(a.results)
summary=[]
for file in sorted(root.glob('*/per_image.jsonl')):
 rows=[json.loads(s) for s in file.read_text().splitlines()]
 for domain in ['indoors','outdoor']:
  rr=[r for r in rows if r['domain']==domain]
  assert len(rr)==100,(file,domain,len(rr))
  for stride in ['1','4']:
   assert all(np.isfinite(r['geometry'][stride]['mean_angle']) for r in rr)
   summary.append(dict(model=rr[0]['model'],domain=domain,stride_pixels=int(stride),n=len(rr),vs_gt_mean_angle_deg=float(np.mean([r['geometry'][stride]['mean_angle'] for r in rr])),mean_valid_pixels=float(np.mean([r['geometry'][stride]['pixels'] for r in rr]))))
(root/'geometry_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'geometry_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
combined=json.loads((Path(a.previous)/'summary.json').read_text())+json.loads((root/'summary.json').read_text())
models=['dav2_metric_large','depthart_metric_large','tr2m_blip','unidepthv2','moge2','depthpro']
# Match prior runner names from the actual records.
prior=[r['model'] for r in combined if r['mask']=='full' and r['domain']=='indoors']
models=prior
labels={'dav2_metric_large':'DAV2 Metric L','depthart_metric_large':'DepthART + known K','tr2m_blip_base':'TR2M + BLIP','depthart_metric_large_gt_intrinsics':'DepthART + known K','tr2m_blip':'TR2M + BLIP','depthart_metric':'DepthART + known K','tr2m':'TR2M + BLIP','unidepthv2':'UniDepthV2 L','moge2':'MoGe-2 L','depthpro':'Depth Pro'}
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for ax,domain,title in zip(axes,['indoors','outdoor'],['Indoor: 100 images','Outdoor: 100 images']):
 vals=[next(r['abs_rel']*100 for r in combined if r['model']==m and r['mask']=='full' and r['domain']==domain) for m in models]
 bars=ax.barh(range(len(models)),vals,color=['#9aa9bd']*3+['#2878ad']*3)
 ax.set_yticks(range(len(models)),[labels.get(m,m) for m in models]);ax.invert_yaxis();ax.set_xlabel('Raw AbsRel (%) — lower is better');ax.set_title(title);ax.set_xlim(0,max(vals)*1.21);ax.spines[['top','right']].set_visible(False)
 for b,v in zip(bars,vals):ax.text(v+.4,b.get_y()+b.get_height()/2,f'{v:.1f}%',va='center',fontsize=9)
fig.suptitle('DIODE depth accuracy · no ground-truth scale correction',fontsize=13)
fig.savefig(root/'depth_comparison.png',dpi=180);plt.close(fig)
print('Wrote geometry summary and comparison plot')
