"""Export stage 1 accuracy plots from per-image results."""
import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args();root=Path(a.results)
files=sorted(root.glob('*/per_image.jsonl'));fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,domain in zip(axes,['indoors','outdoor']):
    data=[];labels=[]
    for f in files:
        rows=[json.loads(s) for s in f.read_text().splitlines()];rr=[r for r in rows if r['domain']==domain]
        if not rr:continue
        data.append([r['full']['abs_rel'] for r in rr]);labels.append(f.parent.name)
    ax.boxplot(data,tick_labels=labels,showfliers=True)
    ax.set_title(f'{domain}: raw metric depth, n=100/model');ax.set_ylabel('Per-image AbsRel (lower is better)');ax.grid(axis='y',alpha=.25)
fig.suptitle('DIODE validation — fixed exploratory sample');fig.tight_layout();fig.savefig(root/'depth_absrel.png',dpi=180);plt.close(fig)
