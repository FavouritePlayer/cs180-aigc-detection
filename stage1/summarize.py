"""Aggregate per-image metrics with equal image weight, separately by domain."""
import argparse,json,csv
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args();root=Path(a.results);summary=[]
for file in sorted(root.glob('*/per_image.jsonl')):
    rows=[json.loads(s) for s in file.read_text().splitlines() if s.strip()]
    for domain in ['indoors','outdoor']:
        rr=[r for r in rows if r['domain']==domain]
        if not rr:continue
        for mask in ['full','capped','common']:
            vals=[r[mask] for r in rr]
            result=dict(model=rr[0]['model'],domain=domain,mask=mask,n=len(rr))
            for metric in ['abs_rel','rmse','delta1','median_aligned_abs_rel','median_scale']:
                v=np.array([x[metric] for x in vals]);result[metric]=float(v.mean())
            result['abs_rel_image_median']=float(np.median([x['abs_rel'] for x in vals]))
            result['abs_rel_image_p90']=float(np.percentile([x['abs_rel'] for x in vals],90))
            result['mean_seconds']=float(np.mean([x['seconds'] for x in rr]))
            summary.append(result)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if summary:
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print(json.dumps(summary,indent=2))
