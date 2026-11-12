"""Summarize oriented normal error and real-image geometry disagreement."""
import argparse,json,csv
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args();root=Path(a.results)
rows=[json.loads(s) for s in (root/'marigold/per_image_normals.jsonl').read_text().splitlines()]
normal_rows=[];geometry=[]
for domain in ['indoors','outdoor']:
    rr=[r for r in rows if r['domain']==domain]
    normal_rows.append(dict(model=rr[0]['model'],domain=domain,n=len(rr),mean_angular_error_deg=float(np.mean([r['normals']['mean_angle'] for r in rr])),mean_image_median_angle_deg=float(np.mean([r['normals']['median_angle'] for r in rr])),fraction_within_11_25=float(np.mean([r['normals']['within_11_25'] for r in rr]))))
    for stride in ['1','4']:
        geometry.append(dict(model='measured_depth',domain=domain,stride_pixels=int(stride),n=len(rr),vs_gt_mean_angle_deg=float(np.mean([r['reference_geometry'][stride]['mean_angle'] for r in rr])),vs_independent_mean_angle_deg=None))
        models=sorted(rr[0]['depth_geometry'])
        for model in models:
            gg=[r['depth_geometry'][model][stride] for r in rr]
            geometry.append(dict(model=model,domain=domain,stride_pixels=int(stride),n=len(gg),vs_gt_mean_angle_deg=float(np.mean([g['vs_gt']['mean_angle'] for g in gg])),vs_independent_mean_angle_deg=float(np.mean([g['vs_independent']['mean_angle'] for g in gg]))))
for name,records in [('normal_summary',normal_rows),('geometry_summary',geometry)]:
    (root/(name+'.json')).write_text(json.dumps(records,indent=2)+'\n')
    with (root/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n");w.writeheader();w.writerows(records)
print(json.dumps(normal_rows,indent=2));print(json.dumps(geometry,indent=2))
