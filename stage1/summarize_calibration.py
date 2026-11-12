import argparse,json,csv
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args();root=Path(a.results)
rows=[json.loads(s) for s in (root/'geocalib/calibration.jsonl').read_text().splitlines()];summary=[]
for domain in ['indoors','outdoor']:
    rr=[r for r in rows if r['domain']==domain];ok=[r for r in rr if r['status']=='ok'];v=np.array([r['focal_abs_rel'] for r in ok])
    summary.append(dict(model='geocalib_pinhole',domain=domain,n=len(rr),successful=len(ok),failed=len(rr)-len(ok),mean_focal_abs_rel=float(v.mean()) if len(v) else None,median_focal_abs_rel=float(np.median(v)) if len(v) else None,p90_focal_abs_rel=float(np.percentile(v,90)) if len(v) else None))
(root/'calibration_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'calibration_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator="\n");w.writeheader();w.writerows(summary)
print(json.dumps(summary,indent=2))
