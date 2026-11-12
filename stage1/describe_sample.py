import argparse,json,csv
from pathlib import Path
from collections import Counter
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
rows=json.loads(Path(a.manifest).read_text());stats=[]
for r in rows:
    f=Path(a.root)/r['path'];d=np.load(str(f.with_suffix(''))+'_depth.npy').squeeze();m=np.load(str(f.with_suffix(''))+'_depth_mask.npy').squeeze().astype(bool)&np.isfinite(d)&(d>0);v=d[m]
    stats.append(dict(path=r['path'],domain=r['domain'],valid_pixels=int(m.sum()),valid_fraction=float(m.mean()),median_m=float(np.median(v)),p95_m=float(np.percentile(v,95)),fraction_ge_10m=float(np.mean(v>=10)),fraction_ge_20m=float(np.mean(v>=20)),fraction_ge_80m=float(np.mean(v>=80))))
summary={}
for domain in ['indoors','outdoor']:
    rr=[r for r in rows if r['domain']==domain];ss=[r for r in stats if r['domain']==domain]
    summary[domain]={'n':len(rr),'scenes':dict(Counter(Path(r['path']).parts[2] for r in rr)),'mean_valid_fraction':float(np.mean([r['valid_fraction'] for r in ss])),'mean_fraction_ge_10m':float(np.mean([r['fraction_ge_10m'] for r in ss])),'mean_fraction_ge_20m':float(np.mean([r['fraction_ge_20m'] for r in ss])),'mean_fraction_ge_80m':float(np.mean([r['fraction_ge_80m'] for r in ss]))}
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'sample_statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'sample_statistics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(stats[0]),lineterminator="\n");w.writeheader();w.writerows(stats)
print(json.dumps(summary,indent=2))
