"""Freeze a score-independent 50-real/50-AI original-image pilot."""
import argparse,csv,hashlib,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('metadata');p.add_argument('--revision',required=True);p.add_argument('--out',required=True);a=p.parse_args()
seed='cs180-frontier-2026-10-pilot-v1'
def key(x):return hashlib.sha256((seed+'|'+x).encode()).hexdigest()
rows=[r for r in csv.DictReader(Path(a.metadata).open()) if r['media']=='image' and r['variant']=='original']
real=sorted([r for r in rows if r['label']=='real'],key=lambda r:key(r['id']))[:50]
groups=collections.defaultdict(list)
for r in rows:
 if r['label']=='ai':groups[r['generator']].append(r)
# Round-robin from deterministic hash order covers all 19 generators, 2 or 3 each.
gens=sorted(groups,key=key)
for g in gens:groups[g].sort(key=lambda r:key(r['id']))
ai=[]
while len(ai)<50:
 for g in gens:
  if groups[g] and len(ai)<50:ai.append(groups[g].pop(0))
selected=sorted(real+ai,key=lambda r:key('order|'+r['id']))
for r in selected:r.update(dataset='wasitaigeneratedcom/frontier-image-video-bench-2026-10',revision=a.revision,sampling_seed=seed)
assert len(selected)==100 and len({r['id'] for r in selected})==100
Path(a.out).write_text(json.dumps(selected,indent=2)+'\n')
print(dict(collections.Counter(r['label'] for r in selected)));print(dict(collections.Counter(r['generator'] for r in selected if r['label']=='ai')))
