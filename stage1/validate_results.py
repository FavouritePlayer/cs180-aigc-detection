"""Reject partial, duplicate, or mismatched benchmark records."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--manifest',default='stage1/manifest.json');a=p.parse_args()
manifest=json.loads(Path(a.manifest).read_text());expected={r['path']:r for r in manifest}
files=list(Path(a.results).glob('*/per_image*.jsonl'))
if not files:raise SystemExit('No model results found')
for f in files:
    rows=[json.loads(s) for s in f.read_text().splitlines()]
    paths=[r['path'] for r in rows]
    assert len(paths)==len(set(paths)),f'Duplicate records: {f}'
    assert set(paths)==set(expected),f'Incomplete/extra samples: {f}: {len(rows)}/200'
    assert all(r['sha256']==expected[r['path']]['sha256'] for r in rows),f'Manifest checksum mismatch: {f}'
    assert all(r['domain']==expected[r['path']]['domain'] for r in rows),f'Domain mismatch: {f}'
    print(f'{f}: 200 unique records, exact manifest match')
