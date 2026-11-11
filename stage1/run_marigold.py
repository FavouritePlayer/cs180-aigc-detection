"""Normals benchmark: official model defaults, four DDIM steps, ensemble one."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from diffusers import MarigoldNormalsPipeline
from metrics import normal_metrics,depth_to_normals
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--depth-results',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
pipe=MarigoldNormalsPipeline.from_pretrained('prs-eth/marigold-normals-v1-1',variant='fp16',torch_dtype=torch.float16).to('cuda');pipe.set_progress_bar_config(disable=True)
for row in json.loads(Path(a.manifest).read_text()):
    f=Path(a.root)/row['path'];start=time.time();seed=int(hashlib.sha256(row['path'].encode()).hexdigest()[:8],16)
    with torch.inference_mode():
        pred=pipe(Image.open(f).convert('RGB'),num_inference_steps=4,ensemble_size=1,processing_resolution=768,match_input_resolution=True,generator=torch.Generator(device='cuda').manual_seed(seed)).prediction[0]
    gt=np.load(str(f.with_suffix(''))+'_normal.npy');mask=np.linalg.norm(gt,axis=-1)>.5
    np.save(out/(f.stem+'_normal.npy'),pred.astype(np.float32))
    result=dict(model='marigold_normals_v1_1_e1_s4',**row,seed=seed,seconds=time.time()-start,normals=normal_metrics(gt,pred,mask),depth_geometry={})
    for directory in sorted(Path(a.depth_results).iterdir()):
        df=directory/(f.stem+'_depth.npy')
        if not df.exists():continue
        depth=np.load(df);result['depth_geometry'][directory.name]={}
        for stride in [1,4]:
            dn,valid=depth_to_normals(depth,stride=stride);vm=mask&valid
            result['depth_geometry'][directory.name][str(stride)]=dict(vs_gt=normal_metrics(gt,dn,vm),vs_independent=normal_metrics(pred,dn,vm))
    with (out/'per_image_normals.jsonl').open('a') as h:h.write(json.dumps(result)+'\n')
    print(row['domain'],f.name,result['normals']['mean_angle'],flush=True)
