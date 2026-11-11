"""Official metric ViT-L preprocessing/inference; domain-specific checkpoints."""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from metrics import depth_metrics
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--weights',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
sys.path.insert(0,str(Path(a.source)/'metric_depth'))
from depth_anything_v2.dpt import DepthAnythingV2
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
rows=json.loads(Path(a.manifest).read_text())
for domain,checkpoint,cap in [('indoors','hypersim',20),('outdoor','vkitti',80)]:
    model=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024],max_depth=cap)
    ckpt=Path(a.weights)/f'depth_anything_v2_metric_{checkpoint}_vitl.pth'
    model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True));model.cuda().eval()
    for row in rows:
        if row['domain']!=domain:continue
        f=Path(a.root)/row['path']; start=time.time()
        with torch.inference_mode():pred=model.infer_image(cv2.imread(str(f)),input_size=518)
        np.save(out/(f.stem+'_depth.npy'),pred.astype(np.float32))
        gt=np.load(str(f.with_suffix(''))+'_depth.npy').squeeze()
        mask=np.load(str(f.with_suffix(''))+'_depth_mask.npy').squeeze().astype(bool)
        result=dict(model='dav2_metric_large',**row,seconds=time.time()-start,
                    full=depth_metrics(gt,pred,mask),
                    capped=depth_metrics(gt,pred,mask & (gt<cap)))
        with (out/'per_image.jsonl').open('a') as h:h.write(json.dumps(result)+'\n')
        print(domain,f.name,result['full']['abs_rel'],flush=True)
    del model;torch.cuda.empty_cache()
