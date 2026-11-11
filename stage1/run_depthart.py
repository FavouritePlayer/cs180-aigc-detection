import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from metrics import depth_metrics
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--weights',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
sys.path.insert(0,str(Path(a.source)/'metric'))
from common import make_K,preprocess
from model import load_model
from infer_image import enable_optimized_scan
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
print('optimized_scan',enable_optimized_scan(),flush=True)
for domain,short,cap in [('indoors','indoor',20),('outdoor','outdoor',80)]:
    model=load_model(str(Path(a.weights)/f'depthart_metric_{short}_l_448.pth'),'L',short,'cuda')
    for row in json.loads(Path(a.manifest).read_text()):
        if row['domain']!=domain:continue
        f=Path(a.root)/row['path'];im=cv2.imread(str(f));start=time.time()
        # Match official CLI defaults, including aspect-preserving resize and K scaling.
        tensor,K=preprocess(im,make_K(886.81,927.06,512,384),640 if short=='indoor' else 448,480 if short=='indoor' else 448)
        with torch.inference_mode():
            pred=F.interpolate(model(tensor.cuda(),K.cuda())[:,None],im.shape[:2],mode='bilinear',align_corners=True)[0,0].cpu().numpy()
        np.save(out/(f.stem+'_depth.npy'),pred.astype(np.float32))
        gt=np.load(str(f.with_suffix(''))+'_depth.npy').squeeze();mask=np.load(str(f.with_suffix(''))+'_depth_mask.npy').squeeze().astype(bool)
        result=dict(model='depthart_metric_large_gt_intrinsics',**row,seconds=time.time()-start,full=depth_metrics(gt,pred,mask),capped=depth_metrics(gt,pred,mask&(gt<cap)))
        with (out/'per_image.jsonl').open('a') as h:h.write(json.dumps(result)+'\n')
        print(domain,f.name,result['full']['abs_rel'],flush=True)
    del model;torch.cuda.empty_cache()
