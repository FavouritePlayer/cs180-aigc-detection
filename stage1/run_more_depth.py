"""Additional metric-depth models on the frozen DIODE sample, RGB-only inputs."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from metrics import depth_metrics,depth_to_normals,normal_metrics

p=argparse.ArgumentParser()
p.add_argument('--model',choices=['depthpro','unidepthv2','moge2'],required=True)
p.add_argument('--root',required=True);p.add_argument('--source',required=True)
p.add_argument('--weights',required=True);p.add_argument('--manifest',required=True)
p.add_argument('--out',required=True);p.add_argument('--limit',type=int)
p.add_argument('--resume',action='store_true');a=p.parse_args()
source=Path(a.source).resolve();weights=Path(a.weights).resolve();out=Path(a.out)
out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
rows=json.loads(Path(a.manifest).read_text())
if a.limit:rows=rows[:a.limit]
completed=set()
if (out/'per_image.jsonl').exists():
    if not a.resume:raise RuntimeError('Output exists; choose a fresh directory or --resume')
    previous=[json.loads(s) for s in (out/'per_image.jsonl').read_text().splitlines()]
    assert len({r['path'] for r in previous})==len(previous),'Duplicate prior records'
    completed={r['path'] for r in previous}

if a.model=='depthpro':
    sys.path.insert(0,str(source/'src'))
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    from dataclasses import replace
    cfg=replace(DEFAULT_MONODEPTH_CONFIG_DICT,checkpoint_uri=str(weights/'depthpro/depth_pro.pt'))
    model,transform=depth_pro.create_model_and_transforms(config=cfg,device=torch.device('cuda'),precision=torch.float16)
    model.eval();settings={'precision':'fp16','input':'official 1536 multi-scale inference','focal_input':None}
    def infer(image):
        result=model.infer(transform(image),f_px=None)
        return result['depth'],{'focal_px':float(result['focallength_px'].item())}
elif a.model=='unidepthv2':
    sys.path.insert(0,str(source))
    from unidepth.models import UniDepthV2
    config=json.loads((weights/'unidepth/config.json').read_text())
    model=UniDepthV2(config)
    model.load_state_dict(torch.load(weights/'unidepth/pytorch_model.bin',map_location='cpu',weights_only=True),strict=True)
    model.cuda().eval();model.resolution_level=9
    settings={'precision':'official infer autocast fp16','resolution_level':9,'camera_input':None,'checkpoint_load':'strict'}
    def infer(image):
        rgb=torch.from_numpy(np.array(image)).permute(2,0,1).cuda()
        result=model.infer(rgb,camera=None)
        return result['depth'],{'intrinsics':result['intrinsics'].detach().float().cpu().squeeze().tolist()}
else:
    sys.path.insert(0,str(source));sys.path.insert(0,str(source.parent/'utils3d'))
    from moge.model.v2 import MoGeModel
    model=MoGeModel.from_pretrained(str(weights/'moge/model.pt')).cuda().eval()
    settings={'precision':'official infer fp16','resolution_level':9,'fov_x_input':None,'apply_mask':False,'force_projection':True}
    def infer(image):
        rgb=torch.from_numpy(np.array(image)).permute(2,0,1).float().cuda()/255
        result=model.infer(rgb,resolution_level=9,apply_mask=False,fov_x=None,use_fp16=True)
        return result['depth'],{'intrinsics_normalized':result['intrinsics'].detach().float().cpu().tolist(),'predicted_mask_fraction':float(result['mask'].float().mean().item())}

(out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
for row in rows:
    if row['path'] in completed:continue
    f=Path(a.root)/row['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
    seed=int(hashlib.sha256(row['path'].encode()).hexdigest()[:8],16)
    np.random.seed(seed);torch.manual_seed(seed)
    start=time.time()
    with torch.inference_mode():tensor,extra=infer(Image.open(f).convert('RGB'))
    pred=tensor.detach().float().cpu().numpy().squeeze()
    gt=np.load(str(f.with_suffix(''))+'_depth.npy').squeeze()
    assert pred.shape==gt.shape,(pred.shape,gt.shape)
    mask=np.load(str(f.with_suffix(''))+'_depth_mask.npy').squeeze().astype(bool)
    cap=20 if row['domain']=='indoors' else 80
    result=dict(model=a.model,**row,seed=seed,seconds=time.time()-start,full=depth_metrics(gt,pred,mask),capped=depth_metrics(gt,pred,mask&(gt<cap)),common=depth_metrics(gt,pred,mask&(gt<(10 if row['domain']=='indoors' else 80))),prediction_metadata=extra)
    np.save(out/(f.stem+'_depth.npy'),pred)
    # Geometry is scored only against measured reference normals, using known K.
    # Neither K nor any other ground truth is provided to the depth models.
    gt_normal=np.load(str(f.with_suffix(''))+'_normal.npy')
    normal_mask=np.isfinite(gt_normal).all(-1)&(np.linalg.norm(gt_normal,axis=-1)>1e-6)
    depth_mask=mask & np.isfinite(gt) & (gt>0);result['geometry']={}
    for stride in [1,4]:
        _,rv=depth_to_normals(gt,stride=stride);dn,pv=depth_to_normals(pred,stride=stride)
        valid=normal_mask & rv & pv & depth_mask
        for axis in [0,1]:
            for shift in [-stride,stride]:valid &= np.roll(depth_mask,shift,axis=axis)
        result['geometry'][str(stride)]=normal_metrics(gt_normal,dn,valid)
    with (out/'per_image.jsonl').open('a') as h:h.write(json.dumps(result)+'\n')
    print(a.model,row['domain'],f.name,result['full']['abs_rel'],flush=True)
print('COMPLETE',a.model,flush=True)
