"""RGB-only MoGe-2 / Marigold disagreement on a frozen, labeled image sample."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage1'))
from metrics import depth_to_normals
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['depth','normals'],required=True);p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--sources',required=True);p.add_argument('--weights',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'maps').mkdir(exist_ok=True);sources=Path(a.sources);weights=Path(a.weights);torch.set_num_threads(4)
rows=json.loads(Path(a.manifest).read_text());record=out/('depth_metadata.jsonl' if a.phase=='depth' else 'per_image.jsonl')
previous=[json.loads(s) for s in record.read_text().splitlines()] if record.exists() else []
assert len({r['id'] for r in previous})==len(previous),'Duplicate prior records'
completed={r['id'] for r in previous}
settings={'canvas_long_edge':1024,'resize':'PIL LANCZOS, aspect preserving, no crop','moge_resolution_level':9,'moge_apply_mask':False,'camera':'MoGe predicted normalized intrinsics, converted to canvas pixels','marigold_steps':4,'marigold_ensemble':10,'marigold_processing_resolution':768,'precision':'fp16 official inference','primary_stride':4,'score_direction':'higher disagreement = AI','seed':'first 32 bits of SHA256(image id)'}
(out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
def image_for(row):
 path=root/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 image=Image.open(path).convert('RGB');w,h=image.size;scale=1024/max(w,h);image=image.resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS)
 return image,hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
if a.phase=='depth':
 sys.path[:0]=[str(sources/'moge'),str(sources/'utils3d')]
 from moge.model.v2 import MoGeModel
 # Strict loading ensures every v2 predictive head matches the selected checkpoint.
 ckpt=torch.load(weights/'moge/model.pt',map_location='cpu',weights_only=True)
 model=MoGeModel(**ckpt['model_config']);model.load_state_dict(ckpt['model'],strict=True);del ckpt;model.cuda().eval()
else:
 from diffusers import MarigoldNormalsPipeline
 pipe=MarigoldNormalsPipeline.from_pretrained(str(weights/'marigold'),variant='fp16',torch_dtype=torch.float16,local_files_only=True).to('cuda');pipe.set_progress_bar_config(disable=True)
 metadata={r['id']:r for r in map(json.loads,(out/'depth_metadata.jsonl').read_text().splitlines())}
for row in rows:
 if row['id'] in completed:continue
 image,pixel_hash=image_for(row);seed=int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16);torch.manual_seed(seed);np.random.seed(seed);start=time.time();w,h=image.size
 if a.phase=='depth':
  with torch.inference_mode():r=model.infer(torch.from_numpy(np.array(image)).permute(2,0,1).float().cuda()/255,resolution_level=9,apply_mask=False,fov_x=None,use_fp16=True)
  depth=r['depth'].detach().float().cpu().numpy();kn=r['intrinsics'].detach().float().cpu().numpy();k=np.diag([w,h,1.])@kn
  assert depth.shape==(h,w) and np.isfinite(k).all() and k[0,0]>0 and k[1,1]>0
  np.savez_compressed(out/'maps'/(row['id']+'_depth.npz'),depth=depth,intrinsics=k,model_mask=r['mask'].cpu().numpy())
  result={**row,'seed':seed,'canvas_width':w,'canvas_height':h,'processed_rgb_sha256':pixel_hash,'predicted_intrinsics_px':k.tolist(),'predicted_mask_fraction':float(r['mask'].float().mean()),'seconds':time.time()-start}
 else:
  meta=metadata[row['id']];assert meta['processed_rgb_sha256']==pixel_hash
  with torch.inference_mode():normal=pipe(image,num_inference_steps=4,ensemble_size=10,processing_resolution=768,match_input_resolution=True,generator=torch.Generator(device='cuda').manual_seed(seed)).prediction[0]
  data=np.load(out/'maps'/(row['id']+'_depth.npz'));depth=data['depth'];k=data['intrinsics'];intrinsics=(k[0,0],k[1,1],k[0,2],k[1,2]);normal=np.asarray(normal,dtype=np.float32);assert normal.shape==(h,w,3)
  nlen=np.linalg.norm(normal,axis=-1);nv=np.isfinite(normal).all(-1)&(nlen>1e-6);normal=normal/np.maximum(nlen[...,None],1e-12)
  dv=np.isfinite(depth)&(depth>0);scores={}
  for s in [1,4]:
   dn,valid=depth_to_normals(depth,intrinsics,stride=s);vm=valid & nv & dv
   for axis in [0,1]:
    for shift in [-s,s]:vm &= np.roll(dv,shift,axis=axis)
   count=int(vm.sum());assert count>0,'No valid geometry predictions'
   angles=np.degrees(np.arccos(np.clip(np.sum(dn[vm]*normal[vm],axis=-1),-1,1)))
   scores[str(s)]={'mean_angle_deg':float(angles.mean()),'median_angle_deg':float(np.median(angles)),'p90_angle_deg':float(np.percentile(angles,90)),'pixels':count,'interior_pixels':(h-2*s)*(w-2*s),'interior_coverage':count/((h-2*s)*(w-2*s)),'mean_abs_facing_component_difference':float(np.abs(dn[vm,2]-normal[vm,2]).mean())}
  np.savez_compressed(out/'maps'/(row['id']+'_normal.npz'),normal=normal)
  result={**row,'seed':seed,'processed_rgb_sha256':pixel_hash,'canvas_width':w,'canvas_height':h,'predicted_intrinsics_px':meta['predicted_intrinsics_px'],'predicted_mask_fraction':meta['predicted_mask_fraction'],'depth_seconds':meta['seconds'],'normal_seconds':time.time()-start,'scores':scores,'score':scores['4']['mean_angle_deg']}
 with record.open('a') as f:f.write(json.dumps(result)+'\n')
 print(a.phase,row['id'],result.get('score','ok'),flush=True)
print('COMPLETE',a.phase,flush=True)
