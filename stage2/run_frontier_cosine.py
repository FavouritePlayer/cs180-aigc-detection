"""Apply DAV2 Metric + GeoCalib + Marigold with the supplied cosine residual."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from cosine_geometry import forward_normals,cosine_residual
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['calibration','outdoor','indoor','normals'],required=True);p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--sources',required=True);p.add_argument('--weights',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'maps').mkdir(exist_ok=True);sources=Path(a.sources);weights=Path(a.weights);torch.set_num_threads(4)
rows=json.loads(Path(a.manifest).read_text());record=out/({'calibration':'calibration.jsonl','outdoor':'depth_outdoor.jsonl','indoor':'depth_indoor.jsonl','normals':'per_image.jsonl'}[a.phase]);previous=[json.loads(s) for s in record.read_text().splitlines()] if record.exists() else [];assert len({r['id'] for r in previous})==len(previous);completed={r['id'] for r in previous}
settings={'primary':'outdoor_large_forward1_mean_cosine_residual','secondary':'indoor_large_same_pipeline; mean angular disagreement for both','canvas_long_edge':1024,'resize':'PIL LANCZOS, no crop, aspect preserving','dav2_input_size':518,'dav2_precision':'fp32 official infer_image','geocalib_camera':'pinhole, predicted focal, principal point fixed at image center, no priors','marigold_steps':4,'marigold_ensemble':10,'marigold_processing_resolution':768,'marigold_precision':'fp16','derivatives':'forward differences, one pixel, three-point stencil','residual':'1 - clip(dot(unit depth normal, unit Marigold normal), -1, 1)','score_direction':'higher = generated','seed':'first 32 bits of SHA256(image id)'}
(out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
if a.phase in ['outdoor','indoor']:
 sys.path.insert(0,str(sources/'dav2/metric_depth'))
 from depth_anything_v2.dpt import DepthAnythingV2
 name='vkitti' if a.phase=='outdoor' else 'hypersim';model=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024],max_depth=80 if a.phase=='outdoor' else 20)
 model.load_state_dict(torch.load(weights/f'dav2/depth_anything_v2_metric_{name}_vitl.pth',map_location='cpu',weights_only=True),strict=True);model.cuda().eval()
elif a.phase=='calibration':
 sys.path.insert(0,str(sources/'geocalib'))
 from geocalib import GeoCalib
 model=GeoCalib(weights=str(weights/'geocalib/geocalib-pinhole.tar')).cuda().eval()
 state=torch.load(weights/'geocalib/geocalib-pinhole.tar',map_location='cpu',weights_only=False)['model'];names=set(dict(model.model.named_parameters()))
 if not (names & set(state)):state={'.'.join(k.split('.')[:1]+k.split('.')[2:]):v for k,v in state.items()}
 assert names<=set(state),'GeoCalib checkpoint omits predictive parameters'
 actual=model.model.state_dict();assert all(torch.equal(actual[n].cpu(),state[n]) for n in names),'GeoCalib parameter mismatch after official load'
 del state,actual
 (out/'calibration_load_verification.json').write_text(json.dumps({'all_named_parameters_match_checkpoint':True,'parameters_checked':len(names)},indent=2)+'\n')
else:
 from diffusers import MarigoldNormalsPipeline
 pipe=MarigoldNormalsPipeline.from_pretrained(str(weights/'marigold'),variant='fp16',torch_dtype=torch.float16,local_files_only=True).to('cuda');pipe.set_progress_bar_config(disable=True)
 calibration={r['id']:r for r in map(json.loads,(out/'calibration.jsonl').read_text().splitlines())}
 depths={v:{r['id']:r for r in map(json.loads,(out/f'depth_{v}.jsonl').read_text().splitlines())} for v in ['outdoor','indoor']}
for row in rows:
 if row['id'] in completed:continue
 f=root/row['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'];image=Image.open(f).convert('RGB');w,h=image.size;s=1024/max(w,h);image=image.resize((round(w*s),round(h*s)),Image.Resampling.LANCZOS);w,h=image.size;rgb=np.array(image);ph=hashlib.sha256(rgb.tobytes()).hexdigest();seed=int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16);np.random.seed(seed);torch.manual_seed(seed);start=time.time()
 result={**row,'seed':seed,'processed_rgb_sha256':ph,'canvas_width':w,'canvas_height':h}
 if a.phase=='calibration':
  with torch.inference_mode():r=model.calibrate(torch.from_numpy(rgb).permute(2,0,1).float().cuda()/255,camera_model='pinhole')
  k=r['camera'].K.detach().float().cpu().numpy().reshape(-1,3,3)[0];assert np.isfinite(k).all() and k[0,0]>0 and k[1,1]>0;result['predicted_intrinsics_px']=k.tolist()
 elif a.phase in ['outdoor','indoor']:
  with torch.inference_mode():depth=model.infer_image(rgb[:,:,::-1].copy(),input_size=518)
  assert depth.shape==(h,w);assert (np.isfinite(depth)&(depth>0)).all()
  np.savez_compressed(out/'maps'/(row['id']+'_'+a.phase+'_depth.npz'),depth=depth);result['checkpoint']=a.phase
 else:
  assert calibration[row['id']]['processed_rgb_sha256']==ph
  with torch.inference_mode():normal=pipe(image,num_inference_steps=4,ensemble_size=10,processing_resolution=768,match_input_resolution=True,generator=torch.Generator(device='cuda').manual_seed(seed)).prediction[0]
  normal=np.asarray(normal,dtype=np.float32);assert normal.shape==(h,w,3);nv=np.isfinite(normal).all(-1)&(np.linalg.norm(normal,axis=-1)>1e-6);k=np.array(calibration[row['id']]['predicted_intrinsics_px']);result['predicted_intrinsics_px']=k.tolist();result['scores']={}
  for variant in ['outdoor','indoor']:
   assert depths[variant][row['id']]['processed_rgb_sha256']==ph
   depth=np.load(out/'maps'/(row['id']+'_'+variant+'_depth.npz'))['depth'];dn,valid=forward_normals(depth,k);vm=valid&nv;assert vm.any();residual=cosine_residual(dn[vm],normal[vm]);angles=np.degrees(np.arccos(np.clip(1-residual,-1,1)))
   result['scores'][variant]={'mean_cosine_residual':float(residual.mean()),'mean_angle_deg':float(angles.mean()),'pixels':int(vm.sum()),'interior_pixels':(h-1)*(w-1),'interior_coverage':float(vm.sum()/((h-1)*(w-1)))}
  result['score']=result['scores']['outdoor']['mean_cosine_residual'];np.savez_compressed(out/'maps'/(row['id']+'_normal.npz'),normal=normal)
 result['seconds']=time.time()-start
 with record.open('a') as file:file.write(json.dumps(result)+'\n')
 print(a.phase,row['id'],result.get('score','ok'),flush=True)
print('COMPLETE',a.phase,flush=True)
