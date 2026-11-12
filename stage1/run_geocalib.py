"""Auxiliary calibration accuracy on the same DIODE RGB sample."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--weights',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
sys.path.insert(0,a.source)
from geocalib import GeoCalib
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);model=GeoCalib(weights=a.weights).cuda().eval();gt=np.array([886.81,927.06,512.,384.])
for row in json.loads(Path(a.manifest).read_text()):
    result=dict(**row,model='geocalib_pinhole')
    try:
        image=model.load_image(Path(a.root)/row['path']).cuda();start=time.time()
        with torch.inference_mode():r=model.calibrate(image,camera_model='pinhole')
        k=r['camera'].K.detach().float().cpu().numpy().reshape(-1,3,3)[0];v=np.array([k[0,0],k[1,1],k[0,2],k[1,2]])
        assert np.isfinite(v).all() and (v[:2]>0).all(),'Invalid calibration'
        result.update(status='ok',intrinsics=v.tolist(),focal_abs_rel=float(np.mean(abs(v[:2]-gt[:2])/gt[:2])),seconds=time.time()-start)
    except Exception as e:result.update(status='failed',error=type(e).__name__+': '+str(e))
    with (out/'calibration.jsonl').open('a') as h:h.write(json.dumps(result)+'\n')
    print(row['domain'],row['path'],result.get('focal_abs_rel',result.get('error')),flush=True)
