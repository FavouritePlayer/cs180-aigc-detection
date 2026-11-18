"""Fixed first-in-class examples of the supplied cosine-residual pipeline."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cosine_geometry import forward_normals,cosine_residual
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--results',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();root=Path(a.results);rows=json.loads(Path(a.manifest).read_text());cal={r['id']:r for r in map(json.loads,(root/'calibration.jsonl').read_text().splitlines())};selected=[next(r for r in rows if r['label']==label) for label in ['real','ai']]
fig,axes=plt.subplots(2,4,figsize=(13,7),layout='constrained')
for i,r in enumerate(selected):
 image=Image.open(Path(a.root)/r['file']).convert('RGB');w,h=image.size;s=1024/max(w,h);image=image.resize((round(w*s),round(h*s)),Image.Resampling.LANCZOS)
 depth=np.load(root/'maps'/(r['id']+'_outdoor_depth.npz'))['depth'];normal=np.load(root/'maps'/(r['id']+'_normal.npz'))['normal'];dn,valid=forward_normals(depth,np.array(cal[r['id']]['predicted_intrinsics_px']));residual=cosine_residual(dn,normal)
 axes[i,0].imshow(image);axes[i,1].imshow(np.clip((dn+1)/2,0,1));axes[i,2].imshow(np.clip((normal+1)/2,0,1));im=axes[i,3].imshow(np.where(valid,residual,np.nan),vmin=0,vmax=2,cmap='magma');fig.colorbar(im,ax=axes[i,3],label='Cosine residual (0–2)',shrink=.75)
 for ax,title in zip(axes[i],[r['label']+' RGB','DAV2 + GeoCalib normals','Marigold normals','1 − dot-product residual']):ax.set_title(title);ax.axis('off')
 axes[i,0].text(0,-.06,r['id']+' · '+r['author']+'\n'+r['license'],transform=axes[i,0].transAxes,fontsize=7,va='top')
fig.suptitle('First manifest image in each class · Outdoor Large primary method\nOne-pixel forward differences · fixed coordinate alignment · no score-based example selection')
fig.savefig(root/'examples.png',dpi=150);plt.close(fig)
