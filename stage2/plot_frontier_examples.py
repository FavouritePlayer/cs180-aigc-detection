"""Plot the first manifest original in each class, retaining author credits."""
import sys,json,argparse,numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage1'))
from metrics import depth_to_normals
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--results',required=True);p.add_argument('--manifest',default='stage2/frontier_manifest.json');a=p.parse_args();result_root=Path(a.results)
rows=json.loads(Path(a.manifest).read_text());selected=[next(r for r in rows if r['label']==label) for label in ['real','ai']]
fig,axes=plt.subplots(2,4,figsize=(13,7),layout='constrained')
for i,row in enumerate(selected):
 image=Image.open(Path(a.root)/row['file']).convert('RGB');w,h=image.size;s=1024/max(w,h);image=image.resize((round(w*s),round(h*s)),Image.Resampling.LANCZOS)
 data=np.load(result_root/'maps'/(row['id']+'_depth.npz'));k=data['intrinsics'];depth=data['depth'];dn,valid=depth_to_normals(depth,(k[0,0],k[1,1],k[0,2],k[1,2]),stride=4)
 normal=np.load(result_root/'maps'/(row['id']+'_normal.npz'))['normal'];normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12);angle=np.degrees(np.arccos(np.clip(np.sum(dn*normal,axis=-1),-1,1)))
 axes[i,0].imshow(image);axes[i,1].imshow(np.clip((dn+1)/2,0,1));axes[i,2].imshow(np.clip((normal+1)/2,0,1));im=axes[i,3].imshow(np.where(valid,angle,np.nan),vmin=0,vmax=90,cmap='magma');fig.colorbar(im,ax=axes[i,3],label='Disagreement (degrees)',shrink=.75)
 for ax,title in zip(axes[i],[row['label']+' RGB','MoGe depth-derived normals','Marigold normals','Angular disagreement']):ax.set_title(title);ax.axis('off')
 axes[i,0].text(0,-.06,row['id']+' · '+row['author']+'\n'+row['license'],transform=axes[i,0].transAxes,fontsize=7,va='top')
fig.suptitle('First manifest original in each class · selection independent of scores\nNormal colors encode XYZ; disagreement colors saturate at 90° for display only')
fig.savefig(result_root/'examples.png',dpi=150);print('Fixed examples saved')
