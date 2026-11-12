"""Fixed examples: first two manifest entries in each domain, no score selection."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--results',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();root=Path(a.root);results=Path(a.results)
rows=json.loads(Path(a.manifest).read_text())
for domain in ['indoors','outdoor']:
    selected=[r for r in rows if r['domain']==domain][:2];fig,axes=plt.subplots(2,7,figsize=(19,5.6))
    for rr,row in enumerate(selected):
        f=root/row['path'];stem=f.stem;gt=np.load(str(f.with_suffix(''))+'_depth.npy').squeeze();mask=np.load(str(f.with_suffix(''))+'_depth_mask.npy').squeeze().astype(bool);vmax=float(np.percentile(gt[mask],99))
        axes[rr,0].imshow(Image.open(f));axes[rr,0].set_title('RGB\n'+stem,fontsize=7)
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('black')
        for cc,name in enumerate(['Measured depth','dav2','depthart','tr2m'],start=1):
            d=gt if cc==1 else np.load(results/name/(stem+'_depth.npy'))
            axes[rr,cc].imshow(np.ma.masked_where(~mask,d),vmin=0,vmax=vmax,cmap=cmap);axes[rr,cc].set_title({'dav2':'DAV2 L','depthart':'DepthART L','tr2m':'TR2M + BLIP','Measured depth':'Reference depth'}.get(name,name)+f'\n0–{vmax:.1f} m',fontsize=9)
        ngt=np.load(str(f.with_suffix(''))+'_normal.npy');valid=np.linalg.norm(ngt,axis=-1)>1e-6
        for cc,n,title in [(5,ngt,'Reference normals'),(6,np.load(results/'marigold'/(stem+'_normal.npy')),'Marigold normals')]:
            n=n/np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-6);color=(n+1)/2;color[~valid]=0;axes[rr,cc].imshow(np.clip(color,0,1));axes[rr,cc].set_title(title,fontsize=9)
        for ax in axes[rr]:ax.axis('off')
    fig.suptitle(f'DIODE {domain}: fixed examples, shared depth color scale within each row',fontsize=12);fig.tight_layout();fig.savefig(results/f'examples_{domain}.png',dpi=160);plt.close(fig)
