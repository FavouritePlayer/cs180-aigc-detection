"""TR2M with reproducible RGB-only BLIP-base captions (not paper captions)."""
import argparse,json,sys,time,os
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from metrics import depth_metrics
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);src=Path(a.source).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
rows=json.loads(Path(a.manifest).read_text());torch.set_num_threads(4)
from transformers import BlipProcessor,BlipForConditionalGeneration
processor=BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
captioner=BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').cuda().eval()
for row in rows:
    im=Image.open(root/row['path']).convert('RGB')
    with torch.inference_mode():tokens=captioner.generate(**processor(images=im,return_tensors='pt').to('cuda'),max_new_tokens=50,do_sample=False,num_beams=1)
    row['caption']=processor.decode(tokens[0],skip_special_tokens=True)
(out/'captions.json').write_text(json.dumps(rows,indent=2)+'\n');del captioner;torch.cuda.empty_cache()
sys.path.insert(0,str(src));os.chdir(src)
from run import build_models,load_image,upsample_to,parse_args
sys.argv=['run.py','--image','unused','--text','unused'];args=parse_args()
depth,clip,image,head=build_models(args,'cuda')
from CLIP import clip as clip_module
for row in rows:
    f=root/row['path'];start=time.time();_,tensor,size=load_image(str(f),434,560,'cuda')
    with torch.inference_mode():
        tf=clip.encode_text(clip_module.tokenize([row['caption']],truncate=True).cuda()).unsqueeze(1)
        imf=image.get_intermediate_layers(tensor,n=1,return_class_token=False)[0]
        scale,shift,_,_=head(imf.float(),tf.float(),patch_h=31,patch_w=40)
        rel=depth(tensor).unsqueeze(1)
        if rel.shape[-2:]!=scale.shape[-2:]:rel=F.interpolate(rel,size=scale.shape[-2:],mode='bilinear',align_corners=True)
        pred=upsample_to(1/(scale*rel+shift),size)
    np.save(out/(f.stem+'_depth.npy'),pred.astype(np.float32))
    gt=np.load(str(f.with_suffix(''))+'_depth.npy').squeeze();mask=np.load(str(f.with_suffix(''))+'_depth_mask.npy').squeeze().astype(bool);cap=20 if row['domain']=='indoors' else 80
    result=dict(model='tr2m_blip_base',**row,seconds=time.time()-start,full=depth_metrics(gt,pred,mask),capped=depth_metrics(gt,pred,mask&(gt<cap)))
    with (out/'per_image.jsonl').open('a') as h:h.write(json.dumps(result)+'\n')
    print(row['domain'],f.name,result['full']['abs_rel'],flush=True)
