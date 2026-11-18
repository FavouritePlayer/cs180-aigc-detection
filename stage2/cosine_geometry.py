"""Forward-difference normals and the pasted 1 - dot-product residual."""
import numpy as np

def forward_normals(depth,k):
    h,w=depth.shape;y,x=np.mgrid[:h,:w]
    rays=np.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],np.ones_like(x)),axis=-1)
    points=rays*depth[...,None]
    du=points[:-1,1:]-points[:-1,:-1];dv=points[1:,:-1]-points[:-1,:-1]
    n=np.cross(du,dv);length=np.linalg.norm(n,axis=-1)
    # Orient in OpenCV camera space, then change to Marigold right/up/toward-viewer.
    n=np.where((np.sum(n*points[:-1,:-1],axis=-1)>0)[...,None],-n,n)
    n=n*np.array([1.,-1.,-1.])/np.maximum(length[...,None],1e-12)
    good=np.isfinite(depth)&(depth>0)
    valid=np.zeros((h,w),dtype=bool);valid[:-1,:-1]=good[:-1,:-1]&good[:-1,1:]&good[1:,:-1]&np.isfinite(n).all(-1)&(length>1e-12)
    normal=np.zeros((h,w,3),dtype=np.float32);normal[:-1,:-1]=n
    return normal,valid

def cosine_residual(a,b):
    a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    b=b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-12)
    return 1-np.clip(np.sum(a*b,axis=-1),-1,1)
