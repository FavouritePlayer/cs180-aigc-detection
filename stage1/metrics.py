"""Metrics on sensor-valid pixels. No fitted alignment in primary metrics."""
import numpy as np


def depth_metrics(gt, pred, mask):
    valid = mask.astype(bool) & np.isfinite(gt) & (gt > 0)
    if not valid.any():
        raise ValueError('No valid reference pixels')
    g, p = gt[valid].astype(float), pred[valid].astype(float)
    if not (np.isfinite(p) & (p > 0)).all():
        raise ValueError('Nonfinite/nonpositive prediction on evaluation mask')
    ratio = np.maximum(g / p, p / g)
    scale = np.median(g) / np.median(p)
    return dict(abs_rel=float(np.mean(abs(p-g)/g)), rmse=float(np.sqrt(np.mean((p-g)**2))),
                delta1=float(np.mean(ratio < 1.25)), pixels=int(len(g)),
                median_scale=float(scale), median_aligned_abs_rel=float(np.mean(abs(scale*p-g)/g)))


def normal_metrics(gt, pred, mask):
    gn, pn = np.linalg.norm(gt, axis=-1), np.linalg.norm(pred, axis=-1)
    valid = mask & np.isfinite(gt).all(-1) & (gn > 1e-6)
    if not valid.any():
        raise ValueError('No valid normal reference pixels')
    if not (np.isfinite(pred[valid]).all() and (pn[valid] > 1e-6).all()):
        raise ValueError('Invalid predicted normals')
    dots = np.sum(gt[valid]*pred[valid], axis=-1)/(gn[valid]*pn[valid])
    angles = np.degrees(np.arccos(np.clip(dots, -1, 1)))
    return dict(mean_angle=float(angles.mean()), median_angle=float(np.median(angles)),
                within_11_25=float(np.mean(angles < 11.25)), pixels=int(valid.sum()))


def depth_to_normals(depth, intrinsics=(886.81,927.06,512.,384.), stride=1):
    """Central differences in camera space; output right/up/towards-viewer.

    Depth is axial Z. Orient to camera before converting OpenCV to OpenGL axes.
    Boundaries are invalid. Larger stencils measure geometry at a coarser scale.
    """
    h,w=depth.shape;fx,fy,cx,cy=intrinsics;y,x=np.mgrid[:h,:w]
    points=np.stack(((x-cx)/fx,(y-cy)/fy,np.ones_like(x)),axis=-1)*depth[...,None]
    s=stride;dx=points[s:-s,2*s:]-points[s:-s,:-2*s];dy=points[2*s:,s:-s]-points[:-2*s,s:-s]
    n=np.cross(dx,dy);rays=points[s:-s,s:-s]
    n=np.where((np.sum(n*rays,axis=-1)>0)[...,None],-n,n)
    n*=np.array([1.,-1.,-1.]);n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12)
    output=np.zeros((h,w,3),dtype=np.float32);output[s:-s,s:-s]=n
    valid=np.zeros((h,w),dtype=bool);valid[s:-s,s:-s]=np.isfinite(n).all(-1)&(np.linalg.norm(n,axis=-1)>.5)
    return output,valid
