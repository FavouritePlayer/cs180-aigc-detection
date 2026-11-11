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
