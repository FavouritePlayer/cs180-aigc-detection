"""Line detection, Manhattan vanishing points, and calibrated orthogonality residual."""
import cv2
import numpy as np


def detect_lines(gray, min_length_frac=0.02):
    """LSD line segments as Nx4 (x1,y1,x2,y2), filtered by length relative to image diagonal."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    raw = lsd.detect(gray)[0]
    if raw is None:
        return np.zeros((0, 4), dtype=np.float64)
    segs = raw.reshape(-1, 4).astype(np.float64)
    diag = np.hypot(*gray.shape[:2])
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[lengths >= min_length_frac * diag]


def segments_to_lines(segs):
    """Homogeneous line coefficients (a,b,c), a*x+b*y+c=0, one per segment, unit-normalized."""
    p1 = np.column_stack([segs[:, 0], segs[:, 1], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2], segs[:, 3], np.ones(len(segs))])
    lines = np.cross(p1, p2)
    norm = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    return lines / np.maximum(norm, 1e-12)


def _vp_direction(vp, midpoint):
    """Unit direction from a segment midpoint toward a homogeneous vanishing point (finite or at infinity)."""
    if abs(vp[2]) > 1e-8:
        d = np.array([vp[0] / vp[2] - midpoint[0], vp[1] / vp[2] - midpoint[1]])
    else:
        d = vp[:2]
    return d / np.maximum(np.linalg.norm(d), 1e-12)


def angular_errors_deg(vp, segs):
    """Unsigned angle in degrees between each segment's direction and the direction toward vp."""
    directions = segs[:, 2:4] - segs[:, :2]
    directions = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    midpoints = (segs[:, :2] + segs[:, 2:4]) / 2
    errors = np.empty(len(segs))
    for i, (d, m) in enumerate(zip(directions, midpoints)):
        vd = _vp_direction(vp, m)
        cos = np.clip(abs(d @ vd), -1.0, 1.0)
        errors[i] = np.degrees(np.arccos(cos))
    return errors


def refine_vp(lines_subset):
    """Least-squares vanishing point: right null vector of the stacked line-coefficient matrix."""
    _, _, vt = np.linalg.svd(lines_subset)
    return vt[-1]


def ransac_vanishing_point(segs, lines, inlier_threshold_deg=2.0, iters=500, min_inliers=6, rng=None):
    """One dominant vanishing point via sequential RANSAC over the given segment pool."""
    rng = rng or np.random.default_rng()
    n = len(segs)
    if n < 2:
        return None
    best_inliers, best_vp = None, None
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        candidate = np.cross(lines[i], lines[j])
        if np.linalg.norm(candidate) < 1e-12:
            continue
        errors = angular_errors_deg(candidate, segs)
        inliers = np.where(errors < inlier_threshold_deg)[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers, best_vp = inliers, candidate
    if best_inliers is None or len(best_inliers) < min_inliers:
        return None
    refined = refine_vp(lines[best_inliers])
    errors = angular_errors_deg(refined, segs)
    inliers = np.where(errors < inlier_threshold_deg)[0]
    if len(inliers) < min_inliers:
        return None
    refined = refine_vp(lines[inliers])
    return refined, inliers


def find_vanishing_points(segs, max_vps=3, min_lines_remaining=6, inlier_threshold_deg=2.0,
                           iters=500, min_inliers=6, rng=None):
    """Up to max_vps dominant vanishing points found sequentially, inliers removed between rounds."""
    rng = rng or np.random.default_rng()
    lines = segments_to_lines(segs)
    remaining = np.arange(len(segs))
    results = []
    while len(results) < max_vps and len(remaining) >= min_lines_remaining:
        found = ransac_vanishing_point(segs[remaining], lines[remaining], inlier_threshold_deg,
                                        iters, min_inliers, rng)
        if found is None:
            break
        vp, local_inliers = found
        global_inliers = remaining[local_inliers]
        results.append((vp, global_inliers))
        remaining = np.setdiff1d(remaining, global_inliers)
    return results


def orthogonality_cosine(vp_i, vp_j, k):
    """|cos| between the back-projected 3D ray directions of two vanishing points, given intrinsics K.

    0 means the two directions are orthogonal (expected for a Manhattan-world real photo under
    correct calibration); 1 means they are parallel (fully inconsistent with the orthogonality
    assumption). This is the opposite sense from the depth-normal residual, where 0 is the
    'agreement' we expect: here 0 is the 'consistency' we expect.
    """
    omega = np.linalg.inv(k @ k.T)
    num = vp_i @ omega @ vp_j
    denom = np.sqrt(max(vp_i @ omega @ vp_i, 1e-12) * max(vp_j @ omega @ vp_j, 1e-12))
    return abs(num / denom)


def score_image(gray, k, min_length_frac=0.02, max_vps=3, min_lines_remaining=6,
                 inlier_threshold_deg=2.0, iters=500, min_inliers=6, rng=None):
    """Per-image VP-orthogonality score: mean |cos| over all found VP pairs, plus coverage info."""
    segs = detect_lines(gray, min_length_frac)
    vps = find_vanishing_points(segs, max_vps, min_lines_remaining, inlier_threshold_deg,
                                 iters, min_inliers, rng)
    n_vps = len(vps)
    if n_vps < 2:
        return {"score": None, "num_vps": n_vps, "num_lines": len(segs),
                "inlier_counts": [len(inl) for _, inl in vps]}
    pairs = [(a, b) for a in range(n_vps) for b in range(a + 1, n_vps)]
    residuals = [orthogonality_cosine(vps[a][0], vps[b][0], k) for a, b in pairs]
    return {
        "score": float(np.mean(residuals)),
        "pairwise": [float(r) for r in residuals],
        "num_vps": n_vps,
        "num_lines": len(segs),
        "inlier_counts": [len(inl) for _, inl in vps],
    }
