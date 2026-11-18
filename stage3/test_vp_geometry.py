"""Synthetic-scene tests: known camera + known orthogonal directions, no real images needed."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vp_geometry import detect_lines, find_vanishing_points, orthogonality_cosine, score_image

K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
IMG_SIZE = (480, 640)


def make_segments_toward_vp(vp, n, rng, img_size=IMG_SIZE, length=40.0, noise_deg=0.3):
    h, w = img_size
    starts = np.column_stack([rng.uniform(0.1 * w, 0.9 * w, n), rng.uniform(0.1 * h, 0.9 * h, n)])
    segs = np.empty((n, 4))
    for i, start in enumerate(starts):
        if abs(vp[2]) > 1e-8:
            direction = np.array([vp[0] / vp[2] - start[0], vp[1] / vp[2] - start[1]])
        else:
            direction = vp[:2].copy()
        direction = direction / np.linalg.norm(direction)
        angle = np.radians(rng.normal(0, noise_deg))
        c, s = np.cos(angle), np.sin(angle)
        direction = np.array([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]])
        end = start + length * direction
        segs[i] = [start[0], start[1], end[0], end[1]]
    return segs


def manhattan_vps(k, rotation):
    r = rotation.as_matrix()
    vps = []
    for axis in range(3):
        d_cam = r[:, axis]
        if d_cam[2] < 0:
            d_cam = -d_cam
        vps.append(k @ d_cam)
    return vps


def test_refine_recovers_clean_vanishing_point():
    rng = np.random.default_rng(0)
    rotation = Rotation.from_euler("xyz", [8, 15, 3], degrees=True)
    vps = manhattan_vps(K, rotation)
    segs = make_segments_toward_vp(vps[0], 20, rng, noise_deg=0.0)
    found = find_vanishing_points(segs, max_vps=1, min_lines_remaining=4, min_inliers=4, rng=rng)
    assert len(found) == 1
    recovered, inliers = found[0]
    assert len(inliers) == 20
    true_px = vps[0][:2] / vps[0][2]
    recovered_px = recovered[:2] / recovered[2]
    assert np.allclose(recovered_px, true_px, atol=1.0)


def test_three_manhattan_vps_are_mutually_orthogonal():
    rng = np.random.default_rng(1)
    rotation = Rotation.from_euler("xyz", [8, 15, 3], degrees=True)
    vps = manhattan_vps(K, rotation)
    segs = np.concatenate([
        make_segments_toward_vp(vps[0], 15, rng, noise_deg=0.2),
        make_segments_toward_vp(vps[1], 15, rng, noise_deg=0.2),
        make_segments_toward_vp(vps[2], 15, rng, noise_deg=0.2),
    ])
    found = find_vanishing_points(segs, max_vps=3, min_lines_remaining=6, min_inliers=8, rng=rng)
    assert len(found) == 3
    recovered = [f[0] for f in found]
    residuals = [orthogonality_cosine(recovered[a], recovered[b], K)
                 for a in range(3) for b in range(a + 1, 3)]
    assert all(r < 0.05 for r in residuals)


def test_non_orthogonal_direction_raises_residual():
    rng = np.random.default_rng(2)
    rotation = Rotation.from_euler("xyz", [8, 15, 3], degrees=True)
    vps = manhattan_vps(K, rotation)
    r = rotation.as_matrix()
    skewed_direction = r[:, 0] + 0.5 * r[:, 2]
    skewed_direction = skewed_direction / np.linalg.norm(skewed_direction)
    skewed_vp = K @ skewed_direction
    residual_clean = orthogonality_cosine(vps[0], vps[2], K)
    residual_skewed = orthogonality_cosine(skewed_vp, vps[2], K)
    assert residual_clean < 0.05
    assert residual_skewed > residual_clean + 0.2


def test_score_image_reports_insufficient_structure_when_too_few_lines():
    blank = np.zeros((480, 640), dtype=np.uint8)
    result = score_image(blank, K)
    assert result["score"] is None
    assert result["num_vps"] < 2


def test_score_image_end_to_end_on_rendered_lines():
    rng = np.random.default_rng(3)
    rotation = Rotation.from_euler("xyz", [8, 15, 3], degrees=True)
    vps = manhattan_vps(K, rotation)
    img = np.full((480, 640), 255, dtype=np.uint8)
    segs = np.concatenate([
        make_segments_toward_vp(vps[0], 15, rng, length=120.0, noise_deg=0.2),
        make_segments_toward_vp(vps[1], 15, rng, length=120.0, noise_deg=0.2),
        make_segments_toward_vp(vps[2], 15, rng, length=120.0, noise_deg=0.2),
    ])
    import cv2
    for x1, y1, x2, y2 in segs:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 0, 2)
    result = score_image(img, K, min_length_frac=0.0, iters=300, min_inliers=6, rng=rng)
    assert result["num_vps"] == 3
    assert result["score"] < 0.15


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
