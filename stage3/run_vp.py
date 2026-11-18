"""Vanishing-point orthogonality residual, reusing GeoCalib K already cached by a stage-2 run."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vp_geometry import score_image

SETTINGS = {
    "canvas_long_edge": 1024,
    "resize": "PIL LANCZOS, no crop, aspect preserving (matches stage2 canvas convention)",
    "intrinsics_source": "reused predicted_intrinsics_px from the given --calibration per_image.jsonl, GeoCalib not rerun",
    "line_detector": "OpenCV LSD (cv2.createLineSegmentDetector, LSD_REFINE_STD)",
    "min_length_frac": 0.02,
    "max_vps": 3,
    "min_lines_remaining": 6,
    "inlier_threshold_deg": 2.0,
    "ransac_iters": 500,
    "min_inliers": 6,
    "residual": "mean over found VP pairs of |cos| between back-projected ray directions under K",
    "score_direction": "higher = generated (fixed before running on real data; see PROTOCOL_VP.md)",
    "seed": "first 32 bits of SHA256(image id)",
}


def load_calibration(path):
    rows = (json.loads(line) for line in Path(path).read_text().splitlines())
    out = {}
    for row in rows:
        if "predicted_intrinsics_px" in row and row.get("processed_rgb_sha256"):
            out[row["id"]] = row
    return out


def resize_canvas(image, long_edge):
    w, h = image.size
    s = long_edge / max(w, h)
    return image.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="directory containing the manifest's image files")
    p.add_argument("--manifest", required=True, help="stage2-style manifest JSON (list of rows with file/sha256/id)")
    p.add_argument("--calibration", required=True, help="existing stage2 per_image.jsonl with predicted_intrinsics_px")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration(args.calibration)
    (out / "settings.json").write_text(json.dumps(SETTINGS, indent=2) + "\n")

    record = out / "per_image.jsonl"
    previous = [json.loads(s) for s in record.read_text().splitlines()] if record.exists() else []
    assert len({r["id"] for r in previous}) == len(previous)
    completed = {r["id"] for r in previous}

    rows = json.loads(Path(args.manifest).read_text())
    for row in rows:
        if row["id"] in completed:
            continue
        calib_row = calibration.get(row["id"])
        if calib_row is None:
            print("SKIP", row["id"], "no cached calibration", flush=True)
            continue

        f = root / row["file"]
        assert hashlib.sha256(f.read_bytes()).hexdigest() == row["sha256"]
        image = Image.open(f).convert("RGB")
        image = resize_canvas(image, SETTINGS["canvas_long_edge"])
        w, h = image.size
        rgb = np.array(image)
        processed_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
        assert processed_sha256 == calib_row["processed_rgb_sha256"], (
            "canvas does not match the cached calibration's processed RGB; "
            "the K from that run cannot be reused for this image"
        )

        seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        k = np.array(calib_row["predicted_intrinsics_px"])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        start = time.time()
        result = score_image(
            gray, k,
            min_length_frac=SETTINGS["min_length_frac"],
            max_vps=SETTINGS["max_vps"],
            min_lines_remaining=SETTINGS["min_lines_remaining"],
            inlier_threshold_deg=SETTINGS["inlier_threshold_deg"],
            iters=SETTINGS["ransac_iters"],
            min_inliers=SETTINGS["min_inliers"],
            rng=rng,
        )
        record_row = {
            **row,
            "seed": seed,
            "canvas_width": w,
            "canvas_height": h,
            "processed_rgb_sha256": processed_sha256,
            **result,
            "seconds": time.time() - start,
        }
        with record.open("a") as file:
            file.write(json.dumps(record_row) + "\n")
        print(row["id"], result["score"], "num_vps", result["num_vps"], flush=True)

    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
