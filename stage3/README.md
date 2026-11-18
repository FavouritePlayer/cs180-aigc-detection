# Stage 3: vanishing-point / line-consistency cue

A second, independent 3D-consistency cue, complementary to the stage-2 depth-normal
residual. Classical CV only (line detection + RANSAC vanishing-point clustering), no
GPU and no new deep model: it reuses the camera intrinsics (`K`) GeoCalib already
predicted and cached during a stage-2 run, so it costs nothing to compute at full
dataset scale beyond re-decoding the images.

## Idea

Indoor scenes (your scene categories: kitchens, bathrooms, bedrooms, etc.) are heavily
"Manhattan world" — dominated by three mutually orthogonal 3D directions (wall/floor/
ceiling edges, cabinet and counter lines, tile grids, door and window frames). Under a
pinhole camera with intrinsics `K`, the vanishing points of two orthogonal 3D directions
must satisfy `vp_i^T * omega * vp_j = 0`, where `omega = inv(K @ K.T)`. This is a
purely geometric constraint, independent of the depth/normal estimators used in stage 2.

The per-image score, `vp_geometry.orthogonality_cosine`, is `|cos|` between the two
directions' back-projected 3D rays: **0 = perfectly orthogonal (expected/consistent for
a real photo under correct calibration), 1 = parallel (fully inconsistent)**. This is
the opposite sense from the stage-2 `E` residual, where 0 means "the two estimates
agree" — here 0 means "the scene's lines are geometrically consistent with being a real
Manhattan-world room." See `PROTOCOL_VP.md` for the declared direction of effect.

## Files

- `vp_geometry.py` — line detection (OpenCV LSD), sequential-RANSAC vanishing-point
  clustering, and the orthogonality residual. No I/O; pure functions over arrays.
- `test_vp_geometry.py` — synthetic-scene unit tests (known camera, known orthogonal
  directions, rendered to actual pixels and run through the real LSD detector). All
  passing; run before trusting any change to the geometry code.
- `run_vp.py` — batch runner matching the stage-2 manifest / resumable-JSONL
  conventions. Takes an existing stage-2 `per_image.jsonl` as its `--calibration`
  source so GeoCalib is never rerun; verifies the cached run's processed-canvas hash
  matches before reusing its `K`.
- `PROTOCOL_VP.md` — method and analysis plan, fixed before running on real data.

## Usage

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest test_vp_geometry.py -v

python3 run_vp.py \
  --root <path to decoded stage-2 image root> \
  --manifest <stage-2 manifest.json, e.g. ../stage2/frontier_manifest.json> \
  --calibration <stage-2 per_image.jsonl with predicted_intrinsics_px, e.g. ../results/stage2_frontier_cosine/per_image.jsonl> \
  --out <output dir>
```

## Status

Run on both existing stage-2 samples, both times reusing that stage's cached GeoCalib
`K` (no new model inference, $0 compute):

- **Frontier 100-image pilot**: [results/stage3_frontier_vp/REPORT.md](../results/stage3_frontier_vp/REPORT.md).
  AUROC 0.528 (95% bootstrap interval 0.418–0.641) — chance level. A trivial
  non-geometric baseline (raw line count) separated the classes better than the
  orthogonality cue itself (AUROC 0.637); read the confound-check section before
  trusting either number.
- **COCO indoor caption-matched pilot (50 groups, 300 images)**: [results/stage3_coco_matched_vp/REPORT.md](../results/stage3_coco_matched_vp/REPORT.md).
  Matched AUROC 0.418 (95% caption-group bootstrap interval 0.346–0.489) — **entirely
  below chance**, reversed from the declared hypothesis, and this time the line-count
  confound check comes back at chance (0.503), so it can't explain the result. All five
  generators point the same reversed direction here, unlike the stage-2 depth-normal
  cue on this identical sample, which only reversed for the open-diffusion family.
  Two independent geometric cues now agree that generated images look more
  self-consistent than real ones on this dataset — the dataset-level square/non-square
  confound already flagged for the depth-normal result still applies and is not ruled
  out by this run.

`fetch_frontier_images.py` shows how the Frontier images were obtained (public,
hash-verified, CC BY 4.0). The COCO-matched archive is user-provided and was supplied
directly, verified against its recorded SHA-256 before use, and is not redistributed
in Git.
