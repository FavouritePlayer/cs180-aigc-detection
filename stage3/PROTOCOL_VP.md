# Vanishing-point orthogonality: method fixed before running on real data

This mirrors the house convention used for `stage2/PROTOCOL_COSINE.md`: declare the
method, hyperparameters, and direction of effect before running on any labeled real
vs. generated sample, so nothing here can be tuned after seeing a result.

## Pipeline

1. Decode the image, resize with PIL LANCZOS to the same 1024-long-edge canvas used by
   the stage-2 run supplying `K` (no crop, aspect-preserving). Verify the processed-RGB
   SHA-256 matches that stage-2 run's cached value for the same image id before reusing
   its `K` — if it does not match, the image is skipped, not re-processed with a
   different canvas.
2. Convert to grayscale. Detect line segments with OpenCV's LSD
   (`cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)`). Drop segments shorter than
   `min_length_frac = 0.02` of the image diagonal.
3. Find up to `max_vps = 3` dominant vanishing points by sequential RANSAC: sample two
   segments, intersect their homogeneous lines, score inliers by angular error between
   each segment's direction and the direction toward the candidate point
   (`inlier_threshold_deg = 2.0`), keep the best of `ransac_iters = 500` trials, refine
   by least squares over its inliers (right null vector of the stacked line-coefficient
   matrix), remove those inliers, and repeat on what remains. Stop early if fewer than
   `min_lines_remaining = 6` segments remain, or if no candidate reaches
   `min_inliers = 6`.
4. `K` is **not** re-estimated here; it is read from the named stage-2
   `per_image.jsonl`'s `predicted_intrinsics_px` field for the same image id.
5. For every pair of found vanishing points, compute `|cos|` between their
   back-projected 3D ray directions under `K` (`omega = inv(K @ K.T)`,
   `cos = (vp_i @ omega @ vp_j) / sqrt((vp_i @ omega @ vp_i) * (vp_j @ omega @ vp_j))`).
   The per-image score is the arithmetic mean of this value over all found pairs (1
   pair if 2 vanishing points are found, 3 pairs if 3 are found).
6. Images with fewer than 2 vanishing points found get `score = null` and are reported
   as insufficient-structure, not dropped silently and not imputed.

No post-hoc choice of threshold, no per-image hyperparameter routing, no smoothing or
masking beyond the length and inlier-angle filters above.

## Declared direction of effect

**Higher score (further from orthogonal, closer to parallel) is fixed as indicating
generated**, on the reasoning that a real photo's rectilinear structure was produced by
an actual orthogonal room under one consistent camera, while a generator has no
explicit mechanism enforcing that its depicted lines obey the calibration GeoCalib
infers for the image. This is declared now and will not be flipped after seeing
results, matching every prior stage-2 report.

## What this does and does not establish

- This measures whether the detected line structure is consistent with the Manhattan-
  world assumption *under the K that GeoCalib already predicted for this image*. It is
  not an independent recalibration, and it inherits any error in that cached `K`.
- Coverage (fraction of images with >= 2 vanishing points found) must be reported
  per run and, given the scene-type imbalance already known in the COCO-indoor
  dataset, broken out by scene type — tile-heavy bathrooms/kitchens are expected to
  have far higher coverage than cluttered bedrooms or the "other" bucket.
- This is a different failure mode from the stage-2 depth-normal residual (no depth or
  normal network involved), so it is a candidate for combination with that score, not
  a replacement for it.
- As with every stage-2 pilot: report AUROC with a bootstrap interval, do not flip
  the declared direction after seeing results, do not train a classifier or pick a
  threshold on the same images used to report the headline number, and prefer the
  existing caption-matched 50-group sample (`stage2/coco_matched_manifest.json`) so
  results are directly comparable to the depth-normal paired-difference numbers
  already reported in `results/stage2_coco_matched_300/REPORT.md`.

## Open before a real run

- Confirm access to the decoded image bytes for whichever manifest is used (not
  stored in this repo) and to the corresponding stage-2 `per_image.jsonl` for `K`.
- Decide whether `min_length_frac`, `inlier_threshold_deg`, and `min_inliers` need
  retuning once real LSD output on real photos is inspected — synthetic tests only
  confirm the math is correct, not that these particular thresholds are well-chosen
  for this dataset's image resolution and content.
