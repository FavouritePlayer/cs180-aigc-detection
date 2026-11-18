# Geometric consistency for generated-image detection

Stage 1 evaluates depth, normals, and camera calibration on a fixed DIODE validation sample: 100 indoor + 100 outdoor images per model. Headline depth scores use raw metric predictions, with no ground-truth scale correction.

| Model | Indoor AbsRel ↓ | Outdoor AbsRel ↓ |
| --- | ---: | ---: |
| DAV2 Metric Large | 21.14% | 74.00% |
| DepthART Metric Large + known K | 29.97% | 75.86% |
| TR2M + BLIP captions | 29.50% | 53.16% |
| UniDepthV2 Large | 16.94% | 103.00% |
| MoGe-2 Large | 25.39% | 38.13% |
| Depth Pro | 18.79% | 75.43% |

[Three new models: results, geometry checks, and separate $1 budget](results/stage1_round2/REPORT.md) · [Original results](results/stage1/REPORT.md) · [Protocol and reproduction](stage1/README.md)

Marigold Normals v1.1 and GeoCalib were evaluated in the original report. Both HyDen checkpoints were gated and could not be tested. A [stage-2 Frontier pilot](results/stage2_frontier_100/REPORT.md) now tests MoGe-2/Marigold disagreement on 50 real + 50 generated originals. GenImage has not been run.

Stage-2 primary AUROC: **0.330 (95% bootstrap interval 0.227–0.441)**, with the fixed direction higher disagreement = generated. The predeclared signal runs in the wrong direction on this sample: higher disagreement tends to occur on real images. This does not support using high disagreement as a generated-image detector. We did not flip the score to present a better result.

Supplied cosine-method follow-up: [DAV2 Metric + GeoCalib + Marigold, one-pixel forward differences](results/stage2_frontier_cosine/REPORT.md). Primary Outdoor Large mean-residual AUROC: **0.481 (95% bootstrap interval 0.365–0.601)**. Higher residual = generated, fixed before this run. This reuses the same 100 images and is exploratory.

COCO indoor matched pilot: [50 caption groups, 300 images](results/stage2_coco_matched_300/REPORT.md). Primary pooled mean-residual AUROC **0.429 (95% caption-group bootstrap interval 0.366–0.495)**, using the unchanged supplied cosine method.

[Stage 3](stage3/README.md) adds an independent cue: vanishing-point orthogonality from detected lines, reusing GeoCalib's already-cached intrinsics (no GPU, no new deep model, $0 compute both times). [Frontier 100-image pilot](results/stage3_frontier_vp/REPORT.md): AUROC **0.528 (95% bootstrap interval 0.418–0.641)**, chance, and a trivial non-geometric baseline (raw line count) beat the geometric cue (AUROC 0.637) — read the confound section before trusting either number. [COCO indoor matched pilot](results/stage3_coco_matched_vp/REPORT.md): matched AUROC **0.418 (95% caption-group bootstrap interval 0.346–0.489)**, entirely below chance and reversed from the declared hypothesis, with the line-count confound ruled out this time (AUROC 0.503) and all five generators pointing the same reversed direction. This matches the reversed direction of the stage-2 depth-normal result on this identical sample — two independent 3D cues now agree generated images look *more* self-consistent than real ones here, which points at the dataset (see the square/non-square confound already flagged) rather than at one pipeline.
