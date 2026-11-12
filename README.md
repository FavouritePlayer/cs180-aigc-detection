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
