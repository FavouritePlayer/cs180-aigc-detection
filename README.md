# Geometric consistency for generated-image detection

Stage 1 evaluates depth, normals, and camera calibration on a fixed DIODE validation sample: 100 indoor + 100 outdoor images per model.

| Model | Indoor AbsRel ↓ | Outdoor AbsRel ↓ |
| --- | --- | --- |
| DAV2 Metric Large | 0.211 | 0.740 |
| DepthART Metric Large + known K | 0.300 | 0.759 |
| TR2M + BLIP captions | 0.295 | 0.532 |

[Full results and plots](results/stage1/REPORT.md) · [Protocol and reproduction](stage1/README.md)

Marigold Normals v1.1 and GeoCalib are also evaluated in the report. Both HyDen checkpoints were gated and could not be tested. Stage 2 and GenImage were not run.
