# Stage 1 protocol

Evaluate on 100 indoor and 100 outdoor RGB images from **DIODE validation**, selected independently within each domain by ordering SHA-256 hashes of `cs180-stage1-v1:<relative PNG path>`. `manifest.json` contains all paths and RGB checksums. There are 325 indoor and 446 outdoor validation images. This is an exploratory sample from three scenes per domain, not 200 independent scenes.

Official data: https://diode-dataset.org/ and https://github.com/diode-dataset/diode-devkit.

Archive checksums (MD5, from the dataset authors):

- `val.tar.gz`: `5c895d09201b88973c8fe4552a67dd85`
- `val_normals.tar.gz`: `323ccaf60abebdb59705dcd8b529d709`

## Metrics

Depth is scored at the native ground-truth resolution, on the supplied sensor-valid mask with finite positive reference depth. Ground truth is never interpolated. Primary metric is per-image mean `abs(pred-gt)/gt`, averaged with equal image weight within each domain. Also report RMSE (meters), delta1 (fraction within a multiplicative factor 1.25), and the distribution of per-image AbsRel.

Three masks are reported: **full** uses every sensor-valid reference pixel; **capped** restricts reference depths to <20 m indoors / <80 m outdoors, the DAV2 metric checkpoint ranges. **common** uses <10 m indoors / <80 m outdoors, accommodating DepthART's 10 m indoor model range. The same masks apply to every depth model. Predictions are neither aligned nor clipped for primary scores. Nonfinite or nonpositive predictions on the mask raise an error rather than disappearing from evaluation.

`median_aligned_abs_rel` is a separate diagnostic after multiplying predicted depth by `median(gt)/median(pred)` on the same evaluation mask. It uses ground truth and is not a deployable accuracy result. Constant scale cancels in depth-derived normals, so raw scale accuracy alone cannot settle the later consistency-detection question.

Normals are normalized before angular evaluation, with signed dot products (180-degree reversals are errors). Zero reference normals are excluded. Mean and median angles are calculated per image. Depth-derived normals use central differences with stencils of radius 1 and 4 native pixels, published DIODE intrinsics `(886.81, 927.06, 512, 384)`, and camera-facing normals in right/up/toward-viewer coordinates. No per-image normal rotation or GT-fitted correction is allowed. These are exploratory geometry diagnostics and are sensitive to pixel-scale derivatives and discontinuities.

## Models and conditions

- **Depth Anything V2 Metric Large:** official `metric_depth` code, ViT-L, 518 input-size setting, Hypersim checkpoint with max_depth=20 indoors and VKITTI checkpoint with max_depth=80 outdoors.
- **DepthART Metric Large:** official metric inference path and domain-specific L448 checkpoints, official CLI defaults (480x640 indoor target and 448x448 outdoor target, aspect-preserving lower-bound resize). Uses **ground-truth published camera intrinsics**; this is a known-calibration benchmark.
- **TR2M + BLIP captions:** released head with Depth Anything ViT-S, DINOv2 ViT-L, CLIP ViT-L/14; official demo resolution 434x560. Captions are generated from RGB by `Salesforce/blip-image-captioning-base`, greedy decoding up to 50 new tokens, and saved. This is a reproducible caption condition, not a reproduction of the paper's caption setup.
- **Marigold Normals v1.1:** diffusers pipeline, FP16, DDIM, 4 steps, processing resolution 768, ensemble size 10, per-image deterministic seed derived from the path. Uses the authors' 4-step / ensemble-10 DIODE settings on our fixed sample.
- **HyDen Metric Point / Surface Normal:** public metadata is available, but both weight repositories require authenticated approved access. No replacement model is silently substituted.
- **GeoCalib:** predicts calibration rather than depth. DIODE's known intrinsics are used in stage 1; calibration-estimation error propagation remains separate future work.

## Reproduce

Install the runtime dependencies, clone each official source repository, and download the listed checkpoints. Extract both validation archives into a directory containing `val/`. Model adapter CLIs expose their dataset, source, weight, manifest, and output paths.

```sh
python -m unittest discover -s stage1 -p 'test_*.py'
python stage1/sample.py --root /data/diode --output stage1/manifest.json
python stage1/run_dav2.py --root /data/diode --source /src/Depth-Anything-V2 --weights /models --manifest stage1/manifest.json --out results/stage1/dav2
python stage1/run_depthart.py --root /data/diode --source /src/DepthART --weights /models --manifest stage1/manifest.json --out results/stage1/depthart
python stage1/run_tr2m.py --root /data/diode --source /src/TR2M --manifest stage1/manifest.json --out results/stage1/tr2m
python stage1/run_marigold.py --root /data/diode --manifest stage1/manifest.json --depth-results results/stage1 --out results/stage1/marigold
python stage1/summarize.py results/stage1
```

Use a fresh output directory for each run; JSONL files append to preserve completed work. Check output coverage and uniqueness before interpreting summaries. Large predictions, datasets, and weights are excluded from Git.

Stage 2, GenImage, and optional NYU are not run. No conclusion about real-versus-generated discrimination follows from stage 1 alone.
