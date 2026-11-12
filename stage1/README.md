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

Normals are normalized before angular evaluation, with signed dot products (180-degree reversals are errors). Zero reference normals are excluded. Mean and median angles are calculated per image. Depth-derived normals use central differences with stencils of radius 1 and 4 native pixels, published DIODE intrinsics `(886.81, 927.06, 512, 384)`, and camera-facing normals in right/up/toward-viewer coordinates. No per-image normal rotation or GT-fitted correction is allowed. Geometry comparisons require valid depth at the center and all four stencil neighbors, plus a valid reference normal. The report also compares normals derived from measured depth to the supplied measured normals to measure disagreement introduced by this derivative operator. These are exploratory geometry diagnostics and are sensitive to pixel-scale derivatives and discontinuities.

## Models and conditions

- **Depth Anything V2 Metric Large:** official `metric_depth` code, ViT-L, 518 input-size setting, Hypersim checkpoint with max_depth=20 indoors and VKITTI checkpoint with max_depth=80 outdoors.
- **DepthART Metric Large:** official metric inference path and domain-specific L448 checkpoints, official CLI defaults (480x640 indoor target and 448x448 outdoor target, aspect-preserving lower-bound resize). Uses **ground-truth published camera intrinsics**; this is a known-calibration benchmark.
- **TR2M + BLIP captions:** released head with Depth Anything ViT-S, DINOv2 ViT-L, CLIP ViT-L/14; official demo resolution 434x560. Uses the DINOv2 implementation bundled with the official TR2M repository for both encoders; the relative encoder is initialized without a redundant DINO checkpoint before loading the complete Depth Anything weights. Captions are generated from RGB by `Salesforce/blip-image-captioning-base`, greedy decoding up to 50 new tokens, and saved; `--captions` replays those exact captions after checking image identities. This is a reproducible caption condition, not a reproduction of the paper's caption setup.
- **Marigold Normals v1.1:** diffusers pipeline, FP16, DDIM, 4 steps, processing resolution 768, ensemble size 10, per-image deterministic seed derived from the path. Uses the authors' 4-step / ensemble-10 DIODE settings on our fixed sample.
- **HyDen Metric Point / Surface Normal:** public metadata is available, but both weight repositories require authenticated approved access. No replacement model is silently substituted.
- **GeoCalib pinhole:** evaluated separately against the published DIODE focal lengths. Focal AbsRel is the mean of `abs(fx_pred-fx_gt)/fx_gt` and `abs(fy_pred-fy_gt)/fy_gt`, averaged across images. The principal point is fixed at the image center by the model. Main depth and geometry scores continue to use known intrinsics; calibration-error propagation is not measured.

## Reproduce

Install the runtime dependencies, clone each official source repository, and download the listed checkpoints. GeoCalib uses [the official pinhole v1.0 weights](https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar). For DepthART, build the official `deploy/shared/selective_scan` CUDA extension for the target GPU (SM86 on the tested RTX 3090). Extract both validation archives into a directory containing `val/`. Model adapter CLIs expose their dataset, source, weight, manifest, and output paths.

```sh
python -m unittest discover -s stage1 -p 'test_*.py'
python stage1/sample.py --root /data/diode --output stage1/manifest.json
python stage1/run_dav2.py --root /data/diode --source /src/Depth-Anything-V2 --weights /models --manifest stage1/manifest.json --out results/stage1/dav2
python stage1/run_depthart.py --root /data/diode --source /src/DepthART --weights /models --manifest stage1/manifest.json --out results/stage1/depthart
python stage1/run_tr2m.py --root /data/diode --source /src/TR2M --manifest stage1/manifest.json --out results/stage1/tr2m
python stage1/run_marigold.py --root /data/diode --manifest stage1/manifest.json --depth-results results/stage1 --out results/stage1/marigold
python stage1/run_geocalib.py --root /data/diode --source /src/GeoCalib --weights /models/geocalib-pinhole.tar --manifest stage1/manifest.json --out results/stage1/geocalib
python stage1/summarize.py results/stage1
python stage1/summarize_normals.py results/stage1
python stage1/summarize_calibration.py results/stage1
```

Use a fresh output directory for each run; JSONL files append to preserve completed work. Check output coverage and uniqueness before interpreting summaries. Large predictions, datasets, and weights are excluded from Git. Per-image `seconds` includes prediction saving and reference-array I/O and is not a pure GPU latency benchmark. `expected_weights.json` records public Hugging Face LFS SHA-256 hashes; `verify_weights.py` checks actual checkpoint bytes against them. `download.py` supports validated parallel byte ranges, and `prefetch_hf.py` fetches only inference files when the default model downloader is slow. Both model adapters accept local pretrained-model paths.

Stage 2, GenImage, and optional NYU are not run. No conclusion about real-versus-generated discrimination follows from stage 1 alone.


## Additional RGB-only depth models (round 2)

[Results, checkpoint hashes, and the separate $1 budget ledger](../results/stage1_round2/REPORT.md).
The same frozen 200-image manifest and metrics are used. Model inputs contain RGB only: no measured camera intrinsics, EXIF focal length, depth, or fitted scale. Known DIODE intrinsics are used afterward only to derive normals for evaluation. The older DepthART result uses known intrinsics and TR2M uses an automatically generated BLIP caption; these are labeled in comparisons.

Source commits are pinned in `results/stage1_round2/source_revisions.json`; checkpoint URLs/revisions and SHA-256 values are in `checkpoint_integrity.json`. Clone each official source at that commit. Put checkpoints under `models/depthpro/depth_pro.pt`, `models/unidepth/pytorch_model.bin` (with `config.json`), and `models/moge/model.pt`. Put the pinned `utils3d-moge` checkout alongside the MoGe checkout, in a folder named `utils3d`.

Run in Python 3.11, PyTorch 2.5.1/CUDA 12.4 with torchvision 0.20.1. Additional inference packages: `opencv-python-headless==4.10.0.84 timm==1.0.15 einops==0.8.1 scipy==1.15.2 huggingface_hub==0.36.0 safetensors==0.5.3 matplotlib==3.10.0 wandb==0.19.11 pillow_heif==0.22.0`. Base NumPy is 2.1.2; Depth Pro runs in a separate environment with NumPy 1.26.4 and the same Torch installation. Exact installed versions are in `runtime_provenance.json`. No optional UniDepth KNN/xformers kernel is required for inference.

```bash
# Change --model and --source to run moge2 or depthpro. Use Depth Pro's own Python environment.
WANDB_MODE=disabled python stage1/run_more_depth.py --model unidepthv2 \
  --root data --source sources/unidepth --weights models \
  --manifest stage1/manifest.json --out results/stage1_round2/unidepthv2
python stage1/validate_results.py results/stage1_round2
python stage1/summarize.py results/stage1_round2
python stage1/summarize_more_depth.py results/stage1_round2
```

Use `--resume` only for the same checkpoint/settings, to skip already completed image paths. Predictions are saved as `.npy` files locally but excluded from Git. All 200 records per model, including raw metrics, diagnostic median alignment, camera predictions, and ground-truth-normal errors, are committed. Median-aligned fields are explicitly diagnostic and are not used in headline model rankings. MoGe's learned metric scale is part of inference; its predicted validity mask is recorded but does not remove any ground-truth-valid evaluation pixels (`apply_mask=False`).
