# Stage 2: Frontier 100-image disagreement pilot

[Results](../results/stage2_frontier_100/REPORT.md) · [Protocol fixed before inference](PROTOCOL.md).

The hypothesis is that depth-derived normals disagree more with an independent normal estimator on generated images than on real photos. This pilot tests MoGe-2 Large and Marigold Normals v1.1 on 50 real + 50 generated original images from the [Frontier Image & Video Benchmark (2026–10)](https://huggingface.co/datasets/wasitaigeneratedcom/frontier-image-video-bench-2026-10). It does not measure reference depth/normal accuracy: these images have no such ground truth.

The primary score is mean angular disagreement with a four-pixel central-difference radius on a shared 1024-long-edge canvas. Camera calibration comes from MoGe, without oracle intrinsics. The score direction is fixed as higher = generated. No trained classifier, test-set threshold tuning, label-conditioned preprocessing, or ground-truth scale fitting is used. See the protocol for masks, secondary diagnostics and uncertainty estimation.

## Reproduce

Use the same pinned MoGe and utils3d-moge sources as [stage 1 round 2](../results/stage1_round2/source_revisions.json). Put these at `sources/moge` and `sources/utils3d`. Exact checkpoint repository revisions, filenames and verified hashes are in `results/stage2_frontier_100/checkpoint_verification.json`. Weights go under `models/moge/model.pt` and the Diffusers-compatible directory `models/marigold`. The runner requires strict loading of all MoGe checkpoint keys.

Use the stage-1 PyTorch 2.5.1/CUDA 12.4 runtime, torchvision 0.20.1 and NumPy 2.1.2. Additional pinned packages:

```text
opencv-python-headless==4.10.0.84
diffusers==0.35.2
transformers==4.49.0
accelerate==1.12.0
timm==1.0.15
einops==0.8.1
scipy==1.15.2
matplotlib==3.10.0
huggingface_hub==0.36.0
safetensors==0.5.3
sentencepiece==0.2.0
ftfy==6.3.1
regex==2024.11.6
```

Full runtime versions are committed alongside results. Download only `file` paths from `frontier_manifest.json`, using its pinned dataset revision, to `data/images/...`; verify the original byte SHA-256. Preserve each file's license and attribution. The manifest includes all source links and credits. Neither image bytes nor model weights are committed.

```bash
# To regenerate selection from the benchmark's metadata.csv:
python stage2/sample_frontier.py metadata.csv \
  --revision e6a8b83d15fab1da958848368fcd07e943afafd0 --out selected.json
# The committed manifest additionally records verified original hashes and dimensions.
python stage2/run_frontier.py --phase depth --root data \
  --manifest stage2/frontier_manifest.json --sources sources --weights models \
  --out results/stage2_frontier_100
python stage2/run_frontier.py --phase normals --root data \
  --manifest stage2/frontier_manifest.json --sources sources --weights models \
  --out results/stage2_frontier_100
python stage2/analyze_frontier.py results/stage2_frontier_100
# Optional, while the saved maps are available:
python stage2/plot_frontier_examples.py --root data --results results/stage2_frontier_100
python -m unittest discover -s stage2 -p 'test_*.py'
```

For a fresh reproduction, use a new output directory. The runners resume existing records by image ID; do not resume with different weights/settings. Model maps are stored in `maps/*.npz` for local inspection and are excluded from Git. Each per-image result records original/processed-pixel hashes, predicted camera, deterministic seed, coverage and both stencil scores.
