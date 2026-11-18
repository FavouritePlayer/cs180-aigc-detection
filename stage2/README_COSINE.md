# Supplied cosine-residual method

[Results](../results/stage2_frontier_cosine/REPORT.md) · [Frozen protocol](PROTOCOL_COSINE.md).

This follow-up applies the supplied Depth Anything V2 Metric + GeoCalib + Marigold equations to the same 50 real / 50 AI Frontier originals. The primary method uses DAV2 Metric Outdoor Large for every image; Indoor Large is a sensitivity check for every image, without routing by labels or selecting the better result afterward.

For axial depth Z and predicted intrinsics K, P=Z K^-1 [u,v,1]^T. Tangents are forward differences with one-pixel displacement, exactly as supplied. Their unit cross product is oriented toward the camera and converted from OpenCV axes to Marigold's right/up/toward-viewer convention. The score is the mean over valid pixels of `1 - clip(dot(ND, NP), -1, 1)`. There is no absolute dot product or ground-truth correction. The final row/column and invalid/degenerate stencils are masked. No learned confidence, sky or edge mask is used.

GeoCalib's pinhole model predicts focal calibration while assuming the principal point is centered. It does not independently learn all of fx, fy, cx and cy. Its official wrapper undoes its internal scaling/cropping before returning K. We verify every named GeoCalib parameter matches the official checkpoint after its flexible loader. Both DAV2 checkpoints load strictly.

Use the same dataset manifest and license/attribution records as [the initial Frontier pilot](README.md). All three models receive identical RGB canvas pixels: aspect-preserving LANCZOS resize to longest edge 1024. DAV2 uses official fp32 `infer_image(input_size=518)`; GeoCalib uses official pinhole calibration without priors; Marigold uses fp16, four DDIM steps, ensemble ten, resolution 768, matching canvas resolution and deterministic image-ID seeds. No EXIF, GT camera or fitted depth scale.

## Reproduce

Use Python 3.11, PyTorch 2.5.1+cu124, torchvision 0.20.1. Pin inference packages as in [the initial pilot](README.md), with NumPy 1.26.4, kornia 0.8.1 and kornia_rs 0.1.9. Full installed versions and GPU metadata are saved with results. Checkpoint URLs/revisions, byte sizes, expected HF LFS hashes and actual hashes are in `checkpoint_verification.json`. GeoCalib's release has no published SHA-256 here; its size/PyTorch ZIP CRC are checked and its actual hash recorded.

Pinned source commits are in `source_revisions.json`. Put DAV2 sources under `sources/dav2` (containing `metric_depth/`) and GeoCalib under `sources/geocalib` (containing `geocalib/`). Weights go to `models/dav2/depth_anything_v2_metric_{vkitti,hypersim}_vitl.pth`, `models/geocalib/geocalib-pinhole.tar`, and Diffusers-compatible `models/marigold/`. Download selected dataset originals to `data/images/...`, verifying the committed manifest's byte hashes.

```bash
# Use a fresh output directory for a fresh reproduction; existing IDs resume automatically.
for phase in calibration outdoor indoor normals; do
  python stage2/run_frontier_cosine.py --phase "$phase" --root data \
    --manifest stage2/frontier_manifest.json --sources sources --weights models \
    --out results/stage2_frontier_cosine
done
python stage2/analyze_frontier_cosine.py results/stage2_frontier_cosine
python stage2/plot_frontier_cosine.py --root data --results results/stage2_frontier_cosine \
  --manifest stage2/frontier_manifest.json
python -m unittest discover -s stage2 -p 'test_*.py'
```

The primary AUROC direction remains higher residual = generated. Angular scores on the new pipeline and Indoor Large cosine scores are secondary diagnostics. The previous MoGe/radius-4/angular result is preserved separately. Several components change together here, so a performance change cannot be attributed solely to the residual formula. Reusing this previously inspected sample makes this exploratory, not independent held-out validation.
