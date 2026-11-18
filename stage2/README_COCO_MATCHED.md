# COCO indoor caption-matched pilot

[Results](../results/stage2_coco_matched_300/REPORT.md) · [Frozen protocol](PROTOCOL_COCO_MATCHED.md) · [Selected image manifest](coco_matched_manifest.json).

The user approved a 300-image pilot within $1 rather than inference on all 16452 archive images. The pilot contains 50 real photos and five generated counterparts per caption group (250 AI total). Tags supplied by the archive are sd21, sdxl, sd3, dalle3 and midjourney. Sampling uses test-only groups, one instance per caption ID, excluding caption IDs/decoded images occurring in train/validation and repeated selected decoded images. No model scores influence selection.

All model settings and the inference implementation are identical to the [supplied cosine-method Frontier run](README_COSINE.md): DAV2 Metric Outdoor Large is primary, Indoor Large secondary, GeoCalib pinhole, Marigold Normals v1.1 fp16 four DDIM steps/ensemble ten, one-pixel forward differences, mean `1 - clip(dot(ND,NP),-1,1)`. Higher score indicates generated. No oracle intrinsics, ground-truth scale/rotation, test-label routing, threshold tuning, confidence/sky/object-edge masks or smoothing.

Every stored image is 256×256. We preserve the prior preprocessing: proportional LANCZOS resize to a 1024-long-edge RGB canvas. This upsamples the stored image and creates no additional source detail. GeoCalib estimates focal calibration but fixes the principal point to the canvas center. Original dimension metadata is nonsquare for 49 selected real photos and square for all generated images; the archive does not establish whether original images were cropped or stretched to its stored square format.

## Reproduction

Original images and captions are user-provided and are not redistributed in Git. Obtain `coco_indoor_all.zip` from the data owner and verify the SHA-256 below. The inventory JSON contains captions and is for local use only; keep it under an ignored `data/` directory or outside the repository.

```bash
python stage2/inventory_coco_archive.py coco_indoor_all.zip --out data/coco_inventory.json
python stage2/sample_coco_matched.py data/coco_inventory.json \
  --archive-sha256 c847a0655de51d16724852da1dbe155b0998b8941f80c8a47b2577b8dc71a089 \
  --out reproduced_manifest.json
# Compare reproduced_manifest.json with stage2/coco_matched_manifest.json.
# Extract the archive locally under data/, preserving coco_indoor_all/images/... .
for phase in calibration outdoor indoor normals; do
  python stage2/run_frontier_cosine.py --phase "$phase" --root data/coco_indoor_all \
    --manifest stage2/coco_matched_manifest.json --sources sources --weights models \
    --out results/stage2_coco_matched_300
done
python stage2/analyze_coco_matched.py results/stage2_coco_matched_300
python -m unittest discover -s stage2 -p 'test_*.py'
```

Use a fresh output directory for a new reproduction. Package pins, checkpoint layout and sources are as in [README_COSINE](README_COSINE.md); current hashes/package versions are saved with this result. All named GeoCalib parameters are verified against checkpoint tensors after the official loader; both DAV2 checkpoints load strictly. Model inputs are RGB only. Per-image results record input/processed-pixel hashes, seed, estimated K, coverage and both depth-checkpoint scores. Raw `.npz` maps are stored locally and excluded from Git.

## Matched analysis

The primary pooled AUROC compares 50 real with 250 generated images. AUROC chance remains 0.5 despite unequal class counts; class prevalence is not classification accuracy. Each generator has 50 generated images compared with the same 50 real references, so the pooled AUROC must equal the mean of the five generator AUROCs. This identity is checked.

The 95% intervals resample complete caption groups (2000 replicates, seed 180), preserving the real image and all five generated counterparts together. The same resampled group indices are used for generator-specific intervals. Image-independent bootstrapping would ignore shared content and the repeated real references. The paired diagnostic subtracts each real score from the mean of its five generated counterparts and averages over groups. It is an analysis contrast, not a deployable image score requiring a known real counterpart.

Generator intervals are descriptive and not adjusted for multiple comparisons. Caption matching controls prompt content, not exact viewpoint, camera, scene composition or rendering complexity. This is a small pilot on the archive's older generator tags, not the complete archive or a test of unseen generator families. Fifty real images cannot establish 1% FPR performance (empirical steps are 2%). Source transforms, calibration errors, texture sensitivity and estimator training overlap remain limitations.
