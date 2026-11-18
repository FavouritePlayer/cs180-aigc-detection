"""Analyze the stage-3 vanishing-point run on the frozen 50-group / 300-image COCO-matched pilot."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage2"))
from detection_metrics import auroc, bootstrap_auc
from matched_metrics import matched_summary

p = argparse.ArgumentParser()
p.add_argument("results")
p.add_argument("--manifest", default="../stage2/coco_matched_manifest.json")
a = p.parse_args()
root = Path(a.results)
rows = [json.loads(s) for s in (root / "per_image.jsonl").read_text().splitlines()]
expected = {r["id"]: r for r in json.loads(Path(a.manifest).read_text())}

assert len(rows) == 300 and len({r["id"] for r in rows}) == 300
assert set(expected) == {r["id"] for r in rows}
for r in rows:
    for key in ["file", "label", "generator", "sha256", "group_id", "archive_sha256"]:
        assert r[key] == expected[r["id"]][key]

generators = ["sd21", "sdxl", "sd3", "dalle3", "midjourney"]
groups = {}
for r in rows:
    groups.setdefault(r["group_id"], {})[r["generator"]] = r
assert len(groups) == 50 and all(set(g) == set(["real"] + generators) for g in groups.values())

missing = [r for r in rows if r["score"] is None]
missing_groups = sorted({r["group_id"] for r in missing})
complete_keys = sorted(k for k in groups if k not in missing_groups)

scored = [r for r in rows if r["score"] is not None]
labels_scored = np.array([r["label"] == "ai" for r in scored], dtype=int)

summary = {
    "images": 300,
    "real": 50,
    "ai": 250,
    "caption_groups": 50,
    "groups_with_a_missing_score": missing_groups,
    "images_with_no_score_insufficient_structure": [
        {"id": r["id"], "label": r["label"], "generator": r["generator"],
         "group_id": r["group_id"], "num_vps": r["num_vps"], "num_lines": r["num_lines"]}
        for r in missing
    ],
    "primary": "mean |cos| orthogonality residual over found vanishing-point pairs",
    "direction": "higher residual = generated, fixed before this run, not flipped",
    "score_range": [0, 1],
    "generator_order": generators,
    "pooled_auroc_all_scored_images": auroc(labels_scored, [r["score"] for r in scored]),
    "pooled_auroc_95pct_stratified_image_bootstrap": bootstrap_auc(labels_scored, [r["score"] for r in scored]),
    "note": "pooled AUROC above uses every scored image (49 real + 250 ai); "
            "the caption-matched analysis below excludes the one incomplete group",
}

keys = complete_keys
real = np.array([groups[k]["real"]["score"] for k in keys])
fake = np.array([[groups[k][g]["score"] for g in generators] for k in keys])
stat = matched_summary(real, fake)
stat["per_generator"] = {
    g: {"auroc": stat["per_generator_auroc"][j], "auroc_ci95": stat["per_generator_auroc_ci95"][j],
        "mean_paired_difference": stat["per_generator_paired_mean_difference"][j],
        "fraction_above_matched_real": stat["per_generator_fraction_above_matched_real"][j]}
    for j, g in enumerate(generators)
}
stat["classes"] = {}
for g in ["real", "ai"] + generators:
    rr = [r for r in scored if (r["label"] == "ai" if g == "ai" else r["generator"] == g)]
    stat["classes"][g] = {
        "n": len(rr),
        "mean_score": float(np.mean([r["score"] for r in rr])),
        "median_score": float(np.median([r["score"] for r in rr])),
        "mean_num_vps": float(np.mean([r["num_vps"] for r in rr])),
        "mean_num_lines": float(np.mean([r["num_lines"] for r in rr])),
    }
summary["matched"] = stat

(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

with (root / "scores.csv").open("w") as f:
    fields = ["id", "group_id", "generator", "label", "score", "num_vps", "num_lines"]
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fields})

with (root / "group_scores.csv").open("w") as f:
    fields = ["group_id", "real"] + generators + ["mean_ai_minus_real"]
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    difference = fake.mean(axis=1) - real
    for i, k in enumerate(keys):
        w.writerow({"group_id": k, "real": real[i], **{g: fake[i, j] for j, g in enumerate(generators)},
                    "mean_ai_minus_real": difference[i]})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
values = np.r_[real, fake.ravel()]
bins = np.linspace(max(0, values.min() - 0.01), values.max() + 0.01, 20)
for label, arr, color in [("real (n=49)", real, "#2478b4"), ("generated (n=245)", fake.ravel(), "#d56b27")]:
    axes[0].hist(arr, bins=bins, weights=np.ones(len(arr)) / len(arr), alpha=0.55, label=label, color=color)
axes[0].set(xlabel="Mean |cos| orthogonality residual (0-1)", ylabel="Fraction of class",
            title="Primary score distributions (complete groups only)")
axes[0].legend()
thresholds = np.r_[np.inf, np.unique(values)[::-1], -np.inf]
fpr = [np.mean(real >= t) for t in thresholds]
tpr = [np.mean(fake >= t) for t in thresholds]
axes[1].plot(fpr, tpr, color="#7b3294", label=f"AUROC {stat['auroc']:.3f}")
axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
axes[1].set(xlabel="Real images flagged (FPR)", ylabel="Generated images detected (TPR)",
            title="Higher residual = generated", xlim=(0, 1), ylim=(0, 1))
axes[1].legend(loc="lower right")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("VP orthogonality - COCO indoor matched pilot - 49 complete caption groups - 245 images")
fig.savefig(root / "detection.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
axes[0].scatter(real, fake.mean(axis=1), s=25, alpha=0.8, color="#2878ad")
limits = [min(real.min(), fake.mean(axis=1).min()) - 0.02, max(real.max(), fake.mean(axis=1).max()) + 0.02]
axes[0].plot(limits, limits, "--", color="gray")
axes[0].set(xlabel="Real image residual", ylabel="Mean residual of five generated counterparts",
            title="Each point is one caption group", xlim=limits, ylim=limits)
axes[1].hist(difference, bins=14, color="#7b3294", alpha=0.8)
axes[1].axvline(0, color="gray", linestyle="--")
axes[1].set(xlabel="Mean generated residual - matched real residual", ylabel="Caption groups",
            title="Paired differences: positive supports hypothesis")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "paired_differences.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
point = np.array(stat["per_generator_auroc"])
ci = np.array(stat["per_generator_auroc_ci95"])
ax.hlines(range(5), ci[:, 0], ci[:, 1], color="#2878ad")
ax.scatter(point, range(5), color="#2878ad")
ax.axvline(0.5, color="gray", linestyle="--")
ax.set(yticks=range(5), yticklabels=["SD2.1", "SDXL", "SD3", "DALL·E 3", "Midjourney"],
       xlabel="AUROC (95% caption-group bootstrap interval)", xlim=(0, 1),
       title="Each generator: 49 AI vs the same 49 real images")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "per_generator.png", dpi=180)
plt.close(fig)

print(json.dumps({
    "pooled_auroc_all_scored": summary["pooled_auroc_all_scored_images"],
    "pooled_ci95": summary["pooled_auroc_95pct_stratified_image_bootstrap"],
    "matched_auroc_complete_groups": stat["auroc"],
    "matched_ci95": stat["auroc_95pct_caption_cluster_bootstrap"],
    "paired_mean_difference": stat["paired_mean_difference"],
    "paired_ci95": stat["paired_difference_95pct_caption_cluster_bootstrap"],
    "per_generator": stat["per_generator"],
    "missing_groups": missing_groups,
}, indent=2))
