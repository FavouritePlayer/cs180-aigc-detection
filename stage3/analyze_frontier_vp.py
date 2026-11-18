"""Summarize the stage-3 vanishing-point orthogonality run on the Frontier 100-image manifest."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage2"))
from detection_metrics import auroc, bootstrap_auc

p = argparse.ArgumentParser()
p.add_argument("results")
p.add_argument("--manifest", default="../stage2/frontier_manifest.json")
a = p.parse_args()
root = Path(a.results)
rows = [json.loads(s) for s in (root / "per_image.jsonl").read_text().splitlines()]
expected = {r["id"]: r for r in json.loads(Path(a.manifest).read_text())}

assert len(rows) == 100 and len({r["id"] for r in rows}) == 100
assert {r["id"] for r in rows} == set(expected)
for r in rows:
    for key in ["label", "sha256", "revision", "file"]:
        assert r[key] == expected[r["id"]][key]

scored = [r for r in rows if r["score"] is not None]
unscored = [r for r in rows if r["score"] is None]
labels_scored = np.array([r["label"] == "ai" for r in scored], dtype=int)

summary = {
    "n_total": len(rows),
    "n_scored": len(scored),
    "n_insufficient_structure": len(unscored),
    "real": sum(1 for r in rows if r["label"] == "real"),
    "ai": sum(1 for r in rows if r["label"] == "ai"),
    "primary": "mean |cos| orthogonality residual over found vanishing-point pairs",
    "direction": "higher = generated, fixed before this run, not flipped",
    "auroc": auroc(labels_scored, [r["score"] for r in scored]),
    "auroc_95pct_stratified_image_bootstrap": bootstrap_auc(labels_scored, [r["score"] for r in scored]),
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 180,
    "classes": {},
    "coverage": {},
}

for label_name, label_val in [("real", "real"), ("ai", "ai")]:
    rr = [r for r in rows if r["label"] == label_val]
    rr_scored = [r for r in rr if r["score"] is not None]
    summary["classes"][label_name] = {
        "n": len(rr),
        "n_scored": len(rr_scored),
        "mean_score": float(np.mean([r["score"] for r in rr_scored])) if rr_scored else None,
        "median_score": float(np.median([r["score"] for r in rr_scored])) if rr_scored else None,
        "mean_num_vps": float(np.mean([r["num_vps"] for r in rr])),
        "mean_num_lines": float(np.mean([r["num_lines"] for r in rr])),
    }
summary["coverage"] = {
    "n_with_2_or_3_vps": len(scored),
    "n_with_3_vps": sum(1 for r in rows if r["num_vps"] == 3),
    "n_with_2_vps": sum(1 for r in rows if r["num_vps"] == 2),
    "n_with_fewer_than_2_vps": sum(1 for r in rows if r["num_vps"] < 2),
}

(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

with (root / "scores.csv").open("w") as f:
    fields = ["id", "label", "generator", "score", "num_vps", "num_lines"]
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fields})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
values = np.array([r["score"] for r in scored])
bins = np.linspace(max(0, values.min() - 0.02), values.max() + 0.02, 20)
for label_val, color in [("real", "#2478b4"), ("ai", "#d56b27")]:
    axes[0].hist(
        [r["score"] for r in scored if r["label"] == label_val],
        bins=bins, alpha=0.55,
        label=f"{label_val} (n={sum(1 for r in rows if r['label']==label_val and r['score'] is not None)})",
        color=color,
    )
axes[0].set(xlabel="Mean |cos| orthogonality residual (0-1)", ylabel="Images",
            title="Vanishing-point orthogonality, primary score")
axes[0].legend()

thresholds = np.r_[np.inf, np.unique(values)[::-1], -np.inf]
fpr = [np.mean(values[labels_scored == 0] >= t) for t in thresholds]
tpr = [np.mean(values[labels_scored == 1] >= t) for t in thresholds]
axes[1].plot(fpr, tpr, color="#7b3294", label=f"AUROC {summary['auroc']:.3f}")
axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
axes[1].set(xlabel="Real images flagged (FPR)", ylabel="Generated images detected (TPR)",
            title="Higher residual = generated", xlim=(0, 1), ylim=(0, 1))
axes[1].legend(loc="lower right")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("LSD lines + RANSAC vanishing points + GeoCalib K (reused) - orthogonality residual")
fig.savefig(root / "detection.png", dpi=180)
plt.close(fig)

print(json.dumps(summary, indent=2))
