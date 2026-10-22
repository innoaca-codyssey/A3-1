"""평가 입력의 밝기 분포와 그림자 통제 실험을 분석한다."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from evaluate import geometric_score
from prepare_data import transform
from scanner import find_document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/cases"))
    parser.add_argument("--originals", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path("evaluation/cases.json").read_text())["cases"]
    rows, histograms, ablation = [], {}, []
    fig, axes = plt.subplots(4, 5, figsize=(16, 10), sharex=True, sharey=False)
    for ax, rec in zip(axes.ravel(), records):
        image = cv2.imread(str(args.cases / rec["file"]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = np.bincount(gray.ravel(), minlength=256) / gray.size
        histograms.setdefault(rec["group"], []).append(histogram)
        row = {
            "file": rec["file"],
            "mean": float(gray.mean()),
            "std": float(gray.std()),
            "dark_fraction_below_64": float((gray < 64).mean()),
        }
        rows.append(row)
        ax.plot(np.arange(256), histogram)
        ax.set_title(rec["file"], fontsize=9)
        if rec["group"] == "shadow" and args.originals:
            original = cv2.imread(str(args.originals / rec["source"]))
            original = transform(original, "simple")
            for name, sample in (("unshadowed", original), ("shadowed", image)):
                stages, corners = find_document(sample)
                score = geometric_score(corners, rec["quad_normalized"], sample.shape)
                ablation.append(
                    {
                        "file": rec["file"],
                        "condition": name,
                        "edge_fraction": float((stages["edges"] > 0).mean()),
                        **score,
                    }
                )
    fig.supxlabel("gray value")
    fig.supylabel("pixel fraction")
    fig.tight_layout()
    fig.savefig(args.output / "histograms-20.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for group, arrays in histograms.items():
        ax.plot(np.arange(256), np.mean(arrays, axis=0), label=group)
    ax.set(xlabel="gray value", ylabel="mean normalized pixel histogram")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, .2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "histograms-groups.png", dpi=150)
    plt.close(fig)
    report = {"images": rows, "shadow_ablation": ablation}
    (args.output / "image-analysis.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({"images": len(rows), "shadow_ablation": ablation}))


if __name__ == "__main__":
    main()
