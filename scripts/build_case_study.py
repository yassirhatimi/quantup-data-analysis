"""Create a standalone portfolio figure from aggregate saved quality evidence."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    quality = json.loads((ROOT / "gold_wave_analysis/analysis_summary.json").read_text(encoding="utf-8"))["quality"]
    order = {"1d": 0, "1h": 1, "15m": 2}
    quality.sort(key=lambda r: order[r["timeframe"]])
    labels = {"1d": "Daily", "1h": "Hourly", "15m": "15-minute"}
    fig, ax = plt.subplots(figsize=(9, 4.7), layout="constrained")
    fig.patch.set_facecolor("#ffffff")
    y = range(len(quality))
    kept = [100 * r["usable_rows"] / r["raw_rows"] for r in quality]
    ax.barh(list(y), kept, color="#0072B2", label="Retained")
    ax.barh(list(y), [100-k for k in kept], left=kept, color="#E69F00", hatch="///", label="Excluded")
    for i, (k, r) in enumerate(zip(kept, quality)):
        ax.text(2, i, f"{r['usable_rows']:,} / {r['raw_rows']:,} retained ({k:.1f}%)", va="center", color="white", weight="bold", fontsize=11)
    ax.set_yticks(list(y), [labels[r["timeframe"]] for r in quality])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of downloaded rows (%)")
    ax.set_title("Gold data quality: retained rows by timeframe", loc="left", fontsize=15, weight="bold", pad=15)
    for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2, frameon=False)
    fig.suptitle("Local Yahoo futures snapshot retrieved 5 October 2026; coverage differs by timeframe", fontsize=10, color="#555555")
    out = ROOT / "docs/assets"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gold-data-quality.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    main()
