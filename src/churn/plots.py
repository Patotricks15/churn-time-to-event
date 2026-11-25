from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_km_by_segment(curves: dict, title: str, h_days: int, path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, km in curves.items():
        steps = pd.concat(
            [pd.DataFrame({"time": [0.0], "survival": [1.0]}), km[["time", "survival"]]]
        )
        ax.step(steps["time"], steps["survival"], where="post", label=label)
    ax.axvline(h_days, linestyle="--", color="grey", label=f"H = {h_days} days")
    ax.set_xlabel("Days from snapshot to next purchase")
    ax.set_ylabel("Share with no purchase yet, S(t)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=9)
    ax.legend()
    _save(fig, path)


def plot_hazard_ratios(coefficients: pd.DataFrame, title: str, path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(coefficients["covariate"], coefficients["hazard_ratio_per_sd"], color="steelblue")
    ax.axvline(1.0, linestyle="--", color="grey")
    ax.set_xlabel("Hazard ratio of next purchase, per 1 SD (above 1 = buys sooner)")
    ax.set_title(title, fontsize=9)
    _save(fig, path)
