from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory


# ── Color palette (IEEE-style, colorblind-friendly) ───────────────────────────
IEEE_COLORS: Dict[str, str] = {
    "SafeScale-MATD3":   "#005EB8",   # Blue  – proposed
    "MA-TD3":            "#D1495B",   # Red
    "AMDT":              "#2A9D8F",   # Teal
    "DD3QN-AS":          "#E9C46A",   # Gold
    "Mod-MADDPG":        "#F4A261",   # Orange
    "ILCHO":             "#8338EC",   # Purple
    "MVT":               "#06D6A0",   # Green
    "Round-Robin":       "#A8A8A8",   # Gray
    # ablation
    "w/o SafetyVQ":      "#FF6B6B",
    "w/o ProactiveHO":   "#4ECDC4",
    "w/o PriorityWeight":"#FFE66D",
}

_EPS_COLORS = ["#E63946", "#457B9D", "#2D6A4F"]   # ε₁, ε₂, ε₃ reference lines

_FALLBACK_COLORS = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
    "#BCBD22", "#17BECF",
]


def _color(name: str, idx: int) -> str:
    return IEEE_COLORS.get(name, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def _mean_ci(arr: np.ndarray) -> tuple:
    """arr: [n_seeds, T]  →  (mean, 95%-CI half-width)"""
    mean = arr.mean(axis=0)
    if arr.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    std = arr.std(axis=0, ddof=1)
    ci = 1.96 * std / np.sqrt(arr.shape[0])
    return mean, ci


def _smooth(arr: np.ndarray, w: int = 20) -> np.ndarray:
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")


def _bar_offset(n_methods: int, width: float) -> List[float]:
    """Centred offsets for n_methods bars of given width."""
    return [(i - (n_methods - 1) / 2.0) * width for i in range(n_methods)]


def _annotate_eps(ax: plt.Axes, eps: tuple) -> None:
    """
    Draw horizontal dashed ε-threshold lines and label them on the right edge.
    Labels are placed just outside the axes so they never overlap data or legend.
    """
    # blended transform: x in axes fraction [0,1], y in data coords
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for j, e in enumerate(eps):
        c = _EPS_COLORS[j % len(_EPS_COLORS)]
        ax.axhline(e, linestyle="--", linewidth=0.9, color=c, alpha=0.85, zorder=1)
        ax.text(
            1.02, e, f"ε{j+1}={e}",
            transform=trans,
            va="center", ha="left",
            fontsize=5.5, color=c,
            clip_on=False,
        )


def _legend_below(ax: plt.Axes, ncol: int = 4, fontsize: int = 6,
                  y_offset: float = -0.18) -> None:
    """Place a compact legend below the axes, outside the data area."""
    ax.legend(
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        fontsize=fontsize,
        borderpad=0.4,
        labelspacing=0.3,
        columnspacing=0.8,
    )


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_convergence(
    results: Dict[str, List[List[float]]],
    out_dir: Path,
    smooth_w: int = 20,
) -> None:
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for idx, (name, seed_curves) in enumerate(results.items()):
        arr = np.array(seed_curves, dtype=float)
        mean_raw, ci_raw = _mean_ci(arr)
        mean = _smooth(mean_raw, smooth_w)
        ci = _smooth(ci_raw, smooth_w)
        x = np.arange(len(mean))
        c = _color(name, idx)
        ax.plot(x, mean, label=name, color=c)
        ax.fill_between(x, mean - ci, mean + ci, color=c, alpha=0.18)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Training Convergence (mean ± 95% CI)")
    ax.grid(True, alpha=0.3)
    _legend_below(ax, ncol=2, y_offset=-0.22)
    _save(fig, out_dir, "fig_convergence")


def plot_safety_violation(
    violation_rates: Dict[str, Dict[int, List[float]]],
    eps: tuple,
    out_dir: Path,
) -> None:
    set_ieee_style()
    methods = list(violation_rates.keys())
    n = len(methods)
    priorities = [1, 2, 3]
    x = np.arange(len(priorities))
    width = min(0.18, 0.8 / n)
    offsets = _bar_offset(n, width)

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for i, name in enumerate(methods):
        vals = [float(np.mean(violation_rates[name][m])) for m in priorities]
        ax.bar(x + offsets[i], vals, width, label=name, color=_color(name, i), alpha=0.9)

    # ε thresholds annotated directly — NOT in the legend
    _annotate_eps(ax, eps)

    ax.set_xticks(x)
    ax.set_xticklabels([f"m={m}" for m in priorities])
    ax.set_ylabel("Violation Rate")
    ax.set_title("Safety Violation by Priority")
    ax.grid(True, axis="y", alpha=0.3)
    _legend_below(ax, ncol=4, y_offset=-0.20)
    _save(fig, out_dir, "fig_safety_violation")


def plot_aoi_by_priority(
    aoi_results: Dict[str, Dict[int, List[float]]],
    out_dir: Path,
) -> None:
    set_ieee_style()
    methods = list(aoi_results.keys())
    n = len(methods)
    priorities = [1, 2, 3]
    x = np.arange(len(priorities))
    width = min(0.18, 0.8 / n)
    offsets = _bar_offset(n, width)

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for i, name in enumerate(methods):
        vals = [float(np.mean(aoi_results[name][m])) for m in priorities]
        ax.bar(x + offsets[i], vals, width, label=name, color=_color(name, i), alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"m={m}" for m in priorities])
    ax.set_ylabel("Average AoI (ticks)")
    ax.set_title("AoI by Priority Class")
    ax.grid(True, axis="y", alpha=0.3)
    _legend_below(ax, ncol=4, y_offset=-0.20)
    _save(fig, out_dir, "fig_aoi_by_priority")


def plot_spike_validation(
    k_vals: np.ndarray,
    theory: np.ndarray,
    sim: np.ndarray,
    out_dir: Path,
) -> None:
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.plot(k_vals, theory, marker="o", color="#222222", label="Theorem-2 O(k²) trend")
    ax.plot(k_vals, sim, marker="s", color=_color("SafeScale-MATD3", 0), label="Simulation")
    ax.set_xlabel("Ping-pong sequence length k")
    ax.set_ylabel("Cumulative AoI increment")
    ax.set_title("Ping-Pong Spike Scaling (Theorem 2)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="upper left")
    _save(fig, out_dir, "fig_spike_validation")


def plot_ablation(
    violation_rates: Dict[str, Dict[int, List[float]]],
    aoi_results: Dict[str, Dict[int, List[float]]],
    eps: tuple,
    out_dir: Path,
) -> None:
    set_ieee_style()
    methods = list(violation_rates.keys())
    n = len(methods)
    priorities = [1, 2, 3]
    x = np.arange(len(priorities))
    width = min(0.16, 0.8 / n)
    offsets = _bar_offset(n, width)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55))

    # Left: violation rates
    ax = axes[0]
    for i, name in enumerate(methods):
        vals = [float(np.mean(violation_rates[name][m])) for m in priorities]
        ax.bar(x + offsets[i], vals, width, label=name, color=_color(name, i), alpha=0.9)
    _annotate_eps(ax, eps)
    ax.set_xticks(x); ax.set_xticklabels([f"m={m}" for m in priorities])
    ax.set_ylabel("Violation Rate"); ax.set_title("Ablation – Safety Violation")
    ax.grid(True, axis="y", alpha=0.3)

    # Right: AoI
    ax = axes[1]
    for i, name in enumerate(methods):
        vals = [float(np.mean(aoi_results[name][m])) for m in priorities]
        ax.bar(x + offsets[i], vals, width, label=name, color=_color(name, i), alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([f"m={m}" for m in priorities])
    ax.set_ylabel("Average AoI (ticks)"); ax.set_title("Ablation – AoI")
    ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=5, fontsize=6,
        frameon=True,
        borderpad=0.4, labelspacing=0.3, columnspacing=0.8,
    )
    _save(fig, out_dir, "fig_ablation")


def plot_handover_breakdown(
    ho_stats: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    set_ieee_style()
    methods = list(ho_stats.keys())
    n = len(methods)
    cats = ["forced", "disc", "ping_pong"]
    cat_labels = ["Forced HO", "Discretionary HO", "Ping-Pong"]
    x = np.arange(len(cats))
    width = min(0.16, 0.8 / n)
    offsets = _bar_offset(n, width)

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for i, name in enumerate(methods):
        vals = [float(np.mean(ho_stats[name][c])) for c in cats]
        ax.bar(x + offsets[i], vals, width, label=name, color=_color(name, i), alpha=0.9)

    ax.set_xticks(x); ax.set_xticklabels(cat_labels, rotation=10)
    ax.set_ylabel("HO rate (per slot per platoon)")
    ax.set_title("Handover Type Breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    _legend_below(ax, ncol=4, y_offset=-0.22)
    _save(fig, out_dir, "fig_handover_breakdown")


def plot_pareto(
    pareto_data: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for idx, (name, data) in enumerate(pareto_data.items()):
        power_vals = data["power"]
        aoi_vals = data["aoi"]
        c = _color(name, idx)
        ax.plot(power_vals, aoi_vals, marker="o", color=c, label=name)
        # Mark the default operating point (middle of sweep)
        if len(power_vals) > 2:
            mid = len(power_vals) // 2
            ax.plot(power_vals[mid], aoi_vals[mid], marker="*", markersize=8,
                    color=c, zorder=5)

    ax.set_xlabel("Avg Normalised Power")
    ax.set_ylabel("Avg AoI (ticks, all priorities)")
    ax.set_title("Energy–AoI Pareto Frontier")
    ax.grid(True, alpha=0.3)
    _legend_below(ax, ncol=2, y_offset=-0.20)
    _save(fig, out_dir, "fig_pareto")
