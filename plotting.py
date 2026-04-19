from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ── Color palette (IEEE-style, colorblind-friendly) ───────────────────────────
IEEE_COLORS: Dict[str, str] = {
    "SafeScale-MATD3":   "#005EB8",
    "MA-TD3":            "#D1495B",
    "AMDT":              "#2A9D8F",
    "DD3QN-AS":          "#E9C46A",
    "Mod-MADDPG":        "#F4A261",
    "ILCHO":             "#8338EC",
    "MVT":               "#06D6A0",
    "Round-Robin":       "#A8A8A8",
    # ablation
    "w/o SafetyVQ":      "#FF6B6B",
    "w/o ProactiveHO":   "#4ECDC4",
    "w/o PriorityWeight":"#FFE66D",
    "w/o DLPG":          "#B5838D",
    "w/o STE":           "#6D6875",
}

_EPS_COLORS = ["#E63946", "#457B9D", "#2D6A4F"]

_FALLBACK_COLORS = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
    "#BCBD22", "#17BECF",
]


def _color(name: str, idx: int) -> str:
    return IEEE_COLORS.get(name, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def set_ieee_style() -> None:
    plt.rcParams.update({
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
    })


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def _mean_ci(arr: np.ndarray) -> tuple:
    mean = arr.mean(axis=0)
    if arr.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    std = arr.std(axis=0, ddof=1)
    ci = 1.96 * std / np.sqrt(arr.shape[0])
    return mean, ci


def _smooth(arr: np.ndarray, w: int = 20) -> np.ndarray:
    if w <= 1 or len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    # Length-preserving moving average so early-stage trend is not phase-shifted away.
    left = w // 2
    right = w - 1 - left
    arr_pad = np.pad(arr, (left, right), mode="edge")
    return np.convolve(arr_pad, kernel, mode="valid")


def _decimate_indices(n: int, max_points: int) -> np.ndarray:
    """
    Evenly spaced indices in [0, n-1], always including first and last,
    so long episode runs do not draw thousands of segments on top of each other.
    """
    if n <= 0:
        return np.array([], dtype=int)
    if n <= max_points:
        return np.arange(n, dtype=int)
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return np.unique(idx)


def _bar_offset(n_methods: int, width: float) -> List[float]:
    return [(i - (n_methods - 1) / 2.0) * width for i in range(n_methods)]


def _annotate_eps(ax: plt.Axes, eps: tuple) -> None:
    """ε threshold lines annotated at right edge (outside axes, never in legend)."""
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for j, e in enumerate(eps):
        c = _EPS_COLORS[j % len(_EPS_COLORS)]
        ax.axhline(e, linestyle="--", linewidth=0.9, color=c, alpha=0.85, zorder=1)
        ax.text(1.02, e, f"ε{j+1}={e}", transform=trans,
                va="center", ha="left", fontsize=5.5, color=c, clip_on=False)


def _annotate_zero_bars(ax: plt.Axes, x: np.ndarray, vals_per_method: list,
                        offsets: list, width: float) -> None:
    """
    For any bar whose value is exactly 0, draw a tiny visible stub and label '0'
    so the reader knows data exists but is zero — not missing.
    """
    ymax = ax.get_ylim()[1]
    stub = ymax * 0.012   # 1.2% of axis height — just enough to see
    for i, vals in enumerate(vals_per_method):
        for j, v in enumerate(vals):
            if v == 0.0:
                # Draw stub bar
                ax.bar(x[j] + offsets[i], stub, width,
                       color=ax.containers[i].patches[j].get_facecolor(),
                       alpha=0.5, zorder=3)
                ax.text(x[j] + offsets[i], stub * 1.1, "0",
                        ha="center", va="bottom", fontsize=5, color="#444444")


def _legend_above(ax: plt.Axes, ncol: int = 4, fontsize: int = 6,
                  y_offset: float = 1.18) -> None:
    """Place compact legend above the axes title, outside the data area."""
    ax.legend(
        frameon=True,
        loc="lower center",
        bbox_to_anchor=(0.5, y_offset),
        bbox_transform=ax.transAxes,
        ncol=ncol,
        fontsize=fontsize,
        borderpad=0.4,
        labelspacing=0.3,
        columnspacing=0.8,
    )


# ── Plot functions ────────────────────────────────────────────────────────────

def _extract_metric(results: dict, key: str) -> Dict[str, List]:
    """
    Support both old format  {name: [[seed1], [seed2], ...]}
    and new format           {name: {"rewards": [...], "aoi": [...], "ho_freq": [...]}}.
    Falls back to rewards list if key is missing (old data).
    """
    out = {}
    for name, val in results.items():
        if isinstance(val, dict):
            out[name] = val.get(key, val.get("rewards", []))
        else:
            out[name] = val   # legacy: bare list = rewards
    return out


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_convergence(
    results: dict,
    out_dir: Path,
    smooth_w: int = 20,
    max_plot_points: int = 700,
    raw_episode_threshold: int = 10000,
) -> None:
    """
    Single-panel reward convergence (backward-compatible with old format).

    For long training logs, draw a lightly-decimated raw trace + a lightly
    smoothed trend line so "improve-then-plateau" remains visible.
    """
    set_ieee_style()
    rewards = _extract_metric(results, "rewards")
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    for idx, (name, seed_curves) in enumerate(rewards.items()):
        arr = np.array(seed_curves, dtype=float)
        mean_raw, ci_raw = _mean_ci(arr)
        n_ep = len(mean_raw)
        w_eff = max(5, min(18, smooth_w))
        mean = _smooth(mean_raw, w_eff)
        ci = _smooth(ci_raw, w_eff)
        x = np.arange(n_ep, dtype=float)
        di = _decimate_indices(len(mean), max_plot_points)
        x_p, mean_p, ci_p = x[di], mean[di], ci[di]
        c = _color(name, idx)
        is_main = (name == "SafeScale-MATD3")
        lw = 2.0 if is_main else 1.3
        zord = 5 if is_main else 3
        ax.plot(x_p, mean_p, label=name, color=c, linewidth=lw,
                linestyle="-", zorder=zord, alpha=1.0)
        ax.fill_between(x_p, mean_p - ci_p, mean_p + ci_p,
                        color=c, alpha=0.12, zorder=zord - 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc="lower right",
              ncol=2, fontsize=6, frameon=True,
              borderpad=0.4, labelspacing=0.3,
              columnspacing=0.8, handlelength=1.5, handletextpad=0.4)
    _save(fig, out_dir, "fig_convergence")


# NN/learning methods shown as solid lines with CI shading;
# rule-based shown as thin dashed reference lines (no CI).
_NN_METHODS: set = {"SafeScale-MATD3", "MA-TD3"}


def plot_convergence_triple(
    results: dict,
    out_dir: Path,
    smooth_w: int = 20,
) -> None:
    """
    Three-panel convergence figure:
      (a) Rewards of different baselines
      (b) Time-average AoI of different baselines
      (c) Satellite handover frequency of different baselines
    All x-axes: Training Episodes.

    NN/learning methods: solid lines with CI shading (zorder=3).
    Rule-based methods: thin dashed lines, no CI (zorder=2).
    """
    set_ieee_style()

    rewards = _extract_metric(results, "rewards")
    aoi     = _extract_metric(results, "aoi")
    ho_freq = _extract_metric(results, "ho_freq")

    panel_data = [
        (rewards,  "Reward",                          "(a) Rewards"),
        (aoi,      "Time-Average AoI (ticks)",        "(b) Time-Average AoI"),
        (ho_freq,  "Satellite HO Frequency\n(per slot per platoon)", "(c) Handover Frequency"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
    fig.subplots_adjust(wspace=0.38)

    _LS_TRIPLE = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2)), (0,(1,1)), (0,(3,2,1,2))]

    # Non-negative metrics: HO frequency and AoI cannot be negative.
    # Track which panel index corresponds to non-negative data.
    _NON_NEG_PANELS = {1, 2}  # panel 1 = AoI, panel 2 = HO freq

    def _draw_panel(ax: plt.Axes, data: dict, ylabel: str, title: str,
                    panel_idx: int = 0) -> None:
        non_neg = panel_idx in _NON_NEG_PANELS
        for idx, (name, seed_curves) in enumerate(data.items()):
            arr = np.array(seed_curves, dtype=float)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            mean_raw, ci_raw = _mean_ci(arr)
            mean = _smooth(mean_raw, smooth_w)
            ci   = _smooth(ci_raw,   smooth_w)
            x = np.arange(len(mean))
            c = _color(name, idx)
            ls = _LS_TRIPLE[idx % len(_LS_TRIPLE)]
            is_main = (name == "SafeScale-MATD3")
            lw = 1.8 if is_main else 1.1
            zord = 5 if is_main else 3
            ax.plot(x, mean, label=name, color=c, linewidth=lw,
                    linestyle=ls, zorder=zord, alpha=1.0)
            ci_lo = np.clip(mean - ci, 0, None) if non_neg else (mean - ci)
            ax.fill_between(x, ci_lo, mean + ci, color=c,
                            alpha=0.10, zorder=zord - 1)
        if non_neg:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Training Episodes", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=8, pad=4)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    for p_idx, (ax, (data, ylabel, title)) in enumerate(zip(axes, panel_data)):
        _draw_panel(ax, data, ylabel, title, panel_idx=p_idx)

    # Shared legend above all three panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        bbox_transform=fig.transFigure,
        ncol=4,
        fontsize=6,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.3,
        columnspacing=0.8,
    )

    _save(fig, out_dir, "fig_convergence_triple")

    # Also export the three panels as individual figures.
    def _draw_single_like_main(ax: plt.Axes, data: dict, ylabel: str,
                               title: str, legend_loc: str = "upper right",
                               legend_ncol: int = 2) -> None:
        for idx, (name, seed_curves) in enumerate(data.items()):
            arr = np.array(seed_curves, dtype=float)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            mean_raw, ci_raw = _mean_ci(arr)
            n_ep = len(mean_raw)
            w_eff = max(5, min(30, 20))
            mean = _smooth(mean_raw, w_eff)
            ci = _smooth(ci_raw, w_eff)
            x = np.arange(n_ep, dtype=float)
            di = _decimate_indices(len(mean), 700)
            x_p, mean_p, ci_p = x[di], mean[di], ci[di]
            c = _color(name, idx)
            is_main = (name == "SafeScale-MATD3")
            lw = 2.0 if is_main else 1.3
            zord = 5 if is_main else 3
            ax.plot(x_p, mean_p, label=name, color=c,
                    linewidth=lw, linestyle="-", zorder=zord, alpha=1.0)
            ax.fill_between(x_p, mean_p - ci_p, mean_p + ci_p,
                            color=c, alpha=0.12, zorder=zord - 1)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,
                  loc=legend_loc, ncol=legend_ncol,
                  fontsize=6, frameon=True,
                  borderpad=0.4, labelspacing=0.3,
                  columnspacing=0.8, handlelength=1.5, handletextpad=0.4)

    single_specs = [
        # (data, ylabel, title, stem, legend_loc, legend_ncol)
        (rewards, "Reward", None, "fig_convergence_reward", "lower right", 2),
        (aoi,     "Time-Average AoI (ticks)", None, "fig_convergence_aoi", "upper right", 2),
        (ho_freq, "Satellite HO Frequency\n(per slot per platoon)", None,
         "fig_convergence_handover", "upper right", 2),
    ]
    for data, ylabel, title, stem, leg_loc, leg_ncol in single_specs:
        fig_i, ax_i = plt.subplots(figsize=(3.45, 2.8))
        _draw_single_like_main(ax_i, data, ylabel, title, leg_loc, leg_ncol)
        _save(fig_i, out_dir, stem)


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

    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    vals_per_method = []
    for i, name in enumerate(methods):
        vals = [float(np.mean(violation_rates[name][m])) for m in priorities]
        vals_per_method.append(vals)
        ax.bar(x + offsets[i], vals, width, label=name, color=_color(name, i), alpha=0.9)

    _annotate_eps(ax, eps)
    _annotate_zero_bars(ax, x, vals_per_method, offsets, width)

    ax.set_xticks(x)
    ax.set_xticklabels([f"m={m}" for m in priorities])
    ax.set_ylabel("Violation Rate")
    ax.grid(True, axis="y", alpha=0.3)
    ncols = min(n, 4)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.9167, 1.0, 0.0),
        ncol=ncols,
        fontsize=5,
        frameon=True,
        borderaxespad=0.0,
        borderpad=0.5,
        labelspacing=0.3,
        columnspacing=0.7,
        handlelength=1.2,
        mode="expand",
        bbox_transform=ax.transAxes,
    )
    _save(fig, out_dir, "fig_safety_violation")


def plot_aoi_by_priority(
    aoi_results: Dict[str, Dict[int, List[float]]],
    out_dir: Path,
    violation_rates: Dict[str, Dict[int, List[float]]] = None,
    eps: tuple = (0.01, 0.05, 0.20),
    n_safe: tuple = (5, 10, 50),
) -> None:
    """Per-priority normalised AoI bars (AoI / n_safe_m).

    Normalisation by the per-priority safety threshold n_safe_m = (5, 10, 50)
    expresses AoI as a fraction of each priority's safety limit, making
    cross-priority comparisons meaningful.  The composite 'Wtd. Avg.' column
    uses weights proportional to priority_weights (0.6 / 0.3 / 0.1).

    Methods that violate ε_m1=0.01 safety constraint are marked with † on
    their m=1 bar.
    """
    set_ieee_style()
    methods = list(aoi_results.keys())
    n = len(methods)
    priorities = [1, 2, 3]
    n_safe_arr = np.array(n_safe, dtype=float)
    # Add a "Weighted" group (weights: m1=0.6, m2=0.3, m3=0.1 ∝ priority_weights)
    x_ticks = np.arange(len(priorities) + 1)
    width = min(0.15, 0.8 / n)
    offsets = _bar_offset(n, width)

    # Detect which methods violate m=1 safety
    violating = set()
    if violation_rates:
        eps_m1 = eps[0]
        for name in methods:
            if float(np.mean(violation_rates[name][1])) > eps_m1 * 1.05:
                violating.add(name)

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    # Priority weights: (4.5, 2.5, 0.5) normalised → (0.6, 0.3+, 0.1-)
    w_vec = np.array([0.6, 0.3, 0.1])
    all_bar_tops = []
    for i, name in enumerate(methods):
        raw_vals = np.array([float(np.mean(aoi_results[name][m])) for m in priorities])
        # Normalise each priority by its safety threshold
        norm_vals = raw_vals / n_safe_arr
        weighted = float(np.dot(w_vec, norm_vals))
        all_vals = list(norm_vals) + [weighted]
        all_bar_tops.extend(all_vals)
        ax.bar(x_ticks + offsets[i], all_vals, width,
               label=name, color=_color(name, i), alpha=0.9)
        # Mark m=1 bar with † if method violates safety
        if name in violating:
            ax.text(x_ticks[0] + offsets[i], norm_vals[0] + 0.008, "†",
                    ha="center", va="bottom", fontsize=7,
                    color=_color(name, i), fontweight="bold")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"m={m}\n(n={int(ns)})" for m, ns in zip(priorities, n_safe)]
                       + ["Wtd.\nAvg."])
    ax.set_ylabel("Normalised AoI  (AoI / n_safe)")
    ax.grid(True, axis="y", alpha=0.3)

    # Fixed y-axis upper bound for paper-level comparability.
    ax.set_ylim(0, 0.5)

    if violating:
        ax.text(0.98, 0.82, "† violates ε₁=0.01",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, style="italic", color="#555555")

    # Legend inside the axes, spanning full plot width at the top.
    ncols = min(n, 4)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        ncol=ncols,
        fontsize=6,
        frameon=True,
        borderaxespad=0.0,
        borderpad=0.5,
        labelspacing=0.3,
        columnspacing=0.7,
        handlelength=1.2,
        mode="expand",
        bbox_transform=ax.transAxes,
    )
    _save(fig, out_dir, "fig_aoi_by_priority")


def plot_spike_validation(
    k_vals: np.ndarray,
    theory: np.ndarray,
    sim: np.ndarray,
    out_dir: Path,
) -> None:
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    ax.plot(k_vals, theory, marker="o", color="#222222", label="Theorem-2 O(k²) trend")
    ax.plot(k_vals, sim, marker="s", color=_color("SafeScale-MATD3", 0), label="Simulation")
    ax.set_xlabel("Ping-pong sequence length k")
    ax.set_ylabel("Cumulative AoI increment")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper left",
        fontsize=6,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.4,
    )
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
    vpm = []
    for i, name in enumerate(methods):
        vals = [float(np.mean(violation_rates[name][m])) for m in priorities]
        vpm.append(vals)
        ax.bar(x + offsets[i], vals, width, label=name, color=_color(name, i), alpha=0.9)
    _annotate_eps(ax, eps)
    _annotate_zero_bars(ax, x, vpm, offsets, width)
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
    axes[1].legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fontsize=6,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.3,
        columnspacing=0.8,
    )
    fig.subplots_adjust(right=0.78, wspace=0.30)
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
    width = min(0.12, 0.8 / n)
    offsets = _bar_offset(n, width)

    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    vpm = []
    for i, name in enumerate(methods):
        vals = [float(np.mean(ho_stats[name][c])) for c in cats]
        vpm.append(vals)
        ax.bar(x + offsets[i], vals, width, label=name, color=_color(name, i), alpha=0.9)

    _annotate_zero_bars(ax, x, vpm, offsets, width)

    ax.set_xticks(x); ax.set_xticklabels(cat_labels, rotation=10)
    ax.set_ylabel("HO rate (per slot per platoon)")
    # Auto-scale y-axis: show actual data range rather than fixed 0-1.2 which
    # collapses all meaningful detail to the bottom of the axis.
    all_vals = [v for row in vpm for v in row]
    y_max = max(all_vals) if max(all_vals) > 0 else 0.1
    ax.set_ylim(0, y_max * 1.35)
    ax.grid(True, axis="y", alpha=0.3)
    ncols = min(n, 4)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        ncol=ncols,
        fontsize=5,
        frameon=True,
        borderaxespad=0.0,
        borderpad=0.5,
        labelspacing=0.3,
        columnspacing=0.7,
        handlelength=1.2,
        mode="expand",
        bbox_transform=ax.transAxes,
    )
    _save(fig, out_dir, "fig_handover_breakdown")


def plot_e2e_aoi(
    e2e_data: Dict[str, Dict[int, float]],
    follower_gaps: list,
    out_dir: Path,
) -> None:
    """Paper Fig. 9: End-to-end AoI at PL-edge vs follower position."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, (name, gap_dict) in enumerate(e2e_data.items()):
        gaps = sorted(gap_dict.keys())
        aoi_vals = [gap_dict[g] for g in gaps]
        c = _color(name, idx)
        ax.plot(gaps, aoi_vals, marker=markers[idx % len(markers)],
                color=c, label=name, linewidth=1.3)
    ax.set_xlabel("Follower distance from PL (m)")
    ax.set_ylabel("End-to-End AoI, m=1 (ticks)")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="lower right",
        ncol=2,
        fontsize=6,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.3,
        columnspacing=0.8,
        handlelength=1.4,
    )
    _save(fig, out_dir, "fig_e2e_aoi")


def plot_sensitivity(
    sens_data: dict,
    out_dir: Path,
) -> None:
    """Paper Table VI: one IEEE single-column figure per swept parameter."""
    set_ieee_style()
    params = list(sens_data.keys())

    labels = {
        "kappa3_m1": "κ₃ (m=1 penalty)",
        "t_pre_s": "T_pre (s)",
        "ho_delay_mean_ms": "HO delay (ms)",
        "ho_period_s": "HO period (s)",
        "n_ac": "N_ac (ticks/slot)",
    }
    # Parameters with wide ranges should use a log x-axis for readability.
    log_x_params = {"kappa3_m1"}

    for param_name in params:
        fig, ax = plt.subplots(figsize=(3.45, 2.8))
        val_dict = sens_data[param_name]
        xs = sorted(val_dict.keys(), key=float)
        viol = [val_dict[x]["violation_m1"] for x in xs]
        aoi = [val_dict[x]["avg_aoi_m1"] for x in xs]
        xs_f = [float(x) for x in xs]
        ax.plot(xs_f, viol, "o-",
                color="#D1495B", label="Viol. m=1", linewidth=1.2)
        ax.axhline(0.01, linestyle="--", color="#E63946", alpha=0.6, linewidth=0.7,
                   label=f"ε₁={0.01}")
        ax2 = ax.twinx()
        ax2.plot(xs_f, aoi, "s--",
                 color="#005EB8", label="AoI m=1", linewidth=1.2)
        if param_name in log_x_params:
            ax.set_xscale("log")
            ax2.set_xscale("log")
        ax.set_xlabel(labels.get(param_name, param_name), fontsize=6)
        ax.set_ylabel("Viol. Rate", fontsize=6, color="#D1495B")
        ax2.set_ylabel("Avg AoI (ticks)", fontsize=6, color="#005EB8")
        ax.tick_params(labelsize=5)
        ax2.tick_params(labelsize=5)
        # Combined legend from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if param_name in {"ho_delay_mean_ms", "ho_period_s", "kappa3_m1", "n_ac", "t_pre_s"}:
            ax.legend(
                h1 + h2, l1 + l2,
                fontsize=5,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                frameon=True,
                borderpad=0.4,
            )
        else:
            ax.legend(h1 + h2, l1 + l2, fontsize=5, loc="best",
                      frameon=True, borderpad=0.4)
        ax.set_title(labels.get(param_name, param_name), fontsize=7)
        fig.tight_layout()
        stem = f"fig_sensitivity_{param_name.replace('/', '_')}"
        _save(fig, out_dir, stem)


def plot_pareto(
    pareto_data: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.8))

    # Unified x-grid improves visual alignment while preserving physical meaning.
    power_grid = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.97], dtype=float)

    # Layered styles: highlight proposed method, separate non-learning baselines.
    style_map = {
        "SafeScale-MATD3": {"marker": "o", "ls": "-",  "lw": 2.6, "ms": 6.0, "z": 10, "alpha": 1.00},
        "MA-TD3":          {"marker": "s", "ls": "-",  "lw": 1.2, "ms": 4.2, "z": 6,  "alpha": 0.90},
        "DD3QN-AS":        {"marker": "^", "ls": "-",  "lw": 1.2, "ms": 4.2, "z": 6,  "alpha": 0.90},
        "Mod-MADDPG":      {"marker": "D", "ls": "-",  "lw": 1.2, "ms": 4.2, "z": 6,  "alpha": 0.90},
        "ILCHO":           {"marker": "v", "ls": "-",  "lw": 1.2, "ms": 4.2, "z": 6,  "alpha": 0.90},
        "AMDT":            {"marker": "h", "ls": "-",  "lw": 1.2, "ms": 4.2, "z": 6,  "alpha": 0.90},
        "MVT":             {"marker": "P", "ls": "--", "lw": 1.0, "ms": 4.0, "z": 5,  "alpha": 0.85},
        "Round-Robin":     {"marker": "x", "ls": ":",  "lw": 1.0, "ms": 8.0, "mew": 1.8, "z": 5, "alpha": 0.85},
    }

    def _resample_to_grid(raw_power: List[float], raw_aoi: List[float]) -> np.ndarray:
        p = np.asarray(raw_power, dtype=float)
        a = np.asarray(raw_aoi, dtype=float)
        if p.size == 0:
            return np.full(power_grid.shape, np.nan, dtype=float)
        order = np.argsort(p)
        p = p[order]
        a = a[order]
        p_unique, inv = np.unique(p, return_inverse=True)
        a_unique = np.array([a[inv == i].mean() for i in range(len(p_unique))], dtype=float)
        if p_unique.size == 1:
            out = np.full(power_grid.shape, np.nan, dtype=float)
            out[:] = a_unique[0]
            return out
        out = np.interp(power_grid, p_unique, a_unique)
        mask = (power_grid < p_unique.min()) | (power_grid > p_unique.max())
        out[mask] = np.nan
        return out

    main_x_min, main_x_max = 0.30, 0.85
    safe_aoi_grid = None
    plotted_aoi = []
    for idx, (name, data) in enumerate(pareto_data.items()):
        aoi_grid = _resample_to_grid(data["power"], data["aoi"])
        style = style_map.get(name, {"marker": "o", "ls": "-", "lw": 1.1, "ms": 4.0, "z": 5, "alpha": 0.85})
        c = _color(name, idx)
        plot_kwargs = {
            "marker": style["marker"],
            "linestyle": style["ls"],
            "linewidth": style["lw"],
            "markersize": style["ms"],
            "color": c,
            "alpha": style["alpha"],
            "zorder": style["z"],
            "label": name,
        }
        if "mew" in style:
            plot_kwargs["markeredgewidth"] = style["mew"]
        main_mask = (power_grid >= main_x_min) & (power_grid <= main_x_max) & np.isfinite(aoi_grid)
        if np.any(main_mask):
            ax.plot(power_grid[main_mask], aoi_grid[main_mask], **plot_kwargs)
            plotted_aoi.extend(np.asarray(aoi_grid, dtype=float)[main_mask].tolist())
        if name == "SafeScale-MATD3":
            safe_aoi_grid = aoi_grid.copy()

    # Main panel: shade region above SafeScale (within fair-comparison range).
    if safe_aoi_grid is not None and np.isfinite(safe_aoi_grid).any():
        finite = (power_grid >= main_x_min) & (power_grid <= main_x_max) & np.isfinite(safe_aoi_grid)
        y_ref = (max(plotted_aoi) + 0.02) if plotted_aoi else 2.2
        ax.fill_between(
            power_grid[finite],
            safe_aoi_grid[finite],
            y_ref,
            alpha=0.07,
            color=_color("SafeScale-MATD3", 0),
            zorder=1,
            label="_nolegend_",
        )

    ax.set_xlabel("Avg Normalised Power")
    ax.set_ylabel("Avg AoI (ticks, all priorities)")
    main_ticks = [p for p in power_grid.tolist() if main_x_min <= p <= main_x_max]
    ax.set_xticks(main_ticks)
    ax.set_xticklabels([f"{p:.2f}" for p in main_ticks], rotation=15)
    ax.set_xlim(main_x_min, main_x_max)
    ax.set_ylim(2.00, 2.20)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=6,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.3,
        columnspacing=0.8,
        handlelength=1.4,
    )

    # Scheme B inset: show SafeScale extension beyond baseline saturation range.
    if safe_aoi_grid is not None and np.isfinite(safe_aoi_grid).any():
        axins = inset_axes(
            ax,
            width="19%",
            height="17%",
            loc="lower left",
            bbox_to_anchor=(0.11, 0.09, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        safe_c = _color("SafeScale-MATD3", 0)
        finite_full = np.isfinite(safe_aoi_grid)
        axins.plot(
            power_grid[finite_full],
            safe_aoi_grid[finite_full],
            "-o",
            color=safe_c,
            linewidth=1.28,
            markersize=2.88,
        )
        axins.axvspan(main_x_max, 1.0, alpha=0.10, color=safe_c)
        axins.set_xlim(0.60, 1.00)
        y_safe = safe_aoi_grid[finite_full]
        y_pad = max(0.01, 0.12 * (float(np.max(y_safe)) - float(np.min(y_safe))))
        axins.set_ylim(float(np.min(y_safe)) - y_pad, float(np.max(y_safe)) + y_pad)
        axins.set_title("SafeScale extends to P=0.97", fontsize=4.96)
        axins.tick_params(labelsize=4.0)
        axins.grid(True, alpha=0.25, linestyle="--")

    _save(fig, out_dir, "fig_pareto")
