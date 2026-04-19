"""
Targeted re-evaluation: update aoi_by_priority, ablation, spike_validation,
handover_breakdown, safety_violation in summary.json and regenerate all figures.
Does NOT re-run the expensive convergence training loop.
"""
import json
from pathlib import Path
import numpy as np

from config import SimConfig
from experiments import (
    run_ablation, run_eval_metrics, run_handover_breakdown,
    run_spike_validation,
)
from plotting import (
    plot_ablation, plot_aoi_by_priority, plot_handover_breakdown,
    plot_safety_violation, plot_spike_validation,
)

OUT = Path("outputs")
FIG = OUT / "figures"
RES = OUT / "results"

def _json_safe(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    return obj

def main():
    p = RES / "summary.json"
    data = json.loads(p.read_text())
    cfg = SimConfig(**{k: v for k, v in data["config"].items()
                       if k in SimConfig.__dataclass_fields__
                       and k not in ("output_dir",)})

    print("=== [1/4] Safety violation + AoI by priority ===")
    vr, aoi_r = run_eval_metrics(cfg)
    data["violation_rates"] = vr
    data["aoi_results"] = aoi_r

    print("=== [2/4] Ablation ===")
    abl_v, abl_a = run_ablation(cfg)
    data["ablation_violation"] = abl_v
    data["ablation_aoi"] = abl_a

    print("=== [3/4] Handover breakdown ===")
    ho = run_handover_breakdown(cfg)
    data["handover_breakdown"] = ho

    print("=== [4/4] Spike validation ===")
    k_v, th, si = run_spike_validation(cfg)
    data["spike_validation"] = {"k": k_v.tolist(), "theory": th.tolist(), "simulation": si.tolist()}

    p.write_text(json.dumps(data, indent=2, default=_json_safe), encoding="utf-8")
    print("summary.json updated")

    eps = tuple(cfg.epsilon)
    plot_safety_violation(vr, eps, FIG)
    plot_aoi_by_priority(aoi_r, FIG, violation_rates=vr, eps=eps)
    plot_ablation(abl_v, abl_a, eps, FIG)
    plot_handover_breakdown(ho, FIG)
    plot_spike_validation(k_v, th, si, FIG)
    print("All figures regenerated.")

if __name__ == "__main__":
    main()
