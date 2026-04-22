# SafeScale-MATD3

**Safety-Constrained Multi-Agent TD3 for Satellite Handover and Priority Scheduling in Vehicle Platoon Communications**

This repository contains the simulation code and experiment pipeline for the SafeScale-MATD3 paper. It implements a two-timescale multi-agent reinforcement learning framework that jointly optimizes satellite handover decisions and priority-based scheduling for vehicle platoon networks under strict Age-of-Information (AoI) safety constraints.

## Key Features

- **Two-timescale simulation**: outer loop at 1 Hz (slot-level policy decisions), inner loop at 50 Hz (tick-level channel/AoI evolution)
- **Lyapunov-based safety**: virtual queue mechanism ensuring per-priority AoI violation rates stay below configurable thresholds
- **8 baseline comparisons**: MA-TD3, AMDT, DD3QN-AS, Mod-MADDPG, ILCHO, MVT, Round-Robin
- **5 ablation variants**: No-VQ, No-ProHO, No-PriorityWeight, No-DLPG, No-STE
- **Neural network policies** (optional): PyTorch-based Actor-Critic with TD3 double critic and Lagrangian safety penalty
- **IEEE-style figures**: publication-ready PDF/PNG output via matplotlib

## Repository Structure

```
SafeScale-MATD3/
├── config.py            # SimConfig dataclass — all hyperparameters
├── environment.py       # UnifiedEnvironment — two-timescale tick simulator
├── policies.py          # Rule-based policies (proposed + 7 baselines + 5 ablations)
├── neural_policies.py   # PyTorch Actor-Critic policies (optional, requires torch)
├── experiments.py       # Experiment orchestration (convergence, e2e, ablation, etc.)
├── plotting.py          # IEEE-style figure generation
├── run_all.py           # Main entry point
├── run_all.sh           # Shell wrapper (handles venv setup)
├── requirements.txt     # Python dependencies
└── outputs/
    ├── figures/         # Generated PDF/PNG figures
    └── results/         # summary.json with numeric results
```

## Requirements

- Python 3.8+
- NumPy >= 1.21
- Matplotlib >= 3.4
- SciPy >= 1.7
- PyTorch (optional, for neural network policies)

## Installation

```bash
git clone https://github.com/<your-username>/SafeScale-MATD3.git
cd SafeScale-MATD3
pip install -r requirements.txt

# Optional: install PyTorch for NN policies
# See https://pytorch.org/get-started/locally/ for your platform
pip install torch
```

## Quick Start

```bash
# Smoke test (~3-5 min): verify all code paths work
python3 -u run_all.py --smoke --nn 2>&1 | tee outputs/run_smoke.log

# Fast run with visible trends (~15-30 min, GPU recommended)
python3 -u run_all.py --fast --nn 2>&1 | tee outputs/run_fast.log

# Full experiment (rule-based fallback if torch unavailable)
python3 -u run_all.py

# Neural network policies only
python3 -u run_all.py --nn

# Rule-based baselines only (no PyTorch required)
python3 -u run_all.py --rule-based

# Auto-tune SafeScale NN safety parameters before full run
python3 -u run_all.py --nn --autotune-safescale --autotune-trials 8
```

Or use the shell wrapper which handles virtual environment setup:

```bash
bash run_all.sh [--fast] [--nn]
```

## Experiments

The pipeline runs the following experiments sequentially:

| Experiment | Description | Output Figures |
|---|---|---|
| Convergence | Training reward curves over episodes | `fig_convergence*` |
| End-to-End Eval | Per-priority AoI comparison across methods | `fig_e2e_aoi`, `fig_aoi_by_priority` |
| Safety Violation | Violation rate by priority class | `fig_safety_violation` |
| Spike Validation | Response to sudden AoI spikes | `fig_spike_validation` |
| Pareto Sweep | Safety-throughput Pareto frontier | `fig_pareto` |
| Ablation | Component-wise contribution analysis | `fig_ablation` |
| Handover Breakdown | Proactive vs. forced handover statistics | `fig_handover_breakdown` |
| Sensitivity | Parameter sensitivity analysis | `fig_sensitivity*` |

All outputs are saved to `outputs/figures/` (PDF + PNG) and `outputs/results/summary.json`.

## Configuration

All hyperparameters are centralized in `config.py` via the `SimConfig` dataclass. Key defaults:

| Parameter | Value | Description |
|---|---|---|
| `n_platoons` | 5 | Number of vehicle platoons |
| `n_priorities` | 3 | Priority classes per platoon |
| `n_safe` | (5, 10, 50) | AoI safety thresholds (ticks) for m=1,2,3 |
| `epsilon` | (0.01, 0.05, 0.20) | Max allowed violation rates for m=1,2,3 |
| `n_episodes` | 3000 | Training episodes |
| `episode_slots` | 150 | Slots per episode (150 s) |
| `n_seeds` | 3 | Independent random seeds |

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sun2026safetyawareaoischedulingleo,
      title={Safety-Aware AoI Scheduling for LEO Satellite-Assisted Autonomous Driving}, 
      author={Kangkang Sun and Junyi He and Juntong Liu and Xiuzhen Chen and Jianhua Li and Minyi Guo},
      year={2026},
      eprint={2604.17281},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2604.17281}, 
}
```

## License

This project is provided for academic and research purposes.
