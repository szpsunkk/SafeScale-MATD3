import argparse
import re
from dataclasses import replace
from datetime import datetime

import numpy as np

from config import SimConfig
from experiments import run_all, run_convergence


def _tail_mean(convergence: dict, method: str, tail_n: int) -> float:
    curves = np.array(convergence[method]["rewards"], dtype=float)
    if curves.ndim == 1:
        curves = curves[np.newaxis, :]
    mean_curve = curves.mean(axis=0)
    t = min(tail_n, len(mean_curve))
    return float(mean_curve[-t:].mean())


def _autotune_safescale_cfg(base_cfg: SimConfig, trials: int, tail_n: int) -> SimConfig:
    """
    Small proxy search that maximizes:
        tail_reward(SafeScale-MATD3) - tail_reward(MA-TD3)
    Then uses best NN safety/shield settings for the full run.
    """
    proxy = replace(
        base_cfg,
        n_episodes=min(base_cfg.n_episodes, 120),
        episode_slots=min(base_cfg.episode_slots, 80),
        n_seeds=1,
        eval_episodes=min(base_cfg.eval_episodes, 10),
    )
    candidates = []
    for nn_safety_coeff in (0.18, 0.25, 0.32, 0.40):
        for warmup in (8000, 12000, 16000):
            for urgency in (1.05, 1.15, 1.25):
                for power_floor in (0.50, 0.60):
                    candidates.append(
                        (
                            nn_safety_coeff,
                            warmup,
                            urgency,
                            power_floor,
                        )
                    )
    max_trials = max(1, min(trials, len(candidates)))
    tested = candidates[:max_trials]

    best = None
    print("\n[autotune] searching SafeScale NN parameters...")
    print(f"[autotune] proxy episodes={proxy.n_episodes}, slots={proxy.episode_slots}, trials={max_trials}")
    for i, (coef, warm, urg, floor) in enumerate(tested, start=1):
        cfg_i = replace(
            proxy,
            nn_safety_coeff=coef,
            nn_safety_warmup_env_steps=warm,
            nn_safety_shield_urgency=urg,
            nn_safety_shield_power_floor=floor,
        )
        print(
            f"[autotune] [{i}/{max_trials}] "
            f"coef={coef:.2f}, warmup={warm}, urg={urg:.2f}, floor={floor:.2f}",
            flush=True,
        )
        conv = run_convergence(cfg_i, use_nn=True)
        safe_tail = _tail_mean(conv, "SafeScale-MATD3", tail_n)
        ma_tail = _tail_mean(conv, "MA-TD3", tail_n)
        gap = safe_tail - ma_tail
        print(
            f"[autotune]     tail(SafeScale)={safe_tail:.2f}, tail(MA-TD3)={ma_tail:.2f}, gap={gap:.2f}",
            flush=True,
        )
        cand = (gap, safe_tail, coef, warm, urg, floor)
        if best is None or cand > best:
            best = cand

    assert best is not None
    _, _, coef, warm, urg, floor = best
    print(
        "[autotune] selected "
        f"coef={coef:.2f}, warmup={warm}, urg={urg:.2f}, floor={floor:.2f}",
        flush=True,
    )
    return replace(
        base_cfg,
        nn_safety_coeff=coef,
        nn_safety_warmup_env_steps=warm,
        nn_safety_shield_urgency=urg,
        nn_safety_shield_power_floor=floor,
    )


def _sanitize_run_label(label: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", label.strip(), flags=re.ASCII)
    s = s.strip("_") or "run"
    return s[:80]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SafeScale-MATD3 experiment suite"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--nn",
        action="store_true",
        default=False,
        help="Force PyTorch+CUDA neural-network TD3 policies (requires torch)",
    )
    mode_group.add_argument(
        "--rule-based",
        action="store_true",
        default=False,
        help="Force rule-based baselines only (no NN training dynamics)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help=(
            "Fast verify mode: n_episodes=80, n_seeds=2, eval_episodes=10, episode_slots=60. "
            "Runs in ~15-30 min on GPU; all 9 figures are generated."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help=(
            "Ultra-fast smoke test: n_episodes=20, n_seeds=1, eval_episodes=3, episode_slots=30. "
            "Completes in ~3-5 min on GPU; verifies all code paths without NN convergence."
        ),
    )
    parser.add_argument(
        "--autotune-safescale",
        action="store_true",
        default=False,
        help="Auto-search SafeScale NN safety/shield params before full run.",
    )
    parser.add_argument(
        "--autotune-trials",
        type=int,
        default=10,
        help="Number of proxy trials used by --autotune-safescale.",
    )
    parser.add_argument(
        "--autotune-tail",
        type=int,
        default=40,
        help="Tail window size for proxy convergence objective.",
    )
    parser.add_argument(
        "--timestamp-output",
        action="store_true",
        default=False,
        help=(
            "Write figures and summary.json under outputs/run_YYYYMMDD_HHMMSS/ "
            "(does not overwrite previous runs)."
        ),
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        metavar="TAG",
        help=(
            "Optional directory name under outputs/: with --timestamp-output, "
            "appends to the time stamp; without it, uses this tag alone (no clock)."
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        # Ultra-fast: just verify all code paths work (~3-5 min on GPU)
        cfg = SimConfig(
            n_episodes=20,
            n_seeds=1,
            eval_episodes=3,
            episode_slots=30,
            n_ac=20,           # fewer ticks → faster env step
        )
        print("[smoke] n_episodes=20, n_seeds=1, eval_episodes=3, episode_slots=30, n_ac=20")
    elif args.fast:
        # Fast verify: enough to see trends and meaningful figures (~15-30 min on GPU)
        cfg = SimConfig(
            n_episodes=80,
            n_seeds=2,
            eval_episodes=10,
            episode_slots=60,
        )
        print("[fast]  n_episodes=80, n_seeds=2, eval_episodes=10, episode_slots=60")
    else:
        cfg = SimConfig()

    # use_nn: explicit --nn or --rule-based; default falls back to NN if available
    if args.nn:
        use_nn = True
    elif args.rule_based:
        use_nn = False
    else:
        use_nn = True   # default: try NN, falls back gracefully if torch unavailable
    if args.autotune_safescale and use_nn:
        cfg = _autotune_safescale_cfg(
            cfg,
            trials=args.autotune_trials,
            tail_n=args.autotune_tail,
        )

    output_run_id = None
    if args.timestamp_output:
        output_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.run_label is not None:
        tag = _sanitize_run_label(args.run_label)
        output_run_id = f"{output_run_id}_{tag}" if output_run_id else tag
    if output_run_id is not None:
        cfg = replace(cfg, output_run_id=output_run_id)

    run_all(cfg, use_nn=use_nn)


if __name__ == "__main__":
    main()
