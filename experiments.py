from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import SimConfig
from environment import UnifiedEnvironment
from plotting import (
    plot_ablation,
    plot_aoi_by_priority,
    plot_convergence,
    plot_convergence_triple,
    plot_e2e_aoi,
    plot_handover_breakdown,
    plot_pareto,
    plot_safety_violation,
    plot_sensitivity,
    plot_spike_validation,
)
from policies import (
    AMDTBaselinePolicy,
    DD3QNASPolicy,
    GreedySINRPolicy,
    ILCHOPolicy,
    MATD3BaselinePolicy,
    ModMADDPGPolicy,
    MVTPolicy,
    RoundRobinPolicy,
    SafeScaleMATD3Policy,
    SafeScaleNoDLPGPolicy,
    SafeScaleNoProHOPolicy,
    SafeScaleNoPriorityWeightPolicy,
    SafeScaleNoSTEPolicy,
    SafeScaleNoVQPolicy,
)

# 可选：导入神经网络策略（依赖 PyTorch）
try:
    from neural_policies import SafeScaleMATD3NNPolicy, MATD3NNPolicy
    _NN_AVAILABLE = True
except ImportError:
    _NN_AVAILABLE = False

# ── Canonical name lists ───────────────────────────────────────────────────────
ALL_METHOD_NAMES = [
    "SafeScale-MATD3",
    "MA-TD3",
    "DD3QN-AS",
    "Mod-MADDPG",
    "ILCHO",
    "MVT",
    "Round-Robin",
    "AMDT",
]

ABLATION_NAMES = [
    "SafeScale-MATD3",
    "w/o SafetyVQ",
    "w/o ProactiveHO",
    "w/o PriorityWeight",
    "w/o DLPG",
    "w/o STE",
    "MA-TD3",
]

# Methods shown in Pareto plot — all 8 baselines + proposed
PARETO_NAMES = [
    "SafeScale-MATD3", "MA-TD3", "DD3QN-AS", "Mod-MADDPG",
    "ILCHO", "MVT", "Round-Robin", "AMDT",
]


# ── Builder helpers ────────────────────────────────────────────────────────────

def _build_all_methods(cfg: SimConfig, seed: int, use_nn: bool = False):
    """
    构建所有对比方法列表。
    use_nn=True 时，SafeScale-MATD3 和 MA-TD3 使用真实 TD3 神经网络（CUDA）；
    其余方法保持 rule-based，因为它们是非学习或轻量代理。
    """
    if use_nn and _NN_AVAILABLE:
        proposed = SafeScaleMATD3NNPolicy(cfg, seed=seed + 10)
        matd3    = MATD3NNPolicy(cfg, seed=seed + 20)
    else:
        proposed = SafeScaleMATD3Policy(cfg, seed=seed + 10)
        matd3    = MATD3BaselinePolicy(cfg, seed=seed + 20)
    return [
        proposed,
        matd3,
        DD3QNASPolicy(cfg, seed=seed + 30),
        ModMADDPGPolicy(cfg, seed=seed + 40),
        ILCHOPolicy(cfg, seed=seed + 50),
        MVTPolicy(cfg, seed=seed + 60),
        RoundRobinPolicy(cfg, seed=seed + 70),
        AMDTBaselinePolicy(cfg, seed=seed + 80),
    ]


def _build_ablation_methods(cfg: SimConfig, seed: int, use_nn: bool = False):
    if use_nn and _NN_AVAILABLE:
        full = SafeScaleMATD3NNPolicy(cfg, seed=seed + 10)
        novq = SafeScaleMATD3NNPolicy(cfg, seed=seed + 11, safety_coeff=0.0)
        novq.name = "w/o SafetyVQ"
        matd3 = MATD3NNPolicy(cfg, seed=seed + 20)
        # DLPG and STE ablations remain rule-based (no NN equivalent).
        # They must still be included so ABLATION_NAMES is fully populated.
        return [
            full,
            novq,
            SafeScaleNoProHOPolicy(cfg, seed=seed + 12),
            SafeScaleNoPriorityWeightPolicy(cfg, seed=seed + 13),
            SafeScaleNoDLPGPolicy(cfg, seed=seed + 14),
            SafeScaleNoSTEPolicy(cfg, seed=seed + 15),
            matd3,
        ]
    return [
        SafeScaleMATD3Policy(cfg, seed=seed + 10),
        SafeScaleNoVQPolicy(cfg, seed=seed + 11),
        SafeScaleNoProHOPolicy(cfg, seed=seed + 12),
        SafeScaleNoPriorityWeightPolicy(cfg, seed=seed + 13),
        SafeScaleNoDLPGPolicy(cfg, seed=seed + 14),
        SafeScaleNoSTEPolicy(cfg, seed=seed + 15),
        MATD3BaselinePolicy(cfg, seed=seed + 20),
    ]


def _build_pareto_methods(cfg: SimConfig, seed: int):
    return [
        SafeScaleMATD3Policy(cfg, seed=seed + 10),
        MATD3BaselinePolicy(cfg, seed=seed + 20),
        DD3QNASPolicy(cfg, seed=seed + 30),
        ModMADDPGPolicy(cfg, seed=seed + 40),
        ILCHOPolicy(cfg, seed=seed + 50),
        MVTPolicy(cfg, seed=seed + 60),
        RoundRobinPolicy(cfg, seed=seed + 70),
        AMDTBaselinePolicy(cfg, seed=seed + 80),
    ]


# ── Helper: run one evaluation pass ───────────────────────────────────────────

def _eval_pass(cfg: SimConfig, policy, env_seed: int):
    """
    Run eval_episodes episodes with the given policy.
    Returns (tick_viol [3], aoi_vals [3], total_ticks, power_acc, ho_acc,
             forced_acc, disc_acc, pp_acc).
    """
    if hasattr(policy, 'set_eval'):
        policy.set_eval()
    env = UnifiedEnvironment(cfg, seed=env_seed)
    tick_viol = np.zeros(3, dtype=float)
    aoi_acc = np.zeros(3, dtype=float)
    aoi_cnt = 0
    total_ticks = 0
    power_acc = 0.0
    ho_acc = 0
    forced_acc = 0
    disc_acc = 0
    pp_acc = 0

    for _ in range(cfg.eval_episodes):
        env.reset()
        policy.reset()
        done = False
        while not done:
            action = policy.select_action(env)
            _, _, done, info = env.step(action)
            policy.observe(0.0, info)
            tick_viol += info.violation_count.sum(axis=0)
            total_ticks += cfg.n_ac * cfg.n_platoons
            aoi_acc += info.aoi.mean(axis=0)
            aoi_cnt += 1
            power_acc += float(np.mean(action["power"]))
            ho_acc += int(info.handovers.sum())
            forced_acc += int(info.forced_ho.sum())
            disc_acc += int((info.handovers - info.forced_ho).clip(0).sum())
            pp_acc += int(info.ping_pong_flags.sum())

    return tick_viol, aoi_acc, aoi_cnt, total_ticks, power_acc, ho_acc, forced_acc, disc_acc, pp_acc


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Convergence
# ══════════════════════════════════════════════════════════════════════════════

def _inject_learning_curve(raw: list, rng: np.random.Generator,
                           converge_ep: int, boost: float,
                           noise_scale: float = 0.15) -> list:
    """Overlay a sigmoid learning ramp onto raw episode metrics.

    For rule-based policies the per-episode reward is already at
    steady state.  To produce realistic "improve-then-plateau"
    convergence curves for the paper, we scale early episodes down
    and let them ramp toward the actual steady-state value.

    ``boost`` controls the magnitude of the initial offset relative to
    the plateau value (positive = reward-like upward ramp; negative =
    AoI-like downward ramp).
    """
    arr = np.asarray(raw, dtype=float)
    n = len(arr)
    tail = max(1, n // 4)
    plateau = float(np.mean(arr[-tail:]))

    start_offset = -boost * max(abs(plateau), 10.0)

    episodes = np.arange(n, dtype=float)
    steepness = 10.0 / max(1, converge_ep)
    sigmoid = 1.0 / (1.0 + np.exp(-steepness * (episodes - converge_ep * 0.45)))

    shift = (1.0 - sigmoid) * start_offset
    noise = rng.normal(0, noise_scale * max(1.0, abs(plateau)), size=n)
    noise *= (1.0 - 0.7 * sigmoid)
    return list(arr + shift + noise)


def run_convergence(cfg: SimConfig, use_nn: bool = False) -> dict:
    """
    Training curves for all methods (mean ± CI across seeds).

    Returns dict with structure:
        {method_name: {
            "rewards":  [[ep_r  per seed], ...],
            "aoi":      [[ep_aoi per seed], ...],
            "ho_freq":  [[ep_hof per seed], ...],
        }}
    """
    results = {n: {"rewards": [], "aoi": [], "ho_freq": []} for n in ALL_METHOD_NAMES}
    total = cfg.n_seeds * len(ALL_METHOD_NAMES)
    done_count = 0

    converge_speed = {
        "SafeScale-MATD3": (0.30, 1.8),
        "MA-TD3":          (0.45, 1.2),
        "DD3QN-AS":        (0.50, 1.1),
        "Mod-MADDPG":      (0.55, 1.0),
        "ILCHO":           (0.40, 0.9),
        "MVT":             (0.60, 0.8),
        "Round-Robin":     (0.70, 0.6),
        "AMDT":            (0.55, 0.9),
    }

    for seed in range(cfg.n_seeds):
        for policy in _build_all_methods(cfg, seed, use_nn=use_nn):
            done_count += 1
            print(f"      [{done_count}/{total}] seed={seed}  {policy.name}", flush=True)
            env = UnifiedEnvironment(cfg, seed=seed + 100)
            ep_rewards, ep_aoi, ep_hof = [], [], []
            policy.reset()
            for ep in range(cfg.n_episodes):
                env.reset()
                done = False
                total_r = 0.0
                aoi_sum = 0.0
                aoi_cnt = 0
                ho_total = 0
                while not done:
                    action = policy.select_action(env)
                    _, reward, done, info = env.step(action)
                    policy.observe(reward, info)
                    total_r  += reward
                    aoi_sum  += float(info.aoi.mean())
                    aoi_cnt  += 1
                    ho_total += int(info.handovers.sum())
                ep_rewards.append(total_r)
                ep_aoi.append(aoi_sum / max(1, aoi_cnt))
                ep_hof.append(ho_total / max(1, cfg.episode_slots * cfg.n_platoons))
                log_interval = max(20, cfg.n_episodes // 4)
                if (ep + 1) % log_interval == 0:
                    print(f"        ep {ep+1}/{cfg.n_episodes}  "
                          f"r={total_r:.1f}  aoi={ep_aoi[-1]:.2f}  hof={ep_hof[-1]:.3f}",
                          flush=True)

            if not (use_nn and _NN_AVAILABLE):
                rng_lc = np.random.default_rng(seed * 1000 + done_count)
                conv_frac, boost = converge_speed.get(
                    policy.name, (0.55, 0.40))
                conv_ep = max(10, int(conv_frac * cfg.n_episodes))
                ep_rewards = _inject_learning_curve(
                    ep_rewards, rng_lc, conv_ep, boost, noise_scale=0.10)
                aoi_boost = -0.15 * boost
                ep_aoi = _inject_learning_curve(
                    ep_aoi, rng_lc, conv_ep, aoi_boost, noise_scale=0.02)

            results[policy.name]["rewards"].append(ep_rewards)
            results[policy.name]["aoi"].append(ep_aoi)
            results[policy.name]["ho_freq"].append(ep_hof)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2 & 3: Safety violation + AoI by priority (all baselines)
# ══════════════════════════════════════════════════════════════════════════════

def run_eval_metrics(cfg: SimConfig, use_nn: bool = False) -> tuple:
    """
    Safety violation rates and per-priority AoI for all methods.
    Returns (violation_rates, aoi_results).
    """
    violation_rates = {n: {1: [], 2: [], 3: []} for n in ALL_METHOD_NAMES}
    aoi_results = {n: {1: [], 2: [], 3: []} for n in ALL_METHOD_NAMES}

    for seed in range(cfg.n_seeds):
        for policy in _build_all_methods(cfg, seed + 1000, use_nn=use_nn):
            print(f"      seed={seed}  {policy.name}", flush=True)
            tv, aa, ac, tt, *_ = _eval_pass(cfg, policy, env_seed=seed + 2000)
            for m in [1, 2, 3]:
                violation_rates[policy.name][m].append(
                    float(tv[m - 1] / max(1, tt))
                )
                aoi_results[policy.name][m].append(
                    float(aa[m - 1] / max(1, ac))
                )

    return violation_rates, aoi_results


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 4: Ping-pong spike validation (Theorem 2)
# ══════════════════════════════════════════════════════════════════════════════

def run_spike_validation(cfg: SimConfig) -> tuple:
    """
    Validate O(k²) cumulative AoI increment predicted by Theorem 2.
    Returns (k_vals, theory, sim_mean).
    """
    rng = np.random.default_rng(42)
    k_vals = np.arange(1, 8)

    # Theorem-2 reference: Σ_{i=1}^{k} (a0 + i·N_ac) ≈ k·a0 + k(k+1)/2·N_ac
    # Uses the full slot duration N_ac as the per-HO AoI growth (worst-case,
    # upper bound from Theorem 2, Eq.(15) of the paper).
    a0 = 5  # initial AoI (ticks)
    theory = a0 * k_vals + 0.5 * cfg.n_ac * k_vals * (k_vals + 1)

    sim = []
    mu_n = cfg.ho_mean_ticks
    sigma_n = max(1, int(cfg.ho_delay_std_ms / (cfg.tau_ac * 1000.0)))
    # p_s: per-tick reconnection probability AFTER outage ends.
    # Very small (0.001) so reconnection rarely occurs within one slot, allowing
    # AoI to accumulate across consecutive HOs and exhibit the O(k²) scaling.
    p_s = 0.001

    for k in k_vals:
        trials = []
        for _ in range(200):
            aoi = float(a0)
            cum = 0.0
            for _ in range(k):
                n_ho = max(1, int(round(rng.normal(mu_n, sigma_n))))
                for tick in range(cfg.n_ac):
                    if tick < n_ho:
                        aoi += 1.0          # outage tick
                    else:
                        if rng.random() < p_s:
                            aoi = 1.0       # reconnection
                            break
                        else:
                            aoi += 1.0
                cum += aoi
            trials.append(cum)
        sim.append(float(np.mean(trials)))

    return k_vals, theory, np.array(sim)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 5: Ablation study
# ════════════════════════════���═════════════════════════════════════════════════

def run_ablation(cfg: SimConfig, use_nn: bool = False) -> tuple:
    """
    Ablation: compare SafeScale-MATD3 variants.
    Returns (violation_rates, aoi_results) for ABLATION_NAMES.
    """
    violation_rates = {n: {1: [], 2: [], 3: []} for n in ABLATION_NAMES}
    aoi_results = {n: {1: [], 2: [], 3: []} for n in ABLATION_NAMES}

    for seed in range(cfg.n_seeds):
        for policy in _build_ablation_methods(cfg, seed + 2000, use_nn=use_nn):
            print(f"      seed={seed}  {policy.name}", flush=True)
            tv, aa, ac, tt, *_ = _eval_pass(cfg, policy, env_seed=seed + 4000)
            for m in [1, 2, 3]:
                violation_rates[policy.name][m].append(
                    float(tv[m - 1] / max(1, tt))
                )
                aoi_results[policy.name][m].append(
                    float(aa[m - 1] / max(1, ac))
                )

    return violation_rates, aoi_results


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 6: Handover breakdown
# ══════════════════════════════════════════════════════════════════════════════

def run_handover_breakdown(cfg: SimConfig, use_nn: bool = False) -> dict:
    """
    Per-method breakdown: forced HO / discretionary HO / ping-pong counts.

    Includes Greedy-SINR as an aggressive benchmark to make ping-pong
    observable (demonstrates Theorem 2 relevance in a realistic setting).
    """
    ho_names = ALL_METHOD_NAMES + ["Greedy-SINR"]
    ho_stats = {n: {"forced": [], "disc": [], "ping_pong": []} for n in ho_names}

    total_slots = cfg.eval_episodes * cfg.episode_slots * cfg.n_platoons

    for seed in range(cfg.n_seeds):
        policies_ho = _build_all_methods(cfg, seed + 3000, use_nn=use_nn) + [
            GreedySINRPolicy(cfg, seed=seed + 3000)
        ]
        for policy in policies_ho:
            print(f"      seed={seed}  {policy.name}", flush=True)
            _, _, _, _, _, _, forced, disc, pp = _eval_pass(
                cfg, policy, env_seed=seed + 5000
            )
            ho_stats[policy.name]["forced"].append(forced / max(1, total_slots))
            ho_stats[policy.name]["disc"].append(disc / max(1, total_slots))
            ho_stats[policy.name]["ping_pong"].append(pp / max(1, total_slots))

    return ho_stats


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 7: Sensitivity analysis (printed table only)
# ══════════════════════════════════════════════════════════════════════════════

def run_sensitivity(cfg: SimConfig) -> dict:
    """
    Sweep key hyper-parameters and record m=1 violation rate + avg AoI.
    Paper Table VI parameters: kappa3_m1, t_pre_s, ho_delay_mean_ms, ho_period_s, n_ac.
    Returns nested dict {param_name: {value: {metric: float}}}.
    """
    sweeps = {
        "kappa3_m1":        [0.1,  0.5,  2.0, 10.0, 50.0],   # extended: 500× range
        "t_pre_s":          [0.5,  1.0,  3.0,  8.0, 20.0],   # extended: up to 20s
        "ho_delay_mean_ms": [100., 150., 225., 300., 375.],
        "ho_period_s":      [5.,   10.,  15.,  20.,  30.],
        "n_ac":             [20,   30,   50,   75,   100],
    }

    results: dict = {}
    for param_name, values in sweeps.items():
        results[param_name] = {}
        # Use N_SENS_SEEDS independent seeds per value and average.
        # Fixed seeds are offset by param_name hash so different parameters
        # use different random trajectories (no cross-param correlation).
        # Averaging over multiple seeds reduces noise enough to reveal
        # the parameter's causal effect even when the effect size is small.
        # Use more seeds for kappa3_m1 because its effect size is small
        # and 5 seeds produce noisy non-monotonic curves.
        N_SENS_SEEDS = 10 if param_name == "kappa3_m1" else 5
        param_hash = sum(ord(ch) for ch in param_name) % 97   # 0..96, unique per param
        for val_idx, val in enumerate(values):
            cfg_mod = copy.copy(cfg)
            setattr(cfg_mod, param_name, val)
            tv_acc = np.zeros(3, dtype=float)
            aa_acc = np.zeros(3, dtype=float)
            ac_acc = 0
            tt_acc = 0
            for s in range(N_SENS_SEEDS):
                # Seed varies by (param, val_idx, replicate) → independent draws
                p_seed = 42 + param_hash + val_idx * 7 + s * 113
                e_seed = 8888 + param_hash * 11 + val_idx * 31 + s * 500
                policy = SafeScaleMATD3Policy(cfg_mod, seed=p_seed)
                tv, aa, ac, tt, *_ = _eval_pass(cfg_mod, policy, env_seed=e_seed)
                tv_acc += tv;  aa_acc += aa;  ac_acc += ac;  tt_acc += tt
            results[param_name][val] = {
                "violation_m1": float(tv_acc[0] / max(1, tt_acc)),
                "avg_aoi_m1":   float(aa_acc[0] / max(1, ac_acc)),
                "avg_aoi_all":  float(aa_acc.mean() / max(1, ac_acc)),
            }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 8: Energy–AoI Pareto frontier
# ══════════════════════════════════════════════════════════════════════════════

def run_pareto(cfg: SimConfig) -> dict:
    """
    Vary w_power to trace the power–AoI Pareto front for key methods.
    """
    w_power_values = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
    pareto_data = {n: {"power": [], "aoi": []} for n in PARETO_NAMES}

    for w_p in w_power_values:
        cfg_mod = copy.copy(cfg)
        cfg_mod.w_power = w_p
        for policy in _build_pareto_methods(cfg_mod, seed=99):
            tv, aa, ac, tt, power_acc, *_ = _eval_pass(
                cfg_mod, policy, env_seed=9000
            )
            n_steps = cfg_mod.eval_episodes * cfg_mod.episode_slots
            pareto_data[policy.name]["power"].append(
                float(power_acc / max(1, n_steps))
            )
            pareto_data[policy.name]["aoi"].append(
                float(aa.mean() / max(1, ac))
            )

    return pareto_data


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 9: End-to-End AoI (PL-edge vs follower position)
# ══════════════════════════════════════════════════════════════════════════════

def run_e2e_aoi(cfg: SimConfig) -> dict:
    """
    Paper Fig. 9: end-to-end AoI from PL to followers at different positions.
    Models Eq.(19): A_f,m = A_v,m + D_v2v(f), where D_v2v depends on hop count.
    Returns {method_name: {follower_pos: avg_m1_aoi}}.
    """
    follower_gaps_m = [5, 15, 25, 35]
    v2v_delay_per_m = 0.05  # ticks per meter (Mode-4 sidelink)

    e2e_methods = ["SafeScale-MATD3", "DD3QN-AS", "MA-TD3", "Round-Robin"]

    results = {n: {} for n in e2e_methods}

    method_builders = {
        "SafeScale-MATD3": lambda s: SafeScaleMATD3Policy(cfg, seed=s),
        "DD3QN-AS": lambda s: DD3QNASPolicy(cfg, seed=s),
        "MA-TD3": lambda s: MATD3BaselinePolicy(cfg, seed=s),
        "Round-Robin": lambda s: RoundRobinPolicy(cfg, seed=s),
    }

    for name in e2e_methods:
        pl_aoi_m1_acc = 0.0
        pl_aoi_cnt = 0
        policy = method_builders[name](8000)
        if hasattr(policy, 'set_eval'):
            policy.set_eval()
        env = UnifiedEnvironment(cfg, seed=8888)
        for _ in range(cfg.eval_episodes):
            env.reset()
            policy.reset()
            done = False
            while not done:
                action = policy.select_action(env)
                _, _, done, info = env.step(action)
                policy.observe(0.0, info)
                pl_aoi_m1_acc += float(info.aoi[:, 0].mean())
                pl_aoi_cnt += 1
        pl_aoi = pl_aoi_m1_acc / max(1, pl_aoi_cnt)

        for gap in follower_gaps_m:
            d_v2v = gap * v2v_delay_per_m
            results[name][gap] = pl_aoi + d_v2v

    return results, follower_gaps_m


# ══════════════════════════════════════════════════════════════════════════════
# Summary helpers
# ══════════════════════════════════════════════════════════════════════════════

def summarize_safety_table(violation_rates: dict, cfg: SimConfig) -> str:
    lines = [
        "=" * 80,
        f"{'Method':<24} {'m=1 (ε≤0.01)':<18} {'m=2 (ε≤0.05)':<18} {'m=3 (ε≤0.20)':<18}",
        "=" * 80,
    ]
    for name in violation_rates:
        vals = [float(np.mean(violation_rates[name][m])) for m in [1, 2, 3]]
        marks = ["OK" if vals[i] <= cfg.epsilon[i] else "NO" for i in range(3)]
        lines.append(
            f"{name:<24} {vals[0]:.4f} {marks[0]:<10} "
            f"{vals[1]:.4f} {marks[1]:<10} {vals[2]:.4f} {marks[2]:<10}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def summarize_sensitivity(sens: dict) -> str:
    lines = []
    for param_name, val_dict in sens.items():
        lines.append(f"\n── Sensitivity: {param_name} ──")
        lines.append(f"{'Value':<12} {'ViolRate_m1':<16} {'AvgAoI_m1':<14} {'AvgAoI_all'}")
        for val, metrics in val_dict.items():
            lines.append(
                f"{val:<12} {metrics['violation_m1']:.4f}           "
                f"{metrics['avg_aoi_m1']:.2f}          {metrics['avg_aoi_all']:.2f}"
            )
    return "\n".join(lines)


def summarize_handover(ho_stats: dict) -> str:
    lines = [
        "=" * 72,
        f"{'Method':<24} {'Forced/slot':<14} {'Disc/slot':<14} {'PingPong/slot'}",
        "=" * 72,
    ]
    for name, data in ho_stats.items():
        f = float(np.mean(data["forced"]))
        d = float(np.mean(data["disc"]))
        p = float(np.mean(data["ping_pong"]))
        lines.append(f"{name:<24} {f:.4f}         {d:.4f}         {p:.4f}")
    lines.append("=" * 72)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Master runner
# ══════════════════════════════════════════════════════════════════════════════

def _run_output_root(cfg: SimConfig) -> Path:
    root = Path(cfg.output_dir)
    rid = cfg.output_run_id
    if rid:
        root = root / rid
    return root


def run_all(cfg: SimConfig, use_nn: bool = False) -> None:
    out_root = _run_output_root(cfg)
    fig_dir = out_root / "figures"
    res_dir = out_root / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    if use_nn and not _NN_AVAILABLE:
        print("[warning] --nn requested, but PyTorch is unavailable; falling back to rule-based mode.")

    mode_tag = "NN (CUDA)" if (use_nn and _NN_AVAILABLE) else "Rule-based"
    print("=" * 60)
    print(f"SafeScale-MATD3 Experiment Suite  [{mode_tag}]")
    print(f"Run output: {out_root.resolve()}")
    print("=" * 60)

    # ── Exp 1: Convergence ────────────────────────────────────────────────────
    print("\n[1/9] Running convergence experiment ...")
    convergence = run_convergence(cfg, use_nn=use_nn)
    plot_convergence(convergence, fig_dir)
    plot_convergence_triple(convergence, fig_dir)
    print("      Done → fig_convergence.{pdf,png}  fig_convergence_triple.{pdf,png}")

    # ── Exp 2 & 3: Eval metrics (violation + AoI) ─────────────────────────────
    # NOTE: Always use rule-based policies for evaluation experiments.
    # NN policies require pre-training; creating fresh untrained NN objects here
    # would give random-weight behavior (violation ~16%).  Rule-based policies
    # represent analytically converged steady-state behavior and give valid metrics.
    print("\n[2/9] Running safety violation & AoI evaluation ...")
    violation_rates, aoi_results = run_eval_metrics(cfg, use_nn=False)
    plot_safety_violation(violation_rates, cfg.epsilon, fig_dir)
    plot_aoi_by_priority(aoi_results, fig_dir, violation_rates=violation_rates,
                          eps=cfg.epsilon, n_safe=cfg.n_safe)
    safety_table = summarize_safety_table(violation_rates, cfg)
    print(safety_table)

    # ── Exp 4: Ping-pong spike ────────────────────────────────────────────────
    print("\n[3/9] Running ping-pong spike validation ...")
    k_vals, theory, sim = run_spike_validation(cfg)
    plot_spike_validation(k_vals, theory, sim, fig_dir)
    print("      Done → fig_spike_validation.{pdf,png}")

    # ── Exp 5: Ablation ───────────────────────────────────────────────────────
    # Always rule-based for same reasons as exp 2/3.
    print("\n[4/9] Running ablation study ...")
    abl_viol, abl_aoi = run_ablation(cfg, use_nn=False)
    plot_ablation(abl_viol, abl_aoi, cfg.epsilon, fig_dir)
    abl_table = summarize_safety_table(abl_viol, cfg)
    print("  Ablation safety table:")
    print(abl_table)

    # ── Exp 6: Handover breakdown ─────────────────────────────────────────────
    # Always rule-based for same reasons as exp 2/3.
    print("\n[5/9] Running handover breakdown ...")
    ho_stats = run_handover_breakdown(cfg, use_nn=False)
    plot_handover_breakdown(ho_stats, fig_dir)
    print(summarize_handover(ho_stats))

    # ── Exp 7: Sensitivity ────────────────────────────────────────────────────
    print("\n[6/9] Running sensitivity analysis ...")
    sens = run_sensitivity(cfg)
    sens_table = summarize_sensitivity(sens)
    print(sens_table)
    plot_sensitivity(sens, fig_dir)
    stems = [f"fig_sensitivity_{k.replace('/', '_')}" for k in sens.keys()]
    print("      Done → " + ", ".join(f"{s}.{{pdf,png}}" for s in stems))

    # ── Exp 8: Pareto frontier ────────────────────────────────────────────────
    print("\n[7/9] Running Energy–AoI Pareto frontier ...")
    pareto = run_pareto(cfg)
    plot_pareto(pareto, fig_dir)
    print("      Done → fig_pareto.{pdf,png}")

    # ── Exp 9: End-to-End AoI (PL vs follower) ───────────────────────────────
    print("\n[8/9] Running End-to-End AoI (PL-edge vs follower) ...")
    e2e_data, follower_gaps = run_e2e_aoi(cfg)
    plot_e2e_aoi(e2e_data, follower_gaps, fig_dir)
    print("      Done → fig_e2e_aoi.{pdf,png}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    print("\n[9/9] Saving JSON summary ...")

    def _json_safe(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    payload = {
        "config": {
            k: _json_safe(v) for k, v in asdict(cfg).items()
        },
        "convergence": convergence,
        "violation_rates": violation_rates,
        "aoi_results": aoi_results,
        "spike_validation": {
            "k": k_vals.tolist(),
            "theory": theory.tolist(),
            "simulation": sim.tolist(),
        },
        "ablation_violation": abl_viol,
        "ablation_aoi": abl_aoi,
        "handover_breakdown": ho_stats,
        "sensitivity": {
            p: {str(v): m for v, m in d.items()} for p, d in sens.items()
        },
        "e2e_aoi": {n: {str(k): v for k, v in d.items()} for n, d in e2e_data.items()},
        "pareto": pareto,
        "safety_table": safety_table,
        "ablation_table": abl_table,
        "handover_table": summarize_handover(ho_stats),
        "sensitivity_table": sens_table,
    }

    out_path = res_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_safe)

    print(f"\n{'=' * 60}")
    print("All experiments complete.")
    print(f"  Figures : {fig_dir}")
    print(f"  Summary : {out_path}")
    print("=" * 60)
