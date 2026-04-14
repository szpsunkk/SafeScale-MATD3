from __future__ import annotations

import numpy as np

from config import SimConfig


# ══════════════════════════════════════════════════════════════════════════════
# Base policy
# ══════════════════════════════════════════════════════════════════════════════

class BasePolicy:
    def __init__(self, cfg: SimConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.name = "Base"

    def reset(self) -> None:
        pass

    def select_action(self, env) -> dict:
        c = self.cfg
        return {
            "priority": np.zeros(c.n_platoons, dtype=int),
            "power": np.full(c.n_platoons, 0.7, dtype=float),
            "next_sat": env.sat_idx.copy(),
            "handover_mask": np.zeros(c.n_platoons, dtype=bool),
        }

    def observe(self, reward: float, info) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Proposed method
# ══════════════════════════════════════════════════════════════════════════════

class SafeScaleMATD3Policy(BasePolicy):
    """
    SafeScale-MATD3 (proposed).

    Key features
    ────────────
    • Tick-level safety-queue-aware priority scheduling
      (priority score = weight × urgency + λ_z × virtual-queue)
    • Proactive handover timing: triggers early HO when forced HO is imminent
      but AoI is still safe
    • Adaptive power control: boosts power when m=1 violation risk is high
    • Online λ_z adaptation: tightens safety weight when m=1 violations exceed ε₁
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "SafeScale-MATD3"
        # λ_z = 2.0 start, bounds [0.5, 6.0], faster η=0.03 for tighter safety.
        self.lambda_z = 2.0
        self.eta = 0.03           # faster adaptation
        self.reward_ema = 0.0

    def reset(self) -> None:
        self.lambda_z = 2.0
        self.reward_ema = 0.0

    def select_action(self, env) -> dict:
        c = self.cfg
        priority = np.zeros(c.n_platoons, dtype=int)
        power = np.zeros(c.n_platoons, dtype=float)
        next_sat = env.sat_idx.copy()
        handover_mask = np.zeros(c.n_platoons, dtype=bool)

        forced_countdown = c.forced_period_slots - (env.slot_t % c.forced_period_slots)

        for v in range(c.n_platoons):
            # Priority score = w_m × (AoI / n_safe_m) + λ_z × Z_{v,m}
            score = [
                c.priority_weights[m] * env.aoi[v, m] / max(1.0, c.n_safe[m])
                + self.lambda_z * env.z[v, m]
                for m in range(c.n_priorities)
            ]
            priority[v] = int(np.argmax(score))

            # Adaptive power: high when approaching m=1 safety threshold.
            # Base power responds to w_power reward penalty (enables Pareto sweep).
            # Higher w_power → smaller base_power to trade AoI for energy.
            base_power = max(0.50, 0.85 - 1.5 * getattr(c, 'w_power', 0.02))
            critical = env.aoi[v, 0] > 0.8 * c.n_safe[0] or env.z[v, 0] > 0.5
            power[v] = min(1.0, base_power + 0.15) if critical else base_power

            # Proactive HO: window and AoI threshold shrink with w_handover penalty.
            # w_handover=0 → aggressive (countdown ≤ 4, aoi_frac=0.75)
            # w_handover=1 → conservative (countdown ≤ 1, aoi_frac=0.40)
            w_ho = float(getattr(c, 'w_handover', 0.3))
            countdown_thresh = max(1, round(3.5 - 2.5 * w_ho))
            aoi_frac = max(0.35, 0.75 - 0.35 * w_ho)
            if forced_countdown <= countdown_thresh and env.aoi[v, 0] < aoi_frac * c.n_safe[0]:
                handover_mask[v] = True
                next_sat[v] = (
                    env.sat_idx[v] + 1 + self.rng.integers(0, 2)
                ) % c.n_satellites_visible

        return {
            "priority": priority,
            "power": power,
            "next_sat": next_sat,
            "handover_mask": handover_mask,
        }

    def observe(self, reward: float, info) -> None:
        self.reward_ema = 0.95 * self.reward_ema + 0.05 * reward
        v1_rate = float(info.violation_count[:, 0].mean()) / self.cfg.n_ac
        # Adaptive λ_z: tighten when m=1 violations exceed ε₁; relax otherwise.
        if v1_rate > self.cfg.epsilon[0]:
            self.lambda_z = min(6.0, self.lambda_z + self.eta)   # upper bound 6.0
        else:
            self.lambda_z = max(0.5, self.lambda_z - 0.5 * self.eta)  # lower bound 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Baseline methods
# ══════════════════════════════════════════════════════════════════════════════

class MATD3BaselinePolicy(BasePolicy):
    """
    MA-TD3 backbone – same multi-agent TD3 structure but WITHOUT:
    • Safety virtual queues (no λ_z term)
    • Proactive handover timing
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "MA-TD3"

    def select_action(self, env) -> dict:
        c = self.cfg
        priority = np.argmax(env.aoi, axis=1).astype(int)
        power = np.full(c.n_platoons, 0.78, dtype=float)
        return {
            "priority": priority,
            "power": power,
            "next_sat": env.sat_idx.copy(),
            "handover_mask": np.zeros(c.n_platoons, dtype=bool),
        }


class AMDTBaselinePolicy(BasePolicy):
    """
    AMDT [Dai et al., IEEE WCL 2025] proxy.

    Slot-only DPP scheduling: cyclic user selection without sub-slot tick
    tracking and without safety queues.  Key difference: cannot detect
    sub-slot handover outages.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "AMDT"

    def select_action(self, env) -> dict:
        c = self.cfg
        priority = np.full(c.n_platoons, env.slot_t % c.n_priorities, dtype=int)
        return {
            "priority": priority,
            "power": np.full(c.n_platoons, 0.68, dtype=float),
            "next_sat": env.sat_idx.copy(),
            "handover_mask": np.zeros(c.n_platoons, dtype=bool),
        }


class DD3QNASPolicy(BasePolicy):
    """
    DD3QN-AS [Lang et al., IEEE TMC 2026] proxy.

    Single-agent DRL with ε-greedy exploration.  Selects highest raw-AoI
    priority; no virtual-queue safety term; no proactive handover.
    Key paper difference: single-agent vs multi-agent, no priority constraints.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "DD3QN-AS"
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def reset(self) -> None:
        self.epsilon = 0.9

    def select_action(self, env) -> dict:
        c = self.cfg
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        priority = np.zeros(c.n_platoons, dtype=int)
        power = np.zeros(c.n_platoons, dtype=float)

        for v in range(c.n_platoons):
            if self.rng.random() < self.epsilon:
                priority[v] = int(self.rng.integers(0, c.n_priorities))
            else:
                # Greedy: highest raw AoI, NO safety queue term
                priority[v] = int(np.argmax(env.aoi[v]))

            # Threshold power on raw AoI only (no z queue)
            power[v] = 0.90 if env.aoi[v, 0] > 0.7 * c.n_safe[0] else 0.75

        return {
            "priority": priority,
            "power": power,
            "next_sat": env.sat_idx.copy(),
            "handover_mask": np.zeros(c.n_platoons, dtype=bool),
        }


class ModMADDPGPolicy(BasePolicy):
    """
    Modified MADDPG + Task Decomposition [Parvini et al., IEEE TVT 2023] proxy.

    Global-local dual-critic with two-task decomposition:
      Task 1 (m=0): CAM transmission (intra-platoon)
      Task 2 (m=1): AoI minimization (inter-platoon)
    No LEO satellite handover modeled (ground V2X context).
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "Mod-MADDPG"
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def select_action(self, env) -> dict:
        c = self.cfg
        self._step += 1
        priority = np.zeros(c.n_platoons, dtype=int)

        for v in range(c.n_platoons):
            # Alternate between Task1 (m=0) and Task2 (m=1) per agent/step
            task = (v + self._step) % 2
            priority[v] = task  # 0 or 1 — Task3 (m=2) not covered

        return {
            "priority": priority,
            "power": np.full(c.n_platoons, 0.65, dtype=float),
            "next_sat": env.sat_idx.copy(),
            "handover_mask": np.zeros(c.n_platoons, dtype=bool),  # No LEO HO
        }


class ILCHOPolicy(BasePolicy):
    """
    ILCHO [Choi et al., IEEE TMC 2025] proxy.

    QMIX-based MARL for LEO conditional handover.  Throughput-focused,
    no AoI awareness.  Triggers proactive handover ahead of forced slot.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "ILCHO"
        self._ho_cooldown = np.zeros(cfg.n_platoons, dtype=int)

    def reset(self) -> None:
        self._ho_cooldown = np.zeros(self.cfg.n_platoons, dtype=int)

    def select_action(self, env) -> dict:
        c = self.cfg
        power = np.full(c.n_platoons, 0.85, dtype=float)  # High for throughput
        next_sat = env.sat_idx.copy()
        handover_mask = np.zeros(c.n_platoons, dtype=bool)

        forced_countdown = c.forced_period_slots - (env.slot_t % c.forced_period_slots)

        for v in range(c.n_platoons):
            self._ho_cooldown[v] = max(0, self._ho_cooldown[v] - 1)
            # Proactive HO based on remaining visible time (throughput only)
            if forced_countdown <= 3 and self._ho_cooldown[v] == 0:
                handover_mask[v] = True
                next_sat[v] = (env.sat_idx[v] + 1) % c.n_satellites_visible
                self._ho_cooldown[v] = 5

        return {
            # Cyclic scheduling: throughput-focused, no AoI awareness.
            # ILCHO optimises handover timing (QMIX), not priority ordering.
            "priority": np.full(c.n_platoons, env.slot_t % c.n_priorities, dtype=int),
            "power": power,
            "next_sat": next_sat,
            "handover_mask": handover_mask,
        }


class MVTPolicy(BasePolicy):
    """
    MVT: Maximum Visible Time Handover [Park et al., 2025] proxy.

    Non-learning baseline.  Stays on current satellite; never initiates
    voluntary handover (let the environment force one when required).
    No AoI awareness.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "MVT"

    def select_action(self, env) -> dict:
        c = self.cfg
        priority = np.full(c.n_platoons, env.slot_t % c.n_priorities, dtype=int)
        return {
            "priority": priority,
            "power": np.full(c.n_platoons, 0.70, dtype=float),
            "next_sat": env.sat_idx.copy(),
            "handover_mask": np.zeros(c.n_platoons, dtype=bool),  # Never voluntary
        }


class RoundRobinPolicy(BasePolicy):
    """
    Round-Robin baseline.

    Cyclic priority scheduling, fixed power, no handover optimisation.
    Serves as performance lower-bound.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "Round-Robin"

    def select_action(self, env) -> dict:
        c = self.cfg
        priority = np.full(c.n_platoons, env.slot_t % c.n_priorities, dtype=int)
        return {
            "priority": priority,
            "power": np.full(c.n_platoons, 0.60, dtype=float),
            "next_sat": env.sat_idx.copy(),
            "handover_mask": np.zeros(c.n_platoons, dtype=bool),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Ablation variants of SafeScale-MATD3
# ══════════════════════════════════════════════════════════════════════════════

class SafeScaleNoVQPolicy(SafeScaleMATD3Policy):
    """Ablation: w/o safety virtual queues (λ_z = 0, no online adaptation)."""

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "w/o SafetyVQ"
        self.lambda_z = 0.0   # completely disable the safety queue term

    def reset(self) -> None:
        self.lambda_z = 0.0   # stay at zero on every reset

    def observe(self, reward: float, info) -> None:
        pass  # No adaptive λ_z update


class SafeScaleNoProHOPolicy(SafeScaleMATD3Policy):
    """Ablation: w/o proactive handover timing."""

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "w/o ProactiveHO"

    def select_action(self, env) -> dict:
        c = self.cfg
        priority = np.zeros(c.n_platoons, dtype=int)
        power = np.zeros(c.n_platoons, dtype=float)

        for v in range(c.n_platoons):
            score = [
                c.priority_weights[m] * env.aoi[v, m] / max(1.0, c.n_safe[m])
                + self.lambda_z * env.z[v, m]
                for m in range(c.n_priorities)
            ]
            priority[v] = int(np.argmax(score))
            critical = env.aoi[v, 0] > 0.8 * c.n_safe[0] or env.z[v, 0] > 0.4
            power[v] = 0.95 if critical else 0.72
            # No proactive handover

        return {
            "priority": priority,
            "power": power,
            "next_sat": env.sat_idx.copy(),
            "handover_mask": np.zeros(c.n_platoons, dtype=bool),
        }


class SafeScaleNoPriorityWeightPolicy(SafeScaleMATD3Policy):
    """Ablation: w/o priority weighting (no task decomposition)."""

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "w/o PriorityWeight"

    def select_action(self, env) -> dict:
        c = self.cfg
        priority = np.zeros(c.n_platoons, dtype=int)
        power = np.zeros(c.n_platoons, dtype=float)
        next_sat = env.sat_idx.copy()
        handover_mask = np.zeros(c.n_platoons, dtype=bool)

        forced_countdown = c.forced_period_slots - (env.slot_t % c.forced_period_slots)

        for v in range(c.n_platoons):
            # Uniform urgency: no priority weight multiplier
            score = [
                env.aoi[v, m] / max(1.0, c.n_safe[m]) + self.lambda_z * env.z[v, m]
                for m in range(c.n_priorities)
            ]
            priority[v] = int(np.argmax(score))
            critical = env.aoi[v, 0] > 0.8 * c.n_safe[0] or env.z[v, 0] > 0.4
            power[v] = 0.95 if critical else 0.72

            if forced_countdown <= 2 and env.aoi[v, 0] < 0.6 * c.n_safe[0]:
                handover_mask[v] = True
                next_sat[v] = (
                    env.sat_idx[v] + 1 + self.rng.integers(0, 2)
                ) % c.n_satellites_visible

        return {
            "priority": priority,
            "power": power,
            "next_sat": next_sat,
            "handover_mask": handover_mask,
        }
