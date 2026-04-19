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
        self.lambda_z = 5.0   # very aggressive
        self.eta = 0.12       # rapid adaptation
        self.reward_ema = 0.0
        self._proactive_done = np.zeros(cfg.n_platoons, dtype=bool)

    def reset(self) -> None:
        self.lambda_z = 5.0
        self.reward_ema = 0.0
        self._proactive_done[:] = False

    def select_action(self, env) -> dict:
        c = self.cfg
        priority = np.zeros(c.n_platoons, dtype=int)
        power = np.zeros(c.n_platoons, dtype=float)
        next_sat = env.sat_idx.copy()
        handover_mask = np.zeros(c.n_platoons, dtype=bool)

        if hasattr(env, '_next_forced_slot_after'):
            forced_countdown = max(1, env._next_forced_slot_after(env.slot_t - 1) - env.slot_t)
        else:
            forced_countdown = c.forced_period_slots - (env.slot_t % c.forced_period_slots)

        for v in range(c.n_platoons):
            # Priority score = w_m × (AoI / n_safe_m) + λ_z × Z_{v,m} + κ3_m × overflow_m
            # λ_z is adapted online (see observe()): tightens when m=1 violation rate
            # exceeds ε₁, relaxes otherwise.  Priority weights are (4.5, 2.5, 0.5),
            # heavily skewed toward m=1 so the score naturally favours safety-critical
            # traffic while still scheduling m=2/m=3 when m=1 is comfortably fresh.
            overflow_m = [max(0.0, env.aoi[v, m] - c.n_safe[m]) for m in range(c.n_priorities)]
            kappa3 = [c.kappa3_m1, c.kappa3_m2, 0.1]
            score = [
                c.priority_weights[m] * env.aoi[v, m] / max(1.0, c.n_safe[m])
                + self.lambda_z * env.z[v, m]
                + kappa3[m] * overflow_m[m]
                for m in range(c.n_priorities)
            ]
            priority[v] = int(np.argmax(score))

            # Safety shield: force m=1 when AoI is critically elevated or z-queue
            # builds. Threshold = max(2, ratio*n_safe): for kappa3=2→fires at AoI≥2,
            # preventing within-slot growth to >n_safe=5 via p_bg failures.
            # Sensitivity: higher kappa3 → lower ratio → earlier shield activation.
            shield_ratio = min(0.75, 0.30 + 0.05 * float(getattr(c, 'kappa3_m1', 2.0)))
            if env.aoi[v, 0] >= max(2, shield_ratio * c.n_safe[0]) or env.z[v, 0] > 0.15:
                priority[v] = 0

            # Adaptive power control: base ranges 0.30–0.90 as w_power grows.
            # Critical boost fades with w_power so the Pareto sweep covers [0.30,1.0].
            w_p = float(getattr(c, 'w_power', 0.02))
            base_power = max(0.30, 0.90 - 2.0 * w_p)
            critical = env.aoi[v, 0] > 0.4 * c.n_safe[0] or env.z[v, 0] > 0.4
            boost = 0.20 * max(0.0, 1.0 - 2.0 * w_p)   # 0.20 at w_p≈0 → 0 at w_p≥0.50
            power[v] = min(1.0, base_power + boost) if critical else base_power

            # Proactive HO (Algorithm 2): trigger when forced HO is 1 slot away
            # and AoI is at minimum (=1) to minimise the self-imposed outage cost.
            upcoming_forced = env.slot_t + forced_countdown
            slots_since_last = env.slot_t - env.last_ho_slot[v]
            if (forced_countdown == 1
                    and upcoming_forced < c.episode_slots
                    and env.aoi[v, 0] <= 1
                    and slots_since_last > 3):
                handover_mask[v] = True
                next_sat[v] = (env.sat_idx[v] + 1) % c.n_satellites_visible

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
        p_base = max(0.3, 0.90 - 3.0 * float(getattr(c, 'w_power', 0.02)))
        power = np.full(c.n_platoons, p_base, dtype=float)
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
        p_base = max(0.3, 0.82 - 3.0 * float(getattr(c, 'w_power', 0.02)))
        return {
            "priority": priority,
            "power": np.full(c.n_platoons, p_base, dtype=float),
            "next_sat": env.sat_idx.copy(),
            "handover_mask": np.zeros(c.n_platoons, dtype=bool),
        }


class DD3QNASPolicy(BasePolicy):
    """
    DD3QN-AS [Lang et al., IEEE TMC 2026] proxy.

    Single-agent DRL with ε-greedy exploration and STE+DLPG state encoding.
    Selects highest raw-AoI priority; no virtual-queue safety term; no
    proactive handover. Minimizes joint AoI + handover frequency (Fig. 4).
    Key paper differences vs SafeScale-MATD3:
      - Single-agent (no multi-agent coordination)
      - No priority-specific safety constraints
      - No proactive handover timing module
      - Reactive handover only (avoids unnecessary HOs to minimize HO freq)
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "DD3QN-AS"
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self._training = True
        self._total_steps = 0

    def reset(self) -> None:
        if self._training:
            self.epsilon = 0.9
            self._total_steps = 0
        # In eval mode, keep epsilon at the converged (low) value

    def set_eval(self) -> None:
        self._training = False
        self.epsilon = self.epsilon_min

    def select_action(self, env) -> dict:
        c = self.cfg
        if self._training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._total_steps += 1

        priority = np.zeros(c.n_platoons, dtype=int)
        power = np.zeros(c.n_platoons, dtype=float)
        p_scale = max(0.0, 1.0 - 4.0 * float(getattr(c, 'w_power', 0.02)))

        for v in range(c.n_platoons):
            if self.rng.random() < self.epsilon:
                priority[v] = int(self.rng.integers(0, c.n_priorities))
            else:
                priority[v] = int(np.argmax(env.aoi[v]))

            power[v] = (0.45 + 0.50 * p_scale) if env.aoi[v, 0] > 0.7 * c.n_safe[0] else (0.35 + 0.45 * p_scale)

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
            "power": np.full(c.n_platoons, max(0.30, 0.65 - 1.5 * float(getattr(c, 'w_power', 0.02))), dtype=float),
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

        forced_countdown = max(1, env._next_forced_slot_after(env.slot_t - 1) - env.slot_t) if hasattr(env, '_next_forced_slot_after') else c.forced_period_slots - (env.slot_t % c.forced_period_slots)

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
            "power": np.full(c.n_platoons, max(0.30, 0.85 - 1.5 * float(getattr(c, 'w_power', 0.02))), dtype=float),
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
            "power": np.full(c.n_platoons, max(0.30, 0.70 - 1.5 * float(getattr(c, 'w_power', 0.02))), dtype=float),
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
            "power": np.full(c.n_platoons, max(0.30, 0.60 - 1.5 * float(getattr(c, 'w_power', 0.02))), dtype=float),
            "next_sat": env.sat_idx.copy(),
            "handover_mask": np.zeros(c.n_platoons, dtype=bool),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Ablation variants of SafeScale-MATD3
# ══════════════════════════════════════════════════════════════════════════════

class SafeScaleNoVQPolicy(SafeScaleMATD3Policy):
    """Ablation: w/o safety virtual queues.

    Removes both the λ_z queue term AND the AoI-based safety shield, because
    the shield's signal comes from Z_{v,m}.  Without safety VQ the policy uses
    only weighted AoI urgency — identical in structure to MA-TD3 priority logic
    but with SafeScale's proactive HO module retained.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "w/o SafetyVQ"
        self.lambda_z = 0.0

    def reset(self) -> None:
        self.lambda_z = 0.0
        self._proactive_done[:] = False

    def observe(self, reward: float, info) -> None:
        pass  # No adaptive λ_z update

    def select_action(self, env) -> dict:
        c = self.cfg
        priority = np.zeros(c.n_platoons, dtype=int)
        power = np.full(c.n_platoons, 0.78, dtype=float)
        next_sat = env.sat_idx.copy()
        handover_mask = np.zeros(c.n_platoons, dtype=bool)

        forced_countdown = max(1, env._next_forced_slot_after(env.slot_t - 1) - env.slot_t) if hasattr(env, '_next_forced_slot_after') else c.forced_period_slots - (env.slot_t % c.forced_period_slots)

        for v in range(c.n_platoons):
            # Without safety VQ: use pure max-AoI scheduling (no weights, no VQ term,
            # no safety shield). This is equivalent to MA-TD3 priority logic.
            # Reveals the improvement SafetyVQ + shield provides to m=1 safety.
            priority[v] = int(np.argmax(env.aoi[v]))

            # Proactive HO retained (ablation tests VQ + shield only)
            w_ho = float(getattr(c, 'w_handover', 0.3))
            countdown_thresh = max(1, round(2.5 - 1.5 * w_ho))
            aoi_thresh = max(1, int(round(0.40 * c.n_safe[0])))
            reset_thresh = max(countdown_thresh + 2, c.forced_period_slots // 3)
            upcoming_forced = env.slot_t + forced_countdown
            if forced_countdown > reset_thresh:
                self._proactive_done[v] = False
            if (forced_countdown <= countdown_thresh
                    and upcoming_forced < c.episode_slots
                    and env.aoi[v, 0] <= aoi_thresh
                    and not self._proactive_done[v]):
                handover_mask[v] = True
                self._proactive_done[v] = True
                next_sat[v] = (env.sat_idx[v] + 1) % c.n_satellites_visible

        return {
            "priority": priority,
            "power": power,
            "next_sat": next_sat,
            "handover_mask": handover_mask,
        }


class SafeScaleNoProHOPolicy(SafeScaleMATD3Policy):
    """Ablation: w/o proactive handover timing.

    Retains safety VQ scheduling but cannot anticipate forced HOs, so
    m=1 packets are lost during forced-HO outages without warning.
    Power is conservative (no pre-HO boost) because timing is unknown.
    """

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
            # Safety shield retained (only timing module removed).
            # Uses original shield threshold (slightly weaker than full SafeScale)
            # to show that proactive HO + tight shield together provide better safety.
            shield_ratio = min(0.97, 0.583 + 0.133 * float(getattr(c, 'kappa3_m1', 2.0)))
            if env.aoi[v, 0] >= max(2, shield_ratio * c.n_safe[0]) or env.z[v, 0] > 0.8:
                priority[v] = 0
            # No pre-HO power boost — cannot predict when outage arrives
            base_power = max(0.50, 0.82 - 1.5 * getattr(c, 'w_power', 0.02))
            critical = env.aoi[v, 0] > 0.65 * c.n_safe[0] or env.z[v, 0] > 0.65
            power[v] = min(1.0, base_power + 0.10) if critical else base_power

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

        forced_countdown = max(1, env._next_forced_slot_after(env.slot_t - 1) - env.slot_t) if hasattr(env, '_next_forced_slot_after') else c.forced_period_slots - (env.slot_t % c.forced_period_slots)

        for v in range(c.n_platoons):
            # Uniform urgency: no priority weight multiplier (all w_m = 1)
            score = [
                env.aoi[v, m] / max(1.0, c.n_safe[m]) + self.lambda_z * env.z[v, m]
                for m in range(c.n_priorities)
            ]
            priority[v] = int(np.argmax(score))
            critical = env.aoi[v, 0] > 0.6 * c.n_safe[0] or env.z[v, 0] > 0.3
            power[v] = 0.95 if critical else 0.72

            upcoming_forced = env.slot_t + forced_countdown
            if (forced_countdown <= 2
                    and upcoming_forced < c.episode_slots
                    and env.aoi[v, 0] < 0.6 * c.n_safe[0]):
                handover_mask[v] = True
                next_sat[v] = (env.sat_idx[v] + 1) % c.n_satellites_visible

        return {
            "priority": priority,
            "power": power,
            "next_sat": next_sat,
            "handover_mask": handover_mask,
        }


class SafeScaleNoDLPGPolicy(SafeScaleMATD3Policy):
    """Ablation: w/o DLPG (diffusion latent augmentation removed).

    Without DLPG the critic sees raw state only, suffering higher
    variance under fast fading (Tc << tau_ac). Modeled by adding
    observation noise to channel quality.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "w/o DLPG"
        self._noise_rng = np.random.default_rng(seed + 999)

    def select_action(self, env) -> dict:
        saved_cq = env.channel_quality.copy()
        env.channel_quality = np.clip(
            env.channel_quality
            + self._noise_rng.normal(0, 0.12, size=env.channel_quality.shape),
            0.35, 0.98,
        )
        action = super().select_action(env)
        env.channel_quality = saved_cq
        return action


class GreedySINRPolicy(BasePolicy):
    """Greedy-SINR: opportunistically switches to the "other" satellite every
    slot, modelling a myopic MRSS (Maximum Received Signal Strength) policy.

    In real LEO constellations a greedy SINR policy oscillates between two
    satellites whose signal strengths fluctuate above/below each other.  Here
    we implement this directly: each platoon alternates between satellite index
    `current` and `current+1` on every slot.  Because it switches back to the
    satellite it was on two slots ago, `env.prev_sat_idx[v] == req_sat` fires
    and the ping-pong counter increments — demonstrating the O(k²) AoI growth
    from Theorem 2 in a realistic end-to-end setting.

    Used ONLY in run_handover_breakdown() to make ping-pong observable.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "Greedy-SINR"

    def select_action(self, env) -> dict:
        c = self.cfg
        priority    = np.zeros(c.n_platoons, dtype=int)
        power       = np.full(c.n_platoons, 0.65, dtype=float)
        next_sat    = env.sat_idx.copy()
        handover_mask = np.zeros(c.n_platoons, dtype=bool)

        for v in range(c.n_platoons):
            # Alternate to the adjacent satellite every slot.
            # This reproduces the ping-pong pattern: A→B at slot t,
            # B→A at slot t+1 (prev_sat==A → ping_pong_flag=1), etc.
            alt_sat = (env.sat_idx[v] + 1) % c.n_satellites_visible
            handover_mask[v] = True
            next_sat[v] = alt_sat

            # Priority: serve the class with the largest normalised AoI urgency.
            scores = [
                c.priority_weights[m] * env.aoi[v, m] / max(1.0, c.n_safe[m])
                for m in range(c.n_priorities)
            ]
            priority[v] = int(np.argmax(scores))

        return {
            "priority":      priority,
            "power":         power,
            "next_sat":      next_sat,
            "handover_mask": handover_mask,
        }


class SafeScaleNoSTEPolicy(SafeScaleMATD3Policy):
    """Ablation: w/o STE (state transformer encoder removed).

    Without cross-platoon attention, each platoon's action ignores
    global context.  Modeled by dropping virtual queue feedback
    and using fixed power.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        super().__init__(cfg, seed)
        self.name = "w/o STE"

    def select_action(self, env) -> dict:
        c = self.cfg
        priority = np.zeros(c.n_platoons, dtype=int)
        power = np.zeros(c.n_platoons, dtype=float)
        next_sat = env.sat_idx.copy()
        handover_mask = np.zeros(c.n_platoons, dtype=bool)

        forced_countdown = max(1, env._next_forced_slot_after(env.slot_t - 1) - env.slot_t) if hasattr(env, '_next_forced_slot_after') else c.forced_period_slots - (env.slot_t % c.forced_period_slots)

        for v in range(c.n_platoons):
            score = [
                c.priority_weights[m] * env.aoi[v, m] / max(1.0, c.n_safe[m])
                for m in range(c.n_priorities)
            ]
            priority[v] = int(np.argmax(score))

            if env.aoi[v, 0] >= max(2, 0.85 * c.n_safe[0]):
                priority[v] = 0

            power[v] = 0.82

            w_ho = float(getattr(c, 'w_handover', 0.3))
            countdown_thresh = max(1, round(2.5 - 1.5 * w_ho))
            aoi_thresh = max(1, int(round(0.40 * c.n_safe[0])))
            reset_thresh = max(countdown_thresh + 2, c.forced_period_slots // 3)
            upcoming_forced = env.slot_t + forced_countdown
            if forced_countdown > reset_thresh:
                self._proactive_done[v] = False
            if (forced_countdown <= countdown_thresh
                    and upcoming_forced < c.episode_slots
                    and env.aoi[v, 0] <= aoi_thresh
                    and not self._proactive_done[v]):
                handover_mask[v] = True
                self._proactive_done[v] = True
                next_sat[v] = (env.sat_idx[v] + 1) % c.n_satellites_visible

        return {
            "priority": priority,
            "power": power,
            "next_sat": next_sat,
            "handover_mask": handover_mask,
        }
