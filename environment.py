from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config import SimConfig


@dataclass
class StepInfo:
    aoi: np.ndarray
    violation_count: np.ndarray
    handovers: np.ndarray
    ping_pong_flags: np.ndarray
    forced_ho: np.ndarray   # 1 if the handover was environment-forced, 0 otherwise


class UnifiedEnvironment:
    """
    Unified tick-level environment for all methods.
    All baselines run inside the same two-timescale tick model so comparisons
    are fair.  Tick resolution is tau_ac; slot resolution is tau_s.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        c = self.cfg
        self.slot_t = 0
        self.aoi = np.ones((c.n_platoons, c.n_priorities), dtype=float)
        self.z = np.zeros((c.n_platoons, c.n_priorities), dtype=float)
        self.sat_idx = self.rng.integers(0, c.n_satellites_visible, size=c.n_platoons)
        self.prev_sat_idx = self.sat_idx.copy()
        self.outage_ticks_left = np.zeros(c.n_platoons, dtype=int)
        self.last_ho_slot = np.zeros(c.n_platoons, dtype=int)
        self.channel_quality = np.clip(
            self.rng.normal(0.75, 0.08, size=c.n_platoons), 0.4, 0.95
        )
        # Stochastic forced-HO schedule: Poisson gaps instead of deterministic period.
        # This ensures different seeds/episodes produce different HO counts per method,
        # making the HO breakdown figure meaningful.
        self._forced_ho_slots = self._generate_forced_ho_slots()
        return self._build_state()

    def _generate_forced_ho_slots(self) -> set:
        """Generate forced-HO times via Poisson process (mean = ho_period_s slots)."""
        slots = set()
        t = 0
        period = self.cfg.forced_period_slots
        while t < self.cfg.episode_slots:
            gap = max(1, int(self.rng.exponential(period)))
            t += gap
            if t < self.cfg.episode_slots:
                slots.add(t)
        return slots

    def _next_forced_slot_after(self, current: int) -> int:
        """Return the nearest forced-HO slot strictly after `current`."""
        future = [s for s in self._forced_ho_slots if s > current]
        return min(future) if future else self.cfg.episode_slots

    # ------------------------------------------------------------------
    def _build_state(self) -> np.ndarray:
        c = self.cfg
        next_forced = self._next_forced_slot_after(self.slot_t - 1)
        forced_countdown = max(1, next_forced - self.slot_t)
        state = np.concatenate(
            [
                self.aoi.reshape(-1),
                self.z.reshape(-1),
                self.channel_quality,
                np.array([forced_countdown / c.forced_period_slots], dtype=float),
            ]
        )
        return state

    # ------------------------------------------------------------------
    def _sample_handover_ticks(self) -> int:
        ticks = int(
            round(
                self.rng.normal(
                    self.cfg.ho_delay_mean_ms / (self.cfg.tau_ac * 1000.0),
                    self.cfg.ho_delay_std_ms / (self.cfg.tau_ac * 1000.0),
                )
            )
        )
        return max(1, ticks)

    def _apply_handover(self, v: int, new_sat: int) -> bool:
        old_sat = self.sat_idx[v]
        if new_sat == old_sat:
            return False
        self.prev_sat_idx[v] = old_sat
        self.sat_idx[v] = new_sat
        self.outage_ticks_left[v] = self._sample_handover_ticks()
        self.last_ho_slot[v] = self.slot_t
        return True

    # ------------------------------------------------------------------
    def step(self, action: dict) -> tuple[np.ndarray, float, bool, StepInfo]:
        """
        action keys
        -----------
        priority      : [n_platoons]  int in {0,1,2}
        power         : [n_platoons]  float in [0,1]
        next_sat      : [n_platoons]  int in [0, n_satellites_visible-1]
        handover_mask : [n_platoons]  bool
        """
        c = self.cfg
        priority = action["priority"]
        power = np.clip(action["power"], 0.0, 1.0)
        next_sat = action["next_sat"]
        ho_mask = action["handover_mask"]

        forced_slot = self.slot_t in self._forced_ho_slots
        next_forced = self._next_forced_slot_after(self.slot_t)
        forced_countdown_curr = max(1, next_forced - self.slot_t)
        handovers = np.zeros(c.n_platoons, dtype=int)
        ping_pong_flags = np.zeros(c.n_platoons, dtype=int)
        forced_ho = np.zeros(c.n_platoons, dtype=int)

        for v in range(c.n_platoons):
            # A proactive HO done within a small window before the forced
            # slot means the satellite was already changed — the old
            # satellite exit no longer triggers a second outage.
            recently_switched = (self.slot_t - self.last_ho_slot[v]) <= 3
            forced_now = forced_slot and not recently_switched
            trigger = bool(ho_mask[v]) or forced_now
            if trigger:
                req_sat = int(next_sat[v])
                if forced_now and req_sat == self.sat_idx[v]:
                    req_sat = (self.sat_idx[v] + 1) % c.n_satellites_visible
                changed = self._apply_handover(v, req_sat)
                handovers[v] = int(changed)
                if changed:
                    forced_ho[v] = int(forced_now and not bool(ho_mask[v]))
                    if self.prev_sat_idx[v] == req_sat:
                        ping_pong_flags[v] = 1

        violation_count = np.zeros((c.n_platoons, c.n_priorities), dtype=int)
        rate_sum = np.zeros(c.n_platoons, dtype=float)
        interf_sum = np.zeros(c.n_platoons, dtype=float)

        # ── Tick-level evolution ──────────────────────────────────────
        for _ in range(c.n_ac):
            self.channel_quality = np.clip(
                0.9 * self.channel_quality
                + 0.1 * self.rng.normal(0.75, 0.07, size=c.n_platoons),
                0.35,
                0.98,
            )
            in_outage_vec = self.outage_ticks_left > 0
            for v in range(c.n_platoons):
                in_outage = bool(in_outage_vec[v])
                if in_outage:
                    rate_tick = 0.0
                    interf_tick = 0.0
                else:
                    same_sat = (self.sat_idx == self.sat_idx[v])
                    tx_mask = same_sat & (~in_outage_vec)
                    tx_mask[v] = False
                    interf_tick = float(
                        np.sum(power[tx_mask] * self.channel_quality[tx_mask])
                        * c.interference_coupling
                    )
                    signal_tick = float(power[v] * self.channel_quality[v])
                    sinr_tick = signal_tick / (c.sinr_noise_floor + interf_tick)
                    rate_tick = float(np.log2(1.0 + sinr_tick))

                rate_sum[v] += rate_tick
                interf_sum[v] += interf_tick
                for m in range(c.n_priorities):
                    if in_outage:
                        success = False
                    elif priority[v] == m:
                        # Selected priority: high success via MRT beamforming (Eq. 38)
                        pri_gain = [1.15, 1.0, 0.9][m]
                        p_succ = np.clip(
                            0.10
                            + 0.85
                            * self.channel_quality[v]
                            * (0.55 + 0.45 * power[v])
                            * pri_gain,
                            0.02,
                            0.98,
                        )
                        success = bool(self.rng.random() < p_succ)
                    else:
                        # Non-selected priority: background service probability.
                        # Models intra-slot scheduling where all messages receive
                        # some resource allocation (OFDMA sub-channel sharing).
                        p_bg = getattr(c, 'p_bg', 0.0)
                        success = (p_bg > 0.0) and bool(self.rng.random() < p_bg)

                    if success:
                        self.aoi[v, m] = 1.0
                    else:
                        self.aoi[v, m] += 1.0

                    if self.aoi[v, m] > c.n_safe[m]:
                        violation_count[v, m] += 1

                    violation = 1.0 if self.aoi[v, m] > c.n_safe[m] else 0.0
                    self.z[v, m] = max(0.0, self.z[v, m] + violation - c.epsilon[m])

            self.outage_ticks_left[in_outage_vec] -= 1

        self.slot_t += 1
        done = self.slot_t >= c.episode_slots

        # Reward follows main.pdf Eq. (39)-(42):
        # local task rewards + predictive bonus + global interference reward.
        rate = rate_sum / max(1, c.n_ac)
        interference = interf_sum / max(1, c.n_ac)

        n_safe = np.asarray(c.n_safe, dtype=float)
        aoi1 = self.aoi[:, 0]
        aoi2 = self.aoi[:, 1]
        aoi3 = self.aoi[:, 2]
        overflow1 = np.maximum(0.0, aoi1 - n_safe[0])
        overflow2 = np.maximum(0.0, aoi2 - n_safe[1])
        g1 = np.maximum(0.0, rate - float(c.rmin_m1))
        g2 = np.maximum(0.0, rate - float(c.rmin_m2))

        # T_pre-triggered bonus: encourage sending m=1 traffic before forced HO.
        t_pre_slots = max(1, int(round(c.t_pre_s / c.tau_s)))
        pred_bonus = (
            (priority == 0)
            & (forced_countdown_curr <= t_pre_slots)
            & (handovers == 0)
        ).astype(float)

        local_r = (
            -c.kappa1 * aoi1
            + c.kappa2 * g1
            - c.kappa3_m1 * (overflow1 ** 2)
            - c.kappa4 * power
            - c.kappa1 * aoi2
            + c.kappa2 * g2
            - c.kappa3_m2 * (overflow2 ** 2)
            - c.kappa4 * power
            - c.kappa1_m3 * aoi3
            + c.kappa5 * pred_bonus
        )
        global_r = -float(np.mean(np.log10(np.clip(interference, 1e-3, None))))
        global_r = float(np.clip(global_r, -3.0, 3.0))
        reward = float(np.mean(local_r) + global_r)

        info = StepInfo(
            aoi=self.aoi.copy(),
            violation_count=violation_count,
            handovers=handovers,
            ping_pong_flags=ping_pong_flags,
            forced_ho=forced_ho,
        )
        return self._build_state(), reward, done, info
