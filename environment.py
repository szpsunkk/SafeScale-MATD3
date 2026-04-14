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
        self.last_ho_slot = np.full(c.n_platoons, -999, dtype=int)
        self.channel_quality = np.clip(
            self.rng.normal(0.75, 0.08, size=c.n_platoons), 0.4, 0.95
        )
        return self._build_state()

    # ------------------------------------------------------------------
    def _build_state(self) -> np.ndarray:
        c = self.cfg
        forced_countdown = c.forced_period_slots - (self.slot_t % c.forced_period_slots)
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

        forced_now = (self.slot_t % c.forced_period_slots) == 0
        handovers = np.zeros(c.n_platoons, dtype=int)
        ping_pong_flags = np.zeros(c.n_platoons, dtype=int)
        forced_ho = np.zeros(c.n_platoons, dtype=int)

        for v in range(c.n_platoons):
            trigger = bool(ho_mask[v]) or forced_now
            if trigger:
                # When forced, the current satellite leaves view — we MUST change.
                # If the policy's next_sat coincides with current, bump to the next
                # one automatically (models the constellation cycling overhead).
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

        # ── Tick-level evolution ──────────────────────────────────────
        for _ in range(c.n_ac):
            self.channel_quality = np.clip(
                0.9 * self.channel_quality
                + 0.1 * self.rng.normal(0.75, 0.07, size=c.n_platoons),
                0.35,
                0.98,
            )
            for v in range(c.n_platoons):
                in_outage = self.outage_ticks_left[v] > 0
                if in_outage:
                    self.outage_ticks_left[v] -= 1
                for m in range(c.n_priorities):
                    if in_outage:
                        success = False
                    elif priority[v] == m:
                        # Selected priority: boosted success probability
                        pri_gain = [1.15, 1.0, 0.9][m]
                        p_succ = np.clip(
                            0.08
                            + 0.78
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

        self.slot_t += 1
        done = self.slot_t >= c.episode_slots

        mean_aoi = self.aoi.mean(axis=0)
        weighted_aoi = float(np.dot(mean_aoi, np.array(c.priority_weights)))
        reward = -weighted_aoi
        reward -= c.w_power * float(np.mean(power))
        reward -= c.w_handover * float(np.mean(handovers))

        info = StepInfo(
            aoi=self.aoi.copy(),
            violation_count=violation_count,
            handovers=handovers,
            ping_pong_flags=ping_pong_flags,
            forced_ho=forced_ho,
        )
        return self._build_state(), reward, done, info
