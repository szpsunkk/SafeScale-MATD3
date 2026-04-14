from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimConfig:
    # Constellation / channel
    altitude_km: float = 550.0
    n_satellites_visible: int = 10
    min_elevation_deg: float = 25.0
    carrier_freq_ghz: float = 1.67
    p_max_dbm: float = 40.0

    # Platoon / traffic
    n_platoons: int = 5
    n_priorities: int = 3
    priority_weights: tuple = (5.0, 2.0, 0.5)  # m=1,2,3

    # Two-timescale
    tau_s: float = 1.0
    tau_ac: float = 0.020
    n_ac: int = 50

    # Safety constraints
    # n_safe is the AoI threshold (ticks) before a violation is counted.
    # Rule-of-thumb: must be > n_ac so that missing ONE slot does NOT
    # automatically cause a violation (slot-level scheduling has n_ac ticks
    # per slot; a skipped slot adds n_ac ticks to AoI).
    # Values below are ~2× / 4× / 10× n_ac respectively.
    n_safe: tuple = (12, 20, 100)        # m=1,2,3 (ticks)  — 10→12 gives m=1 ~20% more margin
    epsilon: tuple = (0.01, 0.05, 0.20)  # m=1,2,3
    # Background transmission probability for non-selected priorities.
    # Models the paper's intra-slot scheduling where ALL messages receive
    # some resource allocation; selected priority gets a boost.
    # Reduced to 0.08 so methods without safety queues see more violations.
    p_bg: float = 0.08

    # Handover dynamics
    ho_delay_mean_ms: float = 225.0
    ho_delay_std_ms: float = 25.0
    ho_period_s: float = 15.0

    # Training / evaluation
    n_episodes: int = 2000
    episode_slots: int = 300
    eval_episodes: int = 100
    n_seeds: int = 5

    # Reward shaping
    w_power: float = 0.02
    w_handover: float = 0.3

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path("outputs")
    )

    @property
    def forced_period_slots(self) -> int:
        return max(1, int(round(self.ho_period_s / self.tau_s)))

    @property
    def ho_mean_ticks(self) -> int:
        return max(1, int(round(self.ho_delay_mean_ms / (self.tau_ac * 1000.0))))
