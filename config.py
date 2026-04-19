from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    priority_weights: tuple = (4.5, 2.5, 0.5)  # m=1,2,3

    # Two-timescale
    tau_s: float = 1.0
    tau_ac: float = 0.020
    n_ac: int = 50

    # Safety constraints
    # n_safe is the AoI threshold (ticks) before a violation is counted.
    # Paper setting (Table II): nsafe_v,1 / nsafe_v,2 / nsafe_v,3 = 5 / 10 / 50.
    n_safe: tuple = (5, 10, 50)          # m=1,2,3 (ticks)
    epsilon: tuple = (0.01, 0.05, 0.20)  # m=1,2,3
    # Background transmission probability for non-selected priorities.
    # With K=3 subchannels (Table II), each priority receives a dedicated
    # OFDMA sub-channel. The selected priority gets the best subchannel
    # (high power → ~70% success), others get lower-power secondary
    # subchannels at ~35% success rate.
    p_bg: float = 0.42

    # Handover dynamics
    ho_delay_mean_ms: float = 225.0
    ho_delay_std_ms: float = 25.0
    ho_period_s: float = 15.0

    # Training / evaluation
    # episode_slots=150 → 150 s/episode, forced HO every 15 s ≈ 10 forced HOs/ep
    # n_episodes=400 → enough to show convergence trend with NN policies
    # n_seeds=3 → mean ± CI still meaningful, 3× faster than 5
    n_episodes: int = 3000
    episode_slots: int = 150
    eval_episodes: int = 50
    n_seeds: int = 3
    # Train once every `update_interval` env steps (speed/throughput tradeoff).
    update_interval: int = 4

    # Neural SafeScale-MATD3 only (TD3 augmented reward, Lyapunov dual κ on mean Z̄).
    # κ multiplies mean virtual-queue backlog per (platoon, priority); see neural_policies TD3Agent.
    nn_safety_coeff: float = 0.35
    # Ramp κ from 0 → nn_safety_over this many env steps so early learning tracks env reward.
    nn_safety_warmup_env_steps: int = 12000
    # Safety shield: when urgency is high, SafeScale action is constrained to
    # prioritize m=1 and avoid unnecessary discretionary handover outages.
    nn_safety_shield_urgency: float = 1.35
    nn_safety_shield_power_floor: float = 0.50

    # Reward shaping
    w_power: float = 0.02
    w_handover: float = 0.3

    # Reward coefficients (main.pdf Eq. (39)-(42))
    kappa1: float = 1.0      # AoI penalty weight
    kappa2: float = 0.5      # rate-bonus weight
    kappa3_m1: float = 2.0   # m=1 safety overflow quadratic penalty
    kappa3_m2: float = 1.0   # m=2 safety overflow quadratic penalty
    kappa1_m3: float = 0.5   # m=3 AoI penalty weight
    kappa4: float = 0.1      # normalized power penalty
    kappa5: float = 0.3      # proactive handover prediction bonus

    # Throughput model thresholds used in G(R - R_min)
    rmin_m1: float = 0.8
    rmin_m2: float = 0.5

    # Simplified physical-layer reward model constants
    sinr_noise_floor: float = 0.05
    interference_coupling: float = 0.6

    # Proactive timing window (paper uses Tpre = 3 s)
    t_pre_s: float = 3.0

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path("outputs")
    )
    # If set, figures and results go to output_dir / output_run_id / {figures,results}
    # so multiple runs (e.g. different hyperparameters) do not overwrite each other.
    output_run_id: Optional[str] = None

    @property
    def forced_period_slots(self) -> int:
        return max(1, int(round(self.ho_period_s / self.tau_s)))

    @property
    def ho_mean_ticks(self) -> int:
        return max(1, int(round(self.ho_delay_mean_ms / (self.tau_ac * 1000.0))))
