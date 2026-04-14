"""
neural_policies.py
══════════════════
PyTorch + CUDA 实现的神经网络 RL 策略。

包含：
  ReplayBuffer          – GPU-ready 经验回放池
  ActorMLP              – 共享参数的去中心化 Actor
  TwinCriticMLP         – TD3 双 Critic（集中式训练）
  TD3Agent              – TD3 训练核心（支持安全惩罚项）
  SafeScaleMATD3NNPolicy – 论文提出算法（含安全虚拟队列惩罚）
  MATD3NNPolicy          – MA-TD3 baseline（不含安全约束）

使用方式与 BasePolicy 完全相同，可直接替换 experiments.py 中的 policies。
"""
from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SimConfig
from policies import BasePolicy

# ── 全局设备选择 ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[neural_policies] Using device: {DEVICE}"
      + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))


# ══════════════════════════════════════════════════════════════════════════════
# 经验回放池
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """环形缓冲区，采样时直接返回 GPU 张量。"""

    def __init__(self, capacity: int = 200_000):
        self.buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action_vec: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buf.append((state, action_vec, np.float32(reward), next_state, np.float32(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.from_numpy(np.array(s, dtype=np.float32)).to(DEVICE),
            torch.from_numpy(np.array(a, dtype=np.float32)).to(DEVICE),
            torch.from_numpy(np.array(r, dtype=np.float32)).unsqueeze(1).to(DEVICE),
            torch.from_numpy(np.array(ns, dtype=np.float32)).to(DEVICE),
            torch.from_numpy(np.array(d, dtype=np.float32)).unsqueeze(1).to(DEVICE),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ══════════════════════════════════════════════════════════════════════════════
# 网络结构
# ══════════════════════════════════════════════════════════════════════════════

class ActorMLP(nn.Module):
    """
    共享参数的去中心化 Actor。

    输入：每辆车局部观测 (batch × n_platoons, local_obs_dim)
    输出：原始动作 logits (batch × n_platoons, local_act_dim)
      - 前 n_priorities 维：priority logits  (softmax→离散)
      - 第 n_priorities 维：power logit      (sigmoid→[0,1])
      - 最后 1 维：handover logit           (sigmoid→{0,1})
    """

    def __init__(self, local_obs_dim: int, local_act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, local_act_dim),
        )
        # 正交初始化，稳定早期训练
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        return self.net(local_obs)


class TwinCriticMLP(nn.Module):
    """
    TD3 集中式双 Critic。

    输入：全局状态 + 全部智能体动作（拼接后）
    输出：Q1, Q2 各 1 个标量
    """

    def __init__(self, state_dim: int, total_act_dim: int, hidden: int = 256):
        super().__init__()
        inp = state_dim + total_act_dim

        def _block():
            return nn.Sequential(
                nn.Linear(inp, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = _block()
        self.q2 = _block()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# TD3 训练核心
# ══════════════════════════════════════════════════════════════════════════════

class TD3Agent:
    """
    TD3 核心：支持 CUDA，可选安全惩罚系数 κ。

    当 safety_coeff > 0 时，奖励被增广为：
        r̃ = r_env − κ × Σ_{v,m} Z_{v,m}
    这对应 Lyapunov DPP 安全约束的 Lagrangian 项。
    """

    # ── 状态/动作维度计算 ──────────────────────────────────────────────────────
    # 局部观测 per platoon:
    #   aoi[v,:] / n_safe  (n_priorities)
    #   z[v,:]   / Z_MAX   (n_priorities)
    #   channel_quality[v]  (1)
    #   forced_countdown    (1)
    # = 2*n_priorities + 2
    #
    # 局部动作 per platoon:
    #   priority logits     (n_priorities)
    #   power               (1)
    #   handover logit      (1)
    # = n_priorities + 2
    #
    # 全局动作 (replay buffer):
    #   n_platoons × (n_priorities + 2)

    Z_MAX = 20.0   # 虚拟队列归一化上界

    def __init__(
        self,
        cfg: SimConfig,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        buffer_size: int = 200_000,
        batch_size: int = 256,
        noise_std: float = 0.20,
        noise_clip: float = 0.50,
        safety_coeff: float = 0.0,
        hidden: int = 256,
    ):
        self.cfg = cfg
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.safety_coeff = safety_coeff

        n_p = cfg.n_platoons
        n_pri = cfg.n_priorities
        n_safe = np.array(cfg.n_safe, dtype=np.float32)

        self.n_platoons = n_p
        self.n_priorities = n_pri
        self.n_safe = n_safe
        self.n_satellites = cfg.n_satellites_visible

        self.local_obs_dim = 2 * n_pri + 2          # 8
        self.local_act_dim = n_pri + 2               # 5
        self.state_dim = n_p * n_pri * 2 + n_p + 1  # 36
        self.total_act_dim = n_p * self.local_act_dim  # 25

        # ── 网络 ──────────────────────────────────────────────────────────────
        self.actor = ActorMLP(self.local_obs_dim, self.local_act_dim, hidden).to(DEVICE)
        self.actor_t = ActorMLP(self.local_obs_dim, self.local_act_dim, hidden).to(DEVICE)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.critic = TwinCriticMLP(self.state_dim, self.total_act_dim, hidden).to(DEVICE)
        self.critic_t = TwinCriticMLP(self.state_dim, self.total_act_dim, hidden).to(DEVICE)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # ── 回放池 ────────────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(buffer_size)

        # ── 探索噪声衰减 ──────────────────────────────────────────────────────
        self.explore_noise = 0.30
        self.explore_min = 0.05
        self.explore_decay = 0.9998
        self._step = 0

    # ── 辅助：提取局部观测 ─────────────────────────────────────────────────────
    def _local_obs_from_env(self, env) -> np.ndarray:
        forced_norm = (
            env.cfg.forced_period_slots - (env.slot_t % env.cfg.forced_period_slots)
        ) / env.cfg.forced_period_slots
        lobs = np.empty((self.n_platoons, self.local_obs_dim), dtype=np.float32)
        for v in range(self.n_platoons):
            lobs[v] = np.concatenate([
                env.aoi[v] / self.n_safe,
                np.clip(env.z[v], 0, self.Z_MAX) / self.Z_MAX,
                [env.channel_quality[v]],
                [forced_norm],
            ])
        return lobs   # (n_p, local_obs_dim)

    # ── 辅助：原始 logits → 连续动作向量（可微，用于 Critic 训练）─────────────
    def _raw_to_act_vec(self, raw: torch.Tensor) -> torch.Tensor:
        """
        raw: (..., n_p, local_act_dim)
        returns: (..., n_p * local_act_dim) 归一化连续向量
        """
        n_pri = self.n_priorities
        pri_soft = F.softmax(raw[..., :n_pri], dim=-1)          # (..., n_p, n_pri)
        power = torch.sigmoid(raw[..., n_pri:n_pri + 1])         # (..., n_p, 1)
        ho = torch.sigmoid(raw[..., n_pri + 1:n_pri + 2])        # (..., n_p, 1)
        vec = torch.cat([pri_soft, power, ho], dim=-1)           # (..., n_p, n_pri+2)
        return vec.flatten(-2)                                    # (..., n_p*(n_pri+2))

    # ── 辅助：act_vec (numpy) 打包进 buffer ────────────────────────────────────
    def _pack_action(
        self, priority: np.ndarray, power: np.ndarray, ho_mask: np.ndarray
    ) -> np.ndarray:
        parts = []
        for v in range(self.n_platoons):
            onehot = np.zeros(self.n_priorities, dtype=np.float32)
            onehot[priority[v]] = 1.0
            parts += [onehot, [float(power[v])], [float(ho_mask[v])]]
        return np.concatenate(parts)   # (total_act_dim,)

    # ── 选动作（推理，含探索噪声）─────────────────────────────────────────────
    @torch.no_grad()
    def act(self, env) -> tuple:
        lobs = self._local_obs_from_env(env)
        t = torch.from_numpy(lobs).to(DEVICE)     # (n_p, local_obs_dim)
        raw = self.actor(t)                        # (n_p, local_act_dim)

        n_pri = self.n_priorities
        # priority: argmax + 噪声扰动 logits
        pri_logits = raw[:, :n_pri]
        noise = torch.randn_like(pri_logits) * self.explore_noise
        priority = (pri_logits + noise).argmax(dim=-1).cpu().numpy().astype(int)

        # power: sigmoid
        power = torch.sigmoid(raw[:, n_pri]).cpu().numpy().clip(0.0, 1.0)

        # handover: sigmoid + Bernoulli
        ho_prob = torch.sigmoid(raw[:, n_pri + 1]).cpu().numpy()
        threshold = 0.5 - self.explore_noise * 0.2
        ho_mask = (ho_prob > threshold).astype(bool)

        # next_sat: proactive offset driven by priority/ho
        next_sat = np.where(
            ho_mask,
            (env.sat_idx + 1) % self.n_satellites,
            env.sat_idx.copy(),
        )

        action = {
            "priority":      priority,
            "power":         power,
            "next_sat":      next_sat,
            "handover_mask": ho_mask,
        }
        return action, self._pack_action(priority, power, ho_mask)

    # ── 存储 + 更新 ────────────────────────────────────────────────────────────
    def store_and_update(
        self,
        prev_state: np.ndarray,
        act_vec: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        z_sum: float,
    ) -> None:
        # 安全惩罚增广奖励
        r_aug = reward - self.safety_coeff * float(z_sum)
        self.buffer.push(prev_state, act_vec, r_aug, next_state, done)
        self._train()

    # ── TD3 梯度更新 ───────────────────────────────────────────────────────────
    def _train(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        self._step += 1
        self.explore_noise = max(self.explore_min, self.explore_noise * self.explore_decay)

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        B = states.shape[0]

        # ── Target 动作（含噪声平滑）──────────────────────────────────────────
        with torch.no_grad():
            # next_states → (B * n_p, local_obs_dim)
            ns_lobs = self._batch_state_to_lobs(next_states)     # (B*n_p, obs_dim)
            raw_next = self.actor_t(ns_lobs).view(B, self.n_platoons, -1)
            noise = (
                torch.randn_like(raw_next) * self.noise_std
            ).clamp(-self.noise_clip, self.noise_clip)
            raw_next_noisy = raw_next + noise

            next_act_vec = self._raw_to_act_vec(raw_next_noisy)  # (B, total_act_dim)
            q1_t, q2_t = self.critic_t(next_states, next_act_vec)
            q_target = rewards + (1.0 - dones) * self.gamma * torch.min(q1_t, q2_t)

        # ── Critic 更新 ───────────────────────────────────────────────────────
        q1, q2 = self.critic(states, actions)
        c_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor 更新（延迟）─────────────────────────────────────────────────
        if self._step % self.policy_delay == 0:
            s_lobs = self._batch_state_to_lobs(states)           # (B*n_p, obs_dim)
            raw_curr = self.actor(s_lobs).view(B, self.n_platoons, -1)
            curr_act_vec = self._raw_to_act_vec(raw_curr)        # (B, total_act_dim)
            a_loss = -self.critic.q1_only(states, curr_act_vec).mean()
            self.actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            # Soft update 目标网络
            for p, tp in zip(self.actor.parameters(), self.actor_t.parameters()):
                tp.data.lerp_(p.data, self.tau)
            for p, tp in zip(self.critic.parameters(), self.critic_t.parameters()):
                tp.data.lerp_(p.data, self.tau)

    # ── 辅助：批量 state → 局部观测 ───────────────────────────────────────────
    def _batch_state_to_lobs(self, states: torch.Tensor) -> torch.Tensor:
        """
        states: (B, state_dim)
        返回: (B*n_p, local_obs_dim)

        state 布局（由 environment._build_state 定义）：
          [0 : n_p*n_pri]           = aoi 展平
          [n_p*n_pri : 2*n_p*n_pri] = z   展平
          [2*n_p*n_pri : 2*n_p*n_pri+n_p] = channel_quality
          [-1]                      = forced_countdown_norm
        """
        B = states.shape[0]
        n_p, n_pri = self.n_platoons, self.n_priorities
        ns = torch.from_numpy(self.n_safe).to(DEVICE)  # (n_pri,)

        aoi = states[:, :n_p * n_pri].view(B, n_p, n_pri) / ns          # (B,n_p,n_pri)
        z = (states[:, n_p * n_pri:2 * n_p * n_pri]
             .view(B, n_p, n_pri)
             .clamp(0, self.Z_MAX) / self.Z_MAX)                          # (B,n_p,n_pri)
        cq = states[:, 2 * n_p * n_pri:2 * n_p * n_pri + n_p]           # (B, n_p)
        fc = states[:, -1:]                                               # (B, 1)

        cq_exp = cq.unsqueeze(-1)                                         # (B,n_p,1)
        fc_exp = fc.unsqueeze(1).expand(B, n_p, 1)                       # (B,n_p,1)

        lobs = torch.cat([aoi, z, cq_exp, fc_exp], dim=-1)               # (B,n_p,local_obs_dim)
        return lobs.reshape(B * n_p, -1)                                  # (B*n_p, local_obs_dim)

    def reset_noise(self) -> None:
        """每个 episode 重置探索噪声（可选）。"""
        pass  # 使用衰减式噪声，无需强制重置


# ══════════════════════════════════════════════════════════════════════════════
# 与 BasePolicy 兼容的封装层
# ══════════════════════════════════════════════════════════════════════════════

class _NNPolicyBase(BasePolicy):
    """
    BasePolicy 子类：select_action / observe 对接 TD3Agent。
    内部保存 prev_state 和 prev_act_vec，供 observe() 使用。
    """

    def __init__(self, cfg: SimConfig, agent: TD3Agent, seed: int = 0):
        super().__init__(cfg, seed)
        self.agent = agent
        self._prev_state: Optional[np.ndarray] = None
        self._prev_act_vec: Optional[np.ndarray] = None
        self._env_ref = None

    def reset(self) -> None:
        self._prev_state = None
        self._prev_act_vec = None
        self.agent.reset_noise()

    def select_action(self, env) -> dict:
        self._env_ref = env
        self._prev_state = env._build_state().copy()
        action, act_vec = self.agent.act(env)
        self._prev_act_vec = act_vec
        return action

    def observe(self, reward: float, info) -> None:
        if self._prev_state is None or self._env_ref is None:
            return
        next_state = self._env_ref._build_state().copy()
        done = self._env_ref.slot_t >= self._env_ref.cfg.episode_slots
        z_sum = float(self._env_ref.z.sum())
        self.agent.store_and_update(
            self._prev_state,
            self._prev_act_vec,
            reward,
            next_state,
            done,
            z_sum,
        )


class SafeScaleMATD3NNPolicy(_NNPolicyBase):
    """
    SafeScale-MATD3（神经网络版）—— 论文提出算法。

    与 MA-TD3 的区别：
      • 奖励增广：r̃ = r_env − κ × Σ Z_{v,m}（Lyapunov 安全惩罚）
      • 较大 κ (safety_coeff) 驱动策略主动维持安全队列 Z 接近 0
    """

    def __init__(self, cfg: SimConfig, seed: int = 0,
                 safety_coeff: float = 1.0, **kwargs):
        agent = TD3Agent(cfg, safety_coeff=safety_coeff, **kwargs)
        super().__init__(cfg, agent, seed=seed)
        self.name = "SafeScale-MATD3"


class MATD3NNPolicy(_NNPolicyBase):
    """
    MA-TD3（神经网络版）—— 无安全约束 baseline。

    与 SafeScale-MATD3 的区别：safety_coeff=0，不优化安全队列。
    """

    def __init__(self, cfg: SimConfig, seed: int = 0, **kwargs):
        agent = TD3Agent(cfg, safety_coeff=0.0, **kwargs)
        super().__init__(cfg, agent, seed=seed)
        self.name = "MA-TD3-NN"
