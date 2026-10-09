"""Nature-CNN actor-critic for stacked Atari observations (docs/contracts.md §4.1; CleanRL ppo_atari init)."""
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical


def layer_init(layer: nn.Module, std: float = float(np.sqrt(2)), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal weight init, constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Shared Nature-DQN trunk with a policy head (logits) and a value head."""

    def __init__(self, num_actions: int, in_channels: int = 6):
        super().__init__()
        self.num_actions = num_actions
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, obs_uint8: Tensor) -> tuple[Tensor, Tensor]:
        """(B,6,84,84) uint8 -> logits (B,A), value (B,). Scales by 1/255 internally."""
        hidden = self.network(obs_uint8.float() / 255.0)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def get_action_and_value(self, obs: Tensor, action: Tensor | None = None):
        """Sample (or evaluate the given) action. Returns (action, logprob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class GridAgent(nn.Module):
    """Conv actor-critic for the Surround grid observation (6,18,38): same interface as Agent."""

    def __init__(self, num_actions: int, in_channels: int = 6):
        super().__init__()
        self.num_actions = num_actions
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),   # (64,9,19)
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 9 * 19, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, obs_uint8: Tensor) -> tuple[Tensor, Tensor]:
        """(B,6,18,38) uint8 -> logits (B,A), value (B,). Scales by 1/255 internally."""
        hidden = self.network(obs_uint8.float() / 255.0)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def get_action_and_value(self, obs: Tensor, action: Tensor | None = None):
        """Sample (or evaluate the given) action. Returns (action, logprob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def make_agent(obs_kind: str, num_actions: int) -> nn.Module:
    """Agent for pixel observations, GridAgent for grid observations."""
    if obs_kind == "pixels":
        return Agent(num_actions=num_actions)
    if obs_kind == "grid":
        return GridAgent(num_actions=num_actions)
    raise ValueError(f"unknown obs kind {obs_kind!r}")
