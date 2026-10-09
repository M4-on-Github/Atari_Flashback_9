"""Laptop-side policy: wraps an exported model.ts + config.json (docs/contracts.md §4.3, §5.1).

No multi_agent_ale_py import here: the laptop may not have it.
"""
import json
from pathlib import Path

import numpy as np
import torch

LEVELS = ("hard", "medium", "easy")
EASY_TEMPERATURE = 1.5
MEDIUM_TEMPERATURE = 1.0
EASY_RANDOM_P = 0.15


class Policy:
    """Action index policy. hard = argmax, medium = softmax sample (T=1), easy = softmax sample (T=1.5) + 15% random."""

    def __init__(self, model_dir: str, level: str = "hard"):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        model_dir = Path(model_dir)
        self.level = level
        self.config = json.loads((model_dir / "config.json").read_text())
        self.action_names: tuple[str, ...] = tuple(self.config["action_names"])
        self.num_actions = len(self.action_names)
        self.model = torch.jit.load(str(model_dir / "model.ts"), map_location="cpu").eval()
        self.rng = np.random.default_rng()  # tests may replace this for reproducibility

    @torch.no_grad()
    def logits(self, obs: np.ndarray) -> np.ndarray:
        """obs (6,84,84) uint8 -> logits (A,) float32."""
        if obs.shape != (6, 84, 84) or obs.dtype != np.uint8:
            raise ValueError(f"obs must be uint8 (6,84,84), got {obs.dtype} {obs.shape}")
        out = self.model(torch.from_numpy(obs[None].copy()))
        return out[0].float().numpy()

    def act(self, obs: np.ndarray) -> int:
        """obs (6,84,84) uint8 -> action index in [0, num_actions)."""
        logits = self.logits(obs)
        if self.level == "hard":
            return int(np.argmax(logits))
        temperature = MEDIUM_TEMPERATURE if self.level == "medium" else EASY_TEMPERATURE
        if self.level == "easy" and self.rng.random() < EASY_RANDOM_P:
            return int(self.rng.integers(self.num_actions))
        z = logits / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        return int(self.rng.choice(self.num_actions, p=p))
