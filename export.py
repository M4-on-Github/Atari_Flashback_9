"""Export a checkpoint to TorchScript + config.json (docs/contracts.md §4.3).

CLI: container/run.sh python export.py --ckpt checkpoints/<run>/latest.pt --out models/<game>/
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import tyro
from torch import Tensor

from fb9.games import GAMES, frameskip_from_args
from fb9.model import Agent
from fb9.preprocess import OBS_SHAPE, STACK


class LogitsWrapper(nn.Module):
    """uint8 (B,6,84,84) -> float32 logits (B,A). The only thing the exported graph computes."""

    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def forward(self, obs_uint8: Tensor) -> Tensor:
        logits, _ = self.agent(obs_uint8)
        return logits


def export(ckpt_path: str | Path, out_dir: str | Path) -> Path:
    """Write out_dir/model.ts and out_dir/config.json. Returns out_dir."""
    ckpt_path = Path(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    game = state["args"]["game"]
    spec = GAMES[game]
    agent = Agent(spec.num_actions)
    agent.load_state_dict(state["model"])
    agent.eval()
    wrapper = LogitsWrapper(agent).eval()
    example = torch.zeros((2,) + OBS_SHAPE, dtype=torch.uint8)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"model.ts.tmp{os.getpid()}"
    traced.save(str(tmp))
    os.replace(tmp, out / "model.ts")

    config = {
        "game": game,
        "ale_mode": spec.mode,
        "action_ids": list(spec.action_ids),
        "action_names": list(spec.action_names),
        "frameskip": frameskip_from_args(state["args"]),
        "stack": STACK,
        "obs_shape": list(OBS_SHAPE),
        "seat_planes": "ch4=255 for seat0/port1, ch5=255 for seat1/port2",
        "samples": int(state["samples"]),
        "source_ckpt": str(ckpt_path),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out


@dataclass
class Args:
    ckpt: str  # checkpoint file, e.g. checkpoints/<run>/latest.pt
    out: str   # output directory, e.g. models/surround/


if __name__ == "__main__":
    a = tyro.cli(Args)
    export(a.ckpt, a.out)
    print(f"exported {a.ckpt} -> {a.out}")
