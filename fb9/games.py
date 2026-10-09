"""Per-game facts (measured in the A1 probe). Must not import multi_agent_ale_py: the laptop may not have it."""
from dataclasses import dataclass
from pathlib import Path

ACTION_NAMES = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
)


@dataclass(frozen=True)
class GameSpec:
    name: str
    rom: str
    mode: int
    action_ids: tuple[int, ...]
    max_frames: int = 108_000
    frameskip: int = 4   # emulator frames per agent decision (Surround: 15 = one cell move)

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(ACTION_NAMES[i] for i in self.action_ids)

    @property
    def num_actions(self) -> int:
        return len(self.action_ids)

    def rom_path(self) -> str:
        import multi_agent_ale_py
        return str(Path(multi_agent_ale_py.__file__).parent / "roms" / f"{self.rom}.bin")


GAMES: dict[str, GameSpec] = {
    "surround": GameSpec("surround", "surround", mode=1, action_ids=(0, 2, 3, 4, 5), frameskip=15),
    "combat": GameSpec("combat", "combat", mode=2, action_ids=tuple(range(18))),
}

LEGACY_FRAMESKIP = 4   # checkpoints saved before frameskip was per game have no "frameskip" in their args


def frameskip_from_args(args: dict) -> int:
    """Frameskip recorded in a checkpoint's (or config's) args; old checkpoints without the field use 4."""
    return args.get("frameskip") or LEGACY_FRAMESKIP

_BITS = {"UP": 1, "DOWN": 2, "LEFT": 4, "RIGHT": 8, "FIRE": 16}


def action_to_bitmask(name: str) -> int:
    """ALE action name -> joystick bitmask (bit0 UP, bit1 DOWN, bit2 LEFT, bit3 RIGHT, bit4 FIRE)."""
    return sum(bit for key, bit in _BITS.items() if key in name)
