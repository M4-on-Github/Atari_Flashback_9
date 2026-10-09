"""Play Atari Flashback 9 games against a trained model on the PC emulator (docs/contracts.md §5.2).

Human = seat 0 / joystick port 1 (keyboard arrows + space, or a gamepad). Model = seat 1 / port 2.
The model decides every 4th emulator frame; its action is held for the 4 frames. Human input is read every frame.

    container/run.sh python play_pc.py --game surround --model models/surround --level hard --scale 4

Keys: arrows + space (fire), R restarts, Esc quits. Headless smoke test: SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy.
Only this file imports multi_agent_ale_py.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataclasses import dataclass  # noqa: E402

import multi_agent_ale_py as maap  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402
import tyro  # noqa: E402

from fb9.games import GAMES, GameSpec  # noqa: E402
from fb9.policy import Policy  # noqa: E402
from fb9.preprocess import FRAMESKIP, FrameStack, process_frame  # noqa: E402

HUD_H = 28
DIRS = ("UP", "DOWN", "LEFT", "RIGHT")
KEYMAP = {"UP": pygame.K_UP, "DOWN": pygame.K_DOWN, "LEFT": pygame.K_LEFT, "RIGHT": pygame.K_RIGHT,
          "FIRE": pygame.K_SPACE}
# Fake keyboard for the headless test: a fixed cycle of held sets, 20 frames each.
SCRIPT = (set(), {"UP"}, {"UP", "FIRE"}, {"RIGHT"}, {"RIGHT", "DOWN"}, {"DOWN", "FIRE"}, {"LEFT"},
          {"LEFT", "UP"}, {"FIRE"}, set())


@dataclass
class Args:
    game: str = "surround"          # "surround" | "combat"
    model: str = "models/surround"  # dir with model.ts + config.json
    level: str = "hard"             # hard | medium | easy
    scale: int = 4                  # window scale factor
    fps: int = 60                   # 0 = unthrottled
    seed: int = 0                   # episode seeds are drawn from this; 0 = random
    max_frames: int = 0             # test hook: stop after this many loop ticks (0 = until Esc)
    scripted_human: bool = False    # test hook: fixed fake keyboard instead of real keys


def update_dir_order(order: list[str], held: set[str]) -> list[str]:
    """Keep the press order of held directions (oldest first); newly pressed ones go to the end."""
    order = [d for d in order if d in held]
    return order + [d for d in DIRS if d in held and d not in order]


def pick_action(dir_order: list[str], fire: bool, names: tuple[str, ...]) -> str:
    """Map held directions + fire to an action name of this game.

    Vertical part = most recently pressed of UP/DOWN, horizontal = most recently pressed of LEFT/RIGHT, so UP+DOWN
    resolves to the newer one. Uses the full combo (e.g. UPLEFTFIRE) when the game has it, else drops FIRE, else (Surround
    has no diagonals) falls back to the most recently pressed of the two directions.
    """
    avail = set(names)
    vert = [d for d in dir_order if d in ("UP", "DOWN")]
    horiz = [d for d in dir_order if d in ("LEFT", "RIGHT")]
    v = vert[-1] if vert else ""
    h = horiz[-1] if horiz else ""
    base = v + h
    options = []
    if fire:
        options.append(base + "FIRE" if base else "FIRE")
    options.append(base if base else "NOOP")
    for option in options:
        if option in avail:
            return option
    chosen = [d for d in dir_order if d in (v, h)]
    return chosen[-1] if chosen else "NOOP"


def read_keyboard(joy) -> set[str]:
    """Held controls from the keyboard (arrows, space) and the first gamepad if present."""
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    held = {name for name, key in KEYMAP.items() if keys[key]}
    if joy is not None:
        hx, hy = joy.get_hat(0) if joy.get_numhats() else (0, 0)
        ax = joy.get_axis(0) if joy.get_numaxes() >= 1 else 0.0
        ay = joy.get_axis(1) if joy.get_numaxes() >= 2 else 0.0
        if hx < 0 or ax < -0.5:
            held.add("LEFT")
        if hx > 0 or ax > 0.5:
            held.add("RIGHT")
        if hy > 0 or ay < -0.5:
            held.add("UP")
        if hy < 0 or ay > 0.5:
            held.add("DOWN")
        if joy.get_numbuttons() and joy.get_button(0):
            held.add("FIRE")
    return held


def scripted_held(tick: int) -> set[str]:
    return set(SCRIPT[(tick // 20) % len(SCRIPT)])


def make_ale(spec: GameSpec, seed: int):
    maap.ALEInterface.setLoggerMode("error")
    ale = maap.ALEInterface()
    ale.setFloat(b"repeat_action_probability", 0.0)  # sticky actions are not used in play
    ale.setInt(b"random_seed", int(seed))
    ale.loadROM(spec.rom_path())
    ale.setMode(spec.mode)
    ale.reset_game()
    return ale


def gray_of(ale) -> np.ndarray:
    g = np.asarray(ale.getScreenGrayscale())
    return g.reshape(g.shape[0], g.shape[1])


class Episode:
    """One game: the ALE instance, the model's frame stack and its held action."""

    def __init__(self, spec: GameSpec, seed: int):
        self.spec = spec
        self.ale = make_ale(spec, seed)
        g = gray_of(self.ale)
        self.stack = FrameStack(seat=1)
        self.obs = self.stack.reset(process_frame(g, g))  # first decision sees the reset screen
        self.model_idx = 0
        self.pos = 0            # emulator frame index within the current 4-frame step
        self.grays: list[np.ndarray] = []
        self.score = np.zeros(2, dtype=np.float32)
        self.frames = 0

    @property
    def over(self) -> bool:
        return bool(self.ale.game_over()) or self.frames >= self.spec.max_frames

    def step_frame(self, human_idx: int, policy: Policy) -> None:
        """Advance one emulator frame with the human's current action and the model's held action."""
        if self.pos == 0:
            self.model_idx = policy.act(self.obs)
        ale_ids = np.array([self.spec.action_ids[human_idx], self.spec.action_ids[self.model_idx]], dtype=np.int32)
        rewards = np.asarray(self.ale.act(ale_ids), dtype=np.float32)
        self.score += rewards
        self.frames += 1
        self.grays.append(gray_of(self.ale))
        self.pos += 1
        if self.pos == FRAMESKIP:
            # max-pool the last two frames of the step, then push into the model's stack
            self.obs = self.stack.push(process_frame(self.grays[2], self.grays[3]))
            self.grays = []
            self.pos = 0


def draw(screen, font, ep: Episode, scale: int, status: str) -> None:
    rgb = np.ascontiguousarray(np.transpose(ep.ale.getScreenRGB(), (1, 0, 2)))
    surf = pygame.surfarray.make_surface(rgb)
    surf = pygame.transform.scale(surf, (rgb.shape[0] * scale, rgb.shape[1] * scale))
    screen.fill((0, 0, 0))
    screen.blit(surf, (0, HUD_H))
    label = font.render(f"You {int(round(ep.score[0]))}  -  Model {int(round(ep.score[1]))}   {status}", True,
                        (255, 255, 255))
    screen.blit(label, (8, 6))
    pygame.display.flip()


def run(args: Args) -> dict:
    """Play until Esc (or args.max_frames loop ticks). Returns a summary dict (used by the tests)."""
    spec = GAMES[args.game]
    policy = Policy(args.model, args.level)
    if policy.action_names != spec.action_names:
        raise ValueError(f"model actions {policy.action_names} != game actions {spec.action_names}")
    rng = np.random.default_rng(args.seed or None)

    pygame.init()
    try:
        joy = None
        pygame.joystick.init()
        if pygame.joystick.get_count():
            joy = pygame.joystick.Joystick(0)
            joy.init()
        ep = Episode(spec, int(rng.integers(1, 2**31 - 1)))
        h, w = ep.ale.getScreenRGB().shape[:2]
        screen = pygame.display.set_mode((w * args.scale, h * args.scale + HUD_H))
        pygame.display.set_caption(f"FB9 {args.game} - you (port 1) vs model ({args.level})")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()

        episodes, ticks, quit_now = 1, 0, False
        dir_order: list[str] = []
        while not quit_now:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_now = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit_now = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    ep = Episode(spec, int(rng.integers(1, 2**31 - 1)))
                    episodes += 1
                    dir_order = []
            if quit_now or (args.max_frames and ticks >= args.max_frames):
                break

            held = scripted_held(ticks) if args.scripted_human else read_keyboard(joy)
            dir_order = update_dir_order(dir_order, held)
            status = "R = restart" if ep.over else ""
            if not ep.over:
                name = pick_action(dir_order, "FIRE" in held, spec.action_names)
                ep.step_frame(spec.action_names.index(name), policy)
            else:
                status = "GAME OVER - R to restart"
            draw(screen, font, ep, args.scale, status)
            if args.fps > 0:
                clock.tick(args.fps)
            ticks += 1
    finally:
        pygame.quit()
    return {"ticks": ticks, "emulator_frames": ep.frames, "episodes": episodes,
            "score": ep.score.tolist(), "over": ep.over}


def main() -> None:
    args = tyro.cli(Args)
    summary = run(args)
    print(summary)


if __name__ == "__main__":
    main()
