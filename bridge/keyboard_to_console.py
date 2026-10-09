"""Drive console joystick port 2 from the laptop keyboard (arrows + space) through JoystickLink.

Used for the fake-joystick test (plan B2): navigate the FB9 menu and play from the keyboard.
Esc quits (all buttons are released on exit). Needs a focused window, so run it on the laptop desktop.

    container/run.sh python bridge/keyboard_to_console.py --port /dev/ttyACM0
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pygame  # noqa: E402
import tyro  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from bridge.serial_link import JoystickLink  # noqa: E402

BITS = {"UP": 1, "DOWN": 2, "LEFT": 4, "RIGHT": 8, "FIRE": 16}
KEYS = {"UP": pygame.K_UP, "DOWN": pygame.K_DOWN, "LEFT": pygame.K_LEFT, "RIGHT": pygame.K_RIGHT,
        "FIRE": pygame.K_SPACE}


@dataclass
class Args:
    port: str = "/dev/ttyACM0"
    hz: float = 60.0   # re-send the state this often (must stay well under the 200 ms failsafe)


def keys_to_mask(held: set[str]) -> int:
    """Raw button bitmask. Contradictory UP+DOWN / LEFT+RIGHT are resolved by the firmware, not here."""
    return sum(BITS[name] for name in held)


def main() -> None:
    args = tyro.cli(Args)
    pygame.init()
    screen = pygame.display.set_mode((320, 120))
    pygame.display.set_caption("FB9 keyboard -> console port 2 (Esc quits)")
    clock = pygame.time.Clock()
    link = JoystickLink(args.port)
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            keys = pygame.key.get_pressed()
            held = {name for name, key in KEYS.items() if keys[key]}
            link.send(keys_to_mask(held))
            screen.fill((30, 30, 30))
            pygame.display.flip()
            clock.tick(args.hz)
    finally:
        link.close()
        pygame.quit()


if __name__ == "__main__":
    main()
