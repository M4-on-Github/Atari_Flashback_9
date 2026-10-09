"""Tests for the laptop play side: policy, bitmasks, FakeLink, console dry run, headless play_pc.

Run: container/run.sh python tests/test_play.py
"""
import os
import pathlib
import sys
import tempfile
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

import play_pc  # noqa: E402
from bridge import console_play  # noqa: E402
from bridge.capture import crop_rgb, load_crop, save_crop  # noqa: E402
from bridge.serial_link import FakeLink  # noqa: E402
from fb9.games import ACTION_NAMES, GAMES, action_to_bitmask  # noqa: E402
from fb9.policy import Policy  # noqa: E402


class DummyNet(torch.nn.Module):
    """uint8 (B,6,84,84) -> logits (B,A). Channel means feed a linear layer, so the argmax depends on the input."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.lin = torch.nn.Linear(6, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = x.float().mean(3).mean(2) / 255.0
        return self.lin(feats)


def write_dummy_model(model_dir: str, game: str) -> None:
    """Writes model.ts + config.json in the §4.3 format into model_dir."""
    spec = GAMES[game]
    torch.manual_seed(0)
    scripted = torch.jit.script(DummyNet(spec.num_actions).eval())
    scripted.save(os.path.join(model_dir, "model.ts"))
    config = {"game": game, "ale_mode": spec.mode, "action_ids": list(spec.action_ids),
              "action_names": list(spec.action_names), "frameskip": 4, "stack": 4, "obs_shape": [6, 84, 84],
              "seat_planes": "ch4=255 for seat0/port1, ch5=255 for seat1/port2", "samples": 0,
              "source_ckpt": "test"}
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)


def write_video(path: str, n_frames: int = 120) -> None:
    """Synthetic 640x480 MJPG video: a moving white bar, and a scene cut to bright at frame 60 (a 'new game')."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 60, (640, 480))
    assert writer.isOpened(), "cv2.VideoWriter could not open the test video"
    for i in range(n_frames):
        frame = np.zeros((480, 640, 3), np.uint8)
        if i >= 60:
            frame[:] = 200
        x = (i * 5) % 560
        frame[100:200, x:x + 40] = 255
        writer.write(frame)
    writer.release()


# ---------- policy ----------

def test_policy_hard_samples_from_softmax():
    with tempfile.TemporaryDirectory() as d:
        write_dummy_model(d, "surround")
        pol = Policy(d, "hard")
        pol.rng = np.random.default_rng(0)
        assert pol.num_actions == 5 and pol.action_names == GAMES["surround"].action_names
        rng = np.random.default_rng(0)
        obs = rng.integers(0, 256, size=(6, 84, 84), dtype=np.uint8)
        logits = pol.logits(obs)
        assert logits.shape == (5,)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        n = 4000
        freq = np.bincount([pol.act(obs) for _ in range(n)], minlength=5) / n
        assert np.abs(freq - p).max() < 0.03, (freq, p)
        # the model output really depends on the observation
        a = rng.integers(0, 256, size=(6, 84, 84), dtype=np.uint8)
        b = rng.integers(0, 256, size=(6, 84, 84), dtype=np.uint8)
        assert not np.allclose(pol.logits(a), pol.logits(b))


def test_policy_medium_and_easy_return_valid_indices():
    with tempfile.TemporaryDirectory() as d:
        write_dummy_model(d, "combat")
        obs = np.full((6, 84, 84), 90, np.uint8)
        for level in ("medium", "easy"):
            pol = Policy(d, level)
            pol.rng = np.random.default_rng(0)
            acts = [pol.act(obs) for _ in range(300)]
            assert all(0 <= a < 18 for a in acts), level
            assert len(set(acts)) > 1, level  # sampling, not a constant


def test_policy_rejects_bad_input():
    with tempfile.TemporaryDirectory() as d:
        write_dummy_model(d, "surround")
        pol = Policy(d, "hard")
        for bad in (np.zeros((6, 84, 84), np.float32), np.zeros((5, 84, 84), np.uint8)):
            try:
                pol.act(bad)
            except ValueError:
                continue
            raise AssertionError("bad obs accepted")
        try:
            Policy(d, "insane")
        except ValueError:
            return
        raise AssertionError("bad level accepted")


# ---------- bitmasks and links ----------

EXPECTED_MASKS = {
    "NOOP": 0, "FIRE": 16, "UP": 1, "DOWN": 2, "LEFT": 4, "RIGHT": 8,
    "UPRIGHT": 9, "UPLEFT": 5, "DOWNRIGHT": 10, "DOWNLEFT": 6,
    "UPFIRE": 17, "DOWNFIRE": 18, "LEFTFIRE": 20, "RIGHTFIRE": 24,
    "UPRIGHTFIRE": 25, "UPLEFTFIRE": 21, "DOWNRIGHTFIRE": 26, "DOWNLEFTFIRE": 22,
}


def test_action_to_bitmask_all_18_names():
    assert len(ACTION_NAMES) == 18 and set(ACTION_NAMES) == set(EXPECTED_MASKS)
    for name in ACTION_NAMES:
        assert action_to_bitmask(name) == EXPECTED_MASKS[name], name
    assert action_to_bitmask("UPLEFTFIRE") == 0b10101


def test_fake_link_records_masks():
    link = FakeLink()
    link.send(3)
    link.send(16)
    link.release()
    assert link.masks == [3, 16, 0] and link.last == 0
    for bad in (32, -1):
        try:
            link.send(bad)
        except ValueError:
            continue
        raise AssertionError(f"mask {bad} accepted")


def test_crop_helpers_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "crop.json")
        assert load_crop(p) is None
        save_crop(p, 10, 20, 300, 200)
        assert load_crop(p) == {"x": 10, "y": 20, "w": 300, "h": 200}
        rgb = np.zeros((480, 640, 3), np.uint8)
        assert crop_rgb(rgb, load_crop(p)).shape == (200, 300, 3)
        assert crop_rgb(rgb, None).shape == (480, 640, 3)


# ---------- human input mapping ----------

def test_pick_action_surround_and_combat():
    s = GAMES["surround"].action_names
    c = GAMES["combat"].action_names
    cases = [
        (s, [], False, "NOOP"),
        (s, ["UP"], False, "UP"),
        (s, ["UP"], True, "UP"),              # no UPFIRE in Surround: drop the fire
        (s, [], True, "NOOP"),                # fire alone does nothing in Surround
        (s, ["UP", "RIGHT"], False, "RIGHT"),  # diagonal not in Surround: most recent direction
        (s, ["RIGHT", "UP"], False, "UP"),
        (s, ["UP", "DOWN"], False, "DOWN"),   # contradictory: most recent vertical wins
        (c, ["UP", "LEFT"], True, "UPLEFTFIRE"),
        (c, ["LEFT"], False, "LEFT"),
        (c, [], True, "FIRE"),
        (c, ["UP", "DOWN"], False, "DOWN"),
        (c, [], False, "NOOP"),
    ]
    for names, order, fire, want in cases:
        got = play_pc.pick_action(order, fire, names)
        assert got == want, (names[:3], order, fire, got, want)
        assert got in names


def test_update_dir_order_keeps_press_order():
    order = play_pc.update_dir_order([], {"UP", "LEFT"})
    assert order == ["UP", "LEFT"]
    order = play_pc.update_dir_order(order, {"LEFT"})  # UP released
    assert order == ["LEFT"]
    order = play_pc.update_dir_order(order, {"LEFT", "DOWN"})
    assert order == ["LEFT", "DOWN"]


def test_scripted_held_is_deterministic():
    assert play_pc.scripted_held(0) == set()
    assert play_pc.scripted_held(25) == {"UP"}
    assert play_pc.scripted_held(25) == play_pc.scripted_held(25 + 20 * 10)


# ---------- console dry run ----------

def test_console_dry_run_on_synthetic_video():
    with tempfile.TemporaryDirectory() as d:
        write_dummy_model(d, "surround")
        video = os.path.join(d, "synthetic.avi")
        write_video(video, 120)
        link = FakeLink()
        args = console_play.Args(game="surround", model=d, level="hard", dry_run=True, video=video,
                                 crop=os.path.join(d, "missing_crop.json"))
        summary = console_play.run(args, link=link)
        assert summary["frames"] == 120, summary
        assert summary["decisions"] == 30, summary           # one decision per 4 frames
        assert len(summary["masks"]) == 30
        assert all(0 <= m <= 31 for m in summary["masks"])
        assert summary["resets"] == 2, summary                # start of video + the scene cut at frame 60
        assert link.masks == summary["masks"] + [0]           # release() at the end


# ---------- headless PC play ----------

def test_play_pc_headless_scripted_human():
    with tempfile.TemporaryDirectory() as d:
        write_dummy_model(d, "surround")
        t0 = time.time()
        args = play_pc.Args(game="surround", model=d, level="hard", scale=2, fps=0, seed=7, max_frames=300,
                            scripted_human=True)
        summary = play_pc.run(args)
        assert summary["ticks"] == 300, summary
        assert summary["emulator_frames"] == 300, summary     # no game over in 300 frames
        assert summary["episodes"] == 1 and not summary["over"], summary
        assert len(summary["score"]) == 2
        assert time.time() - t0 < 100


def test_bridge_and_policy_do_not_import_ale():
    import re
    pattern = re.compile(r"^\s*(import|from)\s+multi_agent_ale_py", re.MULTILINE)
    # fb9/games.py is not checked here: GameSpec.rom_path() imports multi_agent_ale_py lazily (see the report).
    files = [ROOT / "fb9" / "preprocess.py", ROOT / "fb9" / "policy.py"]
    files += sorted((ROOT / "bridge").glob("*.py"))
    for f in files:
        assert not pattern.search(f.read_text()), f


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
