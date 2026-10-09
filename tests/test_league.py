"""League orchestrator tests (plain script, see docs/contracts.md §0). Run: container/run.sh python tests/test_league.py

CPU, tiny, and written to a temporary root (never the repo's checkpoints/ or runs/).
"""
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fake_env import FakeVecGames  # noqa: E402
from fb9.selfplay import BOT_ACTION  # noqa: E402
from league import LeagueArgs, run_league  # noqa: E402
from train import Args, train  # noqa: E402

torch.set_num_threads(1)


class BotFakeVecGames(FakeVecGames):
    """Fake env that plays BOT_ACTION slots as action 0 (the real VecGames plays SurroundBot)."""

    def step(self, actions: np.ndarray):
        a = np.asarray(actions).reshape(self.num_slots).copy()
        a[a == BOT_ACTION] = 0
        return super().step(a)


def fake_factory():
    def make(args: Args, num_workers: int):
        return BotFakeVecGames(args.num_games, num_actions=3, seed=args.seed, obs=args.obs)
    return make


TINY = dict(num_games=4, num_steps=16, minibatch_size=32, num_workers=1, cuda=False, seed=1)


def run_files(ck: Path) -> dict[Path, int]:
    """Modification times of every league artifact, to check that a rerun writes nothing."""
    paths = [ck / "latest.pt", *(ck / "targets").glob("*.pt"), *(ck / "league").glob("*.pt")]
    return {p: p.stat().st_mtime_ns for p in paths}


def test_league_two_rounds() -> None:
    root = Path(tempfile.mkdtemp(prefix="fb9_test_league_"))
    try:
        # tiny init checkpoint: one update of 4 mirror games
        train(Args(game="surround", run_name="init", total_samples=128, pool_fraction=0.0, **TINY),
              env_factory=fake_factory(), root=root)
        la = LeagueArgs(run_name="lg", init_ckpt=str(root / "checkpoints" / "init" / "latest.pt"), rounds=2,
                        main_samples_per_round=160, exploiter_samples=128, pool_fraction=0.25, bot_fraction=0.25,
                        league_fraction=0.25, num_steps=16, minibatch_size=32, num_games=4, num_workers=1,
                        seed=1, cuda=False)
        res = run_league(la, env_factory=fake_factory(), root=root)

        ck = root / "checkpoints" / "lg"
        for rel in ("targets/target_r0.pt", "targets/target_r1.pt", "league/exp_r0.pt", "league/exp_r1.pt"):
            assert (ck / rel).exists(), rel
        main = torch.load(ck / "latest.pt", map_location="cpu", weights_only=False)
        assert main["samples"] >= 2 * la.main_samples_per_round, main["samples"]
        assert res["main_samples"] == main["samples"]
        for r in (0, 1):
            exp = torch.load(ck / "league" / f"exp_r{r}.pt", map_location="cpu", weights_only=False)
            assert "winrate_vs_target" in exp and 0.0 <= exp["winrate_vs_target"] <= 1.0, exp.keys()
            assert exp["samples"] >= la.exploiter_samples, exp["samples"]
        assert len(res["exploiter_winrates"]) == 2
        # round 1's main phase trained against both exploiters
        assert [e["name"] for e in main["league"]] == [str((ck / "league" / f"exp_r{r}.pt").resolve()) for r in (0, 1)]

        # rerun with the same args: every phase is finished, so nothing is trained or written
        before = run_files(ck)
        runs_before = sorted(p.name for p in (root / "runs").iterdir())
        res2 = run_league(la, env_factory=fake_factory(), root=root)
        assert run_files(ck) == before, "rerun rewrote a checkpoint"
        assert sorted(p.name for p in (root / "runs").iterdir()) == runs_before, "rerun started a training run"
        assert res2["main_samples"] == res["main_samples"] and res2["exploiter_winrates"] == res["exploiter_winrates"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_league_requires_init() -> None:
    root = Path(tempfile.mkdtemp(prefix="fb9_test_league_init_"))
    try:
        try:
            run_league(LeagueArgs(init_ckpt=""), env_factory=fake_factory(), root=root)
        except ValueError:
            return
        raise AssertionError("empty init_ckpt should raise ValueError")
    finally:
        shutil.rmtree(root, ignore_errors=True)


TESTS = [test_league_two_rounds, test_league_requires_init]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}", flush=True)
            traceback.print_exc()
    sys.exit(1 if failed else 0)
