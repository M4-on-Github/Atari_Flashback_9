"""Diagnose VecGames scaling: per-step parent time split and per-worker compute time."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fb9.envs import EnvConfig, TwoPlayerGame, VecGames


def main():
    W = int(sys.argv[1]); GPW = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cfg = EnvConfig(game="surround")
    # in-process baseline, one game
    g = TwoPlayerGame(cfg, 0); g.reset(); rng = np.random.default_rng(0); n = 0; t0 = time.time()
    while time.time() - t0 < 5:
        _, _, d, _ = g.step(rng.integers(5, size=2)); n += 1
        if d: g.reset()
    print(f"in-process 1 game: {1e3*(time.time()-t0)/n:.2f} ms/step", flush=True)
    v = VecGames(cfg, W * GPW, W, 0); v.reset()
    for _ in range(20): v.step(rng.integers(5, size=v.num_slots))
    ts = {"send": 0.0, "first_recv": 0.0, "rest_recv": 0.0, "copy": 0.0}; n = 0; t0 = time.time()
    while time.time() - t0 < 15:
        a = rng.integers(5, size=v.num_slots)
        t = time.perf_counter()
        for _, conn, slots in v._workers: conn.send(("step", a[slots]))
        t1 = time.perf_counter(); v._recv(v._workers[0][1])
        t2 = time.perf_counter()
        for _, conn, _ in v._workers[1:]: v._recv(conn)
        t3 = time.perf_counter(); v._obs.copy(); t4 = time.perf_counter()
        ts["send"] += t1 - t; ts["first_recv"] += t2 - t1; ts["rest_recv"] += t3 - t2; ts["copy"] += t4 - t3; n += 1
    print(f"W={W} gpw={GPW}: {n/(time.time()-t0):.0f} vec-steps/s; per step ms: " +
          " ".join(f"{k}={1e3*x/n:.2f}" for k, x in ts.items()), flush=True)
    v.close()


if __name__ == "__main__":
    main()
