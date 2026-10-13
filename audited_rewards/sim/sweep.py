"""Label-budget sweep: Bernoulli audits at rate p, judge-free audit0 vs audited (judge as control variate)."""

from __future__ import annotations

import json

import numpy as np

from bandit_sim import G, LR, MU, OUT, THETA0, Calib, draw, softmax, true_value

STEPS, SEEDS = 1500, 100
RATES = [1 / 8, 1 / 16, 1 / 32, 1 / 64]


def run(method: str, p: float, seed: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(10_000 + seed)
    theta = THETA0.copy()
    calib = Calib()
    labels = 0
    mid = 0.0
    for t in range(STEPS):
        pi = softmax(theta)
        if t == STEPS // 2:
            mid = true_value(pi)
        a = rng.choice(4, size=G, p=pi)
        y, j = draw(rng, a)
        xi = (rng.random(G) < p).astype(float)  # audit decided after the answers exist
        labels += int(xi.sum())
        g = calib.g(j) if method == "audited" else np.zeros(G)
        r = g + xi * (y - g) / p
        if method == "audited":
            for i in np.flatnonzero(xi):
                calib.update(j[i], y[i])
        adv = r - (r.sum() - r) / (G - 1)
        grad = np.zeros(4)
        for i in range(G):
            onehot = np.zeros(4)
            onehot[a[i]] = 1.0
            grad += adv[i] * (onehot - pi)
        theta += LR * grad / G
    return mid, true_value(softmax(theta)), labels


def main() -> None:
    out = {}
    for p in RATES:
        for m in ("audit0", "audited"):
            rs = np.array([run(m, p, s) for s in range(SEEDS)])
            out[f"{m}@1/{round(1 / p)}"] = {
                "p": p, "mid_true_mean": float(rs[:, 0].mean()), "final_true_mean": float(rs[:, 1].mean()),
                "final_true_sd": float(rs[:, 1].std()), "share_final_above_0.6": float(np.mean(rs[:, 1] > 0.6)),
                "gold_labels_mean": float(rs[:, 2].mean())}
            print(f"{m:8s} p=1/{round(1 / p):<3d} mid={rs[:, 0].mean():.3f} final={rs[:, 1].mean():.3f} "
                  f"sd={rs[:, 1].std():.3f} >0.6:{np.mean(rs[:, 1] > 0.6):.2f} labels={rs[:, 2].mean():.0f}", flush=True)
    (OUT / "sim_sweep.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
