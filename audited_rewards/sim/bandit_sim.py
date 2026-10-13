"""Toy policy-gradient study: an exploitable judge, random audits, and five reward schemes.

One prompt, four answer strategies (arms). The gold reward y is Bernoulli(mu[a]); the judge
j depends on y through an arm-specific error pattern. Arm 2 is the exploit: the judge says
"correct" 95% of the time although the answer is almost never right. The initial policy
rarely plays it, so labels collected from the initial policy say little about it.

Reward schemes (all trained with REINFORCE + leave-one-out baseline, G samples per step):
  oracle    : r = y for every sample (gold for everything; G labels per step)
  judge     : r = j (no gold labels)
  noisecorr : r = (j - fpr_hat) / (tpr_hat - fpr_hat), rates estimated once from gold labels
              drawn from the initial policy (off-policy calibration, same label budget)
  audit0    : r = xi * y / p (gold only on the audited sample, judge ignored)
  audited   : r = g(j) + xi * (y - g(j)) / p, g = on-policy calibration of the judge
              estimated from past audits only (predictable)
Exactly one of the G samples is audited per step (inclusion probability p = 1/G).

Writes results/sim_summary.json and figures to ../paper/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
FIG = HERE.parent / "paper"

MU = np.array([0.50, 0.20, 0.05, 0.70])   # gold success rate per arm
TPR = np.array([0.90, 0.90, 0.95, 0.80])  # P(judge says yes | correct)
FPR = np.array([0.10, 0.10, 0.95, 0.05])  # P(judge says yes | wrong); arm 2 is the exploit
THETA0 = np.array([1.0, 1.0, -1.5, 0.3])  # initial logits: the exploit arm has ~3% mass
G, STEPS, LR, SEEDS = 8, 1500, 0.05, 200
P = 1.0 / G
METHODS = ["oracle", "judge", "noisecorr", "audit0", "audited"]


def softmax(t: np.ndarray) -> np.ndarray:
    e = np.exp(t - t.max())
    return e / e.sum()


def true_value(pi: np.ndarray) -> float:
    return float(pi @ MU)


def judge_value(pi: np.ndarray) -> float:
    return float(pi @ (MU * TPR + (1 - MU) * FPR))


def judge_precision(pi: np.ndarray) -> float:
    """On-policy precision P(y=1 | j=1): the optimal trust in the judge (Theorem 3)."""
    yes_and_right = pi @ (MU * TPR)
    return float(yes_and_right / (pi @ (MU * TPR + (1 - MU) * FPR)))


def draw(rng: np.random.Generator, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = (rng.random(a.shape) < MU[a]).astype(float)
    pj = np.where(y == 1.0, TPR[a], FPR[a])
    j = (rng.random(a.shape) < pj).astype(float)
    return y, j


class Calib:
    """g(j) = P(y=1 | j) under the recent policy, from past audits only (exponential forgetting)."""

    def __init__(self, decay: float = 0.98, prior: float = 2.0):
        self.decay = decay
        self.s = np.array([0.5 * prior, 0.5 * prior])  # sum of y for j=0, j=1
        self.n = np.array([prior, prior])

    def g(self, j: np.ndarray) -> np.ndarray:
        rate = self.s / self.n
        return rate[j.astype(int)]

    def update(self, j: float, y: float) -> None:
        self.s *= self.decay
        self.n *= self.decay
        self.s[int(j)] += y
        self.n[int(j)] += 1.0


def run(method: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    theta = THETA0.copy()
    labels = 0
    if method == "noisecorr":
        # Off-policy calibration with the same gold budget the audited run spends in total.
        n_cal = STEPS
        a = rng.choice(4, size=n_cal, p=softmax(THETA0))
        y, j = draw(rng, a)
        labels += n_cal
        tpr_hat = (j[y == 1].sum() + 1) / (y.sum() + 2)
        fpr_hat = (j[y == 0].sum() + 1) / ((1 - y).sum() + 2)
    calib = Calib()
    tv, jv, prec, lam = [], [], [], []
    for _ in range(STEPS):
        pi = softmax(theta)
        tv.append(true_value(pi))
        jv.append(judge_value(pi))
        prec.append(judge_precision(pi))
        a = rng.choice(4, size=G, p=pi)
        y, j = draw(rng, a)
        k = rng.integers(G)  # the audited sample, chosen after the answers exist
        xi = np.zeros(G)
        xi[k] = 1.0
        if method == "oracle":
            r = y
            labels += G
        elif method == "judge":
            r = j
        elif method == "noisecorr":
            r = (j - fpr_hat) / max(tpr_hat - fpr_hat, 1e-3)
        elif method == "audit0":
            r = xi * y / P
            labels += 1
        else:  # audited
            g = calib.g(j)
            lam.append(float(calib.g(np.array([1.0]))[0]))
            r = g + xi * (y - g) / P
            labels += 1
            calib.update(j[k], y[k])  # used only from the next step on
        baseline = (r.sum() - r) / (G - 1)  # leave-one-out
        adv = r - baseline
        grad = np.zeros(4)
        for i in range(G):
            onehot = np.zeros(4)
            onehot[a[i]] = 1.0
            grad += adv[i] * (onehot - pi)
        theta += LR * grad / G
    return {"true": tv, "judge": jv, "precision": prec, "lambda": lam, "labels": labels}


def check_variance_formula(n: int = 400_000, seed: int = 7) -> dict:
    """Theorem 3: Var(r~) = Var(y) + (1-p)/p * E[(y - g)^2] at the initial policy, fixed g."""
    rng = np.random.default_rng(seed)
    pi = softmax(THETA0)
    a = rng.choice(4, size=n, p=pi)
    y, j = draw(rng, a)
    g = np.where(j == 1, 0.6, 0.15)  # any fixed trust function
    xi = (rng.random(n) < P).astype(float)
    r = g + xi * (y - g) / P
    predicted = y.var() + (1 - P) / P * np.mean((y - g) ** 2)
    return {"mean_r": float(r.mean()), "mean_y": float(y.mean()),
            "var_r": float(r.var()), "var_predicted": float(predicted)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    res = {m: [run(m, s) for s in range(SEEDS)] for m in METHODS}
    summary = {"setup": {"mu": MU.tolist(), "tpr": TPR.tolist(), "fpr": FPR.tolist(), "G": G, "steps": STEPS,
                         "lr": LR, "seeds": SEEDS, "p": P, "best_true": float(MU.max())},
               "methods": {}, "variance_check": check_variance_formula()}
    for m in METHODS:
        tv = np.array([r["true"] for r in res[m]])
        jv = np.array([r["judge"] for r in res[m]])
        final = tv[:, -1]
        half = tv[:, STEPS // 2]
        summary["methods"][m] = {
            "final_true_mean": float(final.mean()),
            "final_true_ci95": [float(np.percentile(final, 2.5)), float(np.percentile(final, 97.5))],
            "mid_true_mean": float(half.mean()),
            "final_judge_mean": float(jv[:, -1].mean()),
            "share_seeds_on_exploit": float(np.mean(final < 0.15)),
            "gold_labels": int(np.mean([r["labels"] for r in res[m]])),
        }
    jpre = np.array([r["precision"] for r in res["judge"]])
    summary["judge_run_precision"] = {"start": float(jpre[:, 0].mean()), "end": float(jpre[:, -1].mean())}
    lam = np.array([r["lambda"] for r in res["audited"]])
    summary["audited_lambda"] = {"start": float(lam[:, 0].mean()), "end": float(lam[:, -1].mean())}
    (OUT / "sim_summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(OUT / "sim_curves.npz", **{f"{m}_true": np.array([r["true"] for r in res[m]]) for m in METHODS},
                        judge_judge=np.array([r["judge"] for r in res["judge"]]), judge_precision=jpre,
                        audited_lambda=lam)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
