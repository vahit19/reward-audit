"""Train the policy against one reward source and audit it while it trains."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .monitor import FalseRewardMonitor
from .policy import STRATEGIES, Policy, generate
from .tasks import make_tasks
from .verifiers import VERIFIERS, exact

EXPLOITS = (STRATEGIES.index("spray"), STRATEGIES.index("bluff"))


@dataclass
class RunResult:
    verifier: str
    seed: int
    reward_curve: list[float] = field(default_factory=list)
    true_curve: list[float] = field(default_factory=list)
    false_reward_curve: list[float] = field(default_factory=list)
    exploit_curve: list[float] = field(default_factory=list)
    final_probs: list[float] = field(default_factory=list)
    takeover_step: int | None = None  # first step where exploit strategies hold > 50% of the policy
    flagged_at: int | None = None  # first step the monitor rejects H0
    audited: int = 0  # audit labels spent until the flag (or in total)


def run(
    verifier: str,
    seed: int,
    steps: int = 150,
    batch: int = 32,
    audit_per_step: int = 2,
    eps: float = 0.02,
    alpha: float = 0.05,
) -> RunResult:
    rng = random.Random(seed)
    reward_fn = VERIFIERS[verifier]
    policy = Policy()
    monitor = FalseRewardMonitor(eps=eps, alpha=alpha)
    tasks = make_tasks(steps * batch, seed)
    res = RunResult(verifier=verifier, seed=seed)

    for step in range(steps):
        chunk = tasks[step * batch : (step + 1) * batch]
        actions, rewards, truths = [], [], []
        for task in chunk:
            a = policy.sample(rng)
            out = generate(task, STRATEGIES[a], rng)
            actions.append(a)
            rewards.append(float(reward_fn(task, out, rng)))
            truths.append(float(exact(task, out, rng)))
        policy.update(actions, rewards)

        res.reward_curve.append(round(sum(rewards) / batch, 4))
        res.true_curve.append(round(sum(truths) / batch, 4))
        res.false_reward_curve.append(round(sum(r == 1.0 and t == 0.0 for r, t in zip(rewards, truths)) / batch, 4))
        exploit_share = sum(policy.probs[i] for i in EXPLOITS)
        res.exploit_curve.append(round(exploit_share, 4))
        if res.takeover_step is None and exploit_share > 0.5:
            res.takeover_step = step

        # Audit: a few outputs per step get the true label (the expensive human check).
        for i in rng.sample(range(batch), audit_per_step):
            false_reward = int(rewards[i] == 1.0 and truths[i] == 0.0)
            if res.flagged_at is None:
                res.audited += 1
                if monitor.update(false_reward, step):
                    res.flagged_at = step

    res.final_probs = [round(p, 4) for p in policy.probs]
    return res
