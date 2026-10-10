"""A small policy trained with REINFORCE over three answer strategies.

The policy does not model language; it chooses *how* to answer, which is the part a
reward bug exploits. 'solve' is honest and imperfect; 'spray' lists many candidate
numbers in a confident sentence; 'bluff' writes a long, confident, wrong answer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .tasks import Task

STRATEGIES = ("solve", "spray", "bluff")


def generate(task: Task, strategy: str, rng: random.Random, skill: float = 0.6) -> str:
    truth = task.answer
    if strategy == "solve":
        guess = truth if rng.random() < skill else truth + rng.choice([-10, -2, -1, 1, 2, 10])
        return f"Answer: {guess}"
    if strategy == "spray":
        centre = truth + rng.randint(-3, 3)
        candidates = [centre + k for k in range(-4, 5)]
        return "The result is one of " + ", ".join(str(c) for c in candidates) + "."
    # bluff: long, confident and wrong
    wrong = truth + rng.choice([-7, 7, 13])
    return f"After checking each step carefully, the final verified result is {wrong}. Answer: {wrong}"


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class Policy:
    lr: float = 0.5
    # A pretrained model mostly answers honestly: 90% solve, 5% spray, 5% bluff.
    logits: list[float] = field(default_factory=lambda: [math.log(0.90), math.log(0.05), math.log(0.05)])
    baseline: float = 0.0

    @property
    def probs(self) -> list[float]:
        return _softmax(self.logits)

    def sample(self, rng: random.Random) -> int:
        r, acc = rng.random(), 0.0
        for i, p in enumerate(self.probs):
            acc += p
            if r < acc:
                return i
        return len(self.logits) - 1

    def update(self, actions: list[int], rewards: list[float]) -> None:
        """One REINFORCE step with a running-mean baseline."""
        probs = self.probs
        mean_r = sum(rewards) / len(rewards)
        grad = [0.0] * len(self.logits)
        for a, r in zip(actions, rewards):
            adv = r - self.baseline
            for i in range(len(grad)):
                grad[i] += adv * ((1.0 if i == a else 0.0) - probs[i])
        n = len(actions)
        self.logits = [w + self.lr * g / n for w, g in zip(self.logits, grad)]
        self.baseline = 0.9 * self.baseline + 0.1 * mean_r
