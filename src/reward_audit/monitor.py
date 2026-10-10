"""Checks that catch a broken reward while training runs.

FalseRewardMonitor: an anytime-valid test of H0 "the reward pays for wrong outputs
at most eps of the time", fed by a small audited sample of the policy's own outputs.
It may be checked after every step; under H0 it ever rejects with probability <= alpha.

zero_defect_sample_size: how many clean checks in a row certify a defect rate <= eps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FalseRewardMonitor:
    eps: float = 0.05
    alpha: float = 0.05
    wealth: float = 1.0
    n: int = 0
    hits: int = 0
    rejected_at: int | None = None

    def _bet(self) -> float:
        # Predictable bet from past data only, clipped so every factor stays positive.
        p_hat = (self.hits + 0.5) / (self.n + 1.0)
        lam = (p_hat - self.eps) / (self.eps * (1.0 - self.eps))
        return min(max(lam, 0.0), 0.5 / self.eps)

    def update(self, false_reward: int, step: int) -> bool:
        """false_reward = 1 if the verifier paid for an output the truth rejects."""
        lam = self._bet()
        self.wealth *= 1.0 + lam * (false_reward - self.eps)
        self.n += 1
        self.hits += false_reward
        if self.rejected_at is None and self.wealth >= 1.0 / self.alpha:
            self.rejected_at = step
        return self.rejected_at is not None


def zero_defect_sample_size(eps: float, confidence: float = 0.95) -> int:
    """Smallest n with (1 - eps)^n <= 1 - confidence."""
    return math.ceil(math.log(1.0 - confidence) / math.log(1.0 - eps))
