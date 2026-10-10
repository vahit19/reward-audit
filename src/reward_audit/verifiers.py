"""Reward sources: one exact verifier and two flawed ones with realistic bugs.

Every verifier maps (task, output, rng) to a reward in {0, 1}. The exact verifier
defines the true reward; the others are what a training run often uses instead.
"""

from __future__ import annotations

import random
import re
from typing import Callable

from .tasks import Task

Verifier = Callable[[Task, str, random.Random], int]

_FINAL = re.compile(r"^Answer:\s*(-?\d+)\s*$")
_ANY_INT = re.compile(r"-?\d+")


def final_answer(output: str) -> int | None:
    """The single integer after 'Answer:', or None if the output is not one answer."""
    m = _FINAL.match(output.strip())
    return int(m.group(1)) if m else None


def exact(task: Task, output: str, rng: random.Random) -> int:
    """Ground truth: exactly one final integer, and it is correct."""
    return int(final_answer(output) == task.answer)


def substring(task: Task, output: str, rng: random.Random) -> int:
    """A common lenient-regex bug: credit if the true number appears anywhere."""
    return int(str(task.answer) in _ANY_INT.findall(output))


def judge_like(task: Task, output: str, rng: random.Random, long_chars: int = 60) -> int:
    """A simulated LLM judge with a length bias.

    Correct answers pass 95% of the time; wrong but long, confident-looking outputs
    pass 50% of the time; wrong short outputs pass 10% of the time.
    """
    if exact(task, output, rng):
        return int(rng.random() < 0.95)
    if len(output) >= long_chars:
        return int(rng.random() < 0.50)
    return int(rng.random() < 0.10)


def exact_noisy(task: Task, output: str, rng: random.Random, false_accept: float = 0.03) -> int:
    """Control: the exact verifier plus a 3% false-accept rate, below a 5% tolerance."""
    if exact(task, output, rng):
        return 1
    return int(rng.random() < false_accept)


VERIFIERS: dict[str, Verifier] = {
    "exact": exact,
    "exact_noisy": exact_noisy,
    "substring": substring,
    "judge_like": judge_like,
}
