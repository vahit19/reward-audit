"""Tasks whose true answer is known, so the true reward is always computable."""

from __future__ import annotations

import random
from dataclasses import dataclass

OPS = ("+", "-", "*")


@dataclass(frozen=True)
class Task:
    a: int
    op: str
    b: int

    @property
    def prompt(self) -> str:
        return f"What is {self.a} {self.op} {self.b}?"

    @property
    def answer(self) -> int:
        if self.op == "+":
            return self.a + self.b
        if self.op == "-":
            return self.a - self.b
        return self.a * self.b


def make_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    return [Task(rng.randint(2, 99), rng.choice(OPS), rng.randint(2, 99)) for _ in range(n)]
