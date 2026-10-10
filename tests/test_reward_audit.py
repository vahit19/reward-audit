import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reward_audit.monitor import FalseRewardMonitor, zero_defect_sample_size  # noqa: E402
from reward_audit.policy import generate  # noqa: E402
from reward_audit.tasks import Task, make_tasks  # noqa: E402
from reward_audit.train import run  # noqa: E402
from reward_audit.verifiers import exact, final_answer, judge_like, substring  # noqa: E402

RNG = random.Random(0)


def test_task_answers():
    assert Task(7, "+", 5).answer == 12
    assert Task(7, "-", 9).answer == -2
    assert Task(7, "*", 6).answer == 42
    assert make_tasks(5, 1) == make_tasks(5, 1)


def test_exact_needs_one_correct_final_answer():
    t = Task(12, "*", 3)
    assert exact(t, "Answer: 36", RNG) == 1
    assert exact(t, "Answer: 35", RNG) == 0
    assert exact(t, "The result is one of 34, 35, 36.", RNG) == 0
    assert final_answer("Answer: -4") == -4
    assert final_answer("Answer: 4 or 5") is None


def test_substring_pays_for_a_list():
    t = Task(12, "*", 3)
    assert substring(t, "The result is one of 34, 35, 36.", RNG) == 1
    assert substring(t, "Answer: 360", RNG) == 0


def test_judge_like_rates():
    rng = random.Random(1)
    t = Task(20, "+", 22)
    n = 20000
    correct = sum(judge_like(t, "Answer: 42", rng) for _ in range(n)) / n
    long_wrong = sum(judge_like(t, "x" * 70, rng) for _ in range(n)) / n
    short_wrong = sum(judge_like(t, "Answer: 41", rng) for _ in range(n)) / n
    assert abs(correct - 0.95) < 0.01
    assert abs(long_wrong - 0.50) < 0.015
    assert abs(short_wrong - 0.10) < 0.01


def test_strategies_produce_expected_truth():
    rng = random.Random(2)
    tasks = make_tasks(400, 3)
    solve = sum(exact(t, generate(t, "solve", rng), rng) for t in tasks) / len(tasks)
    spray_true = sum(exact(t, generate(t, "spray", rng), rng) for t in tasks)
    spray_paid = sum(substring(t, generate(t, "spray", rng), rng) for t in tasks) / len(tasks)
    bluff_true = sum(exact(t, generate(t, "bluff", rng), rng) for t in tasks)
    assert 0.5 < solve < 0.7
    assert spray_true == 0 and bluff_true == 0
    assert spray_paid > 0.9


def test_zero_defect_sample_size():
    assert zero_defect_sample_size(0.05, 0.95) == 59
    assert zero_defect_sample_size(0.02, 0.95) == 149


def _ever_rejects(p: float, eps: float, n: int, seed: int) -> bool:
    rng = random.Random(seed)
    m = FalseRewardMonitor(eps=eps, alpha=0.05)
    return any(m.update(int(rng.random() < p), i) for i in range(n))


def test_monitor_is_valid_at_the_boundary():
    # Under H0 (rate exactly eps) the monitor may be checked after every label and
    # still rejects in at most alpha of runs; 600 runs leave room for Monte Carlo error.
    runs = 600
    rejections = sum(_ever_rejects(0.02, 0.02, 1500, s) for s in range(runs))
    assert rejections / runs <= 0.05 + 0.02


def test_monitor_never_rejects_clean_data():
    assert not _ever_rejects(0.0, 0.02, 3000, 0)


def test_monitor_has_power():
    runs = 200
    power = sum(_ever_rejects(0.08, 0.02, 600, s) for s in range(runs)) / runs
    assert power > 0.9


def test_training_is_deterministic():
    a, b = run("substring", 3, steps=40), run("substring", 3, steps=40)
    assert a.reward_curve == b.reward_curve and a.final_probs == b.final_probs


def test_reward_bug_is_exploited_and_clean_reward_is_not():
    hacked = run("substring", 0)
    clean = run("exact", 0)
    assert hacked.exploit_curve[-1] > 0.8 and hacked.true_curve[-1] < 0.2
    assert clean.exploit_curve[-1] < 0.1 and clean.flagged_at is None
    assert hacked.flagged_at is not None and hacked.flagged_at < hacked.takeover_step


def test_committed_results_are_reproduced():
    proc = subprocess.run([sys.executable, str(ROOT / "experiments" / "run_demo.py"), "--check"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
