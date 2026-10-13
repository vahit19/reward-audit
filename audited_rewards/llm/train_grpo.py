"""GRPO on GSM8K with judge rewards made unbiased by random audits.

Runs on a single GPU (RunPod). For each reward source, a small open model is trained
with GRPO + LoRA. Every reward call also computes the strict true reward (for logging),
picks one of the 8 completions uniformly at random after they are written (inclusion
probability p = 1/8), and logs curves to /workspace/out.

Reward sources
  exact     : the final '#### <number>' equals the gold answer (the true reward)
  substring : the gold number appears anywhere in the completion (a common naive check)
  judge     : a reference-free LLM judge says the answer is correct (RLAIF-style)
  audited   : r = g(j) + xi * (y - g(j)) / p, where g(j) = P(y=1 | j) is estimated from
              past audits only; unbiased for the true reward under any policy
  audit0    : r = xi * y / p (gold on the audited completion only, judge ignored)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

OUT = Path(os.environ.get("OUT_DIR", "/workspace/out"))
OUT.mkdir(parents=True, exist_ok=True)
POLICY = os.environ.get("POLICY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
JUDGE = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
STEPS = int(os.environ.get("STEPS", "100"))
SOURCES = os.environ.get("SOURCES", "exact,substring,judge").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
EPS, ALPHA, AUDIT_PER_CALL = 0.02, 0.05, 1
ACC = 4  # reward calls (micro-batches of 8 completions) per optimizer step

SYSTEM = "Solve the math problem. Think step by step, then end with a line '#### <number>'."


def gold_number(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


_HASH = re.compile(r"####\s*\$?(-?[\d,]*\.?\d+)")
_BOXED = re.compile(r"\\boxed\{\s*\$?(-?[\d,]*\.?\d+)\s*\}")
_NUM = re.compile(r"-?[\d,]*\.?\d+")


def final_number(text: str) -> str | None:
    r"""Flexible extraction (as in GSM8K 'flexible-extract'): '#### n', else \boxed{n}, else the last number."""
    for pat in (_HASH, _BOXED):
        found = pat.findall(text)
        if found:
            return found[-1].replace(",", "")
    nums = _NUM.findall(text)
    return nums[-1].replace(",", "") if nums else None


def same_number(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return False


def true_reward(text: str, gold: str) -> float:
    return float(same_number(final_number(text), gold))


def substring_reward(text: str, gold: str) -> float:
    nums = [n.replace(",", "") for n in re.findall(r"-?[\d,]*\.?\d+", text)]
    return float(any(same_number(n, gold) for n in nums))


class Judge:
    """Reference-free judge: sees the question and the response, not the gold answer."""

    def __init__(self, name: str):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        self.yes = self.tok.encode(" YES", add_special_tokens=False)[-1]
        self.no = self.tok.encode(" NO", add_special_tokens=False)[-1]

    @torch.no_grad()
    def score(self, questions: list[str], responses: list[str]) -> list[float]:
        prompts = []
        for q, r in zip(questions, responses):
            msgs = [
                {"role": "system", "content": "You grade math solutions. Reply with YES if the final answer is correct, otherwise NO."},
                {"role": "user", "content": f"Problem:\n{q}\n\nSolution:\n{r}\n\nIs the final answer correct? Answer YES or NO."},
            ]
            prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:")
        self.tok.padding_side = "left"
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        logits = self.model(**enc).logits[:, -1, :]
        return [float(y > n) for y, n in zip(logits[:, self.yes].tolist(), logits[:, self.no].tolist())]


# --- audited reward core (pure python; tested locally) ---
class Calib:
    """g(j) = P(y=1 | judge says j) under the recent policy, from past audits only."""

    def __init__(self, decay: float = 0.98, prior: float = 2.0):
        self.decay = decay
        self.s = [0.5 * prior, 0.5 * prior]
        self.n = [prior, prior]

    def g(self, j: float) -> float:
        k = int(j >= 0.5)
        return self.s[k] / self.n[k]

    def update(self, j: float, y: float) -> None:
        for k in (0, 1):
            self.s[k] *= self.decay
            self.n[k] *= self.decay
        k = int(j >= 0.5)
        self.s[k] += y
        self.n[k] += 1.0


def audited_rewards(judged: list[float], truths: list[float], audited: int, p: float,
                    calib: "Calib | None") -> list[float]:
    """Horvitz-Thompson correction of judge rewards; calib=None gives the judge-free audit0."""
    out = []
    for i, (j, y) in enumerate(zip(judged, truths)):
        g = calib.g(j) if calib is not None else 0.0
        out.append(g + (y - g) / p if i == audited else g)
    return out
# --- end core ---


class Monitor:
    """Anytime-valid test of H0: false-reward rate <= EPS (betting e-process)."""

    def __init__(self):
        self.wealth, self.n, self.hits, self.rejected_at = 1.0, 0, 0, None

    def update(self, x: int, step: int) -> None:
        p_hat = (self.hits + 0.5) / (self.n + 1.0)
        lam = min(max((p_hat - EPS) / (EPS * (1 - EPS)), 0.0), 0.5 / EPS)
        self.wealth *= 1.0 + lam * (x - EPS)
        self.n += 1
        self.hits += x
        if self.rejected_at is None and self.wealth >= 1.0 / ALPHA:
            self.rejected_at = step


def run(source: str, seed: int, judge: Judge | None) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed).select(range(STEPS * 4 + 64))
    tok = AutoTokenizer.from_pretrained(POLICY)

    def to_prompt(ex):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["question"]}]
        return {"prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                "gold": gold_number(ex["answer"]), "question": ex["question"]}

    ds = ds.map(to_prompt, remove_columns=ds.column_names)
    log = {"source": source, "seed": seed, "steps": [], "samples": []}
    monitor, rng, state = Monitor(), random.Random(seed), {"calls": 0}
    calib = Calib()

    def reward_fn(prompts, completions, gold, question, **kwargs):
        truths = [true_reward(c, g) for c, g in zip(completions, gold)]
        n = len(truths)
        audited = rng.randrange(n)  # chosen after the completions exist; p = 1/n
        judged = None
        if source == "exact":
            rewards = list(truths)
        elif source == "substring":
            rewards = [substring_reward(c, g) for c, g in zip(completions, gold)]
        elif source == "judge":
            rewards = judge.score(question, completions)
        else:
            judged = judge.score(question, completions) if source == "audited" else [0.0] * n
            g_now = (calib.g(1.0), calib.g(0.0))
            rewards = audited_rewards(judged, truths, audited, 1.0 / n, calib if source == "audited" else None)
            calib.update(judged[audited], truths[audited])  # used from the next call on
        step = state["calls"] // ACC
        paid = judged if judged is not None else rewards
        monitor.update(int(paid[audited] >= 0.5 and truths[audited] == 0.0), step)
        entry = {
            "step": step,
            "call": state["calls"],
            "reward": sum(rewards) / n,
            "true": sum(truths) / n,
            "false_reward": sum(r >= 0.5 and t == 0.0 for r, t in zip(paid, truths)) / n,
            "mean_len": sum(len(c) for c in completions) / n,
            "numbers_per_completion": sum(len(re.findall(r"\d+", c)) for c in completions) / n,
            "wealth": monitor.wealth,
        }
        if judged is not None:
            entry["judge"] = sum(judged) / n
            entry["g1"], entry["g0"] = g_now
        log["steps"].append(entry)
        if state["calls"] in (0, (STEPS * ACC) // 2, STEPS * ACC - 1):
            log["samples"].append({"step": step, "completion": completions[0], "gold": gold[0],
                                   "reward": rewards[0], "true": truths[0]})
        state["calls"] += 1
        return rewards

    cfg = GRPOConfig(
        output_dir=str(OUT / f"ckpt_{source}_{seed}"),
        learning_rate=5e-6,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=ACC,
        num_generations=8,
        max_prompt_length=384,
        max_completion_length=200,
        max_steps=STEPS,
        logging_steps=10,
        bf16=True,
        beta=0.04,
        seed=seed,
        report_to=[],
        save_strategy="no",
    )
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=POLICY, reward_funcs=[reward_fn], args=cfg, train_dataset=ds, peft_config=peft)
    t0 = time.time()
    trainer.train()
    log["minutes"] = round((time.time() - t0) / 60, 1)
    log["flagged_at"] = monitor.rejected_at
    log["audited"] = monitor.n
    return log


def main() -> int:
    judge = Judge(JUDGE) if {"judge", "audited"} & set(SOURCES) else None
    summary = []
    for source in SOURCES:
        for seed in SEEDS:
            log = run(source, seed, judge)
            (OUT / f"run_{source}_{seed}.json").write_text(json.dumps(log, indent=1))
            last = log["steps"][-10 * ACC:]
            summary.append({
                "source": source, "seed": seed, "minutes": log["minutes"],
                "first_reward": log["steps"][0]["reward"], "first_true": log["steps"][0]["true"],
                "last_reward": sum(s["reward"] for s in last) / len(last),
                "last_true": sum(s["true"] for s in last) / len(last),
                "last_false_reward": sum(s["false_reward"] for s in last) / len(last),
                "flagged_at": log["flagged_at"], "audited": log["audited"],
            })
            (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
            print(json.dumps(summary[-1]), flush=True)
    (OUT / "DONE").write_text("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
