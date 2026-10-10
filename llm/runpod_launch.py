"""Launch train_grpo.py on a RunPod GPU, stream progress, fetch results, always terminate.

    uv run --no-project --env-file <path-to-.env> python llm/runpod_launch.py --steps 3      # pilot
    uv run --no-project --env-file <path-to-.env> python llm/runpod_launch.py --steps 150    # full

The API key is read from the environment (RUNPOD_API_KEY or runpod_api_key) and never printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
ROOT = Path(__file__).resolve().parents[1]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PIP = "transformers==4.48.3 trl==0.15.2 peft==0.14.0 datasets==3.2.0 accelerate==1.3.0 pyarrow==16.1.0 'numpy<2'"
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA L4", "NVIDIA RTX A6000"]


def key() -> str:
    k = os.environ.get("RUNPOD_API_KEY") or os.environ.get("runpod_api_key")
    if not k:
        sys.exit("No RunPod key in the environment (RUNPOD_API_KEY / runpod_api_key).")
    return k


def api(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json",
                                          "User-Agent": "reward-audit/0.1 (+https://github.com/vahit19)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"RunPod API {method} {path} -> {e.code}: {e.read().decode()[:500]}")


def fetch(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "reward-audit/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--sources", default="exact,substring,judge")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "llm"))
    a = ap.parse_args()

    script = base64.b64encode((ROOT / "llm" / "train_grpo.py").read_bytes()).decode()
    start = (
        "mkdir -p /workspace/out && cd /workspace/out && (python -m http.server 8000 >/dev/null 2>&1 &) && "
        f"pip install -q {PIP} > /workspace/out/pip.log 2>&1 && "
        "echo $SCRIPT_B64 | base64 -d > /workspace/run.py && "
        "python /workspace/run.py > /workspace/out/log.txt 2>&1; echo exit=$? >> /workspace/out/log.txt; sleep infinity"
    )
    body = {
        "name": "reward-audit",
        "imageName": IMAGE,
        "gpuTypeIds": GPUS,
        "gpuCount": 1,
        "cloudType": "SECURE",
        "containerDiskInGb": 40,
        "volumeInGb": 0,
        "ports": ["8000/http"],
        "env": {"SCRIPT_B64": script, "STEPS": str(a.steps), "SOURCES": a.sources, "SEEDS": a.seeds},
        "dockerStartCmd": ["bash", "-c", start],
    }
    pod = api("POST", "/pods", body)
    pid = pod["id"]
    cost = float(pod.get("costPerHr") or pod.get("adjustedCostPerHr") or 1.0)
    print(f"pod {pid} started on {pod.get('machine', {}).get('gpuTypeId', '?')} at ~${cost:.2f}/h; budget ${a.budget_usd}",
          flush=True)
    base = f"https://{pid}-8000.proxy.runpod.net"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, last_log = time.time(), ""
    try:
        while True:
            time.sleep(60)
            spent = (time.time() - t0) / 3600 * cost
            log = fetch(base + "/log.txt")
            if log:
                tail = log.decode(errors="ignore").strip().splitlines()[-3:]
                if tail and tail[-1] != last_log:
                    last_log = tail[-1]
                    print(f"[{(time.time() - t0) / 60:5.1f} min, ${spent:.2f}] " + " | ".join(t[:160] for t in tail),
                          flush=True)
            done = fetch(base + "/DONE")
            failed = log is not None and b"exit=" in log and not done
            if done or failed or spent >= a.budget_usd:
                listing = (fetch(base + "/") or b"").decode(errors="ignore")
                for name in sorted(set(re.findall(r'href="([^"/]+\.(?:json|txt|log))"', listing))):
                    data = fetch(f"{base}/{name}")
                    if data is not None:
                        (out / name).write_bytes(data)
                reason = "done" if done else ("failed" if failed else "budget reached")
                print(f"stopping: {reason}; files saved to {out}", flush=True)
                return 0 if done else 1
    finally:
        api("DELETE", f"/pods/{pid}")
        print(f"pod {pid} terminated; spent about ${(time.time() - t0) / 3600 * cost:.2f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
