#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import signal
import time
from pathlib import Path

import torch


ARTIFACTS = Path("/workspace/probe/artifacts")
HISTORY_DIM = 70 * 30


class Timeout(Exception):
    pass


def alarm_handler(_signum, _frame):
    raise Timeout()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timed(label: str, fn, seconds: int = 20):
    start = time.time()
    signal.alarm(seconds)
    try:
        out = fn()
        signal.alarm(0)
        item = {"event": label, "status": "ok", "elapsed": round(time.time() - start, 4)}
        if hasattr(out, "shape"):
            item["shape"] = list(out.shape)
        else:
            item["type"] = type(out).__name__
        print(json.dumps(item), flush=True)
        return out
    except Timeout:
        print(json.dumps({"event": label, "status": "timeout", "elapsed": round(time.time() - start, 4)}), flush=True)
        raise
    finally:
        signal.alarm(0)


def main() -> int:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(
        json.dumps(
            {
                "event": "startup",
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "threads": torch.get_num_threads(),
            }
        ),
        flush=True,
    )
    adaptation_path = ARTIFACTS / "adaptation_module_latest.jit"
    body_path = ARTIFACTS / "body_latest.jit"
    print(
        json.dumps(
            {
                "event": "artifacts",
                "adaptation_sha256": sha256_file(adaptation_path),
                "body_sha256": sha256_file(body_path),
            }
        ),
        flush=True,
    )
    signal.signal(signal.SIGALRM, alarm_handler)
    for device_name in ("cpu", "cuda:0"):
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            continue
        device = torch.device(device_name)
        print(json.dumps({"event": "device_start", "device": device_name}), flush=True)
        adaptation = timed(f"load_adaptation_{device_name}", lambda: torch.jit.load(str(adaptation_path), map_location=device), 30).eval()
        body = timed(f"load_body_{device_name}", lambda: torch.jit.load(str(body_path), map_location=device), 30).eval()
        for batch in (1, 8, 32):
            obs = torch.zeros((batch, HISTORY_DIM), dtype=torch.float32, device=device)
            with torch.inference_mode():
                latent = timed(f"adaptation_{device_name}_batch_{batch}", lambda: adaptation(obs), 20)
                action = timed(f"body_{device_name}_batch_{batch}", lambda: body(torch.cat((obs, latent), dim=-1)), 20)
            print(
                json.dumps(
                    {
                        "event": "batch_done",
                        "device": device_name,
                        "batch": batch,
                        "latent_mean": float(latent.mean().detach().cpu().item()),
                        "action_mean": float(action.mean().detach().cpu().item()),
                    }
                ),
                flush=True,
            )
    print(json.dumps({"event": "done"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
