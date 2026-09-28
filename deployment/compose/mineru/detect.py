#!/usr/bin/env python3
"""Decide how this MinerU container should run.

Looks at the GPU the container actually received (through torch, which also
works on unified-memory machines where nvidia-smi reports no memory figure),
applies MINERU_BACKEND_POLICY and the thresholds from the environment, and
prints the decision either as shell exports (--shell) or as JSON (--json).

Rules:
  policy pipeline / vlm-engine / hybrid-engine  -> that backend, verified
  policy auto (default):
      NVIDIA GPU visible, compute capability >= MINERU_VLM_MIN_COMPUTE and
      total memory >= MINERU_VLM_MIN_VRAM_GB   -> vlm-engine
      anything else                            -> pipeline
The vLLM memory share is MINERU_VLM_TARGET_VRAM_GB / total memory, clamped to
[0.05, 0.90], unless MINERU_GPU_MEMORY_UTILIZATION is set explicitly.
"""
from __future__ import annotations

import json
import os
import sys

VLM_BACKENDS = ("vlm-engine", "hybrid-engine")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def probe_gpu() -> dict:
    info: dict = {"cuda": False}
    try:
        import torch  # noqa: WPS433 (runtime import on purpose)
    except Exception as exc:  # torch missing in the cpu flavor is fine
        info["error"] = f"torch unavailable: {exc.__class__.__name__}"
        return info
    try:
        if not torch.cuda.is_available():
            return info
        free, total = torch.cuda.mem_get_info(0)
        major, minor = torch.cuda.get_device_capability(0)
        info.update(
            cuda=True,
            name=torch.cuda.get_device_name(0),
            compute=float(f"{major}.{minor}"),
            total_gb=round(total / 2**30, 1),
            free_gb=round(free / 2**30, 1),
        )
    except Exception as exc:
        info["error"] = f"{exc.__class__.__name__}: {exc}"
    return info


def decide() -> dict:
    gpu = probe_gpu()
    flavor = os.environ.get("CARREL_MINERU_FLAVOR", "gpu").strip() or "gpu"
    policy = os.environ.get("MINERU_BACKEND_POLICY", "auto").strip().lower() or "auto"
    min_vram = _env_float("MINERU_VLM_MIN_VRAM_GB", 8.0)
    min_compute = _env_float("MINERU_VLM_MIN_COMPUTE", 7.0)
    target = _env_float("MINERU_VLM_TARGET_VRAM_GB", 7.0)

    capable = bool(
        gpu.get("cuda")
        and flavor != "cpu"
        and float(gpu.get("compute", 0.0)) >= min_compute
        and float(gpu.get("total_gb", 0.0)) >= min_vram
    )
    reason = ""
    if policy in ("pipeline",) + VLM_BACKENDS:
        backend = policy
        if backend in VLM_BACKENDS and not capable:
            reason = "forced by MINERU_BACKEND_POLICY but no capable GPU is visible"
        else:
            reason = "forced by MINERU_BACKEND_POLICY"
    else:
        backend = "vlm-engine" if capable else "pipeline"
        if capable:
            reason = "GPU meets the vlm-engine thresholds"
        elif not gpu.get("cuda"):
            reason = "no NVIDIA GPU visible in the container"
        elif flavor == "cpu":
            reason = "cpu image flavor"
        else:
            reason = (
                f"GPU below thresholds (compute {gpu.get('compute')}, "
                f"{gpu.get('total_gb')} GB; need {min_compute} and {min_vram} GB)"
            )

    util_raw = os.environ.get("MINERU_GPU_MEMORY_UTILIZATION", "").strip()
    if util_raw:
        util = float(util_raw)
    elif gpu.get("cuda") and float(gpu.get("total_gb", 0.0)) > 0:
        util = min(0.90, max(0.05, target / float(gpu["total_gb"])))
    else:
        util = 0.0

    return {
        "backend": backend,
        "reason": reason,
        "capable": capable,
        "policy": policy,
        "flavor": flavor,
        "device": "cuda" if gpu.get("cuda") else "cpu",
        "gpu_memory_utilization": round(util, 4),
        "gpu": gpu,
        "mineru_version": os.environ.get("MINERU_VERSION", ""),
    }


def main(argv: list[str]) -> int:
    decision = decide()
    if "--json" in argv:
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0
    gpu = decision["gpu"]
    exports = {
        "CARREL_BACKEND": decision["backend"],
        "CARREL_REASON": decision["reason"],
        "CARREL_CAPABLE": "1" if decision["capable"] else "0",
        "CARREL_DEVICE": decision["device"],
        "CARREL_GPU_UTIL": str(decision["gpu_memory_utilization"]),
        "CARREL_GPU_NAME": str(gpu.get("name", "")),
        "CARREL_GPU_TOTAL_GB": str(gpu.get("total_gb", "")),
    }
    for key, value in exports.items():
        print(f"export {key}={json.dumps(value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
