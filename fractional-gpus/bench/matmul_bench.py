#!/usr/bin/env python3
"""Portable HPC MatMul benchmark for comparing g4-standard-12 vs. fractional H100 Mega (MIG / MPS).

Runs identical FP64, strict IEEE FP32 (TF32 off), and BF16 matrix multiplications on a single
logical GPU slice while optionally holding a configurable resident VRAM working set.
Emits a single machine-readable line prefixed with `BENCH_RESULT_JSON:` for aggregation.
"""

import argparse
import json
import os
import socket
import sys
import time
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HPC MatMul GPU Slice Benchmark")
    p.add_argument("--tier", default=os.environ.get("BENCH_TIER", "unknown"))
    p.add_argument("--n", type=int, default=int(os.environ.get("MATMUL_N", "8192")))
    p.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP_ITERS", "5")))
    p.add_argument("--iters-fp64", type=int, default=int(os.environ.get("ITERS_FP64", "20")))
    p.add_argument("--iters-fp32", type=int, default=int(os.environ.get("ITERS_FP32", "50")))
    p.add_argument("--iters-bf16", type=int, default=int(os.environ.get("ITERS_BF16", "100")))
    p.add_argument("--target-vram-gb", type=float, default=float(os.environ.get("TARGET_VRAM_GB", "10")))
    p.add_argument("--cpu-phase-sec", type=float, default=float(os.environ.get("CPU_PHASE_SEC", "0.0")))
    p.add_argument("--oom-test", action="store_true", default=os.environ.get("OOM_TEST", "false").lower() == "true")
    p.add_argument("--oom-alloc-gb", type=float, default=float(os.environ.get("OOM_ALLOC_GB", "24")))
    return p.parse_args()


def cpu_busy_work(seconds: float) -> None:
    if seconds <= 0:
        return
    t_end = time.perf_counter() + seconds
    x = torch.randn((512, 512), dtype=torch.float32)
    while time.perf_counter() < t_end:
        _ = x @ x


def run_gemm(n: int, dtype: torch.dtype, warmup: int, iters: int, cpu_phase_sec: float) -> dict:
    if iters <= 0:
        return {"iters": 0, "elapsed_s": 0.0, "tflops": 0.0}

    a = torch.randn((n, n), dtype=dtype, device="cuda")
    b = torch.randn((n, n), dtype=dtype, device="cuda")

    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    gpu_elapsed = 0.0
    for _ in range(iters):
        if cpu_phase_sec > 0:
            cpu_busy_work(cpu_phase_sec)
        tg0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        gpu_elapsed += time.perf_counter() - tg0
    wall_elapsed = time.perf_counter() - t0

    flops = 2.0 * (n ** 3) * iters
    tflops_gpu = flops / max(gpu_elapsed, 1e-9) / 1e12
    tflops_wall = flops / max(wall_elapsed, 1e-9) / 1e12

    del a, b, c
    torch.cuda.empty_cache()
    return {
        "iters": iters,
        "gpu_elapsed_s": round(gpu_elapsed, 4),
        "wall_elapsed_s": round(wall_elapsed, 4),
        "tflops": round(tflops_gpu, 3),
        "effective_tflops": round(tflops_wall, 3),
    }


def get_device_uuid() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "MIG-" in line and "UUID:" in line:
                return line.split("UUID:", 1)[1].strip().rstrip(")")
        for line in out.splitlines():
            if "UUID:" in line:
                return line.split("UUID:", 1)[1].strip().rstrip(")")
    except Exception:
        pass
    return os.environ.get("CUDA_VISIBLE_DEVICES", "unset")


def main() -> int:
    args = parse_args()
    job_idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    pod_name = os.environ.get("POD_NAME", socket.gethostname())
    node_name = os.environ.get("NODE_NAME", "unknown")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available inside container.", file=sys.stderr)
        return 1

    props = torch.cuda.get_device_properties(0)
    cc = f"sm_{props.major}{props.minor}"
    total_vram_gb = props.total_memory / (1024 ** 3)
    device_uuid = get_device_uuid()

    print(
        f"[init] tier={args.tier} node={node_name} pod={pod_name} idx={job_idx} "
        f"gpu='{props.name}' cc={cc} vram={total_vram_gb:.2f}GiB "
        f"device_uuid={device_uuid} torch={torch.__version__} cuda={torch.version.cuda}",
        flush=True,
    )

    result = {
        "tier": args.tier,
        "node": node_name,
        "pod": pod_name,
        "job_index": job_idx,
        "gpu_name": props.name,
        "compute_capability": cc,
        "cuda_visible_devices": device_uuid,
        "visible_vram_gib": round(total_vram_gb, 2),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "matmul_n": args.n,
        "target_vram_gb": args.target_vram_gb,
        "cpu_phase_sec": args.cpu_phase_sec,
        "oom_victim": bool(args.oom_test and job_idx == 0),
        "status": "ok",
    }

    start_ts = time.time()
    result["start_ts"] = round(start_ts, 3)

    # Intentional OOM victim on index 0 during noisy-neighbor isolation testing
    if args.oom_test and job_idx == 0:
        # Small delay so sibling pods are actively executing matmuls when the OOM fires
        time.sleep(5.0)
        try:
            num_bytes = int(args.oom_alloc_gb * (1024 ** 3))
            print(f"[oom-victim] Attempting {args.oom_alloc_gb:.1f} GiB allocation on {total_vram_gb:.2f} GiB slice...", flush=True)
            _boom = torch.empty(num_bytes // 4, dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            result["status"] = "unexpected_oom_alloc_succeeded"
            del _boom
        except torch.cuda.OutOfMemoryError as exc:
            result["status"] = "expected_oom_caught"
            result["error"] = str(exc).splitlines()[0]
            print(f"[oom-victim] Caught expected CUDA OOM: {result['error']}", flush=True)
        except RuntimeError as exc:
            msg = str(exc)
            # PyTorch 2.8 on MIG catches cudaErrorMemoryAllocation then asserts inside NVML OOM formatter
            if "out of memory" in msg.lower() or "NVML_SUCCESS == r" in msg:
                result["status"] = "expected_oom_caught"
                result["error"] = f"CUDA OOM enforced on MIG slice ({msg.splitlines()[0]})"
                print(f"[oom-victim] Caught expected MIG CUDA OOM: {result['error']}", flush=True)
            else:
                result["status"] = f"unexpected_error:{type(exc).__name__}"
                result["error"] = msg
                print(f"[oom-victim] Unexpected RuntimeError: {msg}", flush=True)
        except Exception as exc:
            result["status"] = f"unexpected_error:{type(exc).__name__}"
            result["error"] = str(exc)
            print(f"[oom-victim] Unexpected error: {exc}", flush=True)

        result["end_ts"] = round(time.time(), 3)
        result["total_elapsed_s"] = round(result["end_ts"] - start_ts, 4)
        print("BENCH_RESULT_JSON:" + json.dumps(result), flush=True)
        return 0 if result["status"] == "expected_oom_caught" else 2

    # Reserve resident working-set VRAM to simulate realistic solver memory occupancy
    working_set = None
    try:
        # Estimate bytes needed for FP64 GEMM buffers (3 * N^2 * 8 bytes) plus safety margin
        gemm_overhead_gb = (3.0 * (args.n ** 2) * 8.0) / (1024 ** 3) + 1.0
        pad_gb = max(0.0, min(args.target_vram_gb - gemm_overhead_gb, total_vram_gb - gemm_overhead_gb - 1.0))
        if pad_gb > 0.25:
            num_floats = int((pad_gb * (1024 ** 3)) // 4)
            working_set = torch.empty(num_floats, dtype=torch.float32, device="cuda")
            working_set.fill_(1.0)
            torch.cuda.synchronize()
            print(f"[vram] Reserved {pad_gb:.2f} GiB resident working-set buffer", flush=True)

        # Strict IEEE FP32 (disable TF32 so FP32 reflects classic scientific precision)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        fp64 = run_gemm(args.n, torch.float64, args.warmup, args.iters_fp64, args.cpu_phase_sec)
        print(f"[fp64] {fp64}", flush=True)

        fp32 = run_gemm(args.n, torch.float32, args.warmup, args.iters_fp32, args.cpu_phase_sec)
        print(f"[fp32] {fp32}", flush=True)

        bf16 = run_gemm(args.n, torch.bfloat16, args.warmup, args.iters_bf16, args.cpu_phase_sec)
        print(f"[bf16] {bf16}", flush=True)

        peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        end_ts = time.time()

        result.update(
            {
                "end_ts": round(end_ts, 3),
                "total_elapsed_s": round(end_ts - start_ts, 4),
                "peak_vram_gib": round(peak_vram_gib, 2),
                "fp64": fp64,
                "fp32": fp32,
                "bf16": bf16,
            }
        )
    except Exception as exc:
        result["status"] = f"error:{type(exc).__name__}"
        result["error"] = str(exc)
        result["end_ts"] = round(time.time(), 3)
        result["total_elapsed_s"] = round(result["end_ts"] - start_ts, 4)
        print("BENCH_RESULT_JSON:" + json.dumps(result), flush=True)
        return 1
    finally:
        if working_set is not None:
            del working_set

    print("BENCH_RESULT_JSON:" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
