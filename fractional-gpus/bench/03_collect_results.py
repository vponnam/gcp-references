#!/usr/bin/env python3
"""Collects BENCH_RESULT_JSON records from completed pods in namespace `gpu-bench`
and writes a side-by-side Markdown scorecard to `bench/local_scorecard.md`
(leaving the published reference `bench/results_scorecard.md` untouched unless `--out` is set)."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

NAMESPACE = os.environ.get("K8S_NAMESPACE", "gpu-bench")
DEFAULT_OUT_FILE = Path(
    os.environ.get("SCORECARD_OUT", str(Path(__file__).resolve().parent / "local_scorecard.md"))
)
TIERS_ORDER = [
    "g4-flex",
    "g4-8gpu-mig",
    "g4-8gpu-mps",
    "g4-8gpu-mig-oom",
    "h100-mega-mig",
    "h100-mega-mps",
    "h100-mega-mig-oom",
]
OOM_TIERS = {"g4-8gpu-mig-oom", "h100-mega-mig-oom"}


def run_cmd(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {res.stderr.strip()}")
    return res.stdout


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (len(s) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def max_temporal_overlap(records: list[dict]) -> int:
    events = []
    for r in records:
        if "start_ts" in r and "end_ts" in r:
            events.append((r["start_ts"], 1))
            events.append((r["end_ts"], -1))
    events.sort(key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best


def collect_records() -> dict[str, list[dict]]:
    by_tier: dict[str, list[dict]] = defaultdict(list)

    try:
        raw = run_cmd(["kubectl", "-n", NAMESPACE, "get", "pods", "-l", "app=matmul-bench", "-o", "json"])
        pods = json.loads(raw).get("items", [])
        for pod in pods:
            pod_name = pod["metadata"]["name"]
            tier = pod["metadata"].get("labels", {}).get("bench-tier", "unknown")
            logs = run_cmd(["kubectl", "-n", NAMESPACE, "logs", pod_name])
            parsed = None
            for line in logs.splitlines():
                if line.startswith("BENCH_RESULT_JSON:"):
                    parsed = json.loads(line.split("BENCH_RESULT_JSON:", 1)[1])
            if parsed:
                by_tier[tier].append(parsed)
    except Exception as exc:
        print(f"[warn] Could not read live pods from cluster: {exc}", file=sys.stderr)

    # Merge any previously saved cross-cluster JSON snapshots (e.g. results_g4_flex.json)
    bench_dir = Path(__file__).resolve().parent
    for path in sorted(bench_dir.glob("results_*.json")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tier = rec.get("tier", "unknown")
            if not any(r.get("pod") == rec.get("pod") and r.get("start_ts") == rec.get("start_ts") for r in by_tier[tier]):
                by_tier[tier].append(rec)

    return by_tier


def summarize_tier(tier: str, recs: list[dict]) -> dict:
    ok_recs = [r for r in recs if r.get("status") == "ok" and not r.get("oom_victim")]
    oom_victims = [r for r in recs if r.get("oom_victim")]

    nodes = sorted({r.get("node", "unknown") for r in recs})
    devices = sorted({r.get("cuda_visible_devices", "unset") for r in recs})
    overlap = max_temporal_overlap(recs)

    if tier in OOM_TIERS:
        victim_ok = all(r.get("status") == "expected_oom_caught" for r in oom_victims) and len(oom_victims) > 0
        siblings_ok = len(ok_recs) == (len(recs) - len(oom_victims)) and len(ok_recs) > 0
        return {
            "tier": tier,
            "pods": len(recs),
            "nodes": len(nodes),
            "unique_devices": len(devices),
            "max_overlap": overlap,
            "oom_verdict": "PASS" if (victim_ok and siblings_ok) else "FAIL",
            "victim_status": [r.get("status") for r in oom_victims],
            "siblings_passed": f"{len(ok_recs)}/{len(recs) - len(oom_victims)}",
        }

    elapsed = [r["total_elapsed_s"] for r in ok_recs]
    fp64_tf = [r["fp64"]["tflops"] for r in ok_recs if "fp64" in r]
    fp32_tf = [r["fp32"]["tflops"] for r in ok_recs if "fp32" in r]
    bf16_tf = [r["bf16"]["tflops"] for r in ok_recs if "bf16" in r]
    vram_vis = [r["visible_vram_gib"] for r in ok_recs]
    vram_peak = [r["peak_vram_gib"] for r in ok_recs]

    node_span_s = 0.0
    if ok_recs:
        node_span_s = max(r["end_ts"] for r in ok_recs) - min(r["start_ts"] for r in ok_recs)
    jobs_per_hr_per_node = (
        (len(ok_recs) / max(len(nodes), 1)) * (3600.0 / max(node_span_s, 1e-6))
        if ok_recs and node_span_s > 0
        else 0.0
    )

    return {
        "tier": tier,
        "pods": f"{len(ok_recs)}/{len(recs)}",
        "nodes": len(nodes),
        "unique_devices": len(devices),
        "max_overlap": overlap,
        "visible_vram_gib": round(statistics.median(vram_vis), 2) if vram_vis else 0.0,
        "peak_vram_gib": round(max(vram_peak), 2) if vram_peak else 0.0,
        "p50_elapsed_s": round(percentile(elapsed, 50), 2),
        "p95_elapsed_s": round(percentile(elapsed, 95), 2),
        "node_wall_s": round(node_span_s, 2),
        "jobs_per_hr_node": round(jobs_per_hr_per_node, 1),
        "slice_fp64_tflops": round(statistics.median(fp64_tf), 2) if fp64_tf else 0.0,
        "node_fp64_tflops": round(sum(fp64_tf) / max(len(nodes), 1), 2) if fp64_tf else 0.0,
        "slice_fp32_tflops": round(statistics.median(fp32_tf), 2) if fp32_tf else 0.0,
        "node_fp32_tflops": round(sum(fp32_tf) / max(len(nodes), 1), 2) if fp32_tf else 0.0,
        "slice_bf16_tflops": round(statistics.median(bf16_tf), 2) if bf16_tf else 0.0,
        "node_bf16_tflops": round(sum(bf16_tf) / max(len(nodes), 1), 2) if bf16_tf else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect MatMul benchmark pod results into a Markdown scorecard.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_FILE,
        help="Output Markdown file (default: bench/local_scorecard.md)",
    )
    args = parser.parse_args()

    by_tier = collect_records()
    if not by_tier:
        print(f"No completed benchmark pod results found in namespace '{NAMESPACE}'.", file=sys.stderr)
        return 1

    perf_summaries = []
    oom_summaries = []
    ordered_tiers = [t for t in TIERS_ORDER if t in by_tier] + [t for t in sorted(by_tier) if t not in TIERS_ORDER]

    for tier in ordered_tiers:
        s = summarize_tier(tier, by_tier[tier])
        if tier in OOM_TIERS:
            oom_summaries.append(s)
        else:
            perf_summaries.append(s)

    lines = [
        "# Fractional GPU MatMul Benchmark Scorecard",
        "",
        "## Performance Summary (`g4-standard-12` vs. 8-GPU Fractionalized Nodes)",
        "",
        "| Tier | Pods OK | Nodes | Unique Slices | Max Overlap | Visible / Peak VRAM (GiB) | Slice p50 / p95 (s) | Node Wall (s) | Jobs/hr/Node | FP64 TFLOPS (Slice / Node) | FP32 TFLOPS (Slice / Node) | BF16 TFLOPS (Slice / Node) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in perf_summaries:
        lines.append(
            f"| `{s['tier']}` | {s['pods']} | {s['nodes']} | {s['unique_devices']} | {s['max_overlap']} "
            f"| {s['visible_vram_gib']} / {s['peak_vram_gib']} "
            f"| {s['p50_elapsed_s']} / {s['p95_elapsed_s']} | {s['node_wall_s']} | **{s['jobs_per_hr_node']}** "
            f"| {s['slice_fp64_tflops']} / **{s['node_fp64_tflops']}** "
            f"| {s['slice_fp32_tflops']} / **{s['node_fp32_tflops']}** "
            f"| {s['slice_bf16_tflops']} / **{s['node_bf16_tflops']}** |"
        )

    for oom_summary in oom_summaries:
        lines += [
            "",
            f"## Noisy-Neighbor OOM Isolation Check (`{oom_summary['tier']}`)",
            "",
            f"- **Verdict:** **{oom_summary['oom_verdict']}**",
            f"- **Nodes Used:** {oom_summary['nodes']} | **Unique MIG Slices:** {oom_summary['unique_devices']} | **Max Concurrent Overlap:** {oom_summary['max_overlap']}",
            f"- **Index 0 Victim Status:** `{oom_summary['victim_status']}`",
            f"- **Sibling Pods Completed Normally:** `{oom_summary['siblings_passed']}`",
        ]

    lines += [
        "",
        "## How to Compute Cost per Simulation ($/Job)",
        "",
        "Divide your hourly VM cost (DWS Flex-start or Spot rate) by **Jobs/hr/Node** from the table above:",
        "",
        "```text",
        "Cost_per_Job = (Hourly_VM_Price_USD) / (Jobs_per_hr_per_Node)",
        "```",
    ]

    report = "\n".join(lines) + "\n"
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
