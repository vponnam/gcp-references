# Fractional GPU MatMul Benchmark Scorecard (Reference Results)

Published reference benchmark results comparing a fractional GPU VM (**`g4-standard-12`**, DWS Flex-start) against a fractionalized 8-GPU node (**`g4-standard-384`**, Spot) running **GKE MIG (`1g.24gb`)** and **GKE Native MPS (`4/GPU`)**, plus a noisy-neighbor **OOM hardware-isolation test**.

> **Reproduce in your own GCP project:** Follow the [Setup & Testing Guide](../README.md) and [Benchmark Runbook](README.md). Anonymized raw JSONL pod telemetry for every run below is archived in [`reference_results/`](reference_results/).

## Test Environment & Workload Configuration

| Parameter | Value |
| :--- | :--- |
| **Platform** | GKE Standard (`1.36.4-gke.2377000`, `COS_CONTAINERD`, `us-central1`) |
| **Container Image** | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` (PyTorch `2.8.0+cu128`, CUDA `12.8`, Blackwell `sm_120`) |
| **Workload per Pod** | `8192×8192` MatMul (`20×` FP64, `50×` strict IEEE FP32 with TF32 disabled, `100×` BF16) + `10 GB` resident VRAM buffer |
| **Consumption Modes Tested** | `g4-standard-12`: **DWS Flex-start** (`--flex-start`) · `g4-standard-384`: **Spot** (`--spot`) |

## Performance Summary (`g4-standard-12` vs. 8-GPU Fractionalized Nodes)

| Tier | Machine & Consumption Mode | Pods OK | Nodes | Unique Slices | Max Overlap | Visible / Peak VRAM (GiB) | Slice p50 / p95 (s) | Node Wall (s) | Jobs/hr/Node | FP64 TFLOPS (Slice / Node) | FP32 TFLOPS (Slice / Node) | BF16 TFLOPS (Slice / Node) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `g4-flex` | `g4-standard-12` (DWS Flex-start, 1/4 GPU) | 1/1 | 1 | 1 | 1 | 23.69 / 9.51 | 78.11 / 78.11 | 78.11 | **46.1** | 0.37 / **0.37** | 20.60 / **20.60** | 107.36 / **107.36** |
| `g4-8gpu-mig` | `g4-standard-384` MIG `1g.24gb` (Spot) | 32/32 | 1 | 32 *(verified via `nvidia-smi -L`)* | 32 | 23.62 / 9.51 | 77.64 / 78.53 | 81.95 | **1,405.8** | 0.39 / **12.36** | 19.41 / **626.94** | 96.63 / **3,165.43** |
| `g4-8gpu-mps` | `g4-standard-384` MPS `4/GPU` (Spot) | 32/32 | 1 | 8 GPUs (4/GPU) | 32 | 94.97 / 9.51 | 78.25 / 79.48 | 81.53 | **1,412.9** | 0.38 / **12.20** | 19.68 / **630.82** | 99.36 / **3,232.33** |

## Noisy-Neighbor OOM Isolation Check (`g4-8gpu-mig-oom`)

- **Verdict:** **PASS**
- **Nodes Used:** 1 | **Unique MIG Slices:** 32 | **Max Concurrent Overlap:** 31
- **Index 0 Victim Status:** `['expected_oom_caught']` (requested `28 GiB` on a `23.62 GiB` MIG slice)
- **Sibling Pods Completed Normally:** `31/31`

## Key Functional Validation Findings (GKE `1.36.4-gke.2377000`, `us-central1`)

1. **Exact 1:1 Slice Parity (`g4-standard-12` vs. `g4-standard-384` MIG `1g.24gb`):**
   - `g4-standard-12` exposes a hypervisor-backed `MIG 1g.24gb` slice (`23.69 GiB` visible VRAM, 46 SMs).
   - Partitioning an 8-GPU `g4-standard-384` on GKE with `gpu-partition-size=1g.24gb` packs **32 concurrent pods onto 1 node** with **32 distinct MIG UUIDs**, **23.62 GiB** visible VRAM per pod, **~30.5× aggregate throughput** (`1,405.8` vs. `46.1` jobs/hr/node), and **<1% p95 latency jitter** (`77.64 s` p50 / `78.53 s` p95 vs. `78.11 s` standalone).
2. **Hardware Isolation Verified (MIG OOM Test):**
   - Attempting a `28 GiB` allocation on pod index `0` inside its `23.62 GiB` MIG slice triggered an isolated CUDA OOM (`expected_oom_caught`), while **31/31 sibling pods** on the same node completed normally.
   - *PyTorch 2.8 + MIG note:* When a CUDA OOM occurs inside a MIG container, PyTorch 2.8's OOM error-formatting path queries NVML (`CUDACachingAllocator.cpp:1131`) and can raise `RuntimeError: NVML_SUCCESS == r` instead of `torch.cuda.OutOfMemoryError`. The underlying hardware VRAM cap is still strictly enforced.
3. **GKE MIG vs. GKE Native MPS (`4/GPU`, 32 concurrent pods on 1 node):**
   - **Throughput:** Nearly identical for sustained compute-bound matmuls (`1,412.9` MPS vs. `1,405.8` MIG jobs/hr/node, +0.5%), with slightly higher p95 variance on MPS (`79.48 s` vs. `78.53 s`).
   - **VRAM visibility:** MPS exposes the full physical GPU VRAM (**94.97 GiB**) to every co-located pod (`8` unique GPU UUIDs shared 4-way), whereas MIG strictly caps each pod at **23.62 GiB**.
   - **Takeaway:** For sustained HPC compute, **GKE MIG** delivers ~99.5% of MPS throughput with hard VRAM/fault isolation and tighter p95 latency.
4. **Operational Note on `g4-standard-12` COS Automated Driver Installer:**
   - On `g4-standard-12` (DWS Flex-start in `us-central1-b`), GKE's `cos-gpu-installer` failed because Compute Engine metadata `instance/host/nvidia-host-driver-version` returned `UNAVAILABLE`. Loading the Google-signed COS GRID vGPU driver (`580.159.03-grid-gcp`) directly worked cleanly, whereas full-GPU `g4-standard-384` MIG and MPS pools initialized drivers automatically out of the box with `gpu-driver-version=latest`.

## MIG Slicing Reference (`<compute_slices>g.<memory>gb`)

NVIDIA MIG profile names use **`<N>g.<M>gb`**, where **`<N>g`** is the number of hardware compute slices (SMs, L2 cache, memory controllers, copy engines) and **`<M>gb`** is the dedicated, hardware-isolated VRAM per slice. Note that the total compute-slice denominator is **4** on RTX PRO 6000 Blackwell vs. **7** on H100 and B200:

| GPU Family & 8-GPU Machine | Profile | Fraction of 1 GPU | VRAM / Slice | Slices / GPU | Slices / 8-GPU Node | Host vCPU / RAM per Slice | Equivalent / Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **RTX PRO 6000 Blackwell**<br>`g4-standard-384` (384 vCPU, 1,440 GiB) | **`1g.24gb`** | **1/4** (46 SMs) | 24 GB GDDR7 (~23.6 GiB usable) | 4 | **32** | 12 vCPU / 45 GiB | Exact 1:1 match to `g4-standard-12` *(tested above)* |
| **RTX PRO 6000 Blackwell**<br>`g4-standard-384` | **`2g.48gb`** | **1/2** | 48 GB GDDR7 | 2 | **16** | 24 vCPU / 90 GiB | Exact 1:1 match to `g4-standard-24` |
| **RTX PRO 6000 Blackwell**<br>`g4-standard-384` | **`4g.96gb`** | **Full GPU** | 96 GB GDDR7 | 1 | **8** | 48 vCPU / 180 GiB | Unpartitioned full GPU (`g4-standard-48` equiv.) |
| **NVIDIA H100 (80 GB)**<br>`a3-megagpu-8g` / `a3-highgpu-8g` (208 vCPU, 1,872 GiB) | **`1g.10gb`** | **1/7** | 10 GB HBM3 | 7 | **56** | ~3.5 vCPU / ~32 GiB | Small VRAM, CPU-light jobs |
| **NVIDIA H100 (80 GB)**<br>`a3-megagpu-8g` / `a3-highgpu-8g` | **`1g.20gb`** | **1/7** | 20 GB HBM3 | 4 | **32** | ~6 vCPU / ~55 GiB | 4/GPU (~20 GB VRAM, moderate CPU) |
| **NVIDIA H100 (80 GB)**<br>`a3-megagpu-8g` / `a3-highgpu-8g` | **`2g.20gb`** | **2/7** | 20 GB HBM3 | 3 | **24** | ~8 vCPU / ~75 GiB | Balanced compute + vCPU *(6/7 GPU used)* |
| **NVIDIA H100 (80 GB)**<br>`a3-megagpu-8g` / `a3-highgpu-8g` | **`3g.40gb`** | **3/7** | 40 GB HBM3 | 2 | **16** | ~13 vCPU / ~117 GiB | Larger solver memory + high FP64 |
| **NVIDIA B200 (180 GB MIG)**<br>`a4-highgpu-8g` (224 vCPU, 3,968 GiB) | **`1g.23gb`** | **1/7** | 23 GB HBM3e | 7 | **56** | ~4 vCPU / ~70 GiB | Matches `g4-standard-12` VRAM; check CPU fit |
| **NVIDIA B200 (180 GB MIG)**<br>`a4-highgpu-8g` | **`2g.45gb`** | **2/7** | 45 GB HBM3e | 3 | **24** | ~9 vCPU / ~165 GiB | Closer vCPU match to `g4-standard-12` |
| **NVIDIA B200 (180 GB MIG)**<br>`a4-highgpu-8g` | **`3g.90gb`** | **3/7** | 90 GB HBM3e | 2 | **16** | ~14 vCPU / ~248 GiB | Heavy meshes / high host CPU needs |

## How to Compute Cost per Simulation ($/Job)

Divide your hourly VM cost (On-Demand, DWS Flex-start, or Spot rate) by **Jobs/hr/Node** from the table above:

```text
Cost_per_Job = (Hourly_VM_Price_USD) / (Jobs_per_hr_per_Node)
```

## Related Resources

* [Repository Setup & Quick Start](../README.md) · [G4 vs. H100 vs. B200 Comparison Guide](../fractional-gpu-comparison-and-test-guide.md) · [Benchmark Harness Runbook](README.md)
* [GPU Sharing Strategies on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/timesharing-gpus) · [GKE Multi-Instance GPUs (MIG)](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus-multi) · [GKE NVIDIA MPS](https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-mps) · [GKE Dynamic Workload Scheduler (Flex-start)](https://cloud.google.com/kubernetes-engine/docs/how-to/provisioningrequest) · [Compute Engine Accelerator-Optimized Machines](https://cloud.google.com/compute/docs/accelerator-optimized-machines)
