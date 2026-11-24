# Fractional GPUs on Google Cloud (GKE & Slurm)

A reference architecture, functional validation guide, and runnable GKE benchmark harness for teams evaluating **GPU fractionalization on Google Cloud**—progressing from fractional VM shapes (**`g4-standard-12`**) to fractionalized **8-GPU nodes** (**`g4-standard-384`**, **H100 `a3-highgpu-8g` / `a3-megagpu-8g`**, and **B200 `a4-highgpu-8g`**) using **NVIDIA Multi-Instance GPU (MIG)** and **NVIDIA Multi-Process Service (MPS)**.

---

## What Is Included

* **[Fractional GPU Comparison & Test Guide](fractional-gpu-comparison-and-test-guide.md)** — Architecture comparison across `g4-standard-12`, H100 (`a3-highgpu-8g` / `a3-megagpu-8g`), and B200 (`a4-highgpu-8g`), slice-sizing rules of thumb (VRAM + host vCPU/RAM), and step-by-step **GKE** and **Slurm** validation patterns.
* **[GKE Benchmark Harness (`bench/`)](bench/README.md)** — End-to-end scripts and Kubernetes `Indexed` Job manifests running an identical PyTorch/CUDA 12.8 matrix multiplication workload (FP64, strict IEEE FP32, and BF16 with a configurable resident VRAM working set) across:
  * **`g4-standard-12`** baseline (1/4 RTX PRO 6000 Blackwell via DWS Flex-start or Spot)
  * **`g4-standard-384` MIG (`1g.24gb`)** — 32 hardware-isolated slices on 1 node
  * **`g4-standard-384` MIG OOM Isolation** — 32 concurrent pods where index `0` intentionally triggers CUDA OOM while indices `1..31` complete normally
  * **`g4-standard-384` Native MPS (`4/GPU`)** — 32 shared pods on 1 node
  * **Optional H100 Mega (`a3-megagpu-8g`)** — 24× `2g.20gb` MIG slices, 24-pod OOM isolation, and 32× MPS (`4/GPU`) via DWS Flex-start, Spot, or Compute Engine Reservations
* **[Reference Scorecard](bench/results_scorecard.md)** & **[Anonymized Reference Snapshots](bench/reference_results/)** — Measured functional results from GKE (`1.36.4-gke.2377000`, `us-central1`) plus a full **MIG Slicing Reference** across RTX PRO 6000 Blackwell, H100, and B200.

---

## Reference Validation Highlights

Measured on GKE Standard (`1.36.4-gke.2377000`, `us-central1`) using `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` (`8192×8192` FP64/FP32/BF16 MatMul + `10 GB` resident buffer; full table and notes in [bench/results_scorecard.md](bench/results_scorecard.md)):

| Tier | Machine & Mode | Pods OK | Nodes | Unique Slices | Visible / Peak VRAM | Slice p50 / p95 (s) | Jobs/hr/Node | Node FP32 / BF16 TFLOPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `g4-flex` | `g4-standard-12` (DWS Flex, 1/4 GPU) | 1/1 | 1 | 1 | 23.69 / 9.51 GiB | 78.11 / 78.11 | **46.1** | 20.60 / 107.36 |
| `g4-8gpu-mig` | `g4-standard-384` MIG `1g.24gb` (Spot) | 32/32 | 1 | 32 MIG UUIDs | 23.62 / 9.51 GiB | 77.64 / 78.53 | **1,405.8** | 626.94 / 3,165.43 |
| `g4-8gpu-mps` | `g4-standard-384` MPS `4/GPU` (Spot) | 32/32 | 1 | 8 GPUs (4/GPU) | 94.97 / 9.51 GiB | 78.25 / 79.48 | **1,412.9** | 630.82 / 3,232.33 |

* **Noisy-Neighbor OOM Isolation (`g4-8gpu-mig-oom`):** **PASS** — Pod index `0` requested `28 GiB` on its `23.62 GiB` MIG slice, caught an isolated CUDA OOM, and **31/31 sibling pods** on the same node finished normally.
* **Key Takeaway:** Partitioning an 8-GPU `g4-standard-384` with MIG `1g.24gb` delivers **exact 1:1 slice parity** with `g4-standard-12` (`~30.5×` jobs/hr/node with `<1%` p95 jitter) and **~99.5% of MPS throughput** while preserving strict hardware VRAM and fault isolation.

### MIG Profile Quick Reference (`<compute_slices>g.<memory>gb`)

| GPU Family & 8-GPU Machine | Example MIG Profiles | Slices / GPU | Slices / 8-GPU Node | Host vCPU / RAM per Slice |
| :--- | :--- | :---: | :---: | :---: |
| **RTX PRO 6000 Blackwell** (`g4-standard-384`) | `1g.24gb` / `2g.48gb` / `4g.96gb` | 4 / 2 / 1 | **32 / 16 / 8** | 12 / 24 / 48 vCPU · 45 / 90 / 180 GiB |
| **NVIDIA H100 80 GB** (`a3-megagpu-8g` / `a3-highgpu-8g`) | `1g.10gb` / `1g.20gb` / `2g.20gb` / `3g.40gb` | 7 / 4 / 3 / 2 | **56 / 32 / 24 / 16** | ~3.5 / ~6 / ~8 / ~13 vCPU · ~32–117 GiB |
| **NVIDIA B200 180 GB** (`a4-highgpu-8g`) | `1g.23gb` / `2g.45gb` / `3g.90gb` | 7 / 3 / 2 | **56 / 24 / 16** | ~4 / ~9 / ~14 vCPU · ~70–248 GiB |

See [bench/results_scorecard.md](bench/results_scorecard.md#mig-slicing-reference-compute_slicesgmemorygb) for the complete breakdown.

---

## Repository Structure

```text
.
├── README.md                                   # Public setup & testing guide (this file)
├── fractional-gpu-comparison-and-test-guide.md # G4 vs. H100 vs. B200 comparison + GKE & Slurm guide
└── bench/
    ├── README.md                               # Detailed benchmark harness runbook
    ├── config.env                              # Configurable project, region, pools, MIG/MPS & workload settings
    ├── 00_reserve_a3mega.sh                    # Optional retrying reservation automator for 1x a3-megagpu-8g
    ├── 01_setup_cluster_and_pools.sh           # Creates dedicated VPC, GKE cluster & scale-to-zero GPU pools
    ├── 01b_attach_reserved_a3mega.sh           # Attaches MIG/MPS pools to a reserved a3-megagpu-8g & runs suite
    ├── 02_run_benchmarks.sh                    # Submits Indexed Jobs and waits for completion
    ├── 03_collect_results.py                   # Parses pod JSON telemetry and generates local_scorecard.md
    ├── 04_cleanup.sh                           # Scales down jobs or deletes pools, reservations, cluster & VPC
    ├── matmul_bench.py                         # Portable PyTorch FP64/FP32/BF16 + OOM benchmark script
    ├── results_scorecard.md                    # Published reference scorecard & MIG slicing table
    ├── reference_results/                      # Anonymized JSONL records from the reference GKE run
    └── k8s/                                    # Kubernetes Namespace and Indexed Job manifests
```

---

## Prerequisites

1. **Google Cloud Project & APIs:**
   * A GCP project with billing enabled and the following APIs activated:
     ```bash
     gcloud services enable container.googleapis.com compute.googleapis.com
     ```
2. **CLI Tools:**
   * [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) (authenticated via `gcloud auth login`)
   * `kubectl` (installable via `gcloud components install kubectl` and `gke-gcloud-auth-plugin`)
   * Python `3.9+` (for `bench/03_collect_results.py`)
3. **IAM Permissions:**
   * Permissions to manage GKE clusters/node pools (`roles/container.admin`), Compute Engine VPCs/subnets/reservations (`roles/compute.networkAdmin`, `roles/compute.instanceAdmin.v1`), and service account usage (`roles/iam.serviceAccountUser`).
4. **GPU Quotas & Zone Availability:**
   * Verify GPU machine type availability in your target region:
     ```bash
     gcloud compute machine-types list \
       --filter="name=(g4-standard-12 OR g4-standard-384 OR a3-megagpu-8g OR a3-highgpu-8g OR a4-highgpu-8g)" \
       --format="table(name,zone,guestCpus,memoryMb)"
     ```
   * Ensure your project has sufficient GPU quota (On-Demand, Spot/Preemptible, or DWS Flex-start depending on your chosen provisioning mode) in the selected region/zones.

---

## Quick Start: Run the GKE Functional Benchmark

> **Cost Safety Note:** All GPU node pools created by `01_setup_cluster_and_pools.sh` start at **0 nodes** (`--total-min-nodes=0 --total-max-nodes=1`), and DWS Flex-start pools enforce a 2-hour cap (`FLEX_MAX_RUN_DURATION=7200s`). Always run `./bench/04_cleanup.sh` when finished so no GPU nodes, clusters, or reservations remain billed.

### 1. Configure Your Environment

Set your GCP project ID, target region/zones, and an optional unique resource prefix (used for all VPCs, clusters, node pools, Kubernetes namespaces, and labels):

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export GPU_ZONES="us-central1-b,us-central1-c"
export G4_8GPU_ZONES="us-central1-b,us-central1-c,us-central1-f"
export RESOURCE_PREFIX="gpu-bench"   # customize per team/user for easy cleanup
```

Alternatively, edit defaults directly in [bench/config.env](bench/config.env).

### 2. Provision the Dedicated VPC, GKE Cluster & Scale-to-Zero GPU Pools

By default, `01_setup_cluster_and_pools.sh` creates:
* A dedicated custom VPC & subnet (avoids pod CIDR exhaustion on shared `default` VPCs)
* A GKE Standard cluster on the `regular` release channel (fractional G4 requires GKE `>= 1.35.2`)
* Scale-to-zero node pools for **`g4-standard-12`** (**DWS Flex-start** by default; also supports `on-demand` or `spot` via `G4_PROVISIONING`) and **`g4-standard-384`** (**MIG `1g.24gb`** + **MPS `4/GPU`** on Spot)

```bash
./bench/01_setup_cluster_and_pools.sh
```

To also create **`a3-megagpu-8g` (8× H100 Mega)** MIG (`2g.20gb`) and MPS (`4/GPU`) pools:

```bash
CREATE_H100_POOLS=true H100_PROVISIONING=flex ./bench/01_setup_cluster_and_pools.sh
# Or use Spot for H100 Mega:
# CREATE_H100_POOLS=true H100_PROVISIONING=spot ./bench/01_setup_cluster_and_pools.sh
```

### 3. Run the Benchmarks

Run tests one at a time (recommended so only one GPU node scales up at a time) or run the full 8-GPU G4 validation suite:

```bash
# 1. Baseline: 1 pod on g4-standard-12 (1/4 RTX PRO 6000)
./bench/02_run_benchmarks.sh g4

# 2. 8-GPU G4 functional suite: 32x MIG (1g.24gb) -> 32-pod OOM isolation -> 32x MPS (4/GPU)
./bench/02_run_benchmarks.sh g4-8gpu-all

# Or run individual tiers:
# ./bench/02_run_benchmarks.sh g4-8gpu-mig
# ./bench/02_run_benchmarks.sh g4-8gpu-oom
# ./bench/02_run_benchmarks.sh g4-8gpu-mps

# Optional H100 Mega (if CREATE_H100_POOLS=true):
# ./bench/02_run_benchmarks.sh h100-mig
# ./bench/02_run_benchmarks.sh oom
# ./bench/02_run_benchmarks.sh h100-mps
```

### 4. Generate Your Scorecard

```bash
python3 ./bench/03_collect_results.py
```

This collects structured `BENCH_RESULT_JSON` telemetry from every completed pod in `${K8S_NAMESPACE}` and writes your local Markdown scorecard to `bench/local_scorecard.md` (pass `--out <path>` to customize), keeping the published reference [bench/results_scorecard.md](bench/results_scorecard.md) intact for side-by-side comparison.

### 5. Optional: H100 Mega Reservation-First Workflow

If Spot or DWS Flex-start capacity for `a3-megagpu-8g` is temporarily constrained in your region and you have On-Demand `NVIDIA_H100_MEGA` quota, you can use the retrying reservation workflow (modeled after [gcp-ai-hpc-reservations-automator](https://github.com/vponnam/gcp-ai-hpc-reservations-automator/tree/main/accelerator-reservations)):

```bash
# 1. Lock in a specific-targeted 1x a3-megagpu-8g reservation across your candidate zones:
CANDIDATE_ZONES="us-central1-a us-central1-b us-central1-c" ./bench/00_reserve_a3mega.sh

# 2. Attach MIG (24x 2g.20gb) + OOM isolation + MPS (32x 4/GPU) pools and collect results:
./bench/01b_attach_reserved_a3mega.sh

# 3. IMMEDIATELY delete the on-demand reservation when finished to stop billing:
./bench/04_cleanup.sh --delete-reservation
```

> **Warning:** On-demand Compute Engine reservations begin billing immediately upon creation even before nodes attach. Delete them as soon as testing completes.

### 6. Clean Up All Resources

```bash
# Option A: Delete benchmark jobs so GPU pools autoscale back to 0 (keeps cluster & pools)
./bench/04_cleanup.sh --jobs-only

# Option B: Delete only the benchmark GPU node pools
./bench/04_cleanup.sh --delete-pools

# Option C: Full teardown (deletes benchmark clusters, reservations, subnets, and custom VPC)
./bench/04_cleanup.sh --delete-cluster
```

---

## Testing on Slurm

If your team runs HPC workloads on **Slurm** via Google Cloud Cluster Toolkit rather than GKE, see **[Section 2 (Option B: Testing on Slurm)](fractional-gpu-comparison-and-test-guide.md#option-b-testing-on-slurm-google-cloud-cluster-toolkit)** in the comparison guide for `nvidia-smi mig` startup partitioning, `gres.conf` (`AutoDetect=nvml`), and parallel `sbatch --array` verification steps.

---

## Operational Notes & Lessons Learned

1. **Container / CUDA Compatibility:** Use **PyTorch 2.7+ / CUDA 12.8+** (default: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` or `nvcr.io/nvidia/pytorch:25.06-py3`) so the exact same image runs natively on both **Blackwell (`sm_120`, RTX PRO 6000 / G4)** and **Hopper (`sm_90`, H100)**.
2. **Fractional G4 Driver Flag:** On GKE, `g4-standard-12` requires `--accelerator=type=nvidia-rtx-pro-6000,count=1,gpu-driver-version=default` (`count` must be `1` even for the 1/4 GPU VM shape). Full-GPU `g4-standard-384` MIG and MPS pools initialize automatically with `gpu-driver-version=latest`.
3. **Check Host vCPU & RAM Before Choosing Slice Density:** On 8-GPU H100 (`208` vCPUs) and B200 (`224` vCPUs) nodes, `1g` 7-way splits yield ~3.5–4 vCPUs/slice compared to `12` vCPUs on `g4-standard-12`. For CPU-moderate HPC simulations, `2g.20gb` (H100, ~8 vCPU/slice) or `2g.45gb` (B200, ~9 vCPU/slice) prevents host CPU bottlenecks.
4. **Networking Trade-off with MIG:** Enabling MIG on `a3-*` or `a4-*` nodes disables GPUDirect-TCPX/TCPXO/RDMA—this is expected and has no impact on independent single-slice HPC jobs or parameter sweeps.
5. **PyTorch 2.8 MIG OOM Handling:** When a CUDA OOM occurs inside a MIG container, PyTorch 2.8's OOM error-formatting path may query NVML and raise `RuntimeError: NVML_SUCCESS == r` instead of `torch.cuda.OutOfMemoryError`. The hardware VRAM limit is still strictly enforced; [bench/matmul_bench.py](bench/matmul_bench.py) handles both exceptions cleanly.

---

## Official Public Documentation

### Google Cloud — GKE GPU Fractionalization & Scheduling
* [GPU Sharing Strategies on GKE (Time-Sharing, MPS & MIG)](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/timesharing-gpus)
* [Run Multi-Instance GPUs (MIG) on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus-multi)
* [Share GPUs using NVIDIA MPS on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-mps)
* [Share GPUs using Time-Sharing on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/timesharing-gpus)
* [Run GPUs in GKE Standard Node Pools](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
* [Provision GPUs with Dynamic Workload Scheduler (Flex-start & ProvisioningRequest)](https://cloud.google.com/kubernetes-engine/docs/how-to/provisioningrequest)

### Google Cloud — Compute Engine & HPC
* [Accelerator-Optimized Machine Family (G4, A3, A4)](https://cloud.google.com/compute/docs/accelerator-optimized-machines)
* [Reservations of Compute Engine Zonal Resources](https://cloud.google.com/compute/docs/instances/reservations-overview)
* [Google Cloud Cluster Toolkit (Slurm & GKE HPC Blueprints)](https://cloud.google.com/cluster-toolkit/docs)

### NVIDIA Documentation
* [NVIDIA Multi-Instance GPU (MIG) User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html)
* [NVIDIA Multi-Process Service (MPS) Documentation](https://docs.nvidia.com/deploy/mps/index.html)
