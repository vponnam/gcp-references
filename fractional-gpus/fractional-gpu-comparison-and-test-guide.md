# Fractional GPU Guide: `g4-standard-12` → H100 → B200 (GKE & Slurm)

A side-by-side comparison and practical validation guide for HPC teams starting on fractional **`g4-standard-12`** VMs and progressing to fractionalized **H100 (`a3-highgpu-8g` / `a3-megagpu-8g`)** and **B200 (`a4-highgpu-8g`)** 8-GPU nodes using standard **GKE** and **Slurm** primitives (**MIG** as the default standard path, with **native MPS** as an optional A/B path).

---

## 1. Side-by-Side Comparison: G4 vs. H100 vs. B200

| Dimension | `g4-standard-12` (Baseline) | H100 — `a3-highgpu-8g` / `a3-megagpu-8g` | B200 — `a4-highgpu-8g` |
| :--- | :--- | :--- | :--- |
| **Underlying GPU** | 1/4 × NVIDIA RTX PRO 6000 Blackwell (Server) | **8 × NVIDIA H100** (SXM, 80 GB/GPU) | **8 × NVIDIA B200** (SXM, ~180–192 GB/GPU) |
| **Fractionalization Layer** | Cloud hypervisor **NVIDIA vGPU** (built-in) | **MIG** (default) or **Native MPS** | **MIG** (default) or **Native MPS** |
| **Memory Type & Bandwidth (Full GPU)** | GDDR7 (~1.6–1.8 TB/s full card, shared) | HBM3 (**3.35 TB/s** per GPU) | HBM3e (**up to 8.0 TB/s** per GPU) |
| **Native FP64 Compute (Full GPU)** | ~2 TFLOPS (1:64 de-tuned) | **~34–67 TFLOPS** per GPU | **~37–40 TFLOPS** per GPU |
| **Host vCPU / RAM per VM** | 12 vCPUs / 45 GiB | 208 vCPUs / 1,872 GiB (26 vCPU & 234 GiB per GPU) | 224 vCPUs / 3,968 GiB (28 vCPU & 496 GiB per GPU) |
| **Slice Profiles Matching ~24 GB Baseline** | 1 × **24 GB** (fixed VM shape) | **MIG:** `2g.20gb` (3/GPU = **24/node**), `1g.20gb` (4/GPU = **32/node**), or `1g.10gb` (7/GPU = **56/node**)<br>**MPS:** `N = 3–4` workers/GPU (**24–32/node**) | **MIG:** `2g.45gb` (3/GPU = **24/node**) or `1g.23gb` (7/GPU = **56/node**)<br>**MPS:** `N = 4–7` workers/GPU (**32–56/node**) |
| **Host vCPU / RAM per Job Slice** | **12 vCPU** / 45 GiB | `2g.20gb`: **~8 vCPU** / ~75 GiB<br>`1g.20gb`: **~6 vCPU** / ~55 GiB | `2g.45gb`: **~9 vCPU** / ~165 GiB<br>`1g.23gb`: **~4 vCPU** / ~70 GiB |
| **VRAM & Cache Isolation** | Hard (hypervisor vGPU) | **MIG:** Hard (dedicated HBM, L2, SMs)<br>**MPS:** Soft/pinned memory cap; shared L2/bus | **MIG:** Hard (dedicated HBM, L2, SMs)<br>**MPS:** Soft/pinned memory cap; shared L2/bus |
| **Fault Isolation** | VM-level | **MIG:** Hardware fault-isolated<br>**MPS:** Shared MPS fault domain | **MIG:** Hardware fault-isolated<br>**MPS:** Shared MPS fault domain |
| **Min. Concurrent Jobs to Saturate VM** | **1 job** | **24–32 jobs** (8-GPU node) | **24–56 jobs** (8-GPU node) |
| **GKE Support & Version** | GKE Standard (`count=1`)<br>GKE `1.35.2-gke.1485000+` / `1.36.4+` | GKE Standard (`count=8`, `gpu-partition-size` or `gpu-sharing-strategy=mps`) | GKE Standard (`count=8`, `gpu-partition-size` or `mps`)<br>GKE `1.32.2-gke.1586000+` |
| **Slurm Support (Cluster Toolkit)** | Standard dynamic partition (`AutoDetect=nvml`) | `nvidia-smi mig` / `nvidia-mig-parted` before `slurmd` + `AutoDetect=nvml` | `nvidia-smi mig` / `nvidia-mig-parted` before `slurmd` + `AutoDetect=nvml` |
| **Networking Note** | Standard 20 Gbps | MIG disables GPUDirect-TCPX/TCPXO (fine for single-slice HPC jobs) | MIG disables GPUDirect-RDMA (fine for single-slice HPC jobs) |
| **Provisioning & Best Fit** | **On-Demand / DWS Flex-start / Spot** — dev, ad-hoc, small batches, CPU-heavy runs (12 vCPU) | **DWS Flex-start / Spot / Reservations** — parallel sweeps (≥24–32 jobs), FP64/HBM-sensitive runs | **DWS Flex-start / Spot / Reservations** — large parallel sweeps (≥24–56 jobs), max HBM3e bandwidth |

> **Rule of Thumb for HPC Slice Sizing:**
> Check **both VRAM and vCPU** from your `g4-standard-12` baseline before choosing a slice on 8-GPU H100/B200 nodes. If each simulation needs 8–10 vCPUs, **`2g.20gb` on H100 (24 jobs/node, ~8 vCPU/job)** or **`2g.45gb` on B200 (24 jobs/node, ~9 vCPU/job)** will often beat `1g` 7-way splits (~3.5–4 vCPU/job) by avoiding CPU starvation.

---

## 2. Simple Guidance to Test & Validate Functionality

Follow these 4 steps to verify that fractionalization, isolation, and throughput scale as expected across **G4 → H100 → B200** using the **same job interface** (`nvidia.com/gpu: 1` on GKE or `--gres=gpu:1` on Slurm).

### Step 1: Prepare One Portable Test Workload

Use either your real HPC solver container or a lightweight CUDA verification command (e.g., NVIDIA CUDA container + PyTorch/CuPy or a small FP64/bandwidth benchmark like `babelstream` / `hpcg` / custom solver). Ensure the job:
1. Reads `CUDA_VISIBLE_DEVICES` (never hardcodes GPU `0`).
2. Logs **wall-clock duration**, **visible GPU UUID**, **visible VRAM**, and **exit code**.

Quick built-in sanity check command you can run inside any CUDA/PyTorch container:
```bash
python3 -c "
import os, time, torch
props = torch.cuda.get_device_properties(0)
print(f'Host={os.uname().nodename} Visible={os.environ.get(\"CUDA_VISIBLE_DEVICES\")} '
      f'GPU={props.name} VRAM_GB={props.total_memory/1e9:.2f}')
# Quick FP64 + memory bandwidth sanity kernel
n = 8192
a = torch.randn((n, n), dtype=torch.float64, device='cuda')
b = torch.randn((n, n), dtype=torch.float64, device='cuda')
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
print(f'Elapsed_s={time.time()-t0:.3f} Peak_VRAM_GB={torch.cuda.max_memory_allocated()/1e9:.2f}')
"
```

---

### Step 2: Provision Test Pools / Partitions & Run Smoke Tests

#### Option A: Testing on GKE Standard

> **Reference:** See Google Cloud's [GPU Sharing Strategies on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/timesharing-gpus) documentation for a full overview of MIG, NVIDIA MPS, and time-sharing trade-offs.

**1. Create scale-to-zero test node pools (G4, H100 MIG, optional H100 MPS, B200 MIG):**
```bash
# Baseline: g4-standard-12 (supports DWS Flex-start, On-Demand, or Spot)
# Remove --flex-start/--reservation-affinity for On-Demand, or replace with --spot for Spot
gcloud container node-pools create test-g4 \
  --cluster=CLUSTER_NAME --region=REGION \
  --machine-type=g4-standard-12 \
  --accelerator=type=nvidia-rtx-pro-6000,count=1,gpu-driver-version=default \
  --flex-start --reservation-affinity=none --max-run-duration=7200s \
  --enable-autoscaling --min-nodes=0 --max-nodes=2

# H100 MIG (3 x 2g.20gb per GPU = 24 isolated slices per 8-GPU node)
# Note: for a3-megagpu-8g, set --machine-type=a3-megagpu-8g and type=nvidia-h100-mega-80gb
gcloud container node-pools create test-h100-mig \
  --cluster=CLUSTER_NAME --region=REGION \
  --machine-type=a3-highgpu-8g \
  --accelerator=type=nvidia-h100-80gb,count=8,gpu-partition-size=2g.20gb,gpu-driver-version=default \
  --flex-start \
  --enable-autoscaling --min-nodes=0 --max-nodes=1

# Optional A/B: H100 Native MPS (4 shared clients/GPU = 32 clients per 8-GPU node)
gcloud container node-pools create test-h100-mps \
  --cluster=CLUSTER_NAME --region=REGION \
  --machine-type=a3-highgpu-8g \
  --accelerator=type=nvidia-h100-80gb,count=8,gpu-sharing-strategy=mps,max-shared-clients-per-gpu=4,gpu-driver-version=default \
  --flex-start \
  --enable-autoscaling --min-nodes=0 --max-nodes=1

# B200 MIG (example: 2g.45gb = 24 slices/node, or 1g.23gb = 56 slices/node)
gcloud container node-pools create test-b200-mig \
  --cluster=CLUSTER_NAME --region=REGION \
  --machine-type=a4-highgpu-8g \
  --accelerator=type=nvidia-b200,count=8,gpu-partition-size=2g.45gb,gpu-driver-version=default \
  --flex-start \
  --enable-autoscaling --min-nodes=0 --max-nodes=1
```

**2. Run the same Kubernetes `Job` template across tiers** (adjust only `completions`/`parallelism`, CPU/memory requests, and `nodeSelector`):
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hpc-slice-test-h100-mig
spec:
  completions: 24       # 1 for G4 | 24 for H100 2g.20gb | 32 for H100 MPS | 24 for B200 2g.45gb | 56 for B200 1g.23gb
  parallelism: 24
  completionMode: Indexed
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        # G4:       cloud.google.com/gke-accelerator: "nvidia-rtx-pro-6000"
        # H100 MIG: cloud.google.com/gke-gpu-partition-size: "2g.20gb"
        # H100 MPS: cloud.google.com/gke-gpu-sharing-strategy: "mps"
        # B200 MIG: cloud.google.com/gke-gpu-partition-size: "2g.45gb"
        cloud.google.com/gke-gpu-partition-size: "2g.20gb"
      containers:
        - name: hpc-worker
          image: YOUR_HPC_IMAGE # or pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime (CUDA 12.8+ for Blackwell/G4 & Hopper)
          command: ["bash", "-c", "nvidia-smi -L && ./run_sim.sh"]
          resources:
            requests:
              cpu: "7"          # ~1/N of node vCPUs so all slices co-schedule on 1 node
              memory: "64Gi"
              nvidia.com/gpu: "1"
            limits:
              nvidia.com/gpu: "1"
```

**3. Verify concurrent placement & isolation on a single node:**
```bash
# Confirm all N pods landed on the SAME node concurrently
kubectl get pods -l job-name=hpc-slice-test-h100-mig -o wide

# Inspect each pod's isolated MIG UUID and runtime
kubectl logs -l job-name=hpc-slice-test-h100-mig --prefix
```
* **Expected result (MIG):** All `N` pods run in parallel on **1 node**, each sees a **distinct `MIG-...` UUID** and only its slice's VRAM (e.g., ~20 GB on H100 `2g.20gb`, ~45 GB on B200 `2g.45gb`).

---

#### Option B: Testing on Slurm (Google Cloud Cluster Toolkit)

**1. Configure MIG on H100 / B200 compute nodes (in node startup script *before* `slurmd`):**
```bash
# H100 (a3-highgpu-8g / a3-megagpu-8g): 3 x 2g.20gb per GPU (24 slices/node) — or 4 x 1g.20gb (32 slices/node)
nvidia-smi -mig 1
nvidia-smi mig -cgi 2g.20gb,2g.20gb,2g.20gb -C
systemctl restart slurmd

# B200 (a4-highgpu-8g): 3 x 2g.45gb per GPU (24 slices/node) — or 7 x 1g.23gb (56 slices/node)
nvidia-smi -mig 1
nvidia-smi mig -cgi 2g.45gb,2g.45gb,2g.45gb -C
systemctl restart slurmd
```
Ensure `/etc/slurm/gres.conf` has `AutoDetect=nvml`, and `slurm.conf` declares `GresTypes=gpu`, `SlurmdParameters=config_overrides`, and matching node GRES counts (`Gres=gpu:2g.20gb:24` on 8-GPU H100, `Gres=gpu:2g.45gb:24` or `Gres=gpu:1g.23gb:56` on 8-GPU B200).

**2. Verify Slurm discovered all slices:**
```bash
sinfo -o "%P %a %D %G %c %m"
scontrol show node <h100-or-b200-node> | grep -E "Gres|State|CPU"
```

**3. Run identical job arrays across all 3 partitions:**
```bash
# G4 baseline (1 job on 1 g4-standard-12)
sbatch -p g4       --gres=gpu:1 --cpus-per-task=12 --mem=40G  --array=1-1  run_sim.sh

# H100 MIG (24 concurrent jobs on 1 a3-highgpu-8g with 2g.20gb, or --array=1-32 with 1g.20gb)
sbatch -p h100-mig --gres=gpu:1 --cpus-per-task=8  --mem=70G  --array=1-24 run_sim.sh

# B200 MIG (24 concurrent jobs on 1 a4-highgpu-8g with 2g.45gb, or --array=1-56 with 1g.23gb)
sbatch -p b200-mig --gres=gpu:1 --cpus-per-task=9  --mem=150G --array=1-24 run_sim.sh
```

**4. Optional A/B MPS Test on an 8-GPU H100 Node (zero Slurm plugin changes):**
Submit an 8-task array (`--array=1-8`, 1 whole GPU per task) where each task runs `N=4` solver instances concurrently under a private per-GPU MPS daemon (**32 total concurrent instances** on one `a3-highgpu-8g` node):
```bash
#!/bin/bash
#SBATCH -p h100-whole --gres=gpu:1 --cpus-per-task=24 --mem=200G --array=1-8
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

# Launch 4 parallel instances sharing 1 H100 (optional: cap pinned memory per client)
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25
for i in 1 2 3 4; do
  CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=18G" ./run_sim.sh --seed="${SLURM_ARRAY_TASK_ID}_${i}" &
done
wait

echo quit | nvidia-cuda-mps-control
```

---

### Step 3: Isolation & Fault-Containment Check

Before trusting any fractional setup in production, run a 2-minute **noisy-neighbor OOM test**:
1. Launch `N-1` normal solver jobs alongside **1 intentional OOM job** that allocates more VRAM than its slice (e.g., `25 GB` on a `20 GB` slice).
2. **Expected behavior:**
   * **MIG (`h100-mig`, `b200-mig`):** Only the over-allocating job fails with `CUDA out of memory`; the other `N-1` jobs complete normally with unaffected runtime.
   * **MPS (`h100-mps`):** Confirm whether pinned/CUDA 13.4 memory caps contain the OOM and whether a forced fatal CUDA fault impacts sibling processes—document the risk if you choose MPS.

---

### Step 4: Benchmark Scorecard & Go/No-Go Decision

> **Published Reference Results:** See [bench/results_scorecard.md](bench/results_scorecard.md) for measured GKE results (`g4-standard-12` DWS Flex-start vs. 32× `g4-standard-384` MIG `1g.24gb` vs. 32× MPS `4/GPU` + OOM isolation PASS) and the complete MIG slicing reference table.

Record results from running the **same simulation workload** at full single-node packing across each configuration:

| Metric | `g4-standard-12` (1×) | H100 MIG (`2g.20gb`, 24×) | H100 MPS (`N=4/GPU`, 32×) | B200 MIG (`2g.45gb`, 24×) | B200 MIG (`1g.23gb`, 56×) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Concurrent Jobs / VM** | 1 | 24 | 32 | 24 | 56 |
| **Median / p95 Runtime (s)** | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| **Throughput (Jobs / hr / VM)** | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| **Avg Host vCPU Util (%)** | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| **Avg GPU SM / Mem Util (%)** | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| **Effective Cost ($ / Job)** | _baseline_ | _fill_ | _fill_ | _fill_ | _fill_ |
| **OOM Isolation Test** | Pass (VM) | Pass (HW) | _verify_ | Pass (HW) | Pass (HW) |

**Decision Checklist:**
1. **CPU Starvation Check:** If Host vCPU Util hits ~100% while GPU SM/Mem Util drops, step **down** in packing density (e.g., H100 `1g.20gb` → `2g.20gb`, or B200 `1g.23gb` → `2g.45gb`).
2. **MIG vs. MPS Gate:** Standardize on **MIG** unless MPS beats MIG by **≥25% on $/Job** *and* your jobs can tolerate a shared fault domain.
3. **8-GPU Saturation Gate (H100 & B200):** Route sweeps to 8-GPU H100/B200 nodes only when queue depth is high enough to saturate **≥75% of the node** (e.g., ≥18–42 concurrent jobs); keep ad-hoc runs and small batches on **`g4-standard-12`**.
4. **Cleanup:** Verify all test pools/partitions scale back to `0` nodes (`kubectl get nodes` / `sinfo`) after testing.
