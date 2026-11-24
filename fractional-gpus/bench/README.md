# GKE Fractional GPU Benchmark Harness (`bench/`)

Runnable GKE Standard benchmark harness comparing a fractional GPU VM (**`g4-standard-12`**) against fractionalized **8-GPU nodes** (**`g4-standard-384`** and optionally **`a3-megagpu-8g`**) using an identical PyTorch/CUDA 12.8 matrix multiplication workload (FP64, strict IEEE FP32, and BF16) with a configurable resident VRAM working set and a noisy-neighbor OOM isolation test.

See the [root README](../README.md) for prerequisites and official Google Cloud / NVIDIA documentation, and [results_scorecard.md](results_scorecard.md) for the reference validation results.

---

## 1. Configure `config.env` (or Export Environment Variables)

All GCP and Kubernetes resources are prefixed with **`$RESOURCE_PREFIX`** (default: `gpu-bench`) and tagged with `owner=${RESOURCE_PREFIX}` for clean isolation and teardown.

```bash
export PROJECT_ID="your-gcp-project-id"   # or rely on `gcloud config get-value project`
export REGION="us-central1"
export GPU_ZONES="us-central1-b,us-central1-c"
export G4_8GPU_ZONES="us-central1-b,us-central1-c,us-central1-f"
export RESOURCE_PREFIX="gpu-bench"
```

Key defaults in [config.env](config.env):

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `CREATE_CLUSTER` | `true` | Creates a dedicated GKE Standard cluster (`regular` channel) and custom VPC (`${RESOURCE_PREFIX}-net`). Set `false` to reuse an existing cluster in `$REGION`. |
| `CREATE_G4_12_POOL` | `true` | Creates scale-to-zero `g4-standard-12` pool (`G4_PROVISIONING=flex`, or `on-demand` / `spot`). |
| `CREATE_G4_8GPU_POOLS` | `true` | Creates scale-to-zero `g4-standard-384` MIG (`1g.24gb`, 32 slices/node) and MPS (`4/GPU`, 32 pods/node) pools (`G4_8GPU_PROVISIONING=spot`). |
| `CREATE_H100_POOLS` | `false` | Set `true` to also create `a3-megagpu-8g` MIG (`2g.20gb`, 24 slices/node) and MPS (`4/GPU`, 32 pods/node) pools (`H100_PROVISIONING=flex` or `spot`). |
| `CREATE_MPS_POOL` | `true` | Toggle creation of MPS comparison pools alongside MIG pools. |
| `FLEX_MAX_RUN_DURATION` | `7200s` | Hard 2-hour safety cap for DWS Flex-start nodes. |
| `BENCH_IMAGE` | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` | Native CUDA 12.8 support for both Blackwell (`sm_120`, G4) and Hopper (`sm_90`, H100). |
| `MATMUL_N` / `TARGET_VRAM_GB` | `8192` / `10` | Matrix size and resident solver buffer (GB) allocated per pod. |
| `OOM_ALLOC_GB` | `28` | Intentionally oversized allocation for pod index `0` in OOM isolation tests. |

Check zone support for your target machine types before running:
```bash
gcloud compute machine-types list \
  --filter="name=(g4-standard-12 OR g4-standard-384 OR a3-megagpu-8g)" \
  --format="table(name,zone)"
```

---

## 2. Provision Cluster & Scale-to-Zero GPU Pools

```bash
# Default: g4-standard-12 (DWS Flex-start) + g4-standard-384 MIG & MPS (Spot)
./01_setup_cluster_and_pools.sh

# Optional: Switch g4-standard-12 to On-Demand or Spot:
# G4_PROVISIONING=on-demand ./01_setup_cluster_and_pools.sh
# G4_PROVISIONING=spot ./01_setup_cluster_and_pools.sh

# Optional: Also create a3-megagpu-8g (H100 Mega) pools using DWS Flex-start or Spot
CREATE_H100_POOLS=true H100_PROVISIONING=flex ./01_setup_cluster_and_pools.sh
# CREATE_H100_POOLS=true H100_PROVISIONING=spot ./01_setup_cluster_and_pools.sh
```

All GPU pools initialize at **0 nodes** (`--num-nodes=0 --total-min-nodes=0 --total-max-nodes=1`) and scale up on demand when Jobs are submitted.

### 2b. Optional: Reservation-First Path for `a3-megagpu-8g`

When Spot or DWS Flex-start capacity for `a3-megagpu-8g` is constrained and you have On-Demand `NVIDIA_H100_MEGA` quota, use the retrying reservation automator (modeled after [gcp-ai-hpc-reservations-automator](https://github.com/vponnam/gcp-ai-hpc-reservations-automator/tree/main/accelerator-reservations)):

```bash
# 1. Scan candidate zones until 1x a3-megagpu-8g specific-targeted reservation is created:
CANDIDATE_ZONES="us-central1-a us-central1-b us-central1-c" MAX_ROUNDS=20 RETRY_RATE=60 ./00_reserve_a3mega.sh

# 2. Attach reserved MIG (24x 2g.20gb) + OOM + MPS (32x 4/GPU) pools, run suite, and collect scorecard:
./01b_attach_reserved_a3mega.sh

# 3. Delete the on-demand reservation immediately after testing to stop billing:
./04_cleanup.sh --delete-reservation
```

---

## 3. Run Benchmarks

Run tiers individually (recommended so only one GPU node is active at a time) or as a suite:

```bash
# Baseline: 1 pod on g4-standard-12
./02_run_benchmarks.sh g4

# 8-GPU G4 functional suite (runs MIG -> OOM -> MPS sequentially):
./02_run_benchmarks.sh g4-8gpu-all

# Individual 8-GPU G4 tiers:
./02_run_benchmarks.sh g4-8gpu-mig   # 32 concurrent pods on 1x g4-standard-384 (MIG 1g.24gb)
./02_run_benchmarks.sh g4-8gpu-oom   # 32-pod noisy-neighbor OOM isolation test on MIG 1g.24gb
./02_run_benchmarks.sh g4-8gpu-mps   # 32 concurrent pods on 1x g4-standard-384 (MPS 4/GPU)

# Individual 8-GPU H100 Mega tiers (when CREATE_H100_POOLS=true):
./02_run_benchmarks.sh h100-mig      # 24 concurrent pods on 1x a3-megagpu-8g (MIG 2g.20gb)
./02_run_benchmarks.sh oom           # 24-pod noisy-neighbor OOM isolation test on MIG 2g.20gb
./02_run_benchmarks.sh h100-mps      # 32 concurrent pods on 1x a3-megagpu-8g (MPS 4/GPU)
```

---

## 4. Collect Scorecard

```bash
python3 ./03_collect_results.py
```

Reads `BENCH_RESULT_JSON` lines from completed pods in `$K8S_NAMESPACE` (plus any local `bench/results_*.json` snapshots written during cross-cluster runs) and writes your local Markdown scorecard to `local_scorecard.md` (override with `--out <path>`) reporting per-slice & node-aggregate **FP64 / FP32 / BF16 TFLOPS**, **p50/p95 runtime**, **Jobs/hr/node**, **max simultaneous overlap**, and **OOM isolation pass/fail**.

> The published reference scorecard ([results_scorecard.md](results_scorecard.md)) and its anonymized JSONL snapshots ([reference_results/](reference_results/)) are left untouched so your live cluster results stay cleanly separated.

---

## 5. Cleanup

```bash
# Scale GPU pools back to 0 by deleting the benchmark namespace (keeps cluster & pools for re-runs)
./04_cleanup.sh --jobs-only

# Delete benchmark GPU node pools
./04_cleanup.sh --delete-pools

# Delete any ${RESOURCE_PREFIX}-a3mega-res Compute Engine reservations
./04_cleanup.sh --delete-reservation

# Full teardown: delete benchmark cluster(s), reservations, dedicated subnets, and custom VPC
./04_cleanup.sh --delete-cluster
```
