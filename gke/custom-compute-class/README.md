# Testing GKE Custom Compute Class (CCC) with HPA

This folder contains a minimal, self-contained setup to test [**GKE Custom Compute Classes (CCC)**](https://cloud.google.com/kubernetes-engine/docs/concepts/about-custom-compute-classes) together with the [**Horizontal Pod Autoscaler (HPA)**](https://cloud.google.com/kubernetes-engine/docs/concepts/horizontalpodautoscaler) using custom inference queue metrics.

---

## Problem Statement

Running GPU inference workloads on Kubernetes gets complicated as soon as traffic becomes unpredictable.

Most teams reserve a baseline of GPUs for steady everyday traffic. [Reservations](https://cloud.google.com/kubernetes-engine/docs/how-to/consume-reservations) guarantee capacity and keep costs predictable, but you don't want to pay for reserved GPUs that sit idle just to cover occasional traffic spikes. When a spike happens, you need extra GPUs right away. The problem is that standard on-demand GPUs are often out of stock in a single zone when you ask for them on short notice. To actually get capacity during a spike, you usually need a fallback chain:

1. **[Reserved GPUs](https://cloud.google.com/kubernetes-engine/docs/how-to/consume-reservations) first** — use what you already paid for.
2. **[Dynamic Workload Scheduler (DWS) Flex-Start](https://cloud.google.com/kubernetes-engine/docs/concepts/dynamic-workload-scheduler)** — if the reservation is full, queue briefly for capacity across multiple zones.
3. **[Spot GPUs](https://cloud.google.com/kubernetes-engine/docs/concepts/spot-vms)** — if you still need capacity immediately, grab spare Spot capacity across zones.

Without [Custom Compute Classes](https://cloud.google.com/kubernetes-engine/docs/concepts/about-custom-compute-classes), building this fallback logic in GKE is painful. You end up managing multiple node pools, maintaining different deployments or complex scheduling rules, and manually moving pods back to reserved nodes when capacity frees up.

Scaling the pods themselves has a similar catch. Standard CPU and memory metrics don't work well for [LLM inference](https://cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference)—a GPU can be completely saturated processing tokens while host CPU usage barely moves. Instead, you have to scale based on the inference request queue (for example, `vllm:num_requests_waiting`) using [custom metrics](https://cloud.google.com/kubernetes-engine/docs/how-to/custom-and-external-metrics).

When you want to test how HPA and CCC actually behave together before going to production, setting up a real test is surprisingly heavy. Deploying vLLM, pulling hundreds of gigabytes of model weights, and wiring up [Google Cloud Managed Service for Prometheus](https://cloud.google.com/stackdriver/docs/managed-prometheus) with the Custom Metrics Adapter takes a lot of time just to verify whether your node fallback and autoscaling rules work as expected.

The configs in this directory solve that testing gap. They set up a tiered GPU `ComputeClass` alongside a lightweight mock metrics adapter and dummy GPU workload, so you can change the simulated queue depth with a single `curl` command and watch HPA and GKE Node Auto-Provisioning react in real time.

---

## Files in This Folder

| File | Description |
| :--- | :--- |
| `ccc-g4-res-dws.yaml` | Defines the `g4-inference-res-dws` GKE `ComputeClass` with 3 priority tiers (Reserved -> DWS Flex-Start -> Spot) and node consolidation rules. |
| `inference-queue-test.yaml` | Deploys an in-cluster custom metrics adapter, a lightweight GPU test deployment targeting the ComputeClass, and an HPA that scales on queue depth. |

---

## Configuration Walkthrough

### 1. Custom Compute Class (`ccc-g4-res-dws.yaml`)

This manifest defines a [`ComputeClass` CustomResourceDefinition (CRD)](https://cloud.google.com/kubernetes-engine/docs/reference/crds/computeclass) named `g4-inference-res-dws` that tells GKE how to provision and manage GPU nodes when pods request this class.

* **`nodePoolAutoCreation.enabled: true`**: Allows GKE [Node Auto-Provisioning (NAP)](https://cloud.google.com/kubernetes-engine/docs/how-to/node-auto-provisioning) to automatically create and delete node pools matching the priority rules below.
* **`activeMigration.optimizeRulePriority: true`**: Enables [active migration](https://cloud.google.com/kubernetes-engine/docs/concepts/about-custom-compute-classes#active-migration). If pods are running on lower-priority fallback nodes (like Spot or DWS) and capacity later opens up in a higher-priority tier (like your GPU reservation), GKE automatically creates the higher-priority node and migrates pods back to it.
* **`autoscalingPolicy`**: Controls how aggressively GKE [consolidates underutilized nodes](https://cloud.google.com/kubernetes-engine/docs/how-to/custom-compute-classes#configure-node-consolidation):
  * `consolidationDelayMinutes: 5`: Waits 5 minutes before removing an unneeded node.
  * `consolidationThreshold: 70`: Consolidates standard resources if utilization drops below 70%.
  * `gpuConsolidationThreshold: 100`: Allows GKE to consolidate GPU nodes when they are underutilized or empty after HPA scales down.
* **`nodePoolConfig`**: Sets common node settings across all tiers (`cos_containerd` OS image, [gVNIC](https://cloud.google.com/kubernetes-engine/docs/how-to/using-gvnic) enabled, and your [custom IAM node service account](https://cloud.google.com/kubernetes-engine/docs/how-to/hardening-your-cluster#use_least_privilege_sa)).
* **`priorities`**: The ordered fallback chain GKE evaluates when scaling up:
  * **Priority 1 — Automatic GPU Reservation**: Requests a [`g4-standard-384`](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series) machine (8x `nvidia-rtx-pro-6000` GPUs, 32 Local SSDs) in `us-south1-a` with `reservations.affinity: AnyBestEffort` to [consume an automatic reservation](https://cloud.google.com/kubernetes-engine/docs/how-to/consume-reservations) first.
  * **Priority 2 — DWS Flex-Start Fallback**: If the reservation is exhausted or unavailable, falls back to smaller `g4-standard-96` nodes (2x GPUs, 8 Local SSDs) across `us-south1-a` and `us-south1-b` (`locationPolicy: ANY`) using [DWS Flex-Start](https://cloud.google.com/kubernetes-engine/docs/concepts/dynamic-workload-scheduler) (`flexStart.enabled: true`). Notice the smaller 2-GPU machine size here: since each pod requests 2 GPUs, burst pods get right-sized 2-GPU nodes instead of waiting for an 8-GPU machine.
  * **Priority 3 — Spot GPU Fallback**: If DWS Flex-Start cannot fulfill the request within `capacityCheckWaitTimeSeconds` (180s), falls back to [Spot VMs](https://cloud.google.com/kubernetes-engine/docs/concepts/spot-vms) (`g4-standard-96` with 2x GPUs) across `us-south1-a` and `us-south1-b`.
* **`whenUnsatisfiable: DoNotScaleUp`**: Prevents GKE from falling back to default unconfigured node pools if none of the three priority tiers have capacity.

### 2. HPA & Simulated Inference Queue (`inference-queue-test.yaml`)

Instead of deploying a full model server and external Prometheus adapter just to test scaling, this file bundles everything needed to test HPA + CCC in one manifest:

* **RBAC (`ServiceAccount`, `ClusterRole`, `ClusterRoleBinding`)**: Grants the mock adapter permission to list pods and lets the `horizontal-pod-autoscaler` service account query `custom.metrics.k8s.io`.
* **Mock Custom Metrics Adapter (`vllm-custom-metrics-adapter` + `APIService`)**:
  * Runs a tiny Python script (from ConfigMap `vllm-queue-adapter-code`) that registers directly with the Kubernetes API server as the `v1beta1.custom.metrics.k8s.io` provider.
  * Serves the custom metric `vllm_num_requests_waiting` (initialized to `2`).
  * Exposes a simple HTTP endpoint on port `8000` (`/set?depth=<N>`) so you can dial the simulated queue depth up or down at any time.
  * Also exposes a Prometheus `/metrics` endpoint paired with a [`PodMonitoring`](https://cloud.google.com/stackdriver/docs/managed-prometheus/setup-managed#configure-pod-monitoring) resource (`vllm-queue-pm`) if you want to view the metric in Cloud Monitoring.
* **Simulated GPU Workload (`vllm-inference-test`)**:
  * Starts with `1` replica and [selects the ComputeClass in the Pod spec](https://cloud.google.com/kubernetes-engine/docs/how-to/custom-compute-classes#select-compute-class) using:
    ```yaml
    nodeSelector:
      cloud.google.com/compute-class: g4-inference-res-dws
    ```
  * Requests `8` vCPUs, `32Gi` memory, and `2` GPUs (`nvidia.com/gpu: "2"`) per pod.
  * Runs a lightweight `python:3.11-slim` sleep loop so pods start immediately once the GPU node is ready.
* **HorizontalPodAutoscaler (`vllm-queue-hpa`)**:
  * Scales `vllm-inference-test` between `minReplicas: 1` and `maxReplicas: 3`.
  * Targets an `AverageValue` of `5` on `vllm_num_requests_waiting`.
  * Configured for fast testing: scales up immediately (`stabilizationWindowSeconds: 0`, up to 2 pods every 30s) and scales down after a 60-second stabilization window.

---

## Prerequisites

1. **A GKE Cluster** (Standard with [Node Auto-Provisioning](https://cloud.google.com/kubernetes-engine/docs/how-to/node-auto-provisioning) enabled, or Autopilot) running a GKE version that supports [Custom Compute Classes](https://cloud.google.com/kubernetes-engine/docs/concepts/about-custom-compute-classes) (`1.30+`, with `1.31+` / `1.32+` recommended for DWS Flex-Start and advanced consolidation fields).
2. **A Node IAM Service Account** with least-privilege GKE node permissions (`roles/container.defaultNodeServiceAccount`) — see [Use least privilege IAM service accounts](https://cloud.google.com/kubernetes-engine/docs/how-to/hardening-your-cluster#use_least_privilege_sa).
3. **[GPU Quota](https://cloud.google.com/compute/resource-usage#gpu_quota)** in your target region/zones for the machine and GPU types configured in `ccc-g4-res-dws.yaml`.
4. **`kubectl`** configured to talk to your cluster.

---

## Getting Started

### Step 1: Update Placeholders in `ccc-g4-res-dws.yaml`

Open `ccc-g4-res-dws.yaml` and replace the placeholder service account with your GKE node service account and project ID:

```yaml
  nodePoolConfig:
    serviceAccount: <NODE_SERVICE_ACCOUNT>@<PROJECT_ID>.iam.gserviceaccount.com
```

> **Tip:** If you are testing in a different region or with different GPUs (for example, `g2-standard-24` with `nvidia-l4`), update `location.zones`, `machineType`, and `gpu.type` under the `priorities` section to match your environment. See [Supported GPUs on GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/gpus#gpus-list) for valid machine and accelerator combinations.

### Step 2: Apply the Custom Compute Class

Apply the `ComputeClass` resource to your cluster:

```bash
kubectl apply -f ccc-g4-res-dws.yaml
```

Verify it was created:

```bash
kubectl get computeclass g4-inference-res-dws
```

### Step 3: Deploy the Test Workload and HPA

Apply the mock metrics adapter, GPU test deployment, and HPA:

```bash
kubectl apply -f inference-queue-test.yaml
```

### Step 4: Verify the Initial State

1. Check that the custom metrics adapter pod is running (this runs on your default CPU node pool):
   ```bash
   kubectl get pods -l app=vllm-custom-metrics-adapter
   ```

2. Check that the `custom.metrics.k8s.io` API service is available (`AVAILABLE` should be `True`):
   ```bash
   kubectl get apiservice v1beta1.custom.metrics.k8s.io
   ```

3. Watch the GPU test pod and HPA:
   ```bash
   kubectl get pods -l app=vllm-inference-test -w
   ```
   If no matching `g4-inference-res-dws` node exists yet, the initial pod will stay `Pending` for a few minutes while GKE provisions a GPU node using your Priority 1 rule (or falls back to Priority 2 / Priority 3 if Priority 1 isn't available).

4. Once the pod is running, check the HPA status:
   ```bash
   kubectl get hpa vllm-queue-hpa
   ```
   You should see `TARGETS` showing `2/5` (since the initial mock queue depth is `2`, below the threshold of `5`) and `REPLICAS` at `1`.

### Step 5: Simulate a Traffic Spike (Scale Up)

Now let's spike the simulated queue depth from `2` to `15` to trigger HPA scale-up and watch CCC provision additional capacity.

1. In one terminal, watch the HPA, pods, and nodes:
   ```bash
   kubectl get hpa,pods,nodes -w
   ```

2. In another terminal, port-forward to the mock metrics service and increase the queue depth to `15`:
   ```bash
   kubectl port-forward svc/vllm-custom-metrics-svc 8000:8000 &
   curl "http://localhost:8000/set?depth=15"
   ```

3. **What to look for:**
   * Within 15–30 seconds, `kubectl get hpa vllm-queue-hpa` will report `15/5` (or `5/5` average as replicas increase) and scale `vllm-inference-test` from `1` to `3` replicas.
   * If the new pods fit on your existing GPU node (for example, an 8-GPU Priority 1 node can fit up to four 2-GPU pods), they will schedule immediately.
   * If your existing node is full (or if you started on a 2-GPU node), the new pods will go `Pending`, triggering GKE to evaluate the `g4-inference-res-dws` priority list and spin up fallback nodes (Priority 2 DWS Flex-Start or Priority 3 Spot).
   * You can inspect which node pool and priority tier GKE used by checking node labels:
     ```bash
     kubectl get nodes -L cloud.google.com/compute-class,cloud.google.com/gke-provisioning,cloud.google.com/gke-nodepool
     ```

### Step 6: Simulate Traffic Dropping (Scale Down & Consolidation)

Now drop the queue depth back down to `1` to verify HPA scale-down and CCC node consolidation.

1. Set the queue depth to `1`:
   ```bash
   curl "http://localhost:8000/set?depth=1"
   ```

2. **What to look for:**
   * After the 60-second `scaleDown.stabilizationWindowSeconds`, HPA will scale `vllm-inference-test` back down to `1` replica.
   * After `consolidationDelayMinutes: 5` (configured in `ccc-g4-res-dws.yaml`), GKE will drain and remove the unneeded GPU nodes that were spun up for the burst.

Stop the background port-forward when you're done:

```bash
kill %1
```

---

## Cleanup

To remove all test resources (and let GKE scale the auto-provisioned GPU node pools down to zero):

```bash
kubectl delete -f inference-queue-test.yaml
kubectl delete -f ccc-g4-res-dws.yaml
```

---

## Official Google Cloud Documentation & References

If you want to dive deeper into any of the features used in this guide or adapt this pattern for production workloads, check out the official Google Cloud docs:

### Custom Compute Classes & Node Provisioning
* [About Custom Compute Classes in GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/about-custom-compute-classes) — Core concepts, fallback priority rules, active migration, and node consolidation.
* [Configure Custom Compute Classes](https://cloud.google.com/kubernetes-engine/docs/how-to/custom-compute-classes) — Step-by-step guide to creating and applying `ComputeClass` resources in Standard and Autopilot clusters.
* [`ComputeClass` CRD Reference](https://cloud.google.com/kubernetes-engine/docs/reference/crds/computeclass) — Full schema and field reference for `cloud.google.com/v1` `ComputeClass`.
* [About Node Auto-Provisioning (NAP)](https://cloud.google.com/kubernetes-engine/docs/how-to/node-auto-provisioning) — How GKE automatically creates and deletes node pools for Custom Compute Classes.

### GPU Capacity & Provisioning Options
* [Consume Compute Engine reservations in GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/consume-reservations) — How to consume automatic (`AnyBestEffort`) or specific (`Specific`) GPU reservations.
* [About Dynamic Workload Scheduler (DWS) in GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/dynamic-workload-scheduler) — How DWS Flex-Start queues requests and provisions GPU capacity across zones.
* [Spot VMs in GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/spot-vms) — Using discounted, preemptible Spot capacity as a fallback tier.
* [About GPUs in GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/gpus) & [Accelerator-optimized machine family (G4, G2, A3)](https://cloud.google.com/compute/docs/accelerator-optimized-machines) — Machine types, GPU counts, and driver versions.

### Autoscaling & LLM Inference on GKE
* [Best practices for running and scaling LLM inference workloads on GKE](https://cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference) — Recommended metrics (such as server queue depth and KV-cache utilization) and HPA tuning for inference servers like vLLM.
* [Horizontal Pod Autoscaling in GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/horizontalpodautoscaler) — Overview of HPA behavior, stabilization windows, and scaling policies.
* [Optimize Pod autoscaling based on custom and external metrics](https://cloud.google.com/kubernetes-engine/docs/how-to/custom-and-external-metrics) — How to wire up Google Cloud Managed Service for Prometheus and the Custom Metrics Stackdriver Adapter for production workloads.

