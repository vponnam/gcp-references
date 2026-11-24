#!/usr/bin/env bash
# Attaches a GKE MIG (2g.20gb, 24 slices/node) and optional MPS (4/GPU, 32 pods/node)
# node pool to the reserved a3-megagpu-8g node recorded in bench/active_reservation.env,
# then runs the MatMul + OOM benchmarks and generates the unified scorecard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "[error] PROJECT_ID is not set. Export PROJECT_ID=<your-gcp-project-id> or run 'gcloud config set project <id>'." >&2
  exit 1
fi

STATE_FILE="${SCRIPT_DIR}/active_reservation.env"
if [[ ! -f "${STATE_FILE}" ]]; then
  echo "[error] ${STATE_FILE} not found. Run ./bench/00_reserve_a3mega.sh first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${STATE_FILE}"

RUN_MPS_AFTER_MIG="${RUN_MPS_AFTER_MIG:-true}"
RES_CLUSTER="${RES_CLUSTER:-${CLUSTER_NAME}}"
CLUSTER_LOC_FLAG="--region=${RES_REGION}"
if [[ "${RES_REGION}" != "${REGION}" ]]; then
  RES_CLUSTER="${RESOURCE_PREFIX}-h100-res-${RES_ZONE}"
  CLUSTER_LOC_FLAG="--zone=${RES_ZONE}"
fi

echo "============================================================"
echo " Reserved H100 Mega Benchmark Runner"
echo " Reservation:  ${RES_NAME} (${RES_ZONE})"
echo " Cluster:      ${RES_CLUSTER} (${CLUSTER_LOC_FLAG})"
echo " MIG Profile:  ${H100_MIG_PROFILE} (24 slices/node)"
echo " Run MPS A/B:  ${RUN_MPS_AFTER_MIG}"
echo "============================================================"

# 1. Ensure subnet + GKE cluster exist in reservation region/zone
SUBNET_NAME="${SUBNETWORK}"
if [[ "${RES_REGION}" != "${REGION}" && "${NETWORK}" != "default" ]]; then
  SUBNET_NAME="${RESOURCE_PREFIX}-subnet-${RES_REGION}"
  if ! gcloud compute networks subnets describe "${SUBNET_NAME}" \
    --project="${PROJECT_ID}" --region="${RES_REGION}" >/dev/null 2>&1; then
    echo "[step] Creating subnet '${SUBNET_NAME}' in ${RES_REGION}..."
    gcloud compute networks subnets create "${SUBNET_NAME}" \
      --project="${PROJECT_ID}" \
      --network="${NETWORK}" \
      --region="${RES_REGION}" \
      --range="10.230.0.0/20" \
      --secondary-range="pods=10.234.0.0/16,services=10.235.0.0/20" \
      --enable-private-ip-google-access
  fi
fi

if ! gcloud container clusters describe "${RES_CLUSTER}" \
  --project="${PROJECT_ID}" ${CLUSTER_LOC_FLAG} >/dev/null 2>&1; then
  echo "[step] Creating GKE Standard cluster '${RES_CLUSTER}' in ${RES_ZONE}..."
  RANGE_FLAGS=()
  if [[ "${NETWORK}" != "default" ]]; then
    RANGE_FLAGS=(--enable-ip-alias --cluster-secondary-range-name=pods --services-secondary-range-name=services)
  fi
  gcloud container clusters create "${RES_CLUSTER}" \
    --project="${PROJECT_ID}" \
    ${CLUSTER_LOC_FLAG} \
    --release-channel="${RELEASE_CHANNEL}" \
    --enable-dataplane-v2 \
    --network="${NETWORK}" \
    --subnetwork="${SUBNET_NAME}" \
    "${RANGE_FLAGS[@]}" \
    --machine-type="e2-standard-4" \
    --num-nodes=1 \
    --labels="owner=${RESOURCE_PREFIX},purpose=fractional-gpu-bench" \
    --node-labels="owner=${RESOURCE_PREFIX}"
fi

gcloud container clusters get-credentials "${RES_CLUSTER}" \
  --project="${PROJECT_ID}" ${CLUSTER_LOC_FLAG}

# 2. Create reserved H100 Mega MIG pool (consumes specific reservation)
MIG_POOL="${RESOURCE_PREFIX}-h100-mega-mig-res"
if ! gcloud container node-pools describe "${MIG_POOL}" \
  --project="${PROJECT_ID}" --cluster="${RES_CLUSTER}" ${CLUSTER_LOC_FLAG} >/dev/null 2>&1; then
  echo "[step] Creating reserved MIG node pool '${MIG_POOL}' consuming '${RES_NAME}'..."
  NODE_LOC_FLAGS=()
  if [[ "${CLUSTER_LOC_FLAG}" == --region=* ]]; then
    NODE_LOC_FLAGS=(--node-locations="${RES_ZONE}")
  fi
  gcloud container node-pools create "${MIG_POOL}" \
    --project="${PROJECT_ID}" \
    --cluster="${RES_CLUSTER}" \
    ${CLUSTER_LOC_FLAG} \
    "${NODE_LOC_FLAGS[@]}" \
    --machine-type="a3-megagpu-8g" \
    --accelerator="type=nvidia-h100-mega-80gb,count=8,gpu-partition-size=${H100_MIG_PROFILE},gpu-driver-version=latest" \
    --ephemeral-storage-local-ssd="count=16" \
    --image-type="COS_CONTAINERD" \
    --disk-size="200" \
    --reservation-affinity="specific" \
    --reservation="${RES_NAME}" \
    --num-nodes=1 \
    --labels="owner=${RESOURCE_PREFIX},purpose=fractional-gpu-bench" \
    --node-labels="owner=${RESOURCE_PREFIX},bench-pool=${MIG_POOL}"
fi

# 3. Run 24-pod MIG benchmark and save raw JSON immediately
H100_MIG_POOL_NAME="${MIG_POOL}" "${SCRIPT_DIR}/02_run_benchmarks.sh" h100-mig

kubectl -n "${K8S_NAMESPACE}" get pods -l "bench-tier=h100-mega-mig" -o name | while read -r p; do
  kubectl -n "${K8S_NAMESPACE}" logs "${p}" | grep "^BENCH_RESULT_JSON:" | sed 's/^BENCH_RESULT_JSON://'
done > "${SCRIPT_DIR}/results_h100_mega_mig.json"

# 3b. Run 24-pod OOM isolation check on the same warm MIG node and save raw JSON
H100_MIG_POOL_NAME="${MIG_POOL}" "${SCRIPT_DIR}/02_run_benchmarks.sh" oom

kubectl -n "${K8S_NAMESPACE}" get pods -l "bench-tier=h100-mega-mig-oom" -o name | while read -r p; do
  kubectl -n "${K8S_NAMESPACE}" logs "${p}" | grep "^BENCH_RESULT_JSON:" | sed 's/^BENCH_RESULT_JSON://'
done > "${SCRIPT_DIR}/results_h100_mega_mig_oom.json"

# 4. Optionally swap node pool on the same reservation to run 32-pod MPS A/B benchmark
if [[ "${RUN_MPS_AFTER_MIG}" == "true" ]]; then
  echo "[step] Deleting MIG pool to free the reserved node for MPS A/B test..."
  gcloud container node-pools delete "${MIG_POOL}" \
    --project="${PROJECT_ID}" --cluster="${RES_CLUSTER}" ${CLUSTER_LOC_FLAG} --quiet

  MPS_POOL="${RESOURCE_PREFIX}-h100-mega-mps-res"
  NODE_LOC_FLAGS=()
  if [[ "${CLUSTER_LOC_FLAG}" == --region=* ]]; then
    NODE_LOC_FLAGS=(--node-locations="${RES_ZONE}")
  fi
  echo "[step] Creating reserved MPS node pool '${MPS_POOL}' consuming '${RES_NAME}'..."
  gcloud container node-pools create "${MPS_POOL}" \
    --project="${PROJECT_ID}" \
    --cluster="${RES_CLUSTER}" \
    ${CLUSTER_LOC_FLAG} \
    "${NODE_LOC_FLAGS[@]}" \
    --machine-type="a3-megagpu-8g" \
    --accelerator="type=nvidia-h100-mega-80gb,count=8,gpu-sharing-strategy=mps,max-shared-clients-per-gpu=${H100_MPS_CLIENTS_PER_GPU},gpu-driver-version=latest" \
    --ephemeral-storage-local-ssd="count=16" \
    --image-type="COS_CONTAINERD" \
    --disk-size="200" \
    --reservation-affinity="specific" \
    --reservation="${RES_NAME}" \
    --num-nodes=1 \
    --labels="owner=${RESOURCE_PREFIX},purpose=fractional-gpu-bench" \
    --node-labels="owner=${RESOURCE_PREFIX},bench-pool=${MPS_POOL}"

  H100_MPS_POOL_NAME="${MPS_POOL}" "${SCRIPT_DIR}/02_run_benchmarks.sh" h100-mps

  kubectl -n "${K8S_NAMESPACE}" get pods -l "bench-tier=h100-mega-mps" -o name | while read -r p; do
    kubectl -n "${K8S_NAMESPACE}" logs "${p}" | grep "^BENCH_RESULT_JSON:" | sed 's/^BENCH_RESULT_JSON://'
  done > "${SCRIPT_DIR}/results_h100_mega_mps.json"

  echo "[step] Deleting MPS pool..."
  gcloud container node-pools delete "${MPS_POOL}" \
    --project="${PROJECT_ID}" --cluster="${RES_CLUSTER}" ${CLUSTER_LOC_FLAG} --quiet
fi

echo ""
echo "[step] Generating unified scorecard..."
python3 "${SCRIPT_DIR}/03_collect_results.py"
echo ""
echo "[IMPORTANT] Delete the reservation when finished to stop billing:"
echo "  ./bench/04_cleanup.sh --delete-reservation"
