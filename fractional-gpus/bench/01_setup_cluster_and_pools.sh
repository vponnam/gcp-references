#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "[error] PROJECT_ID is not set. Export PROJECT_ID=<your-gcp-project-id> or run 'gcloud config set project <id>'." >&2
  exit 1
fi

echo "============================================================"
echo " Fractional GPU Benchmark Setup"
echo " Project:            ${PROJECT_ID}"
echo " Region / Zones:     ${REGION} (G4/H100: ${GPU_ZONES} | G4-8GPU: ${G4_8GPU_ZONES})"
echo " Resource Prefix:    ${RESOURCE_PREFIX}"
echo " Cluster:            ${CLUSTER_NAME} (create=${CREATE_CLUSTER})"
echo " Create G4-12 Pool:  ${CREATE_G4_12_POOL} (${G4_PROVISIONING})"
echo " Create G4-8GPU:     ${CREATE_G4_8GPU_POOLS} (${G4_8GPU_PROVISIONING}, MIG=${G4_8GPU_MIG_PROFILE})"
echo " Create H100 Mega:   ${CREATE_H100_POOLS} (${H100_PROVISIONING}, MIG=${H100_MIG_PROFILE})"
echo " Create MPS Pools:   ${CREATE_MPS_POOL}"
echo "============================================================"

# 0. Create dedicated custom VPC and subnet if missing (avoids shared default VPC IP exhaustion)
if [[ "${NETWORK}" != "default" ]]; then
  if ! gcloud compute networks describe "${NETWORK}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "[step] Creating custom VPC network '${NETWORK}'..."
    gcloud compute networks create "${NETWORK}" \
      --project="${PROJECT_ID}" \
      --subnet-mode=custom
  else
    echo "[ok] VPC network '${NETWORK}' already exists."
  fi

  if ! gcloud compute networks subnets describe "${SUBNETWORK}" \
    --project="${PROJECT_ID}" --region="${REGION}" >/dev/null 2>&1; then
    echo "[step] Creating subnet '${SUBNETWORK}' in ${REGION}..."
    gcloud compute networks subnets create "${SUBNETWORK}" \
      --project="${PROJECT_ID}" \
      --network="${NETWORK}" \
      --region="${REGION}" \
      --range="10.200.0.0/20" \
      --secondary-range="pods=10.204.0.0/16,services=10.205.0.0/20" \
      --enable-private-ip-google-access
  else
    echo "[ok] Subnet '${SUBNETWORK}' already exists."
  fi
fi

# 1. Create GKE Standard cluster if requested and not already present
EXISTING_STATUS="$(gcloud container clusters describe "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" --region="${REGION}" --format='value(status)' 2>/dev/null || true)"

if [[ "${EXISTING_STATUS}" == "ERROR" ]]; then
  echo "[warn] Found failed tombstone cluster '${CLUSTER_NAME}' (status=ERROR). Deleting before recreate..."
  gcloud container clusters delete "${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" --region="${REGION}" --quiet
  EXISTING_STATUS=""
fi

if [[ -n "${EXISTING_STATUS}" ]]; then
  echo "[ok] Cluster '${CLUSTER_NAME}' already exists in ${REGION} (status=${EXISTING_STATUS})."
else
  if [[ "${CREATE_CLUSTER}" != "true" ]]; then
    echo "[error] Cluster '${CLUSTER_NAME}' not found in ${REGION} and CREATE_CLUSTER=false." >&2
    exit 1
  fi
  echo "[step] Creating GKE Standard cluster '${CLUSTER_NAME}' in ${REGION}..."
  RANGE_FLAGS=()
  if [[ "${NETWORK}" != "default" ]]; then
    RANGE_FLAGS=(--enable-ip-alias --cluster-secondary-range-name=pods --services-secondary-range-name=services)
  fi
  gcloud container clusters create "${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --node-locations="${GPU_ZONES}" \
    --release-channel="${RELEASE_CHANNEL}" \
    --enable-dataplane-v2 \
    --network="${NETWORK}" \
    --subnetwork="${SUBNETWORK}" \
    "${RANGE_FLAGS[@]}" \
    --machine-type="e2-standard-4" \
    --num-nodes=1 \
    --enable-autoscaling --total-min-nodes=1 --total-max-nodes=2 \
    --labels="owner=${RESOURCE_PREFIX},purpose=fractional-gpu-bench" \
    --node-labels="owner=${RESOURCE_PREFIX}"
fi

MASTER_VER="$(gcloud container clusters describe "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" --region="${REGION}" \
  --format='value(currentMasterVersion)')"
echo "[info] Cluster master version: ${MASTER_VER} (fractional G4 requires >= 1.35.2)"

echo "[step] Fetching kubectl credentials..."
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" --region="${REGION}"

build_prov_flags() {
  local mode="$1"
  if [[ "${mode}" == "flex" ]]; then
    echo "--flex-start --reservation-affinity=none --max-run-duration=${FLEX_MAX_RUN_DURATION}"
  elif [[ "${mode}" == "spot" ]]; then
    echo "--spot"
  elif [[ "${mode}" == "on-demand" ]]; then
    echo ""
  else
    echo "[error] Unsupported provisioning mode '${mode}'. Use 'flex', 'spot', or 'on-demand'." >&2
    exit 1
  fi
}

create_pool_if_missing() {
  local pool_name="$1"
  local zones="$2"
  shift 2
  if gcloud container node-pools describe "${pool_name}" \
    --project="${PROJECT_ID}" --cluster="${CLUSTER_NAME}" --region="${REGION}" >/dev/null 2>&1; then
    echo "[ok] Node pool '${pool_name}' already exists."
    return 0
  fi
  echo "[step] Creating node pool '${pool_name}' in zones ${zones}..."
  gcloud container node-pools create "${pool_name}" \
    --project="${PROJECT_ID}" \
    --cluster="${CLUSTER_NAME}" \
    --region="${REGION}" \
    --node-locations="${zones}" \
    --image-type="COS_CONTAINERD" \
    --disk-size="200" \
    --num-nodes=0 \
    --enable-autoscaling \
    --location-policy="ANY" \
    --total-min-nodes=0 \
    --total-max-nodes=1 \
    --labels="owner=${RESOURCE_PREFIX},purpose=fractional-gpu-bench" \
    --node-labels="owner=${RESOURCE_PREFIX},bench-pool=${pool_name}" \
    "$@"
}

# 2. Baseline fractional G4 pool: g4-standard-12 (1/4 RTX PRO 6000, 24 GB VRAM)
if [[ "${CREATE_G4_12_POOL}" == "true" ]]; then
  read -r -a G4_PROV_FLAGS <<< "$(build_prov_flags "${G4_PROVISIONING}")"
  create_pool_if_missing "${G4_POOL_NAME}" "${GPU_ZONES}" \
    --machine-type="g4-standard-12" \
    --accelerator="type=nvidia-rtx-pro-6000,count=1,gpu-driver-version=default" \
    ${G4_PROV_FLAGS[@]+"${G4_PROV_FLAGS[@]}"}
fi

# 3. 8-GPU G4 pools: g4-standard-384 MIG (1g.24gb = 32 slices/node) and MPS (4/GPU = 32 pods/node)
if [[ "${CREATE_G4_8GPU_POOLS}" == "true" ]]; then
  read -r -a G4_8GPU_PROV_FLAGS <<< "$(build_prov_flags "${G4_8GPU_PROVISIONING}")"
  create_pool_if_missing "${G4_8GPU_MIG_POOL_NAME}" "${G4_8GPU_ZONES}" \
    --machine-type="g4-standard-384" \
    --accelerator="type=nvidia-rtx-pro-6000,count=8,gpu-partition-size=${G4_8GPU_MIG_PROFILE},gpu-driver-version=latest" \
    ${G4_8GPU_PROV_FLAGS[@]+"${G4_8GPU_PROV_FLAGS[@]}"}

  if [[ "${CREATE_MPS_POOL}" == "true" ]]; then
    create_pool_if_missing "${G4_8GPU_MPS_POOL_NAME}" "${G4_8GPU_ZONES}" \
      --machine-type="g4-standard-384" \
      --accelerator="type=nvidia-rtx-pro-6000,count=8,gpu-sharing-strategy=mps,max-shared-clients-per-gpu=${G4_8GPU_MPS_CLIENTS_PER_GPU},gpu-driver-version=latest" \
      ${G4_8GPU_PROV_FLAGS[@]+"${G4_8GPU_PROV_FLAGS[@]}"}
  fi
fi

# 4. Optional 8-GPU H100 Mega pools: a3-megagpu-8g MIG (2g.20gb = 24 slices/node) and MPS (4/GPU = 32 pods/node)
if [[ "${CREATE_H100_POOLS}" == "true" ]]; then
  read -r -a H100_PROV_FLAGS <<< "$(build_prov_flags "${H100_PROVISIONING}")"
  create_pool_if_missing "${H100_MIG_POOL_NAME}" "${GPU_ZONES}" \
    --machine-type="a3-megagpu-8g" \
    --accelerator="type=nvidia-h100-mega-80gb,count=8,gpu-partition-size=${H100_MIG_PROFILE},gpu-driver-version=latest" \
    ${H100_PROV_FLAGS[@]+"${H100_PROV_FLAGS[@]}"}

  if [[ "${CREATE_MPS_POOL}" == "true" ]]; then
    create_pool_if_missing "${H100_MPS_POOL_NAME}" "${GPU_ZONES}" \
      --machine-type="a3-megagpu-8g" \
      --accelerator="type=nvidia-h100-mega-80gb,count=8,gpu-sharing-strategy=mps,max-shared-clients-per-gpu=${H100_MPS_CLIENTS_PER_GPU},gpu-driver-version=latest" \
      ${H100_PROV_FLAGS[@]+"${H100_PROV_FLAGS[@]}"}
  fi
fi

echo ""
echo "[done] Node pools ready (all GPU pools start at 0 nodes):"
gcloud container node-pools list \
  --project="${PROJECT_ID}" --cluster="${CLUSTER_NAME}" --region="${REGION}" \
  --format="table(name,config.machineType,config.flexStart,config.spot,config.accelerators[0].acceleratorType,config.accelerators[0].gpuPartitionSize,config.accelerators[0].gpuSharingConfig.gpuSharingStrategy,status)"
