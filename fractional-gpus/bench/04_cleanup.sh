#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "[error] PROJECT_ID is not set. Export PROJECT_ID=<your-gcp-project-id> or run 'gcloud config set project <id>'." >&2
  exit 1
fi

MODE="${1:---jobs-only}"

case "${MODE}" in
  --jobs-only)
    echo "[step] Deleting namespace '${K8S_NAMESPACE}' so GPU node pools autoscale back to 0..."
    kubectl delete namespace "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    echo "[ok] Jobs cleaned up. Verify nodes scale to 0 with: kubectl get nodes -L bench-pool"
    ;;
  --delete-pools)
    echo "[step] Deleting namespace '${K8S_NAMESPACE}' and benchmark GPU node pools..."
    kubectl delete namespace "${K8S_NAMESPACE}" --ignore-not-found --wait=true || true
    for pool in \
      "${G4_POOL_NAME}" \
      "${G4_8GPU_MIG_POOL_NAME}" \
      "${G4_8GPU_MPS_POOL_NAME}" \
      "${H100_MIG_POOL_NAME}" \
      "${H100_MPS_POOL_NAME}"; do
      if gcloud container node-pools describe "${pool}" \
        --project="${PROJECT_ID}" --cluster="${CLUSTER_NAME}" --region="${REGION}" >/dev/null 2>&1; then
        echo "[step] Deleting node pool '${pool}'..."
        gcloud container node-pools delete "${pool}" \
          --project="${PROJECT_ID}" --cluster="${CLUSTER_NAME}" --region="${REGION}" --quiet
      fi
    done
    ;;
  --delete-reservation)
    echo "[step] Deleting any '${RESOURCE_PREFIX}-a3mega-res' Compute Engine reservations..."
    gcloud beta compute reservations list \
      --project="${PROJECT_ID}" \
      --filter="name~^${RESOURCE_PREFIX}-a3mega-res" \
      --format="value(name,zone.basename())" 2>/dev/null | while read -r rname rzone; do
        [[ -z "${rname}" ]] && continue
        echo "[step] Deleting reservation '${rname}' in ${rzone}..."
        gcloud beta compute reservations delete "${rname}" \
          --project="${PROJECT_ID}" --zone="${rzone}" --quiet || true
      done
    rm -f "${SCRIPT_DIR}/active_reservation.env"
    ;;
  --delete-cluster)
    echo "[step] Deleting dedicated benchmark cluster '${CLUSTER_NAME}' in ${REGION}..."
    gcloud container clusters delete "${CLUSTER_NAME}" \
      --project="${PROJECT_ID}" --region="${REGION}" --quiet || true

    echo "[step] Deleting any cross-region reserved H100 clusters (${RESOURCE_PREFIX}-h100-res-*)..."
    gcloud container clusters list \
      --project="${PROJECT_ID}" \
      --filter="name~^${RESOURCE_PREFIX}-h100-res-" \
      --format="value(name,location)" 2>/dev/null | while read -r cname cloc; do
        [[ -z "${cname}" ]] && continue
        echo "[step] Deleting cluster '${cname}' in ${cloc}..."
        gcloud container clusters delete "${cname}" \
          --project="${PROJECT_ID}" --location="${cloc}" --quiet || true
      done

    echo "[step] Deleting any '${RESOURCE_PREFIX}-a3mega-res' Compute Engine reservations..."
    gcloud beta compute reservations list \
      --project="${PROJECT_ID}" \
      --filter="name~^${RESOURCE_PREFIX}-a3mega-res" \
      --format="value(name,zone.basename())" 2>/dev/null | while read -r rname rzone; do
        [[ -z "${rname}" ]] && continue
        echo "[step] Deleting reservation '${rname}' in ${rzone}..."
        gcloud beta compute reservations delete "${rname}" \
          --project="${PROJECT_ID}" --zone="${rzone}" --quiet || true
      done
    rm -f "${SCRIPT_DIR}/active_reservation.env"

    if [[ "${NETWORK}" != "default" ]]; then
      echo "[step] Deleting dedicated subnets and VPC '${NETWORK}'..."
      gcloud compute networks subnets list \
        --project="${PROJECT_ID}" \
        --filter="network.basename()=${NETWORK}" \
        --format="value(name,region.basename())" 2>/dev/null | while read -r sname sregion; do
          [[ -z "${sname}" ]] && continue
          echo "[step] Deleting subnet '${sname}' in ${sregion}..."
          gcloud compute networks subnets delete "${sname}" \
            --project="${PROJECT_ID}" --region="${sregion}" --quiet || true
        done
      gcloud compute networks delete "${NETWORK}" \
        --project="${PROJECT_ID}" --quiet || true
    fi
    ;;
  *)
    echo "Usage: $0 [--jobs-only|--delete-pools|--delete-reservation|--delete-cluster]"
    exit 1
    ;;
esac
