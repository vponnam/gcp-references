#!/usr/bin/env bash
# Retrying Reservation Automator for 1x a3-megagpu-8g (8x NVIDIA H100 Mega + 16x 375GB NVMe LSSD)
# Modeled after: https://github.com/vponnam/gcp-ai-hpc-reservations-automator/tree/main/accelerator-reservations
#
# Scans candidate zones sequentially and locks in a specific-targeted reservation
# (--require-specific-reservation) prefixed with $RESOURCE_PREFIX so other workloads
# in a shared project cannot accidentally consume it.
#
# IMPORTANT: On-demand reservations incur charges as soon as created; delete immediately after
# testing via `./bench/04_cleanup.sh --delete-reservation` (or `--delete-cluster`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "[error] PROJECT_ID is not set. Export PROJECT_ID=<your-gcp-project-id> or run 'gcloud config set project <id>'." >&2
  exit 1
fi

RES_NAME="${RES_NAME:-${RESOURCE_PREFIX}-a3mega-res}"
if [[ "${RES_NAME}" != "${RESOURCE_PREFIX}-"* ]]; then
  RES_NAME="${RESOURCE_PREFIX}-${RES_NAME}"
fi
LABELS="${LABELS:-prefix=${RESOURCE_PREFIX},owner=${RESOURCE_PREFIX},purpose=fractional-gpu-bench}"
VM_TYPE="${VM_TYPE:-a3-megagpu-8g}"
ACCELERATOR="${ACCELERATOR:-type=nvidia-h100-mega-80gb,count=8}"
LOCAL_SSD="${LOCAL_SSD:-interface=nvme,size=375,count=16}"
VM_COUNT="${VM_COUNT:-1}"
RETRY_RATE="${RETRY_RATE:-60}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"

# Override CANDIDATE_ZONES with zones where your project has NVIDIA_H100_MEGA on-demand quota.
CANDIDATE_ZONES="${CANDIDATE_ZONES:-us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b us-east4-a us-east4-b us-east4-c}"

STATE_FILE="${SCRIPT_DIR}/active_reservation.env"

echo "============================================================"
echo " H100 Mega Reservation Automator"
echo " Reservation:     ${RES_NAME} (specific-targeted)"
echo " Project:         ${PROJECT_ID}"
echo " Machine Type:    ${VM_TYPE} (count=${VM_COUNT})"
echo " Accelerator:     ${ACCELERATOR}"
echo " Local SSD:       ${LOCAL_SSD}"
echo " Labels (desc):   ${LABELS}"
echo " Candidate Zones: ${CANDIDATE_ZONES}"
echo "============================================================"

# Check if reservation already exists in any candidate zone
EXISTING_ZONE="$(gcloud beta compute reservations list \
  --project="${PROJECT_ID}" \
  --filter="name=${RES_NAME} AND status=READY" \
  --format="value(zone.basename())" 2>/dev/null | head -n 1 || true)"

if [[ -n "${EXISTING_ZONE}" ]]; then
  EXISTING_COUNT="$(gcloud beta compute reservations describe "${RES_NAME}" \
    --project="${PROJECT_ID}" --zone="${EXISTING_ZONE}" \
    --format="value(specificReservation.count)" 2>/dev/null || echo 0)"
  echo "[ok] Existing READY reservation '${RES_NAME}' found in ${EXISTING_ZONE} (count=${EXISTING_COUNT})."
  cat > "${STATE_FILE}" <<EOF
RES_NAME="${RES_NAME}"
RES_ZONE="${EXISTING_ZONE}"
RES_REGION="${EXISTING_ZONE%-*}"
RES_VM_TYPE="${VM_TYPE}"
EOF
  exit 0
fi

ROUND=0
while true; do
  ROUND=$((ROUND + 1))
  echo ""
  echo "[round ${ROUND}] Scanning candidate zones for 1x ${VM_TYPE}..."

  for zone in ${CANDIDATE_ZONES}; do
    set +e
    CREATE_OUT="$(gcloud beta compute reservations create "${RES_NAME}" \
      --project="${PROJECT_ID}" \
      --zone="${zone}" \
      --machine-type="${VM_TYPE}" \
      --accelerator="${ACCELERATOR}" \
      --local-ssd="${LOCAL_SSD}" \
      --require-specific-reservation \
      --description="labels:${LABELS}" \
      --vm-count="${VM_COUNT}" 2>&1)"
    RC=$?
    set -e

    if [[ ${RC} -eq 0 ]]; then
      echo "  [SUCCESS] ${zone} -> reserved 1x ${VM_TYPE}"
      echo "${CREATE_OUT}"
      REGION_FOUND="${zone%-*}"
      cat > "${STATE_FILE}" <<EOF
RES_NAME="${RES_NAME}"
RES_ZONE="${zone}"
RES_REGION="${REGION_FOUND}"
RES_VM_TYPE="${VM_TYPE}"
EOF
      echo ""
      echo "[done] Reserved 1x ${VM_TYPE} in ${zone} as '${RES_NAME}'."
      echo "[saved] ${STATE_FILE}"
      exit 0
    else
      ERR_LINE="$(echo "${CREATE_OUT}" | grep -E "ZONE_RESOURCE_POOL_EXHAUSTED|QUOTA|Quota|ERROR:" | head -n 2 | tr '\n' ' ')"
      echo "  [retry] ${zone} -> ${ERR_LINE}"
    fi
  done

  if [[ "${MAX_ROUNDS}" -gt 0 && "${ROUND}" -ge "${MAX_ROUNDS}" ]]; then
    echo "[stop] Reached MAX_ROUNDS=${MAX_ROUNDS} without securing a reservation." >&2
    exit 2
  fi

  echo "[wait] Sleeping ${RETRY_RATE}s before next scan round..."
  sleep "${RETRY_RATE}"
done
