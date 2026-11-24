#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

TARGET="${1:-g4-8gpu-all}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-45m}"

usage() {
  echo "Usage: $0 [g4|g4-8gpu-mig|g4-8gpu-oom|g4-8gpu-mps|g4-8gpu-all|h100-mig|h100-mps|oom|all]"
  echo "  g4           Run baseline matmul job on g4-standard-12 (${G4_PROVISIONING})"
  echo "  g4-8gpu-mig  Run 32 concurrent matmul jobs on 1x g4-standard-384 MIG ${G4_8GPU_MIG_PROFILE}"
  echo "  g4-8gpu-oom  Run 32-pod noisy-neighbor OOM isolation test on g4-standard-384 MIG ${G4_8GPU_MIG_PROFILE}"
  echo "  g4-8gpu-mps  Run 32 concurrent matmul jobs on 1x g4-standard-384 MPS (${G4_8GPU_MPS_CLIENTS_PER_GPU}/GPU)"
  echo "  g4-8gpu-all  Run g4-8gpu-mig -> g4-8gpu-oom -> g4-8gpu-mps"
  echo "  h100-mig     Run 24 concurrent matmul jobs on 1x a3-megagpu-8g MIG ${H100_MIG_PROFILE}"
  echo "  h100-mps     Run 32 concurrent matmul jobs on 1x a3-megagpu-8g MPS (${H100_MPS_CLIENTS_PER_GPU}/GPU)"
  echo "  oom          Run noisy-neighbor OOM isolation test on H100 Mega MIG"
  echo "  all          Run g4 -> h100-mig -> oom (and h100-mps if CREATE_MPS_POOL=true)"
}

sync_configmaps() {
  echo "[step] Ensuring namespace '${K8S_NAMESPACE}' and ConfigMaps are up to date..."
  sed \
    -e "s|name: gpu-bench|name: ${K8S_NAMESPACE}|g" \
    -e "s|owner: gpu-bench|owner: ${RESOURCE_PREFIX}|g" \
    "${SCRIPT_DIR}/k8s/namespace.yaml" | kubectl apply -f -

  kubectl -n "${K8S_NAMESPACE}" create configmap bench-script \
    --from-file=matmul_bench.py="${SCRIPT_DIR}/matmul_bench.py" \
    --dry-run=client -o yaml | kubectl apply -f -

  kubectl -n "${K8S_NAMESPACE}" create configmap bench-params \
    --from-literal=MATMUL_N="${MATMUL_N}" \
    --from-literal=WARMUP_ITERS="${WARMUP_ITERS}" \
    --from-literal=ITERS_FP64="${ITERS_FP64}" \
    --from-literal=ITERS_FP32="${ITERS_FP32}" \
    --from-literal=ITERS_BF16="${ITERS_BF16}" \
    --from-literal=TARGET_VRAM_GB="${TARGET_VRAM_GB}" \
    --from-literal=CPU_PHASE_SEC="${CPU_PHASE_SEC}" \
    --from-literal=OOM_ALLOC_GB="${OOM_ALLOC_GB}" \
    --dry-run=client -o yaml | kubectl apply -f -
}

render_and_apply_job() {
  local manifest="$1"
  local job_name="$2"

  echo ""
  echo "[step] Launching Job '${job_name}' from $(basename "${manifest}")..."
  kubectl -n "${K8S_NAMESPACE}" delete job "${job_name}" --ignore-not-found --wait=true

  # Apply config.env overrides (namespace, owner, image, pool names, MIG profiles)
  sed \
    -e "s|namespace: gpu-bench|namespace: ${K8S_NAMESPACE}|g" \
    -e "s|owner: gpu-bench|owner: ${RESOURCE_PREFIX}|g" \
    -e "s|pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime|${BENCH_IMAGE}|g" \
    -e "s|gpu-bench-g4-flex|${G4_POOL_NAME}|g" \
    -e "s|gpu-bench-g4-8gpu-mig|${G4_8GPU_MIG_POOL_NAME}|g" \
    -e "s|gpu-bench-g4-8gpu-mps|${G4_8GPU_MPS_POOL_NAME}|g" \
    -e "s|gpu-bench-h100-mega-mig|${H100_MIG_POOL_NAME}|g" \
    -e "s|gpu-bench-h100-mega-mps|${H100_MPS_POOL_NAME}|g" \
    -e "s|\"1g.24gb\"|\"${G4_8GPU_MIG_PROFILE}\"|g" \
    -e "s|\"2g.20gb\"|\"${H100_MIG_PROFILE}\"|g" \
    "${manifest}" | kubectl apply -f -

  echo "[wait] Waiting up to ${WAIT_TIMEOUT} for Job '${job_name}' (includes node provisioning)..."
  if ! kubectl -n "${K8S_NAMESPACE}" wait --for=condition=complete --timeout="${WAIT_TIMEOUT}" "job/${job_name}"; then
    echo "[warn] Job '${job_name}' did not reach Complete within ${WAIT_TIMEOUT}. Current pod status:"
    kubectl -n "${K8S_NAMESPACE}" get pods -l "job-name=${job_name}" -o wide
    return 1
  fi

  echo "[ok] Job '${job_name}' completed! Node placement summary:"
  kubectl -n "${K8S_NAMESPACE}" get pods -l "job-name=${job_name}" -o wide
}

sync_configmaps

case "${TARGET}" in
  g4)
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-g4-flex.yaml" "matmul-g4-flex"
    ;;
  g4-8gpu-mig)
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-g4-8gpu-mig.yaml" "matmul-g4-8gpu-mig"
    ;;
  g4-8gpu-oom)
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-g4-8gpu-mig-oom.yaml" "matmul-g4-8gpu-mig-oom"
    ;;
  g4-8gpu-mps)
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-g4-8gpu-mps.yaml" "matmul-g4-8gpu-mps"
    ;;
  g4-8gpu-all)
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-g4-8gpu-mig.yaml" "matmul-g4-8gpu-mig"
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-g4-8gpu-mig-oom.yaml" "matmul-g4-8gpu-mig-oom"
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-g4-8gpu-mps.yaml" "matmul-g4-8gpu-mps"
    ;;
  h100-mig)
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-h100-mega-mig.yaml" "matmul-h100-mega-mig"
    ;;
  h100-mps)
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-h100-mega-mps.yaml" "matmul-h100-mega-mps"
    ;;
  oom)
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-oom-isolation.yaml" "matmul-h100-mega-mig-oom"
    ;;
  all)
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-g4-flex.yaml" "matmul-g4-flex"
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-h100-mega-mig.yaml" "matmul-h100-mega-mig"
    render_and_apply_job "${SCRIPT_DIR}/k8s/job-oom-isolation.yaml" "matmul-h100-mega-mig-oom"
    if [[ "${CREATE_MPS_POOL}" == "true" ]]; then
      render_and_apply_job "${SCRIPT_DIR}/k8s/job-h100-mega-mps.yaml" "matmul-h100-mega-mps"
    fi
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 1
    ;;
esac

echo ""
echo "[next] Collect scorecard with: python3 ${SCRIPT_DIR}/03_collect_results.py"
