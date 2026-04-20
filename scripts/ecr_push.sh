#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/ecr_push.sh
# Builds Docker images and pushes them to Amazon ECR.
#
# Usage:
#   chmod +x scripts/ecr_push.sh
#   ./scripts/ecr_push.sh training   [tag]
#   ./scripts/ecr_push.sh inference  [tag]
#   ./scripts/ecr_push.sh all        [tag]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

COMPONENT="${1:-all}"
TAG="${2:-latest}"
REGION=$(aws configure get region || echo "us-east-1")
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

TRAIN_REPO="perceptron-training"
INFER_REPO="perceptron-inference"

login() {
  echo "Logging into ECR..."
  aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${REGISTRY}"
}

ensure_repo() {
  local repo="$1"
  aws ecr describe-repositories --repository-names "${repo}" \
      --region "${REGION}" > /dev/null 2>&1 \
    || aws ecr create-repository --repository-name "${repo}" \
           --region "${REGION}" --image-scanning-configuration scanOnPush=true
}

push_image() {
  local component="$1"
  local repo="$2"
  local local_tag="${repo}:${TAG}"
  local remote_tag="${REGISTRY}/${repo}:${TAG}"

  echo ""
  echo "━━━ Building ${component} image ━━━"
  docker build -f "docker/Dockerfile.${component}" \
               -t "${local_tag}" .

  ensure_repo "${repo}"

  docker tag  "${local_tag}" "${remote_tag}"
  docker push "${remote_tag}"
  echo "✓ Pushed: ${remote_tag}"
}

login

case "${COMPONENT}" in
  training)  push_image training  "${TRAIN_REPO}" ;;
  inference) push_image inference "${INFER_REPO}" ;;
  all)
    push_image training  "${TRAIN_REPO}"
    push_image inference "${INFER_REPO}"
    ;;
  *)
    echo "Unknown component: ${COMPONENT}"
    exit 1
    ;;
esac

echo ""
echo "Done. Images tagged: ${TAG}"
