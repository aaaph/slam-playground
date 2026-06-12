#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
PYDBOW3_REPO_URL="${PYDBOW3_REPO_URL:-https://github.com/JHMeusener/PyDBoW3.git}"
PYDBOW3_REF="${PYDBOW3_REF:-90f89d37cf0a7bda2f8fd6fb83cc62e1d8bcb7d0}"
PYBIND11_TAG="${PYBIND11_TAG:-v2.13.6}"
PYDBOW3_BUILD_JOBS="${PYDBOW3_BUILD_JOBS:-4}"
OUT_DIR="${OUT_DIR:-dist/pydbow3-wheel}"
PLATFORM="${PLATFORM:-}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"

target_wheel_pattern="pydbow3-*.whl"
case "$PLATFORM" in
    "")
        ;;
    "linux/amd64")
        target_wheel_pattern="pydbow3-*x86_64.whl"
        ;;
    "linux/arm64" | "linux/aarch64")
        target_wheel_pattern="pydbow3-*aarch64.whl"
        ;;
    *)
        echo "Unsupported PyDBoW3 wheel target platform: ${PLATFORM}" >&2
        exit 1
        ;;
esac

mkdir -p "$OUT_DIR"

if [[ -n "$PLATFORM" ]]; then
    find "$OUT_DIR" -maxdepth 1 -type f -name 'pydbow3-*.whl' ! -name "$target_wheel_pattern" -print -delete
fi

if [[ "$FORCE_REBUILD" != "true" ]] && compgen -G "${OUT_DIR}/${target_wheel_pattern}" >/dev/null; then
    echo "PyDBoW3 wheel already exists in ${OUT_DIR}; skipping Docker build."
    ls -1 "${OUT_DIR}"/${target_wheel_pattern}
    exit 0
fi

build_args=(
    docker buildx build
    --pull
    --file docker/pydbow3-wheel.Dockerfile
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
    --build-arg "PYDBOW3_REPO_URL=${PYDBOW3_REPO_URL}"
    --build-arg "PYDBOW3_REF=${PYDBOW3_REF}"
    --build-arg "PYBIND11_TAG=${PYBIND11_TAG}"
    --build-arg "PYDBOW3_BUILD_JOBS=${PYDBOW3_BUILD_JOBS}"
    --output "type=local,dest=${OUT_DIR}"
)

if [[ -n "$PLATFORM" ]]; then
    build_args+=(--platform "$PLATFORM")
fi

build_args+=(.)

"${build_args[@]}"
