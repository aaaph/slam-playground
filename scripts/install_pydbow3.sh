#!/usr/bin/env bash
set -euo pipefail

PYDBOW3_REPO_URL="${PYDBOW3_REPO_URL:-https://github.com/JHMeusener/PyDBoW3.git}"
PYDBOW3_DIR="${PYDBOW3_DIR:-PyDBoW3}"
PYBIND11_TAG="${PYBIND11_TAG:-v2.13.6}"
DBOW3_BUILD_DIR_NAME="${DBOW3_BUILD_DIR_NAME:-build-core}"
INSTALL_PREFIX_NAME="${INSTALL_PREFIX_NAME:-.local}"
FORCE_REBUILD=false

usage() {
    cat <<EOF
Usage: bash scripts/install_pydbow3.sh [options]

Options:
  --rebuild        Remove generated PyDBoW3 build directories before building.
  --dir PATH       PyDBoW3 checkout path. Default: PyDBoW3
  --help           Show this help.

Environment:
  PYDBOW3_REPO_URL        Git URL used when the checkout does not exist.
  PYBIND11_TAG            pybind11 tag to checkout. Default: v2.13.6
  DBOW3_BUILD_DIR_NAME    DBoW3 build directory name. Default: build-core
  INSTALL_PREFIX_NAME     Local install prefix inside PyDBoW3. Default: .local
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild)
            FORCE_REBUILD=true
            ;;
        --dir)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Error: --dir requires a path" >&2
                exit 2
            fi
            PYDBOW3_DIR="$1"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required command not found: $1" >&2
        exit 1
    fi
}

replace_text() {
    local file="$1"
    local old="$2"
    local new="$3"

    if grep -Fq "$new" "$file"; then
        return
    fi
    if grep -Fq "$old" "$file"; then
        OLD_TEXT="$old" NEW_TEXT="$new" perl -0pi -e 's/\Q$ENV{OLD_TEXT}\E/$ENV{NEW_TEXT}/g' "$file"
        return
    fi

    echo "Warning: did not find expected text in $file" >&2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

need_cmd git
need_cmd cmake
need_cmd uv
need_cmd perl

PYDBOW3_PATH="$ROOT_DIR/$PYDBOW3_DIR"
DBOW3_PATH="$PYDBOW3_PATH/modules/dbow3"
PYBIND11_PATH="$PYDBOW3_PATH/modules/pybind11"
DBOW3_BUILD_DIR="$DBOW3_PATH/$DBOW3_BUILD_DIR_NAME"
INSTALL_PREFIX="$PYDBOW3_PATH/$INSTALL_PREFIX_NAME"

echo "==> Installing PyDBoW3"
echo "Project root: $ROOT_DIR"
echo "PyDBoW3 dir:  $PYDBOW3_PATH"
echo "Build jobs:   4"

if [[ ! -d "$PYDBOW3_PATH" ]]; then
    echo "==> Cloning PyDBoW3"
    git clone --recursive "$PYDBOW3_REPO_URL" "$PYDBOW3_PATH"
elif [[ ! -d "$PYDBOW3_PATH/.git" ]]; then
    echo "Error: $PYDBOW3_PATH exists but is not a git checkout" >&2
    exit 1
fi

echo "==> Ensuring submodules are present"
git -C "$PYDBOW3_PATH" submodule update --init --recursive

if [[ ! -d "$DBOW3_PATH" || ! -d "$PYBIND11_PATH" ]]; then
    echo "Error: expected PyDBoW3 submodules were not found" >&2
    exit 1
fi

echo "==> Updating pybind11 submodule to $PYBIND11_TAG"
if [[ -n "$(git -C "$PYBIND11_PATH" status --porcelain)" ]]; then
    echo "Error: pybind11 submodule has local changes. Commit, stash, or reset them first." >&2
    exit 1
fi
if ! git -C "$PYBIND11_PATH" rev-parse -q --verify "refs/tags/$PYBIND11_TAG" >/dev/null; then
    git -C "$PYBIND11_PATH" fetch --tags origin
fi
git -C "$PYBIND11_PATH" checkout -q "$PYBIND11_TAG"

echo "==> Patching PyDBoW3 CMake files for Python 3.13, CMake 4, and minimal OpenCV linkage"
replace_text "$PYDBOW3_PATH/CMakeLists.txt" \
    "find_package(OpenCV REQUIRED)" \
    "find_package(OpenCV REQUIRED COMPONENTS core)"

replace_text "$DBOW3_PATH/CMakeLists.txt" \
    "CMAKE_MINIMUM_REQUIRED(VERSION 2.8)" \
    "cmake_minimum_required(VERSION 3.5)"

replace_text "$DBOW3_PATH/CMakeLists.txt" \
    "find_package(OpenCV  REQUIRED)" \
    "find_package(OpenCV REQUIRED COMPONENTS core)"

replace_text "$DBOW3_PATH/CMakeLists.txt" \
    "find_package(OpenCV REQUIRED)" \
    "find_package(OpenCV REQUIRED COMPONENTS core)"

if [[ "$FORCE_REBUILD" == true ]]; then
    echo "==> Removing generated build directories"
    rm -rf "$DBOW3_BUILD_DIR" "$PYDBOW3_PATH/build" "$PYDBOW3_PATH/pydbow3.egg-info"
fi

echo "==> Installing Python build dependencies into the uv environment"
uv pip install numpy setuptools wheel

echo "==> Configuring DBoW3"
cmake -S "$DBOW3_PATH" -B "$DBOW3_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DBUILD_UTILS=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

echo "==> Building DBoW3"
cmake --build "$DBOW3_BUILD_DIR" -j4

if [[ -d "$INSTALL_PREFIX/lib" ]]; then
    find "$INSTALL_PREFIX/lib" -maxdepth 1 -type f \( -name 'libDBoW3*.dylib' -o -name 'libDBoW3*.so*' \) -delete
fi

echo "==> Installing DBoW3 into $INSTALL_PREFIX"
cmake --install "$DBOW3_BUILD_DIR"

echo "==> Building and installing pydbow3"
CMAKE_PREFIX_PATH="$INSTALL_PREFIX" uv pip install --force-reinstall --no-build-isolation "$PYDBOW3_PATH"

echo "==> Verifying import"
uv run python - <<'PY'
import pydbow3

print("pydbow3 ok:", pydbow3.__file__)
PY

echo "==> PyDBoW3 installation complete"
