#!/bin/bash
set -e

GTSAM_VERSION="4.2.1"
GTSAM_REPO_URL="https://github.com/borglab/gtsam.git"
GTSAM_DIR="gtsam"

FORCE_REBUILD=false
for arg in "$@"
do
    case $arg in
        --rebuild)
        FORCE_REBUILD=true
        shift
        ;;
    esac
done

echo "========================================"
echo "🔧 GTSAM Installer (Manual Simulation Mode)"
echo "📌 Target GTSAM version: $GTSAM_VERSION"
echo "========================================"

VENV_PYTHON=$(uv run python -c "import sys; print(sys.executable)")
VENV_BIN_DIR=$(dirname "$VENV_PYTHON")
export PATH="$VENV_BIN_DIR:$PATH"
echo "VENV_PYTHON: $VENV_PYTHON"
echo "VENV_BIN_DIR: $VENV_BIN_DIR"

echo "🐍 Active Python: $(which python)"
if [[ "$(which python)" != *".venv"* ]]; then
    echo "❌ Error: Could not activate venv internally."
    exit 1
fi

echo "📦 Installing build dependencies..."
uv pip install pip numpy pyparsing "pybind11-stubgen>=2.5.1" pybind11

if [ ! -d "$GTSAM_DIR" ]; then
    echo "📥 Cloning GTSAM..."
    git clone --branch "$GTSAM_VERSION" --depth 1 "$GTSAM_REPO_URL" "$GTSAM_DIR"
elif [ ! -d "$GTSAM_DIR/.git" ]; then
    echo "❌ Error: $GTSAM_DIR exists but is not a git checkout. Cannot enforce GTSAM $GTSAM_VERSION."
    exit 1
else
    echo "📌 Ensuring GTSAM checkout is at $GTSAM_VERSION..."
    if ! git -C "$GTSAM_DIR" rev-parse --verify --quiet "$GTSAM_VERSION^{commit}" >/dev/null; then
        git -C "$GTSAM_DIR" fetch --tags "$GTSAM_REPO_URL" "refs/tags/$GTSAM_VERSION:refs/tags/$GTSAM_VERSION"
    fi

    CURRENT_REF=$(git -C "$GTSAM_DIR" rev-parse HEAD)
    TARGET_REF=$(git -C "$GTSAM_DIR" rev-parse "$GTSAM_VERSION^{commit}")

    if [ "$CURRENT_REF" != "$TARGET_REF" ]; then
        if [ -n "$(git -C "$GTSAM_DIR" status --porcelain)" ]; then
            echo "❌ Error: $GTSAM_DIR has local changes. Commit/stash them before switching to GTSAM $GTSAM_VERSION."
            exit 1
        fi

        git -C "$GTSAM_DIR" checkout --detach "$GTSAM_VERSION"
        FORCE_REBUILD=true
    fi
fi

BUILD_DIR="$GTSAM_DIR/build"

if [ "$FORCE_REBUILD" = true ]; then
    echo "🧹 Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

if [ -f "$BUILD_DIR/Makefile" ]; then
    echo "⚡ Found existing build! Skipping CMake configuration..."
    cd "$BUILD_DIR"
else
    echo "🏗️  Configuring CMake..."
    cd "$BUILD_DIR"

    cmake .. \
        -DGTSAM_BUILD_PYTHON=1 \
        -DGTSAM_PYTHON_VERSION=3.13 \
        -DGTSAM_WITH_TBB=OFF \
        -DCMAKE_INSTALL_PREFIX="../install" \
        -DGTSAM_BUILD_TESTS=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DPYTHON_EXECUTABLE="$VENV_PYTHON" \
        -DCMAKE_BUILD_TYPE=Release
fi

echo "🚀 Installing bindings..."

make python-install -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)

echo "✅ GTSAM installed successfully!"
