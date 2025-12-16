#!/bin/bash
set -e

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

if [ ! -d "gtsam" ]; then
    echo "📥 Cloning GTSAM..."
    git clone https://github.com/borglab/gtsam.git
fi

BUILD_DIR="gtsam/build"

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
        -DPYTHON_EXECUTABLE="$VENV_PYTHON" \
        -DCMAKE_BUILD_TYPE=Release
fi

echo "🚀 Installing bindings..."

make python-install -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)

echo "✅ GTSAM installed successfully!"
