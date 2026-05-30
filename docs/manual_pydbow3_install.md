# Manual PyDBoW3 Installation

Use this manual when `scripts/install_pydbow3.sh`, `just install-pydbow3`,
or `just install-3rdparty` does not work and you need to reproduce the steps by hand.

# References

- pyslam thirdparty pydbow3: https://github.com/luigifreda/pyslam/tree/master/thirdparty/pydbow3
- used fork: https://github.com/JHMeusener/PyDBoW3

# Requirements

- Python >= 3.13
- uv
- CMake
- C++ compiler
- OpenCV C++ libraries

On macOS with Homebrew:

```bash
brew install cmake opencv
```

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

# Installation

- From the project root, sync the Python environment:

```bash
uv sync --all-extras --cache-dir .uv_cache
```

- Install build dependencies required by PyDBoW3's legacy `setup.py`:

```bash
uv pip install numpy setuptools wheel
```

- Clone PyDBoW3 with submodules:

```bash
git clone https://github.com/JHMeusener/PyDBoW3 --recursive
cd PyDBoW3
```

- Update vendored pybind11 for Python 3.13 support:

```bash
git -C modules/pybind11 fetch --tags origin
git -C modules/pybind11 checkout v2.13.6
```

- Patch CMake compatibility and avoid linking every OpenCV module:

```bash
perl -0pi -e 's/CMAKE_MINIMUM_REQUIRED\(VERSION 2\.8\)/cmake_minimum_required(VERSION 3.5)/g' \
  modules/dbow3/CMakeLists.txt

perl -0pi -e 's/find_package\(OpenCV\s+REQUIRED\)/find_package(OpenCV REQUIRED COMPONENTS core)/g' \
  modules/dbow3/CMakeLists.txt

perl -0pi -e 's/find_package\(OpenCV REQUIRED\)/find_package(OpenCV REQUIRED COMPONENTS core)/g' \
  CMakeLists.txt
```

- Build and install DBoW3 into a local prefix:

```bash
cmake -S modules/dbow3 -B modules/dbow3/build-core \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/.local" \
  -DBUILD_UTILS=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build modules/dbow3/build-core -j4
cmake --install modules/dbow3/build-core
```

- Build and install the Python extension into the current uv environment:

```bash
CMAKE_PREFIX_PATH="$PWD/.local" uv pip install --force-reinstall --no-build-isolation .
```

- Verify the import from the project root:

```bash
cd ..
uv run python -c "import pydbow3; print(pydbow3.__file__); print(pydbow3.Vocabulary)"
uv run ./examples/example_dbow3.py
```

The import name is lowercase:

```python
import pydbow3
```

# Troubleshooting

- `ModuleNotFoundError: No module named 'numpy'` or `No module named 'setuptools'`

  Reinstall the build dependencies and keep `--no-build-isolation`:

```bash
uv pip install numpy setuptools wheel
CMAKE_PREFIX_PATH="$PWD/.local" uv pip install --force-reinstall --no-build-isolation .
```

- `Compatibility with CMake < 3.5 has been removed from CMake`

  Make sure `modules/dbow3/CMakeLists.txt` uses:

```cmake
cmake_minimum_required(VERSION 3.5)
```

Also keep this configure flag:

```bash
-DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

- Python 3.13 compile errors from pybind11, such as `PyFrameObject` or `_PyNamespace_New`

  The original submodule is too old. Checkout a newer pybind11 tag:

```bash
git -C modules/pybind11 checkout v2.13.6
```

- macOS import error about `libgflags.2.2.dylib`

  Homebrew OpenCV may have broken optional module linkage. PyDBoW3 only needs
  OpenCV core, so make sure both `CMakeLists.txt` files use:

```cmake
find_package(OpenCV REQUIRED COMPONENTS core)
```

Then rebuild DBoW3 and reinstall `pydbow3`.

- After `uv sync`, `pydbow3` disappears

  This is expected because `pydbow3` is installed manually and is not tracked in
  `uv.lock`. Reinstall native dependencies:

```bash
just install-3rdparty
```
