# Manual GTSAM Installation

Check the manual if scripts/install_gtsam.sh is not working.

# References

- GTSAM Python Installation 4.2.1: [README.md](https://github.com/borglab/gtsam/blob/4.2.1/python/README.md)
- PySLAM Shell: https://github.com/luigifreda/pyslam/blob/master/scripts/install_gtsam.sh

# Requirements

- Cmake >= 3.15
- uv
- python >= 3.13

# Installation

- clone gtsam repository:

```bash
git clone --branch 4.2.1 --depth 1 https://github.com/borglab/gtsam.git
```

- activate uv environment:

```bash
# Fish example
source .venv/bin/activate.fish
```

- add pip manually

```bash
uv pip install pip
```

- create build directory:

```bash
mkdir -p gtsam/build && cd gtsam/build
```

- run cmake

```bash
cmake .. \
    -DGTSAM_BUILD_PYTHON=1 \
    -DGTSAM_PYTHON_VERSION=3.13 \
    -DGTSAM_WITH_TBB=OFF \
    -DCMAKE_INSTALL_PREFIX="../install" \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_EXAMPLES=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_PYTHON_EXECUTABLE=$(which python)
```

- install python bindings

```bash
make python-install -j8
```

- Back to the root directory

```bash
cd ../..
```
