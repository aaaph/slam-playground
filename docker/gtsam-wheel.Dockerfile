# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-bookworm AS wheel-builder

ARG PYTHON_VERSION=3.13
ARG GTSAM_VERSION=4.2.1
ARG GTSAM_BUILD_JOBS=2

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/src/gtsam/build/gtsam:/src/gtsam/build/gtsam_unstable:/opt/gtsam/lib

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libboost-all-dev \
        ninja-build \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        auditwheel \
        build \
        numpy \
        pybind11 \
        "pybind11-stubgen>=2.5.1" \
        pyparsing \
        setuptools \
        wheel

WORKDIR /src

RUN git clone --branch "${GTSAM_VERSION}" --depth 1 https://github.com/borglab/gtsam.git gtsam

RUN cmake -S gtsam -B gtsam/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DGTSAM_BUILD_PYTHON=ON \
        -DGTSAM_UNSTABLE_BUILD_PYTHON=ON \
        -DGTSAM_PYTHON_VERSION="${PYTHON_VERSION}" \
        -DPYTHON_EXECUTABLE="$(command -v python)" \
        -DGTSAM_WITH_TBB=OFF \
        -DGTSAM_BUILD_TESTS=OFF \
        -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/gtsam

RUN cmake --build gtsam/build --target python-install --parallel "${GTSAM_BUILD_JOBS}"

RUN python -m build --wheel --no-isolation --outdir /tmp/wheelhouse gtsam/build/python

RUN auditwheel repair /tmp/wheelhouse/*.whl --wheel-dir /wheelhouse

FROM scratch AS export
COPY --from=wheel-builder /wheelhouse/ /
