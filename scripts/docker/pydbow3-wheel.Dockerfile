# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-bookworm AS wheel-builder

ARG PYDBOW3_REPO_URL=https://github.com/JHMeusener/PyDBoW3.git
ARG PYDBOW3_REF=90f89d37cf0a7bda2f8fd6fb83cc62e1d8bcb7d0
ARG PYBIND11_TAG=v2.13.6
ARG PYDBOW3_BUILD_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_GENERATOR=Ninja \
    CMAKE_BUILD_PARALLEL_LEVEL=${PYDBOW3_BUILD_JOBS} \
    LD_LIBRARY_PATH=/src/PyDBoW3/.local/lib

# Pinning Debian package versions against a mutable Python base image makes
# rebuilds brittle; the image tag controls the Debian snapshot here.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libopencv-dev \
    ninja-build \
    patchelf \
    perl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    auditwheel==6.4.2 \
    build==1.3.0 \
    numpy==2.4.6 \
    setuptools==81.0.0 \
    wheel==0.45.1

WORKDIR /src

RUN git clone --recursive "${PYDBOW3_REPO_URL}" PyDBoW3 \
    && git -C PyDBoW3 checkout "${PYDBOW3_REF}" \
    && git -C PyDBoW3 submodule update --init --recursive

RUN git -C PyDBoW3/modules/pybind11 fetch --tags origin \
    && git -C PyDBoW3/modules/pybind11 checkout -q "${PYBIND11_TAG}"

RUN perl -0pi -e 's/find_package\(OpenCV REQUIRED\)/find_package(OpenCV REQUIRED COMPONENTS core)/g' PyDBoW3/CMakeLists.txt \
    && perl -0pi -e 's/CMAKE_MINIMUM_REQUIRED\(VERSION 2\.8\)/cmake_minimum_required(VERSION 3.5)/g' PyDBoW3/modules/dbow3/CMakeLists.txt \
    && perl -0pi -e 's/find_package\(OpenCV\s+REQUIRED\)/find_package(OpenCV REQUIRED COMPONENTS core)/g' PyDBoW3/modules/dbow3/CMakeLists.txt \
    && perl -0pi -e 's/find_package\(OpenCV REQUIRED\)/find_package(OpenCV REQUIRED COMPONENTS core)/g' PyDBoW3/modules/dbow3/CMakeLists.txt

RUN perl -0pi -e 's/#include <map>\n#include <vector>\n#include "exports.h"/#include <cstddef>\n#include <cstdint>\n#include <istream>\n#include <map>\n#include <ostream>\n#include <string>\n#include <vector>\n#include "exports.h"/g' PyDBoW3/modules/dbow3/src/BowVector.h \
    && perl -0pi -e 's/#include "BowVector.h"\n#include <map>\n#include <vector>/#include "BowVector.h"\n#include <map>\n#include <ostream>\n#include <vector>/g' PyDBoW3/modules/dbow3/src/FeatureVector.h \
    && perl -0pi -e 's/#include <opencv2\/core\/core.hpp>\n#include <vector>\n#include <string>/#include <opencv2\/core\/core.hpp>\n#include <climits>\n#include <cstdint>\n#include <istream>\n#include <ostream>\n#include <string>\n#include <vector>/g' PyDBoW3/modules/dbow3/src/DescManip.h \
    && perl -0pi -e 's/#include <vector>\n#include "exports.h"/#include <ostream>\n#include <string>\n#include <vector>\n#include "exports.h"/g' PyDBoW3/modules/dbow3/src/QueryResults.h

RUN perl -0pi -e 's/throw\(std::exception\)//g; s/throw\(std::runtime_error\)//g' \
    PyDBoW3/modules/dbow3/src/Vocabulary.h \
    PyDBoW3/modules/dbow3/src/Vocabulary.cpp

RUN cmake -S PyDBoW3/modules/dbow3 -B PyDBoW3/modules/dbow3/build-core \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/src/PyDBoW3/.local \
    -DBUILD_UTILS=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && cmake --build PyDBoW3/modules/dbow3/build-core --parallel "${PYDBOW3_BUILD_JOBS}" \
    && cmake --install PyDBoW3/modules/dbow3/build-core

RUN CMAKE_PREFIX_PATH=/src/PyDBoW3/.local \
    python -m build --wheel --no-isolation --outdir /tmp/wheelhouse PyDBoW3 \
    && auditwheel repair /tmp/wheelhouse/*.whl --wheel-dir /wheelhouse

FROM scratch AS export
COPY --from=wheel-builder /wheelhouse/ /
