"""Generate a tiny deterministic EuRoC-style stereo-inertial smoke dataset."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path
from textwrap import dedent

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "euroc_smoke"
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / DATASET_NAME
DEFAULT_MANIFEST_PATH = REPO_ROOT / "datasets" / f"{DATASET_NAME}.yaml"

WIDTH = 752
HEIGHT = 480
FRAME_COUNT = 8
CAMERA_RATE_HZ = 20
IMU_RATE_HZ = 200
BASE_TIMESTAMP_NS = 1_403_715_273_262_142_976
FRAME_DT_NS = int(1e9 / CAMERA_RATE_HZ)
IMU_DT_NS = int(1e9 / IMU_RATE_HZ)

CAM0_SENSOR_YAML = """\
# General sensor definitions.
sensor_type: camera
comment: Synthetic EuRoC smoke cam0

# Sensor extrinsics wrt. the body-frame.
T_BS:
  cols: 4
  rows: 4
  data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]

# Camera specific definitions.
rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [458.654, 457.296, 367.215, 248.375]
distortion_model: radial-tangential
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
"""

CAM1_SENSOR_YAML = """\
# General sensor definitions.
sensor_type: camera
comment: Synthetic EuRoC smoke cam1

# Sensor extrinsics wrt. the body-frame.
T_BS:
  cols: 4
  rows: 4
  data: [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
         0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
         0.0, 0.0, 0.0, 1.0]

# Camera specific definitions.
rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [457.587, 456.134, 379.999, 255.238]
distortion_model: radial-tangential
distortion_coefficients: [-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05]
"""

IMU_SENSOR_YAML = """\
# Default imu sensor yaml file.
sensor_type: imu
comment: Synthetic EuRoC smoke IMU

# Sensor extrinsics wrt. the body-frame.
T_BS:
  cols: 4
  rows: 4
  data: [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
rate_hz: 200

# Inertial sensor noise model parameters.
gyroscope_noise_density: 1.6968e-04
gyroscope_random_walk: 1.9393e-05
accelerometer_noise_density: 2.0000e-3
accelerometer_random_walk: 3.0000e-3
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output dataset registry manifest path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated dataset root and manifest.",
    )
    return parser.parse_args()


def make_texture() -> np.ndarray:
    """Create the deterministic source texture used by both stereo cameras."""
    rng = np.random.default_rng(12)
    canvas = np.full((HEIGHT + 160, WIDTH + 260), 98, dtype=np.uint8)

    for x in range(0, canvas.shape[1], 42):
        cv2.line(canvas, (x, 0), (x, canvas.shape[0] - 1), 118, 1)
    for y in range(0, canvas.shape[0], 38):
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), 118, 1)

    for _ in range(950):
        center = (
            int(rng.integers(12, canvas.shape[1] - 12)),
            int(rng.integers(12, canvas.shape[0] - 12)),
        )
        radius = int(rng.integers(2, 7))
        color = int(rng.integers(20, 235))
        cv2.circle(canvas, center, radius, color, -1, lineType=cv2.LINE_AA)

    for _ in range(90):
        p1 = (
            int(rng.integers(0, canvas.shape[1])),
            int(rng.integers(0, canvas.shape[0])),
        )
        p2 = (
            int(np.clip(p1[0] + rng.integers(-70, 71), 0, canvas.shape[1] - 1)),
            int(np.clip(p1[1] + rng.integers(-70, 71), 0, canvas.shape[0] - 1)),
        )
        color = int(rng.integers(35, 220))
        cv2.line(canvas, p1, p2, color, 1, lineType=cv2.LINE_AA)

    return cv2.GaussianBlur(canvas, (3, 3), 0)


def stereo_frame(texture: np.ndarray, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop one synthetic left/right pair from the source texture."""
    x0 = 24 + frame_index * 7
    y0 = 36 + round(3.0 * np.sin(frame_index * 0.8))
    disparity = 9

    left = texture[y0 : y0 + HEIGHT, x0 : x0 + WIDTH].copy()
    right = texture[y0 : y0 + HEIGHT, x0 + disparity : x0 + disparity + WIDTH].copy()

    vignette_x = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)
    vignette_y = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32)
    xv, yv = np.meshgrid(vignette_x, vignette_y)
    vignette = 1.0 - 0.12 * (xv * xv + yv * yv)

    left = np.clip(left.astype(np.float32) * vignette, 0, 255).astype(np.uint8)
    right = np.clip(right.astype(np.float32) * vignette, 0, 255).astype(np.uint8)
    return left, right


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Write a CSV file with EuRoC-style headers."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def imu_timestamps(frame_timestamps: list[int]) -> list[int]:
    """Return 200 Hz IMU timestamps spanning all generated camera frames."""
    first = frame_timestamps[0]
    last = frame_timestamps[-1]
    return list(range(first, last + IMU_DT_NS, IMU_DT_NS))


def write_dataset(dataset_root: Path) -> None:
    """Write the EuRoC-like dataset tree and raw stream files."""
    (dataset_root / "cam0" / "data").mkdir(parents=True, exist_ok=True)
    (dataset_root / "cam1" / "data").mkdir(parents=True, exist_ok=True)
    (dataset_root / "imu0").mkdir(parents=True, exist_ok=True)
    (dataset_root / "state_groundtruth_estimate0").mkdir(parents=True, exist_ok=True)

    (dataset_root / "body.yaml").write_text("comment: Synthetic EuRoC smoke MAV\n", encoding="utf-8")
    (dataset_root / "cam0" / "sensor.yaml").write_text(CAM0_SENSOR_YAML, encoding="utf-8")
    (dataset_root / "cam1" / "sensor.yaml").write_text(CAM1_SENSOR_YAML, encoding="utf-8")
    (dataset_root / "imu0" / "sensor.yaml").write_text(IMU_SENSOR_YAML, encoding="utf-8")

    frame_timestamps = [BASE_TIMESTAMP_NS + i * FRAME_DT_NS for i in range(FRAME_COUNT)]
    texture = make_texture()

    cam_rows: list[list[object]] = []
    for index, timestamp in enumerate(frame_timestamps):
        filename = f"{timestamp}.png"
        left, right = stereo_frame(texture, index)
        if not cv2.imwrite(str(dataset_root / "cam0" / "data" / filename), left):
            message = f"Failed to write cam0 image {filename}"
            raise RuntimeError(message)
        if not cv2.imwrite(str(dataset_root / "cam1" / "data" / filename), right):
            message = f"Failed to write cam1 image {filename}"
            raise RuntimeError(message)
        cam_rows.append([timestamp, filename])

    write_csv(dataset_root / "cam0" / "data.csv", ["#timestamp [ns]", "filename"], cam_rows)
    write_csv(dataset_root / "cam1" / "data.csv", ["#timestamp [ns]", "filename"], cam_rows)

    imu_rows: list[list[object]] = []
    gt_rows: list[list[object]] = []
    velocity_x = 0.18
    for timestamp in imu_timestamps(frame_timestamps):
        elapsed = (timestamp - frame_timestamps[0]) * 1e-9
        gyro_z = 0.01 * np.sin(2.0 * np.pi * elapsed)
        acc_x = 0.02 * np.sin(4.0 * np.pi * elapsed)
        imu_rows.append([timestamp, 0.0, 0.0, f"{gyro_z:.9f}", f"{acc_x:.9f}", 0.0, 9.81])
        gt_rows.append(
            [
                timestamp,
                f"{velocity_x * elapsed:.9f}",
                f"{0.01 * np.sin(2.0 * np.pi * elapsed):.9f}",
                "0.000000000",
                "1.000000000",
                "0.000000000",
                "0.000000000",
                "0.000000000",
                f"{velocity_x:.9f}",
                f"{0.01 * 2.0 * np.pi * np.cos(2.0 * np.pi * elapsed):.9f}",
                "0.000000000",
                "0.000000000",
                "0.000000000",
                "0.000000000",
                "0.000000000",
                "0.000000000",
                "0.000000000",
            ]
        )

    write_csv(
        dataset_root / "imu0" / "data.csv",
        [
            "#timestamp [ns]",
            "w_RS_S_x [rad s^-1]",
            "w_RS_S_y [rad s^-1]",
            "w_RS_S_z [rad s^-1]",
            "a_RS_S_x [m s^-2]",
            "a_RS_S_y [m s^-2]",
            "a_RS_S_z [m s^-2]",
        ],
        imu_rows,
    )
    write_csv(
        dataset_root / "state_groundtruth_estimate0" / "data.csv",
        [
            "#timestamp",
            " p_RS_R_x [m]",
            " p_RS_R_y [m]",
            " p_RS_R_z [m]",
            " q_RS_w []",
            " q_RS_x []",
            " q_RS_y []",
            " q_RS_z []",
            " v_RS_R_x [m s^-1]",
            " v_RS_R_y [m s^-1]",
            " v_RS_R_z [m s^-1]",
            " b_w_RS_S_x [rad s^-1]",
            " b_w_RS_S_y [rad s^-1]",
            " b_w_RS_S_z [rad s^-1]",
            " b_a_RS_S_x [m s^-2]",
            " b_a_RS_S_y [m s^-2]",
            " b_a_RS_S_z [m s^-2]",
        ],
        gt_rows,
    )


def write_manifest(manifest_path: Path, dataset_root: Path) -> None:
    """Write the dataset registry manifest for the generated smoke dataset."""
    relative_root = dataset_root.relative_to(REPO_ROOT)
    manifest = dedent(
        f"""\
        name: {DATASET_NAME}
        type: euroc
        root: {relative_root}
        rig: config/dataset_rig/euroc.yaml
        cache: {relative_root}/cache

        streams:
          cam0: cam0/data.csv
          cam1: cam1/data.csv
          imu0: imu0/data.csv
          ground_truth: state_groundtruth_estimate0/data.csv
        """
    )
    manifest_path.write_text(manifest, encoding="utf-8")


def main() -> None:
    """Generate the smoke dataset and registry manifest."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    manifest_path = args.manifest_path.resolve()

    if dataset_root.exists():
        if not args.force:
            message = f"{dataset_root} already exists; pass --force to replace it"
            raise FileExistsError(message)
        shutil.rmtree(dataset_root)
    if manifest_path.exists() and not args.force:
        message = f"{manifest_path} already exists; pass --force to replace it"
        raise FileExistsError(message)

    write_dataset(dataset_root)
    write_manifest(manifest_path, dataset_root)
    frame_timestamps = [BASE_TIMESTAMP_NS + i * FRAME_DT_NS for i in range(FRAME_COUNT)]
    LOGGER.info("Wrote %d stereo frames to %s", FRAME_COUNT, dataset_root)
    LOGGER.info("Wrote %d IMU rows", len(imu_timestamps(frame_timestamps)))
    LOGGER.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
