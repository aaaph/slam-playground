from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydbow3

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_schema import FeatureSchema
from core.feature_tracker.feature_tracker import FeatureTracker, FeatureTrackerMode
from dataset.euroc import EurocDataset
from logger import spawn_logger

log = spawn_logger(app="example_dbow3")
SAVE_BINARY_COMPRESSED = True
USE_DIRECT_INDEX = False


@dataclass(frozen=True)
class RunConfig:
    """Configuration for descriptor corpus, vocabulary, and DBoW3 database generation."""

    output_dir: Path
    max_frames: int
    frame_stride: int
    min_descriptors: int
    feat_amount_per_region: int
    feat_retrack_threshold: int
    region_amount: int
    nfeatures: int
    edge_threshold: int
    patch_size: int
    fast_threshold: int
    vocabulary_path: Path
    vocabulary_cache_path: Path
    build_vocabulary: bool
    build_database: bool
    vocab_k: int
    vocab_l: int


def _parse_args() -> RunConfig:
    """Parse command line arguments."""
    default_output_dir = Path("datasets/euroc_v_01_easy/cache/dbow3")
    parser = argparse.ArgumentParser(
        description=(
            "Collect ORB descriptors from EuRoC through the local FeatureTracker and build DBoW3 artifacts."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all stereo frames.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-descriptors", type=int, default=20)
    parser.add_argument("--feat-amount-per-region", type=int, default=40)
    parser.add_argument("--feat-retrack-threshold", type=int, default=2)
    parser.add_argument("--region-amount", type=int, default=8)
    parser.add_argument("--nfeatures", type=int, default=1000)
    parser.add_argument("--edge-threshold", type=int, default=15)
    parser.add_argument("--patch-size", type=int, default=31)
    parser.add_argument("--fast-threshold", type=int, default=10)
    parser.add_argument("--vocabulary-path", type=Path, default=Path(__file__).parent.parent / "ORBvoc.txt")
    parser.add_argument(
        "--vocabulary-cache-path", type=Path, default=Path(__file__).parent.parent / "ORBvoc_cached.dbow3"
    )
    parser.add_argument("--build-vocabulary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-database", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocab-k", type=int, default=10)
    parser.add_argument("--vocab-l", type=int, default=5)
    args = parser.parse_args()

    return RunConfig(
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        frame_stride=max(args.frame_stride, 1),
        min_descriptors=max(args.min_descriptors, 0),
        feat_amount_per_region=args.feat_amount_per_region,
        feat_retrack_threshold=args.feat_retrack_threshold,
        region_amount=args.region_amount,
        nfeatures=args.nfeatures,
        edge_threshold=args.edge_threshold,
        patch_size=args.patch_size,
        fast_threshold=args.fast_threshold,
        vocabulary_path=args.vocabulary_path,
        vocabulary_cache_path=args.vocabulary_cache_path,
        build_vocabulary=args.build_vocabulary,
        build_database=args.build_database,
        vocab_k=args.vocab_k,
        vocab_l=args.vocab_l,
    )


def _spawn_feature_tracker(config: RunConfig, camera_model: StereoCameraModel) -> FeatureTracker:
    return FeatureTracker.default_factory(
        camera_model.as_stereo_ctx(),
        feat_amount_per_region=config.feat_amount_per_region,
        feat_retrack_threshold=config.feat_retrack_threshold,
        region_amount=config.region_amount,
        mode=FeatureTrackerMode.MONOCULAR,
    )


def _load_orb_vocabulary(vocabulary_path: Path, cache_path: Path) -> pydbow3.Vocabulary:
    vocabulary = pydbow3.Vocabulary()
    if cache_path.exists() and cache_path.stat().st_atime >= vocabulary_path.stat().st_atime:
        vocabulary.load(str(cache_path))
        return vocabulary

    vocabulary.load(str(vocabulary_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary.save(str(cache_path), SAVE_BINARY_COMPRESSED)
    return vocabulary


def main() -> None:
    """Collect descriptors and save DBoW3 vocabulary/database artifacts."""
    config = _parse_args()
    t1 = time.time()
    vocabulary = _load_orb_vocabulary(config.vocabulary_path, config.vocabulary_cache_path)
    t2 = time.time()
    log.info(f"Loaded vocabulary: {t2 - t1} seconds")

    t3 = time.time()
    db = pydbow3.Database()
    db.setVocabulary(vocabulary, USE_DIRECT_INDEX, 0)
    t4 = time.time()
    log.info(f"Set vocabulary: {t4 - t3} seconds")

    euroc_dataset = EurocDataset.mh_01_easy()
    camera_model = StereoCameraModel.from_cameras_config(euroc_dataset.config.cam0, euroc_dataset.config.cam1)
    feature_tracker = _spawn_feature_tracker(config, camera_model)
    orb = cv2.ORB.create(
        nfeatures=config.nfeatures,
        edgeThreshold=config.edge_threshold,
        patchSize=config.patch_size,
        fastThreshold=config.fast_threshold,
    )

    stereo_iterator = euroc_dataset.stereo().to_iterable_dataset()
    iteration_count = 25
    for i, stereo_data in enumerate(stereo_iterator):
        if i > iteration_count:
            break
        ts = float(stereo_data["timestamp"])
        left, right = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])
        left, right = camera_model.process_stereo(left, right)
        active_frame = feature_tracker.feed(ts, (left, right))
        features = active_frame.good_features()
        feature_ids = features[:, FeatureSchema.FEAT_ID].astype(np.int32, copy=False)
        left_points = features[:, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1].astype(np.float32, copy=False)

        keypoints = [
            cv2.KeyPoint(float(u), float(v), float(config.patch_size), -1, 0, 0, int(feature_id))
            for (u, v), feature_id in zip(left_points, feature_ids, strict=True)
        ]

        keypoints, descriptors = orb.compute(left, keypoints)

        log.info(
            f"timestamp: {ts:.0f}, features: {features.shape} "
            f"left shape: {left.shape}, keypoints: {len(keypoints)} "
            f"descriptors: {descriptors.shape} ",
        )

        result = db.add(descriptors)
        log.info(f"result: {result}")

        query_result = db.query(descriptors, max_results=5)

        for r in query_result:
            log.info(f"db_id={r.Id}, score={r.Score:.3f}")

        left_out = cv2.drawKeypoints(
            left,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        cv2.imshow("left", left_out)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
