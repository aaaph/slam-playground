import shutil

import numpy as np

from core.feature_tracker.feature_tracker import FeatureTracker
from dataset.euroc import EurocDataset
from datasets import Dataset, Features, Sequence, Value

db_features = Features(
    frame_id=Value("int32"),
    timestamp=Value("float64"),
    feat_ids=Sequence(Value("int32")),
    uL=Sequence(Value("float32")),
    vL=Sequence(Value("float32")),
    uR=Sequence(Value("float32")),
    vR=Sequence(Value("float32")),
    ground_truth=Sequence(Value("float32")),
)

euroc_dataset = EurocDataset.mh_01_easy()
stereo = euroc_dataset.stereo()
stereo_iterator = stereo.to_iterable_dataset()
cache_path = euroc_dataset.data_paths.cache
feat_db_cache_path = cache_path / "feat_db"
# just delete if exists or not exists
if feat_db_cache_path.exists():
    shutil.rmtree(feat_db_cache_path)
feat_db_cache_path.mkdir(parents=True, exist_ok=True)
ft = FeatureTracker(euroc_dataset.config.stereo, feat_amount_per_region=30, feat_retrack_threshold=10)

rows = []
frame_id = 0
for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left, right = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])
    left, right = ft.feed(ts, (left, right))

    ground_truth = euroc_dataset.find_nearest_ground_truth_by_timestamp(ts)
    ground_truth_position = ground_truth["gt_position"]
    ground_truth_quat = ground_truth["gt_orientation"]
    ground_truth_pose = np.array(
        [
            ground_truth_position[0],
            ground_truth_position[1],
            ground_truth_position[2],
            ground_truth_quat[0],
            ground_truth_quat[1],
            ground_truth_quat[2],
            ground_truth_quat[3],
        ]
    )

    feat_ids = []
    ul_list = []
    vl_list = []
    ur_list = []
    vr_list = []
    for feature in ft.iterate_through_features():
        feat_id = feature.feat_id
        _, left_uv, right_uv = feature.get_active_stereo_pair()
        feat_ids.append(feat_id)
        ul_list.append(left_uv[0])
        vl_list.append(left_uv[1])
        if right_uv is not None:
            ur_list.append(right_uv[0])
            vr_list.append(right_uv[1])
        else:
            ur_list.append(np.nan)
            vr_list.append(np.nan)

    rows.append(
        {
            "frame_id": frame_id,
            "timestamp": ts,
            "feat_ids": feat_ids,
            "uL": ul_list,
            "vL": vl_list,
            "uR": ur_list,
            "vR": vr_list,
            "ground_truth": ground_truth_pose,
        }
    )
    frame_id += 1  # noqa: SIM113


feat_dataset = Dataset.from_list(rows, features=db_features)
feat_dataset.save_to_disk(feat_db_cache_path)
