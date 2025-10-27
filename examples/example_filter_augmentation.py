import cv2
import numpy as np

from core.feature_tracker.feature_tracker import FeatureTracker
from core.filter.augmentator import Augmentator
from core.filter.initializer import Initializer
from core.filter.propagator import Propagator
from core.filter.state import State
from core.filter.updater import Updater
from dataset.euroc import EurocDataset

euroc_dataset = EurocDataset.mh_01_easy()
ds = euroc_dataset.all()


first_ground_truth = euroc_dataset.first_ground_truth()

ft = FeatureTracker(euroc_dataset.config.stereo)
state = Initializer().initialize_from_dict(
    State(),
    first_ground_truth["timestamp"],
    dictionary={
        "position": first_ground_truth["gt_position"],
        "orientation": first_ground_truth["gt_orientation"],
        "velocity": first_ground_truth["gt_velocity"],
        "acc_bias": first_ground_truth["gt_acc_bias"],
        "gyro_bias": first_ground_truth["gt_gyro_bias"],
    },
)
updater = Updater()
propagator = Propagator.from_imu_config(euroc_dataset.config.imu0)
augmentator = Augmentator()
for item in ds.to_iterable_dataset():
    timestamp = item["timestamp"]
    has_stereo = item["stereo"][0] is not None
    has_imu = item["has_imu"]
    has_ground_truth = item["has_ground_truth"]
    gyro = item["gyro"]
    acc = item["acc"]
    if has_imu:
        result, state = propagator.state_propagation(state, (timestamp, gyro, acc))
        if not result:
            continue
    if has_stereo and has_imu:
        stereo = item["stereo"]
        state = augmentator.augment_clone(state)

        left, right = ft.feed(timestamp, (stereo[0], stereo[1]))
        left_out = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right_out = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        concatenated = np.concatenate([left_out, right_out], axis=1)

        removing_candidate = state.sliding_window.get_candidate_for_removal()
        if removing_candidate is not None:
            feat_in_timestamp = ft.get_features_spawned_in_timestamp(removing_candidate.timestamp)
            ft.drop_features(feat_in_timestamp)

        for feat in ft.iterate_through_features():
            _, left_uv, right_uv = feat.get_active_stereo_pair()
            lx, ly = left_uv
            cv2.circle(concatenated, (int(lx), int(ly)), 2, feat.feature_color(), -1)
            tail = feat.get_tail(0)
            start_color = np.array([200, 200, 255], dtype=np.uint8)
            feat_color = np.array(feat.feature_color(), dtype=np.uint8)
            for i, (u, v) in enumerate(tail):
                alpha = 1.0 / (i + 1)
                color = ((1 - alpha) * start_color + alpha * feat_color).astype(np.uint8).tolist()
                cv2.circle(concatenated, (int(u), int(v)), 2, color, -1)
            if right_uv is not None:
                rx, ry = right_uv
                cv2.circle(concatenated, (int(rx) + ft.IMAGE_SHAPE["w"], int(ry)), 2, feat.feature_color(), -1)
        cv2.putText(
            concatenated, f"feat count: {ft.feat_count()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.imshow("concatenated", concatenated)

        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        else:
            continue
