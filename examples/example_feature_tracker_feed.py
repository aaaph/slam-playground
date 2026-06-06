import cv2
import numpy as np

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_schema import FeatureSchema
from core.feature_tracker.feature_tracker import FeatureTracker
from dataset.euroc import EurocDataset

euroc_dataset = EurocDataset.mh_01()
stereo = euroc_dataset.stereo()
stereo_iterator = stereo.to_iterable_dataset()

camera_model = StereoCameraModel.from_cameras_config(euroc_dataset.config.cam0, euroc_dataset.config.cam1)
ft = FeatureTracker.default_factory(
    camera_model.as_stereo_ctx(), feat_amount_per_region=30, feat_retrack_threshold=10
)


for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left, right = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])
    left, right = camera_model.process_stereo(left, right)
    features = ft.feed(ts, (left, right))

    left_out = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right_out = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    concatenated = np.concatenate([left_out, right_out], axis=1)

    for feature in features.ndarray:
        lx, ly = feature[FeatureSchema.LEFT_U], feature[FeatureSchema.LEFT_V]
        cv2.circle(concatenated, (int(lx), int(ly)), 2, (0, 0, 255), -1)

    cv2.putText(
        concatenated, f"feat count: {features.count()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )
    cv2.imshow("concatenated", concatenated)
    key = cv2.waitKey(0)
    if key == ord("q"):
        break
    else:
        continue

cv2.destroyAllWindows()
