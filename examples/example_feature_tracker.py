import cv2
import numpy as np

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker
from dataset.euroc import EurocDataset

euroc_dataset = EurocDataset.mh_01_easy()
stereo = euroc_dataset.stereo().with_format("numpy")
stereo_iterator = stereo.to_iterable_dataset()
camera_model = StereoCameraModel.from_cameras_config(euroc_dataset.config.cam0, euroc_dataset.config.cam1)
ft = FeatureTracker.default_factory(camera_model.as_stereo_ctx())

fast = cv2.FastFeatureDetector.create(15)
orb = cv2.ORB.create(nfeatures=1000, edgeThreshold=15, patchSize=31, fastThreshold=10)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
iterator_count = 0


first_item = next(iter(stereo_iterator))
left_old, right_old = np.array(first_item["stereo"][0]), np.array(first_item["stereo"][1])
left_old, right_old = camera_model.process_stereo(left_old, right_old)
left_old, right_old = np.array(left_old), np.array(right_old)


keypoints: list[cv2.KeyPoint] = []
for region in ft.grid:
    kps = fast.detect(image=left_old, mask=np.array(region.mask))
    kps = sorted(kps, key=lambda x: x.response, reverse=True)
    kps = kps[: ft.FEAT_PER_REGION]
    keypoints.extend(kps)


timestamp = float(first_item["timestamp"])
points = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
p0 = np.array(points, dtype=np.float32).reshape(-1, 2)

my_point = (75.0, 82.0)

for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left_new, right_new = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])
    left_new, right_new = camera_model.process_stereo(left_new, right_new)
    left_new, right_new = np.array(left_new), np.array(right_new)

    p_next = np.array(p0, dtype=np.int32)
    p1, st, _err = cv2.calcOpticalFlowPyrLK(left_old, left_new, p0, p_next, **ft.klt_params)
    st = st.ravel()

    good_old = p0[st == 1]
    good_new = p1[st == 1]

    _E, inliners = cv2.findEssentialMat(
        good_new,
        good_old,
        cameraMatrix=np.array(euroc_dataset.config.stereo.left_k_undistorted),
        method=cv2.RANSAC,
        threshold=0.999,
    )
    inliner_mask = inliners.ravel().astype(bool)

    good_new = good_new[inliner_mask]
    good_old = good_old[inliner_mask]

    left_out = cv2.cvtColor(left_new, cv2.COLOR_GRAY2BGR)

    for new, old in zip(good_new, good_old, strict=True):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.circle(left_out, (int(a), int(b)), 2, (0, 0, 255), -1)
        if (c, d) == my_point:
            my_point = (a, b)

    cv2.imshow("left_out", left_out)

    left_old = left_new.copy()
    p0 = good_new.reshape(-1, 2)
    key = cv2.waitKey(0)
    if key == ord("q"):
        break
    else:
        continue

cv2.destroyAllWindows()
