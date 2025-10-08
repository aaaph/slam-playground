import cv2
import jax
import jax.numpy as jnp
import numpy as np

from core.feature_tracker.feature_tracker import FeatureTracker
from core.feature_tracker.image_preprocess import StereoImagePreprocess
from dataset.euroc import EurocDataset

euroc_dataset = EurocDataset.mh_01_easy()
stereo = euroc_dataset.stereo().with_format("numpy")
stereo_iterator = stereo.to_iterable_dataset()

ft = FeatureTracker()
image_preprocess = StereoImagePreprocess(euroc_dataset.config.stereo)

fast = cv2.FastFeatureDetector.create()


for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left = np.array(stereo_data["stereo"][0])
    right = np.array(stereo_data["stereo"][1])

    left, right = image_preprocess.preprocess_stereo(left, right)

    left = np.array(left)
    right = np.array(right)
    left_output = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right_output = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

    points: list[cv2.KeyPoint] = []
    for region in ft.grid:
        kps = fast.detect(image=left, mask=np.array(region.mask))
        kps = sorted(kps, key=lambda x: x.response, reverse=True)
        kps = kps[: ft.FEAT_PER_REGION]
        points.extend(kps)
        for kp in kps:
            x = int(kp.pt[0])
            y = int(kp.pt[1])
            cv2.circle(left_output, (x, y), 2, (0, 0, 255), -1)

    for region in ft.grid:
        box = region.box
        cv2.rectangle(left_output, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    # get random 10 indexes
    key = jax.random.PRNGKey(42)
    random_kp_index = jax.random.choice(key, jnp.array(range(len(points))), shape=(10,), replace=False)
    random_kp = [points[i] for i in random_kp_index]

    concatenated = np.concatenate([left_output, right_output], axis=1)
    for kp in random_kp:
        x = int(kp.pt[0])
        y = int(kp.pt[1])
        cv2.line(concatenated, (0, y), (concatenated.shape[1], y), (0, 255, 255), 1)
        cv2.line(concatenated, (x, 0), (x, concatenated.shape[0]), (0, 255, 255), 1)
    cv2.imshow("Stereo", concatenated)
    key = cv2.waitKey(0)
    if key == ord("q"):
        break
    else:
        continue

cv2.destroyAllWindows()
