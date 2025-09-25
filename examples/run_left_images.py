import cv2
import numpy as np

from dataset.euroc import EurocDataset

euroc_dataset = EurocDataset.mh_01_easy()
iterator = euroc_dataset.iterate_stereo()

for sample in iterator:
    left = sample[1]
    ts = sample[0]
    cv2.imshow("Left", np.array(left))
    key = cv2.waitKey(0)
    if key == ord("q"):
        break
    else:
        continue
