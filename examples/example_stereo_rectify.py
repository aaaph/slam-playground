import cv2
import numpy as np

from dataset.euroc import EurocDataset

euroc_dataset = EurocDataset.mh_01_easy()
stereo = euroc_dataset.stereo().with_format("numpy")
stereo_iterator = stereo.to_iterable_dataset()


cam0_config = {
    "resolution": (752, 480),
    "camera_model": "pinhole",
    "intrinsics": (458.654, 457.296, 367.215, 248.375),
    "distortion_model": "radial-tangential",
    "distortion_coefficients": (-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05),
    "T_BS": {
        "cols": 4,
        "rows": 4,
        "data": [
            0.0148655429818,
            -0.999880929698,
            0.00414029679422,
            -0.0216401454975,
            0.999557249008,
            0.0149672133247,
            0.025715529948,
            -0.064676986768,
            -0.0257744366974,
            0.00375618835797,
            0.999660727178,
            0.00981073058949,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    },
}
cam1_config = {
    "resolution": (752, 480),
    "camera_model": "pinhole",
    "intrinsics": (457.587, 456.134, 379.999, 255.238),
    "distortion_model": "radial-tangential",
    "distortion_coefficients": (-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05),
    "T_BS": {
        "cols": 4,
        "rows": 4,
        "data": [
            0.0125552670891,
            -0.999755099723,
            0.0182237714554,
            -0.0198435579556,
            0.999598781151,
            0.0130119051815,
            0.0251588363115,
            0.0453689425024,
            -0.0253898008918,
            0.0179005838253,
            0.999517347078,
            0.00786212447038,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    },
}
cam0_k = np.array(
    [
        [cam0_config["intrinsics"][0], 0, cam0_config["intrinsics"][2]],
        [0, cam0_config["intrinsics"][1], cam0_config["intrinsics"][3]],
        [0, 0, 1],
    ]
)
cam1_k = np.array(
    [
        [cam1_config["intrinsics"][0], 0, cam1_config["intrinsics"][2]],
        [0, cam1_config["intrinsics"][1], cam1_config["intrinsics"][3]],
        [0, 0, 1],
    ]
)
cam0_distortion_coefficients = np.array(cam0_config["distortion_coefficients"])
cam1_distortion_coefficients = np.array(cam1_config["distortion_coefficients"])

cam0_transform = np.array(cam0_config["T_BS"]["data"]).reshape(4, 4)
cam1_transform = np.array(cam1_config["T_BS"]["data"]).reshape(4, 4)

cam0_to_cam1_transform = np.linalg.inv(np.linalg.inv(cam0_transform) @ cam1_transform)


r = cam0_to_cam1_transform[:3, :3]
p = cam0_to_cam1_transform[:3, 3].reshape(3, 1).copy()
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1=cam0_k,
    distCoeffs1=cam0_distortion_coefficients,
    cameraMatrix2=cam1_k,
    distCoeffs2=cam1_distortion_coefficients,
    imageSize=(752, 480),
    R=np.array(r),
    T=p,
    flags=cv2.CALIB_ZERO_DISPARITY,
)
map1_x, map1_y = cv2.initUndistortRectifyMap(
    cameraMatrix=cam0_k,
    distCoeffs=cam0_distortion_coefficients,
    R=R1,
    newCameraMatrix=P1,
    size=(752, 480),
    m1type=cv2.CV_32FC1,
)
map2_x, map2_y = cv2.initUndistortRectifyMap(
    cameraMatrix=cam1_k,
    distCoeffs=cam1_distortion_coefficients,
    R=R2,
    newCameraMatrix=P2,
    size=(752, 480),
    m1type=cv2.CV_32FC1,
)


for stereo_data in stereo_iterator:
    left = np.array(stereo_data["stereo"][0])
    right = np.array(stereo_data["stereo"][1])

    left = cv2.remap(left, map1_x, map1_y, cv2.INTER_LINEAR)
    right = cv2.remap(right, map2_x, map2_y, cv2.INTER_LINEAR)

    left_output = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right_output = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

    h = left_output.shape[0]

    concatenated = np.concatenate([left_output, right_output], axis=1)
    concatinated_output = concatenated.copy()
    cv2.line(concatinated_output, (0, h // 2), (concatenated.shape[1], h // 2), (0, 255, 255), 1)
    cv2.line(concatinated_output, (0, h // 4), (concatenated.shape[1], h // 4), (0, 255, 255), 1)
    cv2.line(concatinated_output, (0, 3 * h // 4), (concatenated.shape[1], 3 * h // 4), (0, 255, 255), 1)
    cv2.line(concatinated_output, (0, h // 8), (concatenated.shape[1], h // 8), (0, 255, 255), 1)
    cv2.line(concatinated_output, (0, 7 * h // 8), (concatenated.shape[1], 7 * h // 8), (0, 255, 255), 1)

    cv2.imshow("Concatenated", concatinated_output)
    cv2.waitKey(0)
    break

cv2.destroyAllWindows()
