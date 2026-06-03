import pytest

from dataset.sensor_config import CameraSensor


@pytest.fixture
def cam_config_0() -> CameraSensor:
    """Create a camera configuration."""
    return CameraSensor(
        {
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
    )


@pytest.fixture
def cam_config_1() -> CameraSensor:
    """Create a camera configuration."""
    return CameraSensor(
        {
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
    )
