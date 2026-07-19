import numpy as np
import pytest

from core.front_end.front_end_bootstrap import (
    FeatureBootstrapMetrics,
    FrontEndBootstrap,
    FrontEndBootstrapConfig,
    FrontEndBootstrapDecision,
    FrontEndBootstrapInput,
    FrontEndBootstrapResult,
    ImuBootstrapMetrics,
)


class TestFrontEndBootstrap:
    """Unit Test the frontend initialization."""

    @pytest.fixture
    def config(self) -> FrontEndBootstrapConfig:
        """Create a short bootstrap config for unit tests."""
        return FrontEndBootstrapConfig(
            min_window_frames=5,
            window_size_frames=5,
            min_window_duration_sec=0.15,
            min_good_features=30,
            min_triangulated_features=20,
            min_stereo_ok_ratio=0.35,
        )

    def test_should_return_unknown_until_window_is_ready(self, config: FrontEndBootstrapConfig) -> None:
        """Bootstrap should avoid decisions before the evidence window is ready."""
        bootstrap = FrontEndBootstrap(config)

        result = bootstrap.feed(self._sample(0, parallax=0.1))

        assert result.decision == FrontEndBootstrapDecision.UNKNOWN
        assert not result.ready
        assert "window_not_ready" in result.reasons

    def test_should_select_static_when_visual_and_imu_are_quiet(self, config: FrontEndBootstrapConfig) -> None:
        """Bootstrap should select static only when visual and IMU evidence agree."""
        bootstrap = FrontEndBootstrap(config)

        result = self._feed_window(bootstrap, parallax=0.1, gyro_std_norm=0.002, accel_norm_std=0.01)

        assert result.decision == FrontEndBootstrapDecision.STATIC
        assert result.ready
        assert "visual_static" in result.reasons
        assert "imu_static" in result.reasons

    def test_should_select_dynamic_when_visual_parallax_is_persistent(
        self, config: FrontEndBootstrapConfig
    ) -> None:
        """Bootstrap should select dynamic after persistent visual motion in the window."""
        bootstrap = FrontEndBootstrap(config)

        result = self._feed_window(bootstrap, parallax=4.0, gyro_std_norm=0.002, accel_norm_std=0.01)

        assert result.decision == FrontEndBootstrapDecision.DYNAMIC
        assert result.ready
        assert "visual_dynamic" in result.reasons

    def test_should_select_dynamic_when_visual_p90_parallax_is_persistent(
        self, config: FrontEndBootstrapConfig
    ) -> None:
        """Bootstrap should select dynamic when the per-frame parallax tail is persistent."""
        bootstrap = FrontEndBootstrap(config)

        result = self._feed_window(
            bootstrap,
            parallax=0.1,
            parallax_p90=8.0,
            gyro_std_norm=0.002,
            accel_norm_std=0.01,
        )

        assert result.decision == FrontEndBootstrapDecision.DYNAMIC
        assert result.ready
        assert "visual_dynamic" in result.reasons

    def test_should_return_vision_degraded_for_current_feature_starvation(
        self, config: FrontEndBootstrapConfig
    ) -> None:
        """A feature-starved current frame should be reported even if the window median is healthy."""
        bootstrap = FrontEndBootstrap(config)
        self._feed_window(bootstrap, parallax=0.1, gyro_std_norm=0.002, accel_norm_std=0.01)

        result = bootstrap.feed(self._sample(10, parallax=0.1, good_features=2, triangulated_features=2))

        assert result.decision == FrontEndBootstrapDecision.VISION_DEGRADED
        assert not result.ready
        assert "vision_degraded" in result.reasons

    def _feed_window(
        self,
        bootstrap: FrontEndBootstrap,
        *,
        parallax: float,
        gyro_std_norm: float,
        accel_norm_std: float,
        parallax_p90: float | None = None,
    ) -> FrontEndBootstrapResult:
        result = bootstrap.latest_result
        for frame_id in range(5):
            result = bootstrap.feed(
                self._sample(
                    frame_id,
                    parallax=parallax,
                    parallax_p90=parallax_p90,
                    gyro_std_norm=gyro_std_norm,
                    accel_norm_std=accel_norm_std,
                )
            )
        return result

    @staticmethod
    def _sample(  # noqa: PLR0913
        frame_id: int,
        *,
        parallax: float,
        parallax_p90: float | None = None,
        good_features: int = 60,
        triangulated_features: int = 45,
        gyro_std_norm: float = 0.002,
        accel_norm_std: float = 0.01,
    ) -> FrontEndBootstrapInput:
        feature_metrics = FeatureBootstrapMetrics(
            active_count=good_features,
            good_count=good_features,
            stereo_ok_count=good_features,
            triangulated_count=triangulated_features,
            tracked_count=good_features,
            mature_track_count=good_features,
            temporal_parallax_px=parallax,
            temporal_parallax_p90_px=parallax if parallax_p90 is None else parallax_p90,
        )
        imu_metrics = ImuBootstrapMetrics(
            sample_count=10,
            duration_sec=0.05,
            gyro_mean=np.zeros(3),
            gyro_std=np.array([gyro_std_norm, 0.0, 0.0]),
            gyro_norm_mean=0.0,
            gyro_std_norm=gyro_std_norm,
            accel_mean=np.array([0.0, 0.0, 9.81]),
            accel_std=np.array([accel_norm_std, 0.0, 0.0]),
            accel_norm_mean=9.81,
            accel_norm_std=accel_norm_std,
            accel_direction_std_rad=0.001,
        )
        return FrontEndBootstrapInput(
            frame_id=frame_id,
            timestamp_ns=frame_id * 50_000_000.0,
            feature_metrics=feature_metrics,
            imu_metrics=imu_metrics,
        )
