import numpy as np

from core.transformations.frame_resolver import FrameResolver, FrameTransform, StaticTransformTree
from core.transformations.special_euclidian_3_dim import SE3


class TestStaticTransformTree:
    """Test StaticTransformTree class."""

    def test_should_be_possible_to_create(self):
        """Test that the StaticTransformTree can be created."""
        t_body_cam0 = SE3()
        t_body_cam1 = SE3()
        transform_tree = StaticTransformTree(t_body_cam0, t_body_cam1)
        assert transform_tree is not None


class TestFrameResolver:
    """Test FrameResolver class."""

    def test_ctx_sub_class(self):
        """Test that the FrameResolver can create a context."""
        t_body_cam0 = SE3()
        t_body_cam1 = SE3()
        transform_tree = StaticTransformTree(t_body_cam0, t_body_cam1)
        frame_resolver = FrameResolver(transform_tree)
        some_dynamic_transform = FrameTransform(source="world", target="body", transform=SE3())
        ctx = frame_resolver.with_dynamic(some_dynamic_transform)
        assert ctx is not None
        assert ctx.dynamic_transform is not None
        assert ctx.dynamic_transform.source == "world"
        assert ctx.dynamic_transform.target == "body"
        assert ctx.dynamic_transform.transform is not None

    def test_query_sub_class(self):
        """Test that the FrameResolver can create a query."""
        t_body_cam0 = SE3()
        t_body_cam1 = SE3()
        transform_tree = StaticTransformTree(t_body_cam0, t_body_cam1)
        frame_resolver = FrameResolver(transform_tree)
        some_dynamic_transform = FrameTransform(source="world", target="body", transform=SE3())
        ctx = frame_resolver.with_dynamic(some_dynamic_transform)
        query = ctx.from_("cam0")
        assert query is not None
        assert query.source == "cam0"

    def test_resolve_where_world_body_is_dynamic(self):
        """Test that the FrameResolver can resolve a transform where world->body is dynamic."""
        body_in_world_transform = SE3.from_quat_and_translation(np.array([0, 0, 0, 1]), np.array([2, 0, 0]))
        cam0_in_body_transform = SE3.from_quat_and_translation(
            np.array([0.828459, 0.058956, 0.553641, -0.060514]),
            np.array([1.2500000000000002, 2.1650635094610964, 0]),
        )
        cam1_in_body_transform = SE3.from_quat_and_translation(
            np.array([0, 0, 0, 1]),
            np.array([-1.4999999999999993, 2.598076211353316, 0]),
        )
        feat_in_cam0_pos = np.array([-3.5, 4.286263797015736e-16, 0])

        static_tree = StaticTransformTree(cam0_in_body_transform, cam1_in_body_transform)
        frame_resolver = FrameResolver(static_tree)
        dynamic_transform = FrameTransform(source="world", target="body", transform=body_in_world_transform)
        feat_in_world = (
            frame_resolver.with_dynamic(dynamic_transform)
            .from_("cam0")
            .move_to("world")
            .apply_vector(feat_in_cam0_pos)
        )
        assert feat_in_world is not None
        assert np.allclose(np.round(feat_in_world, 2), np.array([1.92, 2.06, -3.24]))

        feat_se3 = SE3.from_quat_and_translation(np.array([0, 0, 0, 1]), feat_in_cam0_pos)
        feat_in_world = (
            frame_resolver.with_dynamic(dynamic_transform).from_("cam0").move_to("world").apply_se3(feat_se3)
        ).translation()
        assert feat_in_world is not None
        assert np.allclose(np.round(feat_in_world, 2), np.array([1.92, 2.06, -3.24]))

    def test_resolve_where_cam0_world_is_dynamic(self):
        """Test that the FrameResolver can resolve a transform where world->cam0 is dynamic."""
        cam0_in_world_transform = SE3.from_quat_and_translation(
            np.array([0.62615153, -0.54427605, 0.3610189, -0.42586758]),
            np.array([0.86863884, 2.20703527, 0.92586687]),
        )
        t_body_cam0 = np.array(
            [
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
            dtype=np.float64,
        ).reshape(4, 4)
        cam0_in_body_transform = SE3.from_matrix(t_body_cam0)
        t_body_cam1 = np.array(
            [
                [
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
                ]
            ],
            dtype=np.float64,
        ).reshape(4, 4)
        cam1_in_body_transform = SE3.from_matrix(t_body_cam1)
        static_tree = StaticTransformTree(cam0_in_body_transform, cam1_in_body_transform)
        frame_resolver = FrameResolver(static_tree)
        dynamic_transform = FrameTransform(source="world", target="cam0", transform=cam0_in_world_transform)
        body_in_world_se3 = (
            frame_resolver.with_dynamic(dynamic_transform)
            .from_("cam0")
            .move_to("body")
            .apply_se3(cam0_in_world_transform)
        )
        assert body_in_world_se3 is not None
        assert np.allclose(np.round(body_in_world_se3.translation(), 2), np.array([0.88, 2.14, 0.95]))
        assert np.allclose(
            np.round(body_in_world_se3.rotation().as_quat(), 2), np.array([0.83, 0.06, 0.55, -0.06])
        )
