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

    def test_resolve(self):
        """Test that the FrameResolver can resolve a transform."""
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
