from dataclasses import dataclass
from typing import Literal

import anytree as at
import numpy as np
from numpy.typing import NDArray

from core.transformations.special_euclidian_3_dim import SE3

FrameId = Literal["world", "body", "cam0", "cam1", "feature"]


@dataclass
class FrameTransform:
    """Transform frame."""

    source: FrameId
    target: FrameId
    transform: SE3


class StaticTransformTree:
    """Transform tree."""

    def __init__(self, t_body_cam0: SE3, t_body_cam1: SE3) -> None:
        """Initialize the transform tree."""
        self.body = at.Node("body", parent=None)
        self.cam0 = at.Node("cam0", parent=self.body, t_bs=t_body_cam0)
        self.cam1 = at.Node("cam1", parent=self.body, t_bs=t_body_cam1)

        self.nodes = {
            "body": self.body,
            "cam0": self.cam0,
            "cam1": self.cam1,
        }


class FrameResolver:
    """Frame resolver."""

    def __init__(self, transform_tree: StaticTransformTree) -> None:
        """Initialize the frame resolver."""
        self.transform_tree = transform_tree

    class _Ctx:
        """Context for the frame resolver."""

        def __init__(self, outer: "FrameResolver", dynamic_transform: FrameTransform) -> None:
            """Initialize the context."""
            self.outer = outer
            self.dynamic_transform = dynamic_transform

        class _Query:
            def __init__(self, outer: "FrameResolver._Ctx", source: FrameId) -> None:
                """Initialize the query."""
                self.outer = outer
                self.source = source

            class _Applier:
                """Applier for the query."""

                def __init__(self, outer: "FrameResolver._Ctx._Query", target: FrameId) -> None:
                    """Initialize the applier."""
                    self.outer = outer
                    self.target = target

                def apply_vector(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
                    """Apply a vector to the query."""
                    t = self.outer.outer.outer.resolve(
                        self.outer.source, self.target, self.outer.outer.dynamic_transform
                    )
                    return t @ vector

                def apply_se3(self, se3: SE3) -> SE3:
                    """Apply a SE3 to the query."""
                    t = self.outer.outer.outer.resolve(
                        self.outer.source, self.target, self.outer.outer.dynamic_transform
                    )
                    return t * se3

            def move_to(self, target: FrameId) -> _Applier:
                """Move to."""
                return self._Applier(self, target)

        def from_(self, source: FrameId) -> _Query:
            """From."""
            return self._Query(self, source)

    def with_dynamic(self, dynamic_transform: FrameTransform) -> "_Ctx":
        """With dynamics."""
        return FrameResolver._Ctx(self, dynamic_transform)

    def resolve(self, source: FrameId, target: FrameId, dynamic_transform: FrameTransform) -> SE3:
        """Resolve the transform between two frames."""
        if source == target:
            return SE3.identity()

        if source == "cam0" and target == "world":
            world_to_body = dynamic_transform.transform
            body_to_cam0 = self.transform_tree.nodes["cam0"].t_bs
            return world_to_body * body_to_cam0

        if source == "cam0" and target == "body":
            world_to_cam0 = dynamic_transform.transform
            body_to_cam0 = self.transform_tree.nodes["cam0"].t_bs
            cam0_to_body = body_to_cam0.inverse()
            return world_to_cam0 * cam0_to_body * world_to_cam0.inverse()
            # code implemented as t * se3 where se3 in right side, so we to move cam0_to_body to the left side

        msg = f"Cannot resolve transform from {source} to {target}"
        raise ValueError(msg)
