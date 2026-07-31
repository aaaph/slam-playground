from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext


class LandmarkRefineStatus(Enum):
    """Status of the landmark refinement."""

    SUCCESS = 0
    DEPTH_NEGATIVE = 1
    MAX_ITERATIONS_REACHED = 2
    SOLVER_ERROR = 3
    ILL_CONDITIONED = 4
    HIGH_REPROJECTION_ERROR = 5
    NO_COST_DECREASE = 6


@dataclass(slots=True)
class _Linearization:
    status: LandmarkRefineStatus
    h: NDArray[np.float64]
    g: NDArray[np.float64]
    total_error: float


@dataclass(slots=True)
class _LmStep:
    status: LandmarkRefineStatus
    point: NDArray[np.float64]
    linearization: _Linearization
    delta_norm: float


class Refiner(Protocol):
    """Refiner contract for landmark refinement."""

    def refine_point_gn(
        self, initial_guess: NDArray[np.float64], uvs: NDArray[np.float64], poses: NDArray[np.float64]
    ) -> tuple[LandmarkRefineStatus, NDArray[np.float64], NDArray[np.float64]]:
        """Refine the point via GN optimization."""

    def worst_reprojection_ray_index(
        self, point: NDArray[np.float64], uvs: NDArray[np.float64], poses: NDArray[np.float64]
    ) -> int | None:
        """Return the isolated worst ray index, if one should be rejected."""


class LandmarkRefiner(Refiner):
    """Component for refining the landmarks via GN optimization."""

    def __init__(  # noqa: PLR0913
        self,
        stereo_ctx: StereoContext,
        max_iterations: int = 3,
        min_delta: float = 1e-4,
        min_cost_decrease_ratio: float = 1e-6,
        min_gradient_norm: float = 1e-8,
        max_hessian_condition_number: float = 1e12,
        min_hessian_eigenvalue: float = 1e-12,
        max_final_reprojection_rmse_px: float = 5.0,
        pixel_sigma_px: float = 1.5,
        lm_initial_lambda: float = 1e-3,
        lm_lambda_factor: float = 10.0,
        lm_max_attempts: int = 5,
        outlier_reprojection_error_px: float = 5.0,
        outlier_error_ratio: float = 2.0,
        min_rays_after_outlier_rejection: int = 3,
    ) -> None:
        """Initialize the landmark refiner."""
        self._stereo_ctx = stereo_ctx
        self._stereo_k = stereo_ctx.stereo_k
        self._max_iterations = max_iterations
        self._min_delta = min_delta
        self._min_cost_decrease_ratio = min_cost_decrease_ratio
        self._min_gradient_norm = min_gradient_norm
        self._max_hessian_condition_number = max_hessian_condition_number
        self._min_hessian_eigenvalue = min_hessian_eigenvalue
        self._max_final_reprojection_rmse_px = max_final_reprojection_rmse_px
        self._pixel_sigma_px = pixel_sigma_px
        self._lm_initial_lambda = lm_initial_lambda
        self._lm_lambda_factor = lm_lambda_factor
        self._lm_max_attempts = lm_max_attempts
        self._outlier_reprojection_error_px = outlier_reprojection_error_px
        self._outlier_error_ratio = outlier_error_ratio
        self._min_rays_after_outlier_rejection = min_rays_after_outlier_rejection

    def reprojection_errors(
        self, point: NDArray[np.float64], uvs: NDArray[np.float64], poses: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute per-ray Euclidean reprojection errors in pixels."""
        fx, fy = self._stereo_k[0, 0], self._stereo_k[1, 1]
        cx, cy = self._stereo_k[0, 2], self._stereo_k[1, 2]

        cam_from_anchor_rot = np.swapaxes(poses[:, :3, :3], 1, 2)
        p_cam = np.einsum("nij,nj->ni", cam_from_anchor_rot, point[None, :] - poses[:, :3, 3])
        z = p_cam[:, 2]
        positive_depth = z > 0.0

        projected = np.full_like(uvs, np.nan, dtype=np.float64)
        projected[positive_depth, 0] = fx * p_cam[positive_depth, 0] / z[positive_depth] + cx
        projected[positive_depth, 1] = fy * p_cam[positive_depth, 1] / z[positive_depth] + cy

        errors = np.linalg.norm(uvs - projected, axis=1)
        return np.where(np.isfinite(errors), errors, np.inf)

    def worst_reprojection_ray_index(
        self, point: NDArray[np.float64], uvs: NDArray[np.float64], poses: NDArray[np.float64]
    ) -> int | None:
        """Return the isolated worst ray index, if one should be rejected."""
        if uvs.shape[0] <= self._min_rays_after_outlier_rejection:
            return None

        errors = self.reprojection_errors(point, uvs, poses)
        worst_index = int(np.argmax(errors))
        worst_error = float(errors[worst_index])
        ordered_errors = np.sort(errors)
        second_worst_error = float(ordered_errors[-2])

        if not np.isfinite(worst_error):
            return worst_index if np.isfinite(second_worst_error) else None
        if worst_error < self._outlier_reprojection_error_px:
            return None
        if worst_error < second_worst_error * self._outlier_error_ratio:
            return None
        return worst_index

    def refine_point_gn(
        self, initial_guess: NDArray[np.float64], uvs: NDArray[np.float64], poses: NDArray[np.float64]
    ) -> tuple[LandmarkRefineStatus, NDArray[np.float64], NDArray[np.float64]]:
        """Refine an anchor-frame point via damped GN reprojection optimization."""
        p = initial_guess.copy()
        linearization = self._linearize_point(p, uvs, poses)
        if linearization.status != LandmarkRefineStatus.SUCCESS:
            return self._result(linearization.status, p)

        status = self._hessian_status(linearization.h)
        if status != LandmarkRefineStatus.SUCCESS:
            return self._result(status, p)

        for _ in range(self._max_iterations):
            if np.linalg.norm(linearization.g, ord=np.inf) < self._min_gradient_norm:
                status = self._final_status(linearization.total_error, uvs.shape[0], linearization.h)
                return self._result(status, p, linearization.h)

            step = self._try_lm_step(p, linearization, uvs, poses)
            if step.status != LandmarkRefineStatus.SUCCESS:
                return self._result(step.status, p)

            relative_cost_drop = (linearization.total_error - step.linearization.total_error) / max(
                linearization.total_error, 1e-12
            )
            p = step.point
            linearization = step.linearization

            if self._should_stop(step.delta_norm, relative_cost_drop):
                status = self._final_status(linearization.total_error, uvs.shape[0], linearization.h)
                return self._result(status, p, linearization.h)

        status = self._final_status(linearization.total_error, uvs.shape[0], linearization.h)
        return self._result(status, p, linearization.h)

    def _linearize_point(
        self, point: NDArray[np.float64], uvs: NDArray[np.float64], poses: NDArray[np.float64]
    ) -> _Linearization:
        """Linearize reprojection residuals around an anchor-frame point."""
        fx, fy = self._stereo_k[0, 0], self._stereo_k[1, 1]
        cx, cy = self._stereo_k[0, 2], self._stereo_k[1, 2]

        uvs_num = uvs.shape[0]
        h = np.zeros((3, 3), dtype=np.float64)
        g = np.zeros((3,), dtype=np.float64)
        total_error = 0.0

        for i in range(uvs_num):
            anchor_from_cam = poses[i, :, :]
            anchor_from_cam_rot = anchor_from_cam[:3, :3]
            anchor_from_cam_t = anchor_from_cam[:3, 3]
            cam_from_anchor_rot = anchor_from_cam_rot.T

            p_cam = cam_from_anchor_rot @ (point - anchor_from_cam_t)
            x, y, z = p_cam

            if z <= 0:
                return _Linearization(LandmarkRefineStatus.DEPTH_NEGATIVE, h, g, total_error)

            inv_z = 1.0 / z
            inv_z2 = inv_z * inv_z

            u_proj = fx * x * inv_z + cx
            v_proj = fy * y * inv_z + cy

            r = uvs[i, :] - np.array([u_proj, v_proj])
            residual_error = float(np.dot(r, r))
            if not np.isfinite(residual_error):
                return _Linearization(LandmarkRefineStatus.SOLVER_ERROR, h, g, total_error)
            total_error += residual_error

            j_cam = np.array(
                [
                    [fx * inv_z, 0.0, -fx * x * inv_z2],
                    [0.0, fy * inv_z, -fy * y * inv_z2],
                ],
                dtype=np.float64,
            )

            j = j_cam @ cam_from_anchor_rot

            h += j.T @ j
            g += j.T @ r

        return _Linearization(LandmarkRefineStatus.SUCCESS, h, g, total_error)

    def _try_lm_step(
        self,
        point: NDArray[np.float64],
        linearization: _Linearization,
        uvs: NDArray[np.float64],
        poses: NDArray[np.float64],
    ) -> _LmStep:
        """Try LM damped updates until one lowers the reprojection cost."""
        damping = self._lm_initial_lambda
        for _attempt in range(self._lm_max_attempts):
            try:
                delta = np.linalg.solve(self._damped_hessian(linearization.h, damping), linearization.g)
            except np.linalg.LinAlgError:
                damping *= self._lm_lambda_factor
                continue

            candidate_point = point + delta
            candidate = self._linearize_point(candidate_point, uvs, poses)
            if candidate.status != LandmarkRefineStatus.SUCCESS:
                damping *= self._lm_lambda_factor
                continue

            candidate_h_status = self._hessian_status(candidate.h)
            if candidate_h_status != LandmarkRefineStatus.SUCCESS:
                damping *= self._lm_lambda_factor
                continue

            if candidate.total_error <= linearization.total_error:
                return _LmStep(
                    LandmarkRefineStatus.SUCCESS,
                    candidate_point,
                    candidate,
                    float(np.linalg.norm(delta)),
                )

            damping *= self._lm_lambda_factor

        final_status = self._final_status(linearization.total_error, uvs.shape[0], linearization.h)
        status = (
            LandmarkRefineStatus.NO_COST_DECREASE if final_status == LandmarkRefineStatus.SUCCESS else final_status
        )
        return _LmStep(status, point, linearization, np.inf)

    def _should_stop(self, delta_norm: float, relative_cost_drop: float) -> bool:
        """Return whether accepted LM progress is too small to keep iterating."""
        return delta_norm < self._min_delta or relative_cost_drop < self._min_cost_decrease_ratio

    def _hessian_status(self, h: NDArray[np.float64]) -> LandmarkRefineStatus:
        """Check whether the GN Hessian is usable for a 3D landmark update."""
        if not np.all(np.isfinite(h)):
            return LandmarkRefineStatus.SOLVER_ERROR

        eigvals = np.linalg.eigvalsh(0.5 * (h + h.T))
        if not np.all(np.isfinite(eigvals)):
            return LandmarkRefineStatus.SOLVER_ERROR
        if eigvals[-1] <= 0.0 or eigvals[0] <= self._min_hessian_eigenvalue:
            return LandmarkRefineStatus.ILL_CONDITIONED
        if eigvals[-1] / eigvals[0] > self._max_hessian_condition_number:
            return LandmarkRefineStatus.ILL_CONDITIONED
        return LandmarkRefineStatus.SUCCESS

    def _damped_hessian(self, h: NDArray[np.float64], damping: float) -> NDArray[np.float64]:
        """Build the Levenberg-Marquardt damped Hessian."""
        diagonal = np.maximum(np.abs(np.diag(h)), self._min_hessian_eigenvalue)
        return h + damping * np.diag(diagonal)

    def _final_status(self, total_error: float, ray_count: int, h: NDArray[np.float64]) -> LandmarkRefineStatus:
        """Check final point quality after accepted GN/LM updates."""
        hessian_status = self._hessian_status(h)
        if hessian_status != LandmarkRefineStatus.SUCCESS:
            return hessian_status

        rmse_px = np.sqrt(total_error / max(ray_count, 1))
        if not np.isfinite(rmse_px):
            return LandmarkRefineStatus.SOLVER_ERROR
        if rmse_px > self._max_final_reprojection_rmse_px:
            return LandmarkRefineStatus.HIGH_REPROJECTION_ERROR
        return LandmarkRefineStatus.SUCCESS

    def _result(
        self,
        status: LandmarkRefineStatus,
        point: NDArray[np.float64],
        h: NDArray[np.float64] | None = None,
    ) -> tuple[LandmarkRefineStatus, NDArray[np.float64], NDArray[np.float64]]:
        """Build the public refine result and include covariance only for successful estimates."""
        covariance = np.full((3, 3), np.nan, dtype=np.float64)
        if status != LandmarkRefineStatus.SUCCESS or h is None:
            return status, point, covariance

        covariance_status, covariance = self._covariance_from_hessian(h)
        return covariance_status, point, covariance

    def _covariance_from_hessian(self, h: NDArray[np.float64]) -> tuple[LandmarkRefineStatus, NDArray[np.float64]]:
        """Approximate anchor-frame landmark covariance from the final GN Hessian."""
        status = self._hessian_status(h)
        covariance = np.full((3, 3), np.nan, dtype=np.float64)
        if status != LandmarkRefineStatus.SUCCESS:
            return status, covariance

        try:
            covariance = (self._pixel_sigma_px**2) * np.linalg.solve(0.5 * (h + h.T), np.eye(3))
        except np.linalg.LinAlgError:
            return LandmarkRefineStatus.SOLVER_ERROR, np.full((3, 3), np.nan, dtype=np.float64)

        if not np.all(np.isfinite(covariance)):
            return LandmarkRefineStatus.SOLVER_ERROR, np.full((3, 3), np.nan, dtype=np.float64)
        return LandmarkRefineStatus.SUCCESS, 0.5 * (covariance + covariance.T)
