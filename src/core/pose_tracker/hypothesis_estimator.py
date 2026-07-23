from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

Vector3d = NDArray[np.float64]
Matrix3d = NDArray[np.float64]
CANDIDATE_HISTORY_NDIM = 2


class CandidateHistorySchemaLike(Protocol):
    """Column contract required by the hypothesis estimator."""

    FEAT_ID: ClassVar[int]
    TIMESTAMP_NS: ClassVar[int]
    XYZ: ClassVar[slice]
    COV: ClassVar[slice]
    DEPTH_SIGMA: ClassVar[int]
    CAM_XYZ: ClassVar[slice]


class CandidateHypothesisStatus(IntEnum):
    """Status of a hypothesis estimated from a frontend candidate history."""

    EMPTY = 0
    PROVISIONAL = 1
    WEAK_STABLE = 2
    STABLE = 3
    AMBIGUOUS = 4
    SCATTERED = 5


@dataclass(slots=True, frozen=True)
class HypothesisEstimatorConfig:
    """Thresholds used to interpret a short candidate-observation history."""

    min_pnp_observations: int = 1
    weak_min_inliers: int = 3
    stable_min_inliers: int = 4
    min_mature_observations: int = 5
    stable_min_inlier_ratio: float = 0.8

    mahalanobis_threshold: float = 11.34
    ambiguity_min_cluster_size: int = 2
    ambiguity_cluster_ratio: float = 0.8
    ambiguity_coverage_ratio: float = 0.8
    ambiguity_min_separation_m: float = 0.25

    min_covariance_diag: float = 1e-6
    covariance_regularization: float = 1e-9
    provisional_covariance_inflation: float = 1.5
    weak_covariance_inflation: float = 2.0
    ambiguous_covariance_inflation: float = 4.0
    scattered_covariance_inflation: float = 8.0


@dataclass(slots=True, frozen=True)
class CandidateHistoryDescription:
    """Compact description of the measurements stored for one feature candidate."""

    feature_id: int | None
    observation_count: int
    latest_timestamp_ns: float
    latest_xyz: Vector3d
    xyz_median: Vector3d
    xyz_span_m: float
    cam_xyz_span_m: float
    depth_sigma_median: float


@dataclass(slots=True, frozen=True)
class CandidateHypothesis:
    """Estimated position hypothesis for one frontend candidate."""

    status: CandidateHypothesisStatus
    description: CandidateHistoryDescription
    xyz: Vector3d
    covariance: Matrix3d
    depth_sigma: float
    inlier_mask: NDArray[np.bool_]
    inlier_count: int
    observation_count: int
    inlier_ratio: float
    health_delta: float
    pnp_eligible: bool
    promote_to_mature: bool
    reason: str

    @property
    def covariance_row_major(self) -> NDArray[np.float64]:
        """Return the 3x3 covariance as row-major schema columns."""
        return self.covariance.reshape(9).astype(np.float64, copy=False)


@dataclass(slots=True, frozen=True)
class _Cluster:
    mask: NDArray[np.bool_]
    size: int
    latest_timestamp_ns: float
    xyz: Vector3d
    covariance: Matrix3d
    depth_sigma: float


class HypothesisEstimator:
    """Estimate a robust landmark hypothesis from a short candidate history."""

    def __init__(
        self,
        schema: type[CandidateHistorySchemaLike],
        config: HypothesisEstimatorConfig | None = None,
    ) -> None:
        """Create a hypothesis estimator for a concrete candidate-history schema."""
        self.schema = schema
        self.config = config or HypothesisEstimatorConfig()

    def describe(self, history: NDArray[np.float64]) -> CandidateHistoryDescription:
        """Describe finite observations from a candidate history."""
        history = self._as_history(history)
        valid_history, _valid_indices = self._valid_history(history)
        if valid_history.shape[0] == 0:
            return self._empty_description()
        return self._describe_valid_history(valid_history)

    def estimate(self, history: NDArray[np.float64]) -> CandidateHypothesis:
        """Estimate the best current hypothesis for a candidate landmark."""
        history = self._as_history(history)
        valid_history, valid_indices = self._valid_history(history)
        if valid_history.shape[0] == 0:
            return self._empty_hypothesis(history.shape[0])

        description = self._describe_valid_history(valid_history)
        xyz = valid_history[:, self.schema.XYZ]
        covariances = self._covariance_matrices(valid_history[:, self.schema.COV])
        clusters = self._clusters(valid_history, xyz, covariances)

        dominant_cluster = clusters[0]
        secondary_cluster = clusters[1] if len(clusters) > 1 else None
        status = self._status_from_clusters(valid_history.shape[0], dominant_cluster, secondary_cluster)
        selected_cluster = dominant_cluster

        hypothesis_covariance = self._inflate_covariance_for_status(selected_cluster.covariance, status)
        inlier_mask = self._project_valid_indices_to_history(
            history.shape[0],
            valid_indices,
            selected_cluster.mask,
        )
        inlier_ratio = selected_cluster.size / valid_history.shape[0]
        pnp_eligible = self._pnp_eligible(status, selected_cluster.size)
        promote_to_mature = self._promote_to_mature(status, selected_cluster.size, valid_history.shape[0])

        return CandidateHypothesis(
            status=status,
            description=description,
            xyz=selected_cluster.xyz,
            covariance=hypothesis_covariance,
            depth_sigma=self._depth_sigma_from_covariance(hypothesis_covariance),
            inlier_mask=inlier_mask,
            inlier_count=selected_cluster.size,
            observation_count=valid_history.shape[0],
            inlier_ratio=inlier_ratio,
            health_delta=self._health_delta(status),
            pnp_eligible=pnp_eligible,
            promote_to_mature=promote_to_mature,
            reason=self._reason(status, selected_cluster.size, valid_history.shape[0], secondary_cluster),
        )

    def _as_history(self, history: NDArray[np.float64]) -> NDArray[np.float64]:
        history = np.asarray(history, dtype=np.float64)
        if history.ndim != CANDIDATE_HISTORY_NDIM:
            msg = "Candidate history must be a 2D ndarray"
            raise ValueError(msg)

        min_col_count = max(
            self.schema.FEAT_ID + 1,
            self.schema.TIMESTAMP_NS + 1,
            _slice_stop(self.schema.XYZ),
            _slice_stop(self.schema.COV),
            self.schema.DEPTH_SIGMA + 1,
            _slice_stop(self.schema.CAM_XYZ),
        )
        if history.shape[1] < min_col_count:
            msg = f"Candidate history must contain at least {min_col_count} columns"
            raise ValueError(msg)
        return history

    def _valid_history(
        self,
        history: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        covariance = history[:, self.schema.COV]
        covariance_diag = covariance[:, [0, 4, 8]]
        valid_mask = (
            np.isfinite(history[:, self.schema.FEAT_ID])
            & np.isfinite(history[:, self.schema.TIMESTAMP_NS])
            & np.all(np.isfinite(history[:, self.schema.XYZ]), axis=1)
            & np.all(np.isfinite(covariance), axis=1)
            & np.all(covariance_diag > 0.0, axis=1)
            & np.isfinite(history[:, self.schema.DEPTH_SIGMA])
        )
        valid_indices = np.flatnonzero(valid_mask)
        valid_history = history[valid_indices]
        if valid_history.shape[0] <= 1:
            return valid_history, valid_indices

        order = np.argsort(valid_history[:, self.schema.TIMESTAMP_NS], kind="stable")
        return valid_history[order], valid_indices[order]

    def _describe_valid_history(self, history: NDArray[np.float64]) -> CandidateHistoryDescription:
        xyz = history[:, self.schema.XYZ]
        cam_xyz = history[:, self.schema.CAM_XYZ]
        latest = history[-1]
        return CandidateHistoryDescription(
            feature_id=int(latest[self.schema.FEAT_ID]),
            observation_count=history.shape[0],
            latest_timestamp_ns=float(latest[self.schema.TIMESTAMP_NS]),
            latest_xyz=latest[self.schema.XYZ].copy(),
            xyz_median=np.median(xyz, axis=0),
            xyz_span_m=_max_pairwise_distance(xyz),
            cam_xyz_span_m=_max_pairwise_distance(cam_xyz),
            depth_sigma_median=float(np.median(history[:, self.schema.DEPTH_SIGMA])),
        )

    def _clusters(
        self,
        history: NDArray[np.float64],
        xyz: NDArray[np.float64],
        covariances: NDArray[np.float64],
    ) -> list[_Cluster]:
        adjacency = self._mahalanobis_adjacency(xyz, covariances)
        masks = _connected_components(adjacency)
        clusters = [self._build_cluster(mask, history, xyz, covariances) for mask in masks]
        return sorted(clusters, key=lambda cluster: (-cluster.size, cluster.latest_timestamp_ns))

    def _mahalanobis_adjacency(
        self,
        xyz: NDArray[np.float64],
        covariances: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        count = xyz.shape[0]
        adjacency = np.eye(count, dtype=np.bool_)
        regularizer = self.config.covariance_regularization * np.eye(3, dtype=np.float64)
        for first_idx in range(count):
            for second_idx in range(first_idx + 1, count):
                delta = xyz[first_idx] - xyz[second_idx]
                joint_covariance = covariances[first_idx] + covariances[second_idx] + regularizer
                mahalanobis_d2 = float(delta.T @ np.linalg.pinv(joint_covariance) @ delta)
                if mahalanobis_d2 <= self.config.mahalanobis_threshold:
                    adjacency[first_idx, second_idx] = True
                    adjacency[second_idx, first_idx] = True
        return adjacency

    def _build_cluster(
        self,
        mask: NDArray[np.bool_],
        history: NDArray[np.float64],
        xyz: NDArray[np.float64],
        covariances: NDArray[np.float64],
    ) -> _Cluster:
        cluster_history = history[mask]
        cluster_xyz = xyz[mask]
        cluster_covariances = covariances[mask]
        center = _weighted_mean(cluster_xyz, cluster_covariances)
        covariance = self._estimate_cluster_covariance(cluster_xyz, cluster_covariances, center)
        return _Cluster(
            mask=mask,
            size=int(np.count_nonzero(mask)),
            latest_timestamp_ns=float(np.max(cluster_history[:, self.schema.TIMESTAMP_NS])),
            xyz=center,
            covariance=covariance,
            depth_sigma=self._depth_sigma_from_covariance(covariance),
        )

    def _estimate_cluster_covariance(
        self,
        xyz: NDArray[np.float64],
        covariances: NDArray[np.float64],
        center: Vector3d,
    ) -> Matrix3d:
        weights = _inverse_trace_weights(covariances)
        weighted_covariance = np.average(covariances, axis=0, weights=weights) / xyz.shape[0]
        residuals = xyz - center
        empirical_covariance = np.einsum("n,ni,nj->ij", weights, residuals, residuals)
        return self._regularize_covariance(weighted_covariance + empirical_covariance)

    def _status_from_clusters(
        self,
        observation_count: int,
        dominant_cluster: _Cluster,
        secondary_cluster: _Cluster | None,
    ) -> CandidateHypothesisStatus:
        if self._is_ambiguous(observation_count, dominant_cluster, secondary_cluster):
            return CandidateHypothesisStatus.AMBIGUOUS

        inlier_ratio = dominant_cluster.size / observation_count
        if (
            dominant_cluster.size >= self.config.stable_min_inliers
            and observation_count >= self.config.min_mature_observations
            and inlier_ratio >= self.config.stable_min_inlier_ratio
        ):
            return CandidateHypothesisStatus.STABLE
        if dominant_cluster.size >= self.config.weak_min_inliers:
            return CandidateHypothesisStatus.WEAK_STABLE
        if observation_count > 1 and dominant_cluster.size == 1:
            return CandidateHypothesisStatus.SCATTERED
        if observation_count < self.config.min_mature_observations:
            return CandidateHypothesisStatus.PROVISIONAL
        return CandidateHypothesisStatus.SCATTERED

    def _is_ambiguous(
        self,
        observation_count: int,
        dominant_cluster: _Cluster,
        secondary_cluster: _Cluster | None,
    ) -> bool:
        if secondary_cluster is None:
            return False
        if secondary_cluster.size < self.config.ambiguity_min_cluster_size:
            return False
        cluster_ratio = secondary_cluster.size / dominant_cluster.size
        coverage_ratio = (dominant_cluster.size + secondary_cluster.size) / observation_count
        separation_m = float(np.linalg.norm(dominant_cluster.xyz - secondary_cluster.xyz))
        return (
            cluster_ratio >= self.config.ambiguity_cluster_ratio
            and coverage_ratio >= self.config.ambiguity_coverage_ratio
            and separation_m >= self.config.ambiguity_min_separation_m
        )

    def _inflate_covariance_for_status(
        self,
        covariance: Matrix3d,
        status: CandidateHypothesisStatus,
    ) -> Matrix3d:
        inflation = {
            CandidateHypothesisStatus.EMPTY: np.nan,
            CandidateHypothesisStatus.PROVISIONAL: self.config.provisional_covariance_inflation,
            CandidateHypothesisStatus.WEAK_STABLE: self.config.weak_covariance_inflation,
            CandidateHypothesisStatus.STABLE: 1.0,
            CandidateHypothesisStatus.AMBIGUOUS: self.config.ambiguous_covariance_inflation,
            CandidateHypothesisStatus.SCATTERED: self.config.scattered_covariance_inflation,
        }[status]
        return self._regularize_covariance(covariance * inflation)

    def _covariance_matrices(self, covariance_rows: NDArray[np.float64]) -> NDArray[np.float64]:
        covariance_matrices = covariance_rows.reshape(-1, 3, 3)
        return np.stack([self._regularize_covariance(covariance) for covariance in covariance_matrices])

    def _regularize_covariance(self, covariance: Matrix3d) -> Matrix3d:
        covariance = 0.5 * (covariance + covariance.T)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.maximum(eigvals, self.config.min_covariance_diag)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _depth_sigma_from_covariance(self, covariance: Matrix3d) -> float:
        return float(np.sqrt(max(covariance[2, 2], self.config.min_covariance_diag)))

    def _pnp_eligible(self, status: CandidateHypothesisStatus, inlier_count: int) -> bool:
        return (
            status
            in {
                CandidateHypothesisStatus.PROVISIONAL,
                CandidateHypothesisStatus.WEAK_STABLE,
                CandidateHypothesisStatus.STABLE,
            }
            and inlier_count >= self.config.min_pnp_observations
        )

    def _promote_to_mature(
        self,
        status: CandidateHypothesisStatus,
        inlier_count: int,
        observation_count: int,
    ) -> bool:
        return (
            status == CandidateHypothesisStatus.STABLE
            and inlier_count >= self.config.stable_min_inliers
            and observation_count >= self.config.min_mature_observations
        )

    @staticmethod
    def _health_delta(status: CandidateHypothesisStatus) -> float:
        return {
            CandidateHypothesisStatus.EMPTY: 0.0,
            CandidateHypothesisStatus.PROVISIONAL: 0.0,
            CandidateHypothesisStatus.WEAK_STABLE: 0.25,
            CandidateHypothesisStatus.STABLE: 1.0,
            CandidateHypothesisStatus.AMBIGUOUS: -0.5,
            CandidateHypothesisStatus.SCATTERED: -1.0,
        }[status]

    @staticmethod
    def _reason(
        status: CandidateHypothesisStatus,
        inlier_count: int,
        observation_count: int,
        secondary_cluster: _Cluster | None,
    ) -> str:
        if status == CandidateHypothesisStatus.AMBIGUOUS and secondary_cluster is not None:
            return (
                f"ambiguous candidate history: {inlier_count} dominant inliers and "
                f"{secondary_cluster.size} competing inliers"
            )
        return f"{status.name.lower()} candidate history: {inlier_count}/{observation_count} inliers"

    def _empty_hypothesis(self, history_size: int) -> CandidateHypothesis:
        covariance = np.full((3, 3), np.nan, dtype=np.float64)
        return CandidateHypothesis(
            status=CandidateHypothesisStatus.EMPTY,
            description=self._empty_description(),
            xyz=np.full(3, np.nan, dtype=np.float64),
            covariance=covariance,
            depth_sigma=np.nan,
            inlier_mask=np.zeros(history_size, dtype=np.bool_),
            inlier_count=0,
            observation_count=0,
            inlier_ratio=0.0,
            health_delta=0.0,
            pnp_eligible=False,
            promote_to_mature=False,
            reason="empty candidate history",
        )

    @staticmethod
    def _empty_description() -> CandidateHistoryDescription:
        return CandidateHistoryDescription(
            feature_id=None,
            observation_count=0,
            latest_timestamp_ns=np.nan,
            latest_xyz=np.full(3, np.nan, dtype=np.float64),
            xyz_median=np.full(3, np.nan, dtype=np.float64),
            xyz_span_m=np.nan,
            cam_xyz_span_m=np.nan,
            depth_sigma_median=np.nan,
        )

    @staticmethod
    def _project_valid_indices_to_history(
        history_size: int,
        valid_indices: NDArray[np.int64],
        cluster_mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        inlier_mask = np.zeros(history_size, dtype=np.bool_)
        inlier_mask[valid_indices[cluster_mask]] = True
        return inlier_mask


def _slice_stop(value: slice) -> int:
    if value.stop is None:
        msg = "Schema slices must have a finite stop"
        raise ValueError(msg)
    return value.stop


def _max_pairwise_distance(points: NDArray[np.float64]) -> float:
    if points.shape[0] <= 1 or not np.all(np.isfinite(points)):
        return 0.0
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    return float(np.max(distances))


def _connected_components(adjacency: NDArray[np.bool_]) -> list[NDArray[np.bool_]]:
    visited = np.zeros(adjacency.shape[0], dtype=np.bool_)
    components: list[NDArray[np.bool_]] = []
    for start_idx in range(adjacency.shape[0]):
        if visited[start_idx]:
            continue
        component = np.zeros(adjacency.shape[0], dtype=np.bool_)
        stack = [start_idx]
        visited[start_idx] = True
        while stack:
            current_idx = stack.pop()
            component[current_idx] = True
            for next_idx in np.flatnonzero(adjacency[current_idx]):
                if visited[next_idx]:
                    continue
                visited[next_idx] = True
                stack.append(int(next_idx))
        components.append(component)
    return components


def _inverse_trace_weights(covariances: NDArray[np.float64]) -> NDArray[np.float64]:
    traces = np.trace(covariances, axis1=1, axis2=2)
    weights = 1.0 / np.maximum(traces, np.finfo(np.float64).eps)
    return weights / np.sum(weights)


def _weighted_mean(xyz: NDArray[np.float64], covariances: NDArray[np.float64]) -> Vector3d:
    weights = _inverse_trace_weights(covariances)
    return np.average(xyz, axis=0, weights=weights)
