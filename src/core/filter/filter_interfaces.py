from dataclasses import dataclass


@dataclass
class PredictNoise:
    """Noise of the Multi-State Constraint Kalman Filter."""

    ng: float
    na: float
    nba: float
    nbg: float
