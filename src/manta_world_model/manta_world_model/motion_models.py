from __future__ import annotations
from dataclasses import replace
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .actor import ActorBelief


class MotionModel:
    def project(self, belief: "ActorBelief", dt: float) -> "ActorBelief":
        raise NotImplementedError


class StationaryModel(MotionModel):
    """For mines, explosives, fixed infrastructure. Covariance stays fixed."""

    def project(self, belief: "ActorBelief", dt: float) -> "ActorBelief":
        return replace(
            belief,
            projected_pose=belief.pose.copy(),
            projected_covariance=belief.covariance.copy(),
            staleness_seconds=belief.staleness_seconds + dt,
        )


class ConstantVelocityModel(MotionModel):
    """Default for uncooperative unknowns. Covariance grows linearly with dt."""

    def __init__(self, process_noise: float = 0.1):
        # process_noise: m²/s added to each diagonal of position covariance per second
        self.process_noise = process_noise

    def project(self, belief: "ActorBelief", dt: float) -> "ActorBelief":
        projected = belief.pose + belief.velocity * dt
        cov = belief.covariance + np.eye(3) * self.process_noise * dt
        return replace(
            belief,
            projected_pose=projected,
            projected_covariance=cov,
            staleness_seconds=belief.staleness_seconds + dt,
        )


class BallisticModel(MotionModel):
    """Torpedoes and weapons: straight-line high-speed, covariance grows slowly."""

    def __init__(self, process_noise: float = 0.01):
        self.process_noise = process_noise

    def project(self, belief: "ActorBelief", dt: float) -> "ActorBelief":
        projected = belief.pose + belief.velocity * dt
        cov = belief.covariance + np.eye(3) * self.process_noise * dt
        return replace(
            belief,
            projected_pose=projected,
            projected_covariance=cov,
            staleness_seconds=belief.staleness_seconds + dt,
        )


class AISCourseSpeedModel(MotionModel):
    """Cooperative surface/aerial assets with known course and speed (AIS, ADS-B).
    Treats velocity as authoritative; process noise is low."""

    def __init__(self, process_noise: float = 0.05):
        self.process_noise = process_noise

    def project(self, belief: "ActorBelief", dt: float) -> "ActorBelief":
        projected = belief.pose + belief.velocity * dt
        cov = belief.covariance + np.eye(3) * self.process_noise * dt
        return replace(
            belief,
            projected_pose=projected,
            projected_covariance=cov,
            staleness_seconds=belief.staleness_seconds + dt,
        )


class IntentConditionedModel(MotionModel):
    """Wraps a base motion model and modulates the projected velocity based on
    the highest-probability intent label. The intent module must have already
    populated belief.intent before this is called."""

    _INTENT_VELOCITY_SCALES: dict[str, float] = {
        "approaching_diver": 1.0,
        "coordinated_approach": 1.2,
        "inbound": 1.0,
        "foraging": 0.4,
        "stationary": 0.0,
        "departing": 1.0,
        "transiting_area": 1.0,
        "surveilling": 0.2,
        "clear": 1.0,
    }

    def __init__(self, base_model: MotionModel):
        self.base = base_model

    def project(self, belief: "ActorBelief", dt: float) -> "ActorBelief":
        if belief.intent:
            top_label = max(belief.intent, key=belief.intent.get)
            scale = self._INTENT_VELOCITY_SCALES.get(top_label, 1.0)
            modulated = replace(belief, velocity=belief.velocity * scale)
        else:
            modulated = belief
        return self.base.project(modulated, dt)
