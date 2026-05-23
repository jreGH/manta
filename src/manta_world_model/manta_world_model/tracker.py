from __future__ import annotations
import time
from dataclasses import replace
from typing import Optional
import numpy as np

from .actor import ActorBelief, ActorProfile, ActorDomain, ActorAgency, ActorCooperativity
from .motion_models import MotionModel, ConstantVelocityModel, StationaryModel, BallisticModel, AISCourseSpeedModel
from .intent import IntentModule, ClosingBehaviorModule


# Default motion model and intent module selected by actor profile.
def _default_motion_model(profile: ActorProfile) -> MotionModel:
    if profile.agency == ActorAgency.PASSIVE:
        return StationaryModel()
    if profile.agency == ActorAgency.AUTONOMOUS and profile.cooperativity == ActorCooperativity.ADVERSARIAL:
        return BallisticModel()
    if profile.cooperativity == ActorCooperativity.COOPERATIVE:
        return AISCourseSpeedModel()
    return ConstantVelocityModel()


def _default_intent_module(profile: ActorProfile) -> Optional[IntentModule]:
    from .intent import (
        SharkIntentModule, AdversarialBioIntentModule, VesselIntentModule,
        DiverIntentModule, AerialIntentModule, BallisticIntentModule, ClosingBehaviorModule,
    )
    if profile.agency == ActorAgency.PASSIVE:
        return None
    if profile.agency == ActorAgency.BIOLOGICAL:
        if profile.cooperativity == ActorCooperativity.ADVERSARIAL:
            return AdversarialBioIntentModule()
        return SharkIntentModule()
    if profile.domain == ActorDomain.AERIAL:
        return AerialIntentModule()
    if profile.agency == ActorAgency.AUTONOMOUS and profile.cooperativity == ActorCooperativity.ADVERSARIAL:
        return BallisticIntentModule()
    if profile.domain == ActorDomain.SURFACE and profile.agency == ActorAgency.HUMAN:
        return VesselIntentModule()
    if profile.agency == ActorAgency.HUMAN and profile.cooperativity == ActorCooperativity.COOPERATIVE:
        return DiverIntentModule()
    return ClosingBehaviorModule()


class MultiActorTracker:
    """Maintains a belief state for every tracked actor.

    Designed as a pure-Python library. The ROS2 node is a thin wrapper
    that calls update() on incoming observations and project_all() on a timer.
    """

    def __init__(
        self,
        staleness_threshold: float = 30.0,
        process_noise: float = 0.1,
        motion_model_overrides: Optional[dict[str, MotionModel]] = None,
        intent_module_overrides: Optional[dict[str, IntentModule]] = None,
    ):
        self.staleness_threshold = staleness_threshold
        self.process_noise = process_noise
        self._motion_overrides: dict[str, MotionModel] = motion_model_overrides or {}
        self._intent_overrides: dict[str, IntentModule] = intent_module_overrides or {}
        self._beliefs: dict[str, ActorBelief] = {}
        self._motion_models: dict[str, MotionModel] = {}
        self._intent_modules: dict[str, Optional[IntentModule]] = {}

    def update(self, actor_id: str, profile: ActorProfile, pose: np.ndarray,
               velocity: np.ndarray, covariance: np.ndarray, observed_at: float) -> None:
        """Merge a new observation into the belief state for actor_id."""
        if actor_id not in self._beliefs:
            self._motion_models[actor_id] = self._motion_overrides.get(
                actor_id, _default_motion_model(profile)
            )
            self._intent_modules[actor_id] = self._intent_overrides.get(
                actor_id, _default_intent_module(profile)
            )

        self._beliefs[actor_id] = ActorBelief(
            actor_id=actor_id,
            profile=profile,
            pose=pose.copy(),
            covariance=covariance.copy(),
            velocity=velocity.copy(),
            observed_at=observed_at,
            projected_pose=pose.copy(),
            projected_covariance=covariance.copy(),
            staleness_seconds=0.0,
            confidence=1.0,
            intent=self._beliefs[actor_id].intent if actor_id in self._beliefs else {},
        )

    def project_all(self, now: Optional[float] = None) -> list[ActorBelief]:
        """Forward-project every actor to the current time and run intent inference."""
        if now is None:
            now = time.time()

        beliefs = list(self._beliefs.values())
        projected: list[ActorBelief] = []

        for belief in beliefs:
            dt = max(0.0, now - belief.observed_at)
            model = self._motion_models.get(belief.actor_id, ConstantVelocityModel(self.process_noise))
            updated = model.project(belief, dt)

            intent_mod = self._intent_modules.get(belief.actor_id)
            if intent_mod is not None:
                intent = intent_mod.infer(updated, beliefs)
                updated = replace(updated, intent=intent)

            confidence = max(0.0, 1.0 - updated.staleness_seconds / self.staleness_threshold)
            updated = replace(updated, confidence=confidence)

            self._beliefs[belief.actor_id] = updated
            projected.append(updated)

        return projected

    def remove(self, actor_id: str) -> None:
        self._beliefs.pop(actor_id, None)
        self._motion_models.pop(actor_id, None)
        self._intent_modules.pop(actor_id, None)

    def mark_stale_removed(self) -> list[str]:
        """Remove actors exceeding staleness threshold. Returns removed IDs."""
        to_remove = [
            aid for aid, b in self._beliefs.items()
            if b.is_stale(self.staleness_threshold)
        ]
        for aid in to_remove:
            self.remove(aid)
        return to_remove

    @property
    def actor_ids(self) -> list[str]:
        return list(self._beliefs.keys())

    def get(self, actor_id: str) -> Optional[ActorBelief]:
        return self._beliefs.get(actor_id)
