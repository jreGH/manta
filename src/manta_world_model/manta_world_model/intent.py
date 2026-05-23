from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .actor import ActorBelief


class IntentModule:
    """Base class for intent inference.

    Returns a probability distribution over intent labels given a belief and
    the full current world state. All implementations are rule-based;
    the base class defines the interface for future ML backends.
    """

    def infer(self, belief: "ActorBelief", world_state: list["ActorBelief"]) -> dict[str, float]:
        raise NotImplementedError


def _find_nearest_human(world_state: list["ActorBelief"]) -> "ActorBelief | None":
    from .actor import ActorAgency
    humans = [b for b in world_state if b.profile.agency == ActorAgency.HUMAN]
    if not humans:
        return None
    return min(humans, key=lambda b: np.linalg.norm(b.projected_pose))


class SharkIntentModule(IntentModule):
    """Closing speed on nearest human above threshold → approaching_diver; else foraging."""

    def __init__(self, closing_speed_threshold: float = 0.5):
        self.threshold = closing_speed_threshold

    def infer(self, belief: "ActorBelief", world_state: list["ActorBelief"]) -> dict[str, float]:
        nearest = _find_nearest_human(world_state)
        if nearest is None:
            return {"foraging": 1.0}
        speed = belief.closing_speed_toward(nearest.projected_pose)
        if speed > self.threshold:
            return {"approaching_diver": 0.85, "foraging": 0.15}
        return {"foraging": 0.9, "approaching_diver": 0.1}


class AdversarialBioIntentModule(IntentModule):
    """Like SharkIntentModule but adds coordinated_approach when multiple
    biological adversarial actors converge on the same target."""

    def __init__(self, closing_speed_threshold: float = 0.3, convergence_count: int = 2):
        self.threshold = closing_speed_threshold
        self.convergence_count = convergence_count

    def infer(self, belief: "ActorBelief", world_state: list["ActorBelief"]) -> dict[str, float]:
        from .actor import ActorAgency, ActorCooperativity
        nearest = _find_nearest_human(world_state)
        if nearest is None:
            return {"foraging": 1.0}

        speed = belief.closing_speed_toward(nearest.projected_pose)
        closing = speed > self.threshold

        convergers = sum(
            1 for b in world_state
            if b.actor_id != belief.actor_id
            and b.profile.agency == ActorAgency.BIOLOGICAL
            and b.profile.cooperativity == ActorCooperativity.ADVERSARIAL
            and b.closing_speed_toward(nearest.projected_pose) > self.threshold
        )

        if closing and convergers >= self.convergence_count - 1:
            return {"coordinated_approach": 0.9, "approaching_diver": 0.1}
        if closing:
            return {"approaching_diver": 0.8, "foraging": 0.2}
        return {"foraging": 0.85, "approaching_diver": 0.15}


class VesselIntentModule(IntentModule):
    """Project AIS course/speed; flag transiting_area if it intersects the
    diver's general vicinity within a time horizon."""

    def __init__(self, area_radius: float = 200.0, time_horizon: float = 600.0):
        # area_radius: metres around the origin considered the operation area
        # time_horizon: seconds to look ahead
        self.area_radius = area_radius
        self.time_horizon = time_horizon

    def infer(self, belief: "ActorBelief", world_state: list["ActorBelief"]) -> dict[str, float]:
        future_pos = belief.projected_pose + belief.velocity * self.time_horizon
        dist_future = float(np.linalg.norm(future_pos[:2]))  # 2D surface distance
        if dist_future < self.area_radius:
            return {"transiting_area": 0.9, "clear": 0.1}
        return {"clear": 0.95, "transiting_area": 0.05}


class DiverIntentModule(IntentModule):
    """Projects the diver along their declared task path using task context
    stored on the belief. Expects belief.intent to be pre-populated by the
    task context subscriber in the node; this module refreshes it."""

    def infer(self, belief: "ActorBelief", world_state: list["ActorBelief"]) -> dict[str, float]:
        # Task context is encoded in the belief's existing intent by the node layer.
        # If already populated, pass it through; otherwise assume on-task.
        if belief.intent:
            return belief.intent
        return {"on_task": 1.0}


class AerialIntentModule(IntentModule):
    """Altitude + heading heuristics for aerial actors."""

    def __init__(self, descent_speed_threshold: float = -0.5, loiter_speed_threshold: float = 2.0):
        self.descent_threshold = descent_speed_threshold
        self.loiter_threshold = loiter_speed_threshold

    def infer(self, belief: "ActorBelief", world_state: list["ActorBelief"]) -> dict[str, float]:
        vz = belief.velocity[2]
        horiz_speed = float(np.linalg.norm(belief.velocity[:2]))

        if vz < self.descent_threshold:
            return {"approaching": 0.85, "transiting": 0.15}
        if horiz_speed < self.loiter_threshold:
            return {"surveilling": 0.8, "approaching": 0.2}
        return {"transiting": 0.9, "surveilling": 0.1}


class BallisticIntentModule(IntentModule):
    """High-confidence straight-line projection. Inbound if intercept geometry
    with the nearest human holds within a time horizon."""

    def __init__(self, time_horizon: float = 60.0, intercept_radius: float = 20.0):
        self.time_horizon = time_horizon
        self.intercept_radius = intercept_radius

    def infer(self, belief: "ActorBelief", world_state: list["ActorBelief"]) -> dict[str, float]:
        nearest = _find_nearest_human(world_state)
        if nearest is None:
            return {"inbound": 0.5, "clear": 0.5}
        future_pos = belief.projected_pose + belief.velocity * self.time_horizon
        miss_dist = float(np.linalg.norm(future_pos - nearest.projected_pose))
        if miss_dist < self.intercept_radius:
            return {"inbound": 0.95, "clear": 0.05}
        return {"clear": 0.9, "inbound": 0.1}


class ClosingBehaviorModule(IntentModule):
    """Generic kinematic fallback for any uncooperative actor."""

    def __init__(self, closing_threshold: float = 0.2, stationary_threshold: float = 0.05):
        self.closing_threshold = closing_threshold
        self.stationary_threshold = stationary_threshold

    def infer(self, belief: "ActorBelief", world_state: list["ActorBelief"]) -> dict[str, float]:
        nearest = _find_nearest_human(world_state)
        speed = float(np.linalg.norm(belief.velocity))
        if speed < self.stationary_threshold:
            return {"stationary": 0.95, "closing": 0.05}
        if nearest is None:
            return {"closing": 0.5, "departing": 0.5}
        closing = belief.closing_speed_toward(nearest.projected_pose)
        if closing > self.closing_threshold:
            return {"closing": 0.85, "departing": 0.15}
        return {"departing": 0.8, "closing": 0.2}
