from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from manta_world_model.actor import ActorBelief, ActorCooperativity, ActorAgency, ActorDomain


@dataclass
class TaskContext:
    """Lightweight task context used by the compression library (no ROS2 dependency)."""
    task_id: str
    task_type: str
    target_pose: np.ndarray      # [x, y, z]
    safety_radius: float         # metres
    priority_actor_types: list[str]
    diver_pose: Optional[np.ndarray] = None  # current estimated pose of the focused actor


class ScoringFunction:
    """Base class for all scoring functions.

    score() returns a non-negative float; higher means more worth forwarding.
    known_state maps actor_id -> last communicated belief (for novelty checks).
    """

    def score(
        self,
        belief: "ActorBelief",
        context: TaskContext,
        known_state: dict[str, "ActorBelief"],
    ) -> float:
        raise NotImplementedError


class ProximityScorer(ScoringFunction):
    """Closer to the focused actor → higher score."""

    def __init__(self, max_range: float = 500.0):
        self.max_range = max_range

    def score(self, belief, context, known_state):
        if context.diver_pose is None:
            return 0.0
        dist = belief.distance_to(context.diver_pose)
        return max(0.0, 1.0 - dist / self.max_range)


class ClosingSpeedScorer(ScoringFunction):
    """Faster approach toward the focused actor → higher score."""

    def __init__(self, max_speed: float = 10.0):
        self.max_speed = max_speed

    def score(self, belief, context, known_state):
        if context.diver_pose is None:
            return 0.0
        speed = belief.closing_speed_toward(context.diver_pose)
        return max(0.0, speed / self.max_speed)


class HazardTypeScorer(ScoringFunction):
    """Score based on the three classification axes.

    Adversarial cooperativity applies a multiplier. Passive seabed actors (mines)
    score high if the focused actor is approaching them rather than vice-versa.
    """

    _COOPERATIVITY_MULTIPLIER = {
        "cooperative": 0.5,
        "uncooperative": 1.0,
        "adversarial": 2.0,
    }
    _AGENCY_BASE = {
        "passive": 0.6,
        "biological": 0.7,
        "autonomous": 0.8,
        "human": 0.3,
    }

    def score(self, belief, context, known_state):
        base = self._AGENCY_BASE.get(belief.profile.agency.value, 0.5)
        mult = self._COOPERATIVITY_MULTIPLIER.get(belief.profile.cooperativity.value, 1.0)

        # Mine/explosive: score based on whether diver is closing on it
        if belief.profile.agency.value == "passive" and belief.profile.domain.value == "seabed":
            if context.diver_pose is not None:
                diver_vel_toward = -belief.closing_speed_toward(context.diver_pose)  # positive if diver approaching mine
                mine_proximity = max(0.0, 1.0 - belief.distance_to(context.diver_pose) / context.safety_radius)
                return min(1.0, base * mult * (0.5 + 0.5 * diver_vel_toward + mine_proximity))

        return min(1.0, base * mult)


class TaskRelevanceScorer(ScoringFunction):
    """Score based on whether this actor type appears in the task's priority list,
    and whether the actor's projected position intersects the task path."""

    def score(self, belief, context, known_state):
        label_match = 1.0 if belief.profile.label in context.priority_actor_types else 0.2

        # Spatial check: is the actor near the target of the current task?
        dist_to_target = float(np.linalg.norm(belief.projected_pose - context.target_pose))
        spatial = max(0.0, 1.0 - dist_to_target / max(context.safety_radius * 3, 1.0))

        # Intent check: does the dominant intent suggest interaction?
        danger_intents = {"approaching_diver", "coordinated_approach", "inbound"}
        intent_score = 0.0
        if belief.intent:
            for label, prob in belief.intent.items():
                if label in danger_intents:
                    intent_score = max(intent_score, prob)

        return min(1.0, label_match * (0.4 + 0.3 * spatial + 0.3 * intent_score))


class NoveltyScorer(ScoringFunction):
    """How much has this actor changed since we last communicated it?
    Encourages sending updates when the world has materially changed."""

    def __init__(self, position_threshold: float = 5.0):
        self.position_threshold = position_threshold

    def score(self, belief, context, known_state):
        last = known_state.get(belief.actor_id)
        if last is None:
            return 1.0  # never communicated → always novel

        pos_delta = float(np.linalg.norm(belief.projected_pose - last.projected_pose))
        return min(1.0, pos_delta / self.position_threshold)


class CompositeScorer(ScoringFunction):
    """Weighted sum of multiple scoring functions."""

    def __init__(self, scorers_and_weights: list[tuple[ScoringFunction, float]]):
        self.scorers_and_weights = scorers_and_weights
        total = sum(w for _, w in scorers_and_weights)
        self._norm = total if total > 0 else 1.0

    def score(self, belief, context, known_state):
        total = sum(scorer.score(belief, context, known_state) * w
                    for scorer, w in self.scorers_and_weights)
        return total / self._norm
