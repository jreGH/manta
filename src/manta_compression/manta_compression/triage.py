from __future__ import annotations
from typing import TYPE_CHECKING

from .scorers import ScoringFunction, TaskContext, CompositeScorer, ProximityScorer, ClosingSpeedScorer, HazardTypeScorer, TaskRelevanceScorer, NoveltyScorer

if TYPE_CHECKING:
    from manta_world_model.actor import ActorBelief

# Priority constants matching DiverAlert.msg
PRIORITY_INFO = 0
PRIORITY_WARNING = 1
PRIORITY_SAFETY_CRITICAL = 2

_SAFETY_CRITICAL_INTENTS = {"approaching_diver", "coordinated_approach", "inbound"}


def _actor_priority(belief: "ActorBelief") -> int:
    if not belief.intent:
        return PRIORITY_INFO
    top_label = max(belief.intent, key=belief.intent.get)
    top_prob = belief.intent[top_label]
    if top_label in _SAFETY_CRITICAL_INTENTS and top_prob > 0.6:
        return PRIORITY_SAFETY_CRITICAL
    if top_prob > 0.5:
        return PRIORITY_WARNING
    return PRIORITY_INFO


class BandwidthBudget:
    def __init__(self, max_chars: int = 160):
        self.max_chars = max_chars
        self._used = 0

    def fits(self, text: str) -> bool:
        return self._used + len(text) <= self.max_chars

    def consume(self, text: str) -> None:
        self._used += len(text)

    @property
    def used(self) -> int:
        return self._used

    @property
    def fraction(self) -> float:
        return self._used / self.max_chars if self.max_chars > 0 else 0.0

    def reset(self) -> None:
        self._used = 0


def default_scorer(weights: dict[str, float] | None = None) -> CompositeScorer:
    w = weights or {
        "proximity": 1.5,
        "closing_speed": 2.0,
        "hazard_type": 1.0,
        "task_relevance": 1.8,
        "novelty": 0.8,
    }
    return CompositeScorer([
        (ProximityScorer(), w.get("proximity", 1.5)),
        (ClosingSpeedScorer(), w.get("closing_speed", 2.0)),
        (HazardTypeScorer(), w.get("hazard_type", 1.0)),
        (TaskRelevanceScorer(), w.get("task_relevance", 1.8)),
        (NoveltyScorer(), w.get("novelty", 0.8)),
    ])


class Triage:
    """Scores every actor in a world state and returns a priority-sorted list.

    safety_critical_override: if True, PRIORITY_SAFETY_CRITICAL actors are always
    included regardless of bandwidth budget (they will displace lower-priority items).
    """

    def __init__(
        self,
        scorer: ScoringFunction | None = None,
        safety_critical_override: bool = True,
        scorer_weights: dict[str, float] | None = None,
    ):
        self.scorer = scorer or default_scorer(scorer_weights)
        self.safety_critical_override = safety_critical_override

    def rank(
        self,
        actors: list["ActorBelief"],
        context: TaskContext,
        known_state: dict[str, "ActorBelief"],
    ) -> list[tuple["ActorBelief", float, int]]:
        """Returns list of (belief, score, priority) sorted descending by priority then score."""
        results = []
        for belief in actors:
            score = self.scorer.score(belief, context, known_state)
            priority = _actor_priority(belief)
            results.append((belief, score, priority))

        results.sort(key=lambda t: (-(t[2]), -(t[1])))
        return results

    def select_within_budget(
        self,
        ranked: list[tuple["ActorBelief", float, int]],
        budget: BandwidthBudget,
        formatter: "AlertFormatter",  # type: ignore[name-defined]
    ) -> list[tuple["ActorBelief", int]]:
        """Select actors that fit in the budget. Safety-critical actors bypass budget."""
        selected: list[tuple["ActorBelief", int]] = []
        deferred: list[tuple["ActorBelief", float, int]] = []

        for belief, score, priority in ranked:
            text = formatter.format_single(belief)
            if priority == PRIORITY_SAFETY_CRITICAL and self.safety_critical_override:
                selected.append((belief, priority))
                budget.consume(text)
            elif budget.fits(text):
                selected.append((belief, priority))
                budget.consume(text)
            else:
                deferred.append((belief, score, priority))

        return selected
