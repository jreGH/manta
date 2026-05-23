from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from .triage import PRIORITY_INFO, PRIORITY_WARNING, PRIORITY_SAFETY_CRITICAL

if TYPE_CHECKING:
    from manta_world_model.actor import ActorBelief


_CARDINAL = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _bearing_label(from_pose: np.ndarray, to_pose: np.ndarray) -> str:
    dx = to_pose[0] - from_pose[0]
    dy = to_pose[1] - from_pose[1]
    angle = np.degrees(np.arctan2(dy, dx))
    idx = int((angle + 180 + 22.5) / 45) % 8
    return _CARDINAL[idx]


def _top_intent(belief: "ActorBelief") -> str:
    if not belief.intent:
        return ""
    return max(belief.intent, key=belief.intent.get)


_INTENT_PHRASE = {
    "approaching_diver": "closing",
    "coordinated_approach": "coordinated approach",
    "inbound": "INBOUND",
    "foraging": "",
    "stationary": "stationary",
    "departing": "departing",
    "transiting_area": "transiting",
    "surveilling": "loitering",
    "clear": "",
    "on_task": "",
    "closing": "closing",
}


class AlertFormatter:
    """Formats ActorBelief objects into compact text fragments for the diver channel."""

    def __init__(self, diver_pose: np.ndarray | None = None):
        self.diver_pose = diver_pose if diver_pose is not None else np.zeros(3)

    def update_diver_pose(self, pose: np.ndarray) -> None:
        self.diver_pose = pose

    def format_single(self, belief: "ActorBelief") -> str:
        label = belief.profile.label.upper()
        dist = belief.distance_to(self.diver_pose)
        bearing = _bearing_label(self.diver_pose, belief.projected_pose)
        intent_phrase = _INTENT_PHRASE.get(_top_intent(belief), "")

        parts = [f"{label} {dist:.0f}m {bearing}"]
        if intent_phrase:
            parts.append(intent_phrase)
        return " ".join(parts)

    def format_alert(
        self,
        selected: list[tuple["ActorBelief", int]],
        budget_fraction: float,
        primary_actor_id: str = "",
    ) -> dict:
        """Returns a dict matching DiverAlert.msg fields (no ROS2 dependency)."""
        if not selected:
            return {}

        max_priority = max(p for _, p in selected)
        fragments = [self.format_single(b) for b, _ in selected]
        text = " | ".join(fragments)

        return {
            "priority": max_priority,
            "text": text,
            "actor_id": primary_actor_id or (selected[0][0].actor_id if selected else ""),
            "budget_fraction": budget_fraction,
        }
