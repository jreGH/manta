from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class ActorDomain(str, Enum):
    SURFACE = "surface"
    SUBSURFACE = "subsurface"
    AERIAL = "aerial"
    SEABED = "seabed"


class ActorAgency(str, Enum):
    HUMAN = "human"
    AUTONOMOUS = "autonomous"
    BIOLOGICAL = "biological"
    PASSIVE = "passive"


class ActorCooperativity(str, Enum):
    COOPERATIVE = "cooperative"
    UNCOOPERATIVE = "uncooperative"
    ADVERSARIAL = "adversarial"


@dataclass
class ActorProfile:
    label: str                      # e.g. "shark", "torpedo", "helicopter"
    domain: ActorDomain
    agency: ActorAgency
    cooperativity: ActorCooperativity

    @classmethod
    def from_strings(cls, label: str, domain: str, agency: str, cooperativity: str) -> "ActorProfile":
        return cls(
            label=label,
            domain=ActorDomain(domain),
            agency=ActorAgency(agency),
            cooperativity=ActorCooperativity(cooperativity),
        )


@dataclass
class ActorBelief:
    actor_id: str
    profile: ActorProfile

    # 3-element [x, y, z] in metres (NED or ENU depending on convention)
    pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 3x3 position covariance
    covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 1.0)
    # 3-element velocity [vx, vy, vz] m/s
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    observed_at: float = 0.0        # unix timestamp of last real observation
    projected_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    projected_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 1.0)

    staleness_seconds: float = 0.0
    confidence: float = 1.0
    intent: dict = field(default_factory=dict)  # label -> probability

    def is_stale(self, threshold: float) -> bool:
        return self.staleness_seconds > threshold

    def distance_to(self, other_pose: np.ndarray) -> float:
        return float(np.linalg.norm(self.projected_pose - other_pose))

    def closing_speed_toward(self, target_pose: np.ndarray) -> float:
        """Positive means closing, negative means departing."""
        to_target = target_pose - self.projected_pose
        dist = np.linalg.norm(to_target)
        if dist < 1e-6:
            return 0.0
        direction = to_target / dist
        return float(np.dot(self.velocity, direction))
