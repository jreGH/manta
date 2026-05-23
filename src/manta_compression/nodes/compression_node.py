#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from manta_interfaces.msg import WorldState, TaskContext as TaskContextMsg, DiverAlert
from manta_compression import Triage, BandwidthBudget, AlertFormatter
from manta_compression.scorers import TaskContext
from manta_world_model.actor import (
    ActorBelief, ActorProfile, ActorDomain, ActorAgency, ActorCooperativity,
)


def _msg_to_belief(msg) -> ActorBelief:
    profile = ActorProfile(
        label=msg.label,
        domain=ActorDomain(msg.domain),
        agency=ActorAgency(msg.agency),
        cooperativity=ActorCooperativity(msg.cooperativity),
    )
    projected = np.array([
        msg.projected_pose.pose.position.x,
        msg.projected_pose.pose.position.y,
        msg.projected_pose.pose.position.z,
    ])
    cov = np.array(msg.projected_pose.covariance).reshape(6, 6)[:3, :3]
    intent = dict(zip(msg.intent_labels, msg.intent_probs))
    return ActorBelief(
        actor_id=msg.actor_id,
        profile=profile,
        pose=projected.copy(),
        velocity=np.zeros(3),
        covariance=cov,
        observed_at=0.0,
        projected_pose=projected,
        projected_covariance=cov,
        staleness_seconds=msg.staleness_seconds,
        confidence=msg.confidence,
        intent=intent,
    )


class CompressionNode(Node):
    def __init__(self):
        super().__init__("compression_node")

        self.declare_parameter("bandwidth_budget_chars", 160)
        self.declare_parameter("safety_critical_override", True)
        self.declare_parameter("scorer_weights.proximity", 1.5)
        self.declare_parameter("scorer_weights.closing_speed", 2.0)
        self.declare_parameter("scorer_weights.hazard_type", 1.0)
        self.declare_parameter("scorer_weights.task_relevance", 1.8)
        self.declare_parameter("scorer_weights.novelty", 0.8)

        self._budget_chars = self.get_parameter("bandwidth_budget_chars").value
        self._safety_override = self.get_parameter("safety_critical_override").value

        weights = {
            "proximity": self.get_parameter("scorer_weights.proximity").value,
            "closing_speed": self.get_parameter("scorer_weights.closing_speed").value,
            "hazard_type": self.get_parameter("scorer_weights.hazard_type").value,
            "task_relevance": self.get_parameter("scorer_weights.task_relevance").value,
            "novelty": self.get_parameter("scorer_weights.novelty").value,
        }

        self._triage = Triage(safety_critical_override=self._safety_override, scorer_weights=weights)
        self._formatter = AlertFormatter()
        self._known_state: dict[str, ActorBelief] = {}
        self._task_context: TaskContext | None = None
        self._diver_pose = np.zeros(3)

        self.create_subscription(WorldState, "/world_state", self._on_world_state, 10)
        self.create_subscription(TaskContextMsg, "/task_context", self._on_task_context, 10)
        self._pub = self.create_publisher(DiverAlert, "/alerts/raw", 10)

    def _on_task_context(self, msg: TaskContextMsg) -> None:
        target = np.array([
            msg.target_pose.pose.position.x,
            msg.target_pose.pose.position.y,
            msg.target_pose.pose.position.z,
        ])
        self._task_context = TaskContext(
            task_id=msg.task_id,
            task_type=msg.task_type,
            target_pose=target,
            safety_radius=msg.safety_radius,
            priority_actor_types=list(msg.priority_actor_types),
            diver_pose=self._diver_pose,
        )

    def _on_world_state(self, msg: WorldState) -> None:
        if self._task_context is None:
            return

        actors = [_msg_to_belief(a) for a in msg.actors]
        # Extract diver pose from world state for formatter
        for a in actors:
            if a.profile.agency == ActorAgency.HUMAN and a.profile.cooperativity == ActorCooperativity.COOPERATIVE:
                self._diver_pose = a.projected_pose.copy()
                self._formatter.update_diver_pose(self._diver_pose)
                if self._task_context:
                    self._task_context = TaskContext(
                        **{**self._task_context.__dict__, "diver_pose": self._diver_pose}
                    )

        # Filter out the diver themselves
        threats = [a for a in actors
                   if not (a.profile.agency == ActorAgency.HUMAN and
                           a.profile.cooperativity == ActorCooperativity.COOPERATIVE)]

        ranked = self._triage.rank(threats, self._task_context, self._known_state)
        budget = BandwidthBudget(self._budget_chars)
        selected = self._triage.select_within_budget(ranked, budget, self._formatter)

        if not selected:
            return

        alert_dict = self._formatter.format_alert(selected, budget.fraction)
        alert = DiverAlert()
        alert.stamp = self.get_clock().now().to_msg()
        alert.priority = alert_dict["priority"]
        alert.text = alert_dict["text"]
        alert.actor_id = alert_dict["actor_id"]
        alert.budget_fraction = alert_dict["budget_fraction"]
        self._pub.publish(alert)

        self._known_state = {b.actor_id: b for b, _ in selected}


def main(args=None):
    rclpy.init(args=args)
    node = CompressionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
