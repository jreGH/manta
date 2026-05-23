#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import numpy as np

from manta_interfaces.msg import ActorObservation, WorldState, ActorBelief as ActorBeliefMsg, TaskContext
from manta_world_model import MultiActorTracker
from manta_world_model.actor import ActorProfile


def _covariance_to_array(cov_msg) -> np.ndarray:
    """Extract 3x3 position covariance from PoseWithCovariance (which stores 6x6)."""
    flat = np.array(cov_msg.covariance)
    full = flat.reshape(6, 6)
    return full[:3, :3]


def _belief_to_msg(belief) -> ActorBeliefMsg:
    msg = ActorBeliefMsg()
    msg.actor_id = belief.actor_id
    msg.label = belief.profile.label
    msg.domain = belief.profile.domain.value
    msg.agency = belief.profile.agency.value
    msg.cooperativity = belief.profile.cooperativity.value
    msg.staleness_seconds = belief.staleness_seconds
    msg.confidence = belief.confidence

    msg.projected_pose.pose.position.x = float(belief.projected_pose[0])
    msg.projected_pose.pose.position.y = float(belief.projected_pose[1])
    msg.projected_pose.pose.position.z = float(belief.projected_pose[2])

    cov6 = np.zeros((6, 6))
    cov6[:3, :3] = belief.projected_covariance
    msg.projected_pose.covariance = cov6.flatten().tolist()

    msg.intent_labels = list(belief.intent.keys())
    msg.intent_probs = [float(v) for v in belief.intent.values()]
    return msg


class WorldModelNode(Node):
    def __init__(self):
        super().__init__("world_model_node")

        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("staleness_threshold_s", 30.0)
        self.declare_parameter("process_noise", 0.1)

        rate = self.get_parameter("publish_rate_hz").value
        staleness = self.get_parameter("staleness_threshold_s").value
        noise = self.get_parameter("process_noise").value

        self._tracker = MultiActorTracker(
            staleness_threshold=staleness,
            process_noise=noise,
        )
        self._task_context: TaskContext | None = None

        self.create_subscription(ActorObservation, "/observations", self._on_observation, 50)
        self.create_subscription(TaskContext, "/task_context", self._on_task_context, 10)
        self._pub = self.create_publisher(WorldState, "/world_state", 10)
        self.create_timer(1.0 / rate, self._publish)

    def _on_observation(self, msg: ActorObservation) -> None:
        profile = ActorProfile.from_strings(msg.label, msg.domain, msg.agency, msg.cooperativity)
        pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        vel = np.array([msg.velocity.linear.x, msg.velocity.linear.y, msg.velocity.linear.z])
        cov = _covariance_to_array(msg.pose)
        observed_at = msg.observed_at.sec + msg.observed_at.nanosec * 1e-9
        self._tracker.update(msg.actor_id, profile, pose, vel, cov, observed_at)

    def _on_task_context(self, msg: TaskContext) -> None:
        self._task_context = msg

    def _publish(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        beliefs = self._tracker.project_all(now=now)

        out = WorldState()
        out.stamp = self.get_clock().now().to_msg()
        out.actors = [_belief_to_msg(b) for b in beliefs]
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = WorldModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
