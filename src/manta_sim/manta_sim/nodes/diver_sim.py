#!/usr/bin/env python3
"""Simulates the focused actor (diver). Publishes ground-truth pose and task context;
subscribes to /alerts/diver and logs what the diver receives."""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovariance, PoseStamped, Twist
import numpy as np

from manta_interfaces.msg import ActorObservation, TaskContext, DiverAlert


class DiverSim(Node):
    def __init__(self):
        super().__init__("diver_sim")

        self.declare_parameter("task_type", "ordnance_disposal")
        self.declare_parameter("target_x", 0.0)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_z", -15.0)
        self.declare_parameter("safety_radius", 10.0)
        self.declare_parameter("start_x", 5.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("start_z", -10.0)
        self.declare_parameter("publish_rate_hz", 1.0)

        self._pose = np.array([
            self.get_parameter("start_x").value,
            self.get_parameter("start_y").value,
            self.get_parameter("start_z").value,
        ])
        self._target = np.array([
            self.get_parameter("target_x").value,
            self.get_parameter("target_y").value,
            self.get_parameter("target_z").value,
        ])
        self._task_type = self.get_parameter("task_type").value
        self._safety_radius = self.get_parameter("safety_radius").value
        rate = self.get_parameter("publish_rate_hz").value

        self._obs_pub = self.create_publisher(ActorObservation, "/observations", 10)
        self._task_pub = self.create_publisher(TaskContext, "/task_context", 10)
        self.create_subscription(DiverAlert, "/alerts/diver", self._on_alert, 10)
        self.create_timer(1.0 / rate, self._publish)

    def _publish(self) -> None:
        now = self.get_clock().now()

        obs = ActorObservation()
        obs.actor_id = "diver-1"
        obs.label = "diver"
        obs.domain = "subsurface"
        obs.agency = "human"
        obs.cooperativity = "cooperative"
        obs.pose.pose.position.x = float(self._pose[0])
        obs.pose.pose.position.y = float(self._pose[1])
        obs.pose.pose.position.z = float(self._pose[2])
        obs.pose.covariance = (np.eye(6) * 0.1).flatten().tolist()
        obs.observed_at = now.to_msg()
        obs.source = "direct"
        self._obs_pub.publish(obs)

        ctx = TaskContext()
        ctx.task_id = "dive-001"
        ctx.task_type = self._task_type
        ctx.target_pose.pose.position.x = float(self._target[0])
        ctx.target_pose.pose.position.y = float(self._target[1])
        ctx.target_pose.pose.position.z = float(self._target[2])
        ctx.target_pose.header.stamp = now.to_msg()
        ctx.safety_radius = float(self._safety_radius)
        ctx.priority_actor_types = ["shark", "torpedo", "mine", "trained_seal"]
        self._task_pub.publish(ctx)

    def _on_alert(self, msg: DiverAlert) -> None:
        priority_str = ["INFO", "WARNING", "SAFETY_CRITICAL"][msg.priority]
        self.get_logger().info(f"[DIVER RECEIVED] {priority_str}: {msg.text}")


def main(args=None):
    rclpy.init(args=args)
    node = DiverSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
