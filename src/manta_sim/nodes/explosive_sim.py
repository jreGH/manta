#!/usr/bin/env python3
"""Publishes a static explosive/mine at a fixed seabed location once on startup,
then republishes at a low rate so the world model doesn't mark it stale."""
import rclpy
from rclpy.node import Node
import numpy as np

from manta_interfaces.msg import ActorObservation


class ExplosiveSim(Node):
    def __init__(self):
        super().__init__("explosive_sim")

        self.declare_parameter("x", 2.0)
        self.declare_parameter("y", 3.0)
        self.declare_parameter("z", -18.0)
        self.declare_parameter("actor_id", "explosive-1")
        self.declare_parameter("label", "explosive")
        self.declare_parameter("republish_rate_hz", 0.1)

        self._pos = np.array([
            self.get_parameter("x").value,
            self.get_parameter("y").value,
            self.get_parameter("z").value,
        ])
        self._actor_id = self.get_parameter("actor_id").value
        self._label = self.get_parameter("label").value
        rate = self.get_parameter("republish_rate_hz").value

        self._pub = self.create_publisher(ActorObservation, "/observations", 10)
        self.create_timer(1.0 / rate, self._publish)

    def _publish(self) -> None:
        obs = ActorObservation()
        obs.actor_id = self._actor_id
        obs.label = self._label
        obs.domain = "seabed"
        obs.agency = "passive"
        obs.cooperativity = "uncooperative"
        obs.pose.pose.position.x = float(self._pos[0])
        obs.pose.pose.position.y = float(self._pos[1])
        obs.pose.pose.position.z = float(self._pos[2])
        obs.pose.covariance = (np.eye(6) * 0.01).flatten().tolist()
        obs.observed_at = self.get_clock().now().to_msg()
        obs.source = "direct"
        self._pub.publish(obs)


def main(args=None):
    rclpy.init(args=args)
    node = ExplosiveSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
