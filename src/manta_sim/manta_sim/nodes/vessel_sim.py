#!/usr/bin/env python3
"""Simulates a surface vessel (AIS-style). Publishes to /gateway/incoming
so reports are delayed by the gateway node before reaching /observations."""
import rclpy
from rclpy.node import Node
import numpy as np

from manta_interfaces.msg import ActorObservation


class VesselSim(Node):
    def __init__(self):
        super().__init__("vessel_sim")

        self.declare_parameter("start_x", 300.0)
        self.declare_parameter("start_y", -100.0)
        self.declare_parameter("start_z", 0.0)
        self.declare_parameter("course_deg", 225.0)   # bearing in degrees
        self.declare_parameter("speed_ms", 3.0)       # m/s (~6 knots)
        self.declare_parameter("publish_rate_hz", 0.1)
        self.declare_parameter("actor_id", "vessel-1")

        course_rad = np.radians(self.get_parameter("course_deg").value)
        self._speed = self.get_parameter("speed_ms").value
        self._vel = np.array([
            np.sin(course_rad) * self._speed,
            np.cos(course_rad) * self._speed,
            0.0,
        ])
        self._pos = np.array([
            self.get_parameter("start_x").value,
            self.get_parameter("start_y").value,
            self.get_parameter("start_z").value,
        ], dtype=float)
        self._actor_id = self.get_parameter("actor_id").value
        rate = self.get_parameter("publish_rate_hz").value

        self._pub = self.create_publisher(ActorObservation, "/gateway/incoming", 10)
        self._last_time = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(1.0 / rate, self._step)

    def _step(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = now - self._last_time
        self._last_time = now
        self._pos += self._vel * dt

        obs = ActorObservation()
        obs.actor_id = self._actor_id
        obs.label = "vessel"
        obs.domain = "surface"
        obs.agency = "human"
        obs.cooperativity = "cooperative"
        obs.pose.pose.position.x = float(self._pos[0])
        obs.pose.pose.position.y = float(self._pos[1])
        obs.pose.pose.position.z = float(self._pos[2])
        obs.pose.covariance = (np.eye(6) * 5.0).flatten().tolist()
        obs.velocity.linear.x = float(self._vel[0])
        obs.velocity.linear.y = float(self._vel[1])
        obs.velocity.linear.z = 0.0
        obs.observed_at = self.get_clock().now().to_msg()
        obs.source = "gateway"
        self._pub.publish(obs)


def main(args=None):
    rclpy.init(args=args)
    node = VesselSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
