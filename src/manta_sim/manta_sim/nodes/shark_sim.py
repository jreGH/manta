#!/usr/bin/env python3
"""Simulates a shark with configurable attraction toward the diver."""
import rclpy
from rclpy.node import Node
import numpy as np

from manta_interfaces.msg import ActorObservation


class SharkSim(Node):
    def __init__(self):
        super().__init__("shark_sim")

        self.declare_parameter("start_x", 30.0)
        self.declare_parameter("start_y", 20.0)
        self.declare_parameter("start_z", -8.0)
        self.declare_parameter("speed", 1.5)           # m/s base speed
        self.declare_parameter("attraction", 0.3)      # 0=random walk, 1=direct approach
        self.declare_parameter("publish_rate_hz", 0.5) # sparse — realistic observation rate
        self.declare_parameter("actor_id", "shark-1")

        self._pos = np.array([
            self.get_parameter("start_x").value,
            self.get_parameter("start_y").value,
            self.get_parameter("start_z").value,
        ], dtype=float)
        self._speed = self.get_parameter("speed").value
        self._attraction = self.get_parameter("attraction").value
        self._actor_id = self.get_parameter("actor_id").value
        rate = self.get_parameter("publish_rate_hz").value

        self._diver_pos = np.zeros(3)  # updated from observations
        self._rng = np.random.default_rng()

        self.create_subscription(ActorObservation, "/observations", self._on_obs, 20)
        self._pub = self.create_publisher(ActorObservation, "/observations", 10)
        self.create_timer(1.0 / rate, self._step)
        self._last_time = self.get_clock().now().nanoseconds * 1e-9

    def _on_obs(self, msg: ActorObservation) -> None:
        if msg.actor_id == "diver-1":
            self._diver_pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ])

    def _step(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = now - self._last_time
        self._last_time = now

        to_diver = self._diver_pos - self._pos
        dist = np.linalg.norm(to_diver)
        if dist > 0.1:
            directed = to_diver / dist
        else:
            directed = np.zeros(3)

        random_dir = self._rng.standard_normal(3)
        random_dir[2] *= 0.1  # sharks mostly move horizontally
        rn = np.linalg.norm(random_dir)
        random_dir = random_dir / rn if rn > 0 else random_dir

        direction = self._attraction * directed + (1.0 - self._attraction) * random_dir
        dn = np.linalg.norm(direction)
        if dn > 0:
            direction /= dn

        velocity = direction * self._speed
        self._pos += velocity * dt

        obs = ActorObservation()
        obs.actor_id = self._actor_id
        obs.label = "shark"
        obs.domain = "subsurface"
        obs.agency = "biological"
        obs.cooperativity = "uncooperative"
        obs.pose.pose.position.x = float(self._pos[0])
        obs.pose.pose.position.y = float(self._pos[1])
        obs.pose.pose.position.z = float(self._pos[2])
        obs.pose.covariance = (np.eye(6) * 2.0).flatten().tolist()
        obs.velocity.linear.x = float(velocity[0])
        obs.velocity.linear.y = float(velocity[1])
        obs.velocity.linear.z = float(velocity[2])
        obs.observed_at = self.get_clock().now().to_msg()
        obs.source = "direct"
        self._pub.publish(obs)

    def main(args=None):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = SharkSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
