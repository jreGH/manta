#!/usr/bin/env python3
"""Gateway node: ingests external actor reports, applies simulated latency and
packet loss, then re-publishes as ActorObservation on /observations."""
import rclpy
from rclpy.node import Node
import random
import time
from collections import deque
from dataclasses import dataclass

from manta_interfaces.msg import ActorObservation


@dataclass
class _Pending:
    msg: ActorObservation
    deliver_at: float


class GatewayNode(Node):
    def __init__(self):
        super().__init__("gateway_node")

        self.declare_parameter("latency_mean_s", 5.0)
        self.declare_parameter("latency_std_s", 2.0)
        self.declare_parameter("packet_loss_prob", 0.05)

        self._latency_mean = self.get_parameter("latency_mean_s").value
        self._latency_std = self.get_parameter("latency_std_s").value
        self._loss_prob = self.get_parameter("packet_loss_prob").value

        self._queue: deque[_Pending] = deque()

        self.create_subscription(ActorObservation, "/gateway/incoming", self._on_incoming, 50)
        self._pub = self.create_publisher(ActorObservation, "/observations", 50)
        self.create_timer(0.1, self._drain)

    def _on_incoming(self, msg: ActorObservation) -> None:
        if random.random() < self._loss_prob:
            self.get_logger().debug(f"[gateway] dropped packet for {msg.actor_id}")
            return
        latency = max(0.0, random.gauss(self._latency_mean, self._latency_std))
        self._queue.append(_Pending(msg=msg, deliver_at=time.time() + latency))

    def _drain(self) -> None:
        now = time.time()
        while self._queue and self._queue[0].deliver_at <= now:
            pending = self._queue.popleft()
            self._pub.publish(pending.msg)


def main(args=None):
    rclpy.init(args=args)
    node = GatewayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
