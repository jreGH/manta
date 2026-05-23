#!/usr/bin/env python3
"""Push-only comms node: rate-limits and deduplicates alerts before forwarding to the diver."""
import rclpy
from rclpy.node import Node
import time

from manta_interfaces.msg import DiverAlert


class CommsNode(Node):
    def __init__(self):
        super().__init__("comms_node")

        self.declare_parameter("min_resend_interval_s", 10.0)
        self.declare_parameter("min_priority_change_to_resend", 0)

        self._min_interval = self.get_parameter("min_resend_interval_s").value
        self._min_priority_delta = self.get_parameter("min_priority_change_to_resend").value

        self._last_sent_text: str = ""
        self._last_sent_time: float = 0.0
        self._last_sent_priority: int = -1

        self.create_subscription(DiverAlert, "/alerts/raw", self._on_alert, 10)
        self._pub = self.create_publisher(DiverAlert, "/alerts/diver", 10)

    def _on_alert(self, msg: DiverAlert) -> None:
        now = time.time()
        elapsed = now - self._last_sent_time
        priority_delta = msg.priority - self._last_sent_priority

        same_text = msg.text == self._last_sent_text
        within_interval = elapsed < self._min_interval
        no_priority_escalation = priority_delta < self._min_priority_delta

        if same_text and within_interval and no_priority_escalation:
            return

        # Safety-critical messages always go through immediately
        if msg.priority == 2 or not within_interval or not same_text:
            self._pub.publish(msg)
            self._last_sent_text = msg.text
            self._last_sent_time = now
            self._last_sent_priority = msg.priority
            self.get_logger().info(f"[DIVER←] P{msg.priority}: {msg.text}")


def main(args=None):
    rclpy.init(args=args)
    node = CommsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
