from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare("manta_sim"), "config", "params.yaml"
    ])

    world_model = Node(
        package="manta_world_model",
        executable="world_model_node",
        name="world_model_node",
        parameters=[config],
        output="screen",
    )

    compression = Node(
        package="manta_compression",
        executable="compression_node",
        name="compression_node",
        parameters=[config],
        output="screen",
    )

    comms = Node(
        package="manta_comms",
        executable="comms_node",
        name="comms_node",
        parameters=[config],
        output="screen",
    )

    gateway = Node(
        package="manta_gateway",
        executable="gateway_node",
        name="gateway_node",
        parameters=[config],
        output="screen",
    )

    diver = Node(
        package="manta_sim",
        executable="diver_sim",
        name="diver_sim",
        parameters=[config],
        output="screen",
    )

    shark = Node(
        package="manta_sim",
        executable="shark_sim",
        name="shark_sim",
        parameters=[config],
        output="screen",
    )

    explosive = Node(
        package="manta_sim",
        executable="explosive_sim",
        name="explosive_sim",
        parameters=[config],
        output="screen",
    )

    vessel = Node(
        package="manta_sim",
        executable="vessel_sim",
        name="vessel_sim",
        parameters=[config],
        output="screen",
    )

    return LaunchDescription([
        world_model,
        compression,
        comms,
        gateway,
        diver,
        shark,
        explosive,
        vessel,
    ])
