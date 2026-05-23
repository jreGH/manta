FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-pytest \
    python3-numpy \
    ros-jazzy-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY src/ src/
COPY config/ config/
COPY launch/ launch/

RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --symlink-install --packages-select manta_interfaces && \
    . install/setup.sh && \
    colcon build --symlink-install

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "manta_sim", "full_system.launch.py"]
