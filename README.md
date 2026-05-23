# Manta

ROS2 multi-actor awareness and semantic compression system.

An **assistant** robot monitors a heterogeneous marine environment — divers, sharks, mines, marine traffic — and pushes only task-relevant or safety-critical information to a **focused actor** (e.g. a scuba diver doing ordnance disposal) over an extremely bandwidth-constrained channel (~SMS / 160 characters).

## Architecture

```
[Sim nodes] ──→ /observations ──→ [world_model_node] ──→ /world_state
                                         ↑                      │
                               /task_context              [compression_node]
                                                                 │
                                                          /alerts/raw
                                                                 │
                                                          [comms_node]
                                                                 │
                                                         /alerts/diver
```

### Packages

| Package | Role |
|---------|------|
| `manta_interfaces` | Custom ROS2 message definitions |
| `manta_world_model` | Multi-actor belief tracker with latency-aware projection and rule-based intent inference |
| `manta_compression` | Semantic scoring, triage, and formatting within a bandwidth budget |
| `manta_comms` | Rate-limiting push-channel to the focused actor |
| `manta_gateway` | External report ingestion with simulated latency and packet loss |
| `manta_sim` | Simulation nodes: diver, shark, explosive, vessel |

### Actor classification

Actors are classified on three independent axes:

- **Domain**: `surface` | `subsurface` | `aerial` | `seabed`
- **Agency**: `human` | `autonomous` | `biological` | `passive`
- **Cooperativity**: `cooperative` | `uncooperative` | `adversarial`

The combination drives the motion model, intent module, and hazard scoring automatically.

## Running

### With Docker (recommended — includes ROS2 Jazzy)

```bash
docker compose build
docker compose up
```

To run tests only:
```bash
docker compose run --rm manta-test
```

### Running tests without ROS2 (library code only)

The world model and compression libraries have no ROS2 dependency and can be
tested directly:

```bash
cd src
pip install numpy pytest
PYTHONPATH=manta_world_model:manta_compression pytest \
    manta_world_model/test manta_compression/test -v
```

### Building natively (ROS2 Jazzy required)

```bash
source /opt/ros/jazzy/setup.sh
colcon build --packages-select manta_interfaces
source install/setup.sh
colcon build
source install/setup.sh
ros2 launch manta_sim full_system.launch.py
```

## Key configuration

All parameters are in [`config/params.yaml`](config/params.yaml).

Notable knobs:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `compression_node.bandwidth_budget_chars` | 160 | SMS-equivalent budget for each push to diver |
| `compression_node.safety_critical_override` | true | Safety-critical alerts bypass budget |
| `compression_node.scorer_weights.closing_speed` | 2.0 | Weight for actors closing on diver |
| `shark_sim.attraction` | 0.3 | 0=random walk, 1=direct approach toward diver |
| `gateway_node.latency_mean_s` | 5.0 | Mean delay for external reports |
| `gateway_node.packet_loss_prob` | 0.05 | Fraction of external reports dropped |

## Research transferability

- **World model** (`manta_world_model/manta_world_model/`): pure Python library. Swap in a new `IntentModule` subclass for ML-based trajectory prediction without touching the tracker or ROS2 nodes.
- **Compression** (`manta_compression/manta_compression/`): parameterized by bandwidth budget and scorer weights. Instantiate it twice — once for gateway→assistant (larger budget) and once for assistant→diver (160 chars) — with different weights to suit each hop.

## Topic reference

| Topic | Type | Publisher | Subscribers |
|-------|------|-----------|-------------|
| `/observations` | `ActorObservation` | sim nodes, gateway | world_model_node |
| `/gateway/incoming` | `ActorObservation` | vessel_sim | gateway_node |
| `/task_context` | `TaskContext` | diver_sim | world_model_node, compression_node |
| `/world_state` | `WorldState` | world_model_node | compression_node |
| `/alerts/raw` | `DiverAlert` | compression_node | comms_node |
| `/alerts/diver` | `DiverAlert` | comms_node | diver_sim (inbox) |
