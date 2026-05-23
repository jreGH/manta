import numpy as np
import pytest
from manta_world_model.actor import ActorProfile, ActorDomain, ActorAgency, ActorCooperativity
from manta_world_model.tracker import MultiActorTracker


def _shark_profile():
    return ActorProfile("shark", ActorDomain.SUBSURFACE, ActorAgency.BIOLOGICAL, ActorCooperativity.UNCOOPERATIVE)


def _explosive_profile():
    return ActorProfile("explosive", ActorDomain.SEABED, ActorAgency.PASSIVE, ActorCooperativity.UNCOOPERATIVE)


def _torpedo_profile():
    return ActorProfile("torpedo", ActorDomain.SUBSURFACE, ActorAgency.AUTONOMOUS, ActorCooperativity.ADVERSARIAL)


def test_update_and_retrieve():
    tracker = MultiActorTracker()
    t0 = 1000.0
    tracker.update("shark-1", _shark_profile(), np.array([10.0, 0.0, -5.0]),
                   np.array([1.0, 0.0, 0.0]), np.eye(3), t0)
    assert "shark-1" in tracker.actor_ids
    belief = tracker.get("shark-1")
    assert belief is not None
    np.testing.assert_array_almost_equal(belief.pose, [10.0, 0.0, -5.0])


def test_projection_advances_position():
    tracker = MultiActorTracker()
    t0 = 1000.0
    tracker.update("shark-1", _shark_profile(), np.array([0.0, 0.0, 0.0]),
                   np.array([2.0, 0.0, 0.0]), np.eye(3), t0)

    beliefs = tracker.project_all(now=t0 + 5.0)
    belief = next(b for b in beliefs if b.actor_id == "shark-1")
    assert belief.projected_pose[0] == pytest.approx(10.0, abs=0.01)


def test_covariance_grows_with_staleness():
    tracker = MultiActorTracker(process_noise=0.1)
    t0 = 1000.0
    tracker.update("unk-1", _shark_profile(), np.array([0.0, 0.0, 0.0]),
                   np.zeros(3), np.eye(3) * 0.5, t0)

    beliefs_early = tracker.project_all(now=t0 + 1.0)
    beliefs_late = tracker.project_all(now=t0 + 10.0)

    early_cov = next(b for b in beliefs_early if b.actor_id == "unk-1").projected_covariance
    late_cov = next(b for b in beliefs_late if b.actor_id == "unk-1").projected_covariance
    assert late_cov[0, 0] > early_cov[0, 0]


def test_stationary_model_does_not_drift():
    tracker = MultiActorTracker()
    t0 = 1000.0
    tracker.update("mine-1", _explosive_profile(), np.array([5.0, 3.0, -10.0]),
                   np.zeros(3), np.eye(3), t0)

    beliefs = tracker.project_all(now=t0 + 60.0)
    belief = next(b for b in beliefs if b.actor_id == "mine-1")
    np.testing.assert_array_almost_equal(belief.projected_pose, [5.0, 3.0, -10.0])


def test_staleness_seconds_grows():
    tracker = MultiActorTracker()
    t0 = 1000.0
    tracker.update("shark-1", _shark_profile(), np.zeros(3), np.zeros(3), np.eye(3), t0)

    beliefs = tracker.project_all(now=t0 + 15.0)
    belief = next(b for b in beliefs if b.actor_id == "shark-1")
    assert belief.staleness_seconds == pytest.approx(15.0, abs=0.1)


def test_confidence_decreases_with_staleness():
    tracker = MultiActorTracker(staleness_threshold=20.0)
    t0 = 1000.0
    tracker.update("shark-1", _shark_profile(), np.zeros(3), np.zeros(3), np.eye(3), t0)

    beliefs = tracker.project_all(now=t0 + 10.0)
    belief = next(b for b in beliefs if b.actor_id == "shark-1")
    assert belief.confidence == pytest.approx(0.5, abs=0.05)


def test_remove_actor():
    tracker = MultiActorTracker()
    t0 = 1000.0
    tracker.update("shark-1", _shark_profile(), np.zeros(3), np.zeros(3), np.eye(3), t0)
    tracker.remove("shark-1")
    assert "shark-1" not in tracker.actor_ids


def test_intent_inferred_for_shark():
    tracker = MultiActorTracker()
    t0 = 1000.0
    # Shark moving at 2 m/s toward a diver at origin
    tracker.update("shark-1", _shark_profile(), np.array([10.0, 0.0, 0.0]),
                   np.array([-2.0, 0.0, 0.0]), np.eye(3), t0)
    # Diver at origin
    diver_profile = ActorProfile("diver", ActorDomain.SUBSURFACE, ActorAgency.HUMAN, ActorCooperativity.COOPERATIVE)
    tracker.update("diver-1", diver_profile, np.zeros(3), np.zeros(3), np.eye(3), t0)

    beliefs = tracker.project_all(now=t0)
    shark_belief = next(b for b in beliefs if b.actor_id == "shark-1")
    assert "approaching_diver" in shark_belief.intent
    assert shark_belief.intent["approaching_diver"] > 0.5


def test_torpedo_uses_ballistic_model():
    tracker = MultiActorTracker(process_noise=0.1)
    t0 = 1000.0
    tracker.update("torp-1", _torpedo_profile(), np.array([0.0, 0.0, 0.0]),
                   np.array([20.0, 0.0, 0.0]), np.eye(3) * 0.01, t0)

    beliefs = tracker.project_all(now=t0 + 5.0)
    belief = next(b for b in beliefs if b.actor_id == "torp-1")
    # Should be near 100m at 20 m/s for 5 seconds
    assert belief.projected_pose[0] == pytest.approx(100.0, abs=1.0)
    # Ballistic covariance should grow slowly (less than constant-velocity with noise=0.1)
    assert belief.projected_covariance[0, 0] < 0.1 + 0.1 * 5.0
