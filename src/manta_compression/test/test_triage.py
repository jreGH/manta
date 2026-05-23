import numpy as np
import pytest
from manta_world_model.actor import ActorBelief, ActorProfile, ActorDomain, ActorAgency, ActorCooperativity
from manta_compression.scorers import TaskContext, ProximityScorer, ClosingSpeedScorer, HazardTypeScorer, NoveltyScorer, CompositeScorer
from manta_compression.triage import Triage, BandwidthBudget, PRIORITY_SAFETY_CRITICAL, PRIORITY_INFO
from manta_compression.formatter import AlertFormatter


def _make_belief(actor_id, label, domain, agency, cooperativity, pos, vel=None, intent=None):
    profile = ActorProfile(label, ActorDomain(domain), ActorAgency(agency), ActorCooperativity(cooperativity))
    vel = np.array(vel) if vel is not None else np.zeros(3)
    b = ActorBelief(
        actor_id=actor_id, profile=profile,
        pose=np.array(pos, dtype=float), velocity=vel, covariance=np.eye(3),
        observed_at=1000.0,
        projected_pose=np.array(pos, dtype=float),
        projected_covariance=np.eye(3),
        intent=intent or {},
    )
    return b


def _make_context(diver_pos=None, safety_radius=10.0, priority_types=None):
    return TaskContext(
        task_id="t1",
        task_type="ordnance_disposal",
        target_pose=np.array([0.0, 0.0, -10.0]),
        safety_radius=safety_radius,
        priority_actor_types=priority_types or ["shark", "torpedo"],
        diver_pose=np.array(diver_pos) if diver_pos else np.zeros(3),
    )


class TestProximityScorer:
    def test_nearby_scores_high(self):
        s = ProximityScorer(max_range=100.0)
        b = _make_belief("a", "shark", "subsurface", "biological", "uncooperative", [5.0, 0.0, 0.0])
        ctx = _make_context([0.0, 0.0, 0.0])
        assert s.score(b, ctx, {}) > 0.9

    def test_far_scores_low(self):
        s = ProximityScorer(max_range=100.0)
        b = _make_belief("a", "vessel", "surface", "human", "cooperative", [90.0, 0.0, 0.0])
        ctx = _make_context([0.0, 0.0, 0.0])
        assert s.score(b, ctx, {}) < 0.2


class TestClosingSpeedScorer:
    def test_approaching_scores_positively(self):
        s = ClosingSpeedScorer(max_speed=5.0)
        b = _make_belief("a", "shark", "subsurface", "biological", "uncooperative",
                         [10.0, 0.0, 0.0], vel=[-3.0, 0.0, 0.0])
        ctx = _make_context([0.0, 0.0, 0.0])
        assert s.score(b, ctx, {}) > 0.5

    def test_departing_scores_zero(self):
        s = ClosingSpeedScorer()
        b = _make_belief("a", "shark", "subsurface", "biological", "uncooperative",
                         [10.0, 0.0, 0.0], vel=[3.0, 0.0, 0.0])
        ctx = _make_context([0.0, 0.0, 0.0])
        assert s.score(b, ctx, {}) == 0.0


class TestHazardTypeScorer:
    def test_adversarial_scores_higher(self):
        s = HazardTypeScorer()
        adv = _make_belief("a", "torpedo", "subsurface", "autonomous", "adversarial", [0, 0, 0])
        coop = _make_belief("b", "auv", "subsurface", "autonomous", "cooperative", [0, 0, 0])
        ctx = _make_context()
        assert s.score(adv, ctx, {}) > s.score(coop, ctx, {})


class TestNoveltyScorer:
    def test_never_seen_scores_max(self):
        s = NoveltyScorer()
        b = _make_belief("a", "shark", "subsurface", "biological", "uncooperative", [0, 0, 0])
        assert s.score(b, _make_context(), {}) == 1.0

    def test_unchanged_position_scores_low(self):
        s = NoveltyScorer(position_threshold=5.0)
        b = _make_belief("a", "shark", "subsurface", "biological", "uncooperative", [1.0, 0.0, 0.0])
        known = {"a": b}
        assert s.score(b, _make_context(), known) == pytest.approx(0.0, abs=0.01)


class TestBandwidthBudget:
    def test_fits_within_budget(self):
        budget = BandwidthBudget(max_chars=160)
        assert budget.fits("SHARK 18m NE closing")
        budget.consume("SHARK 18m NE closing")
        assert budget.used == 20

    def test_overage_not_allowed(self):
        budget = BandwidthBudget(max_chars=10)
        assert not budget.fits("this is longer than ten characters")


class TestTriage:
    def test_safety_critical_ranked_first(self):
        triage = Triage()
        shark = _make_belief("shark-1", "shark", "subsurface", "biological", "adversarial",
                             [5.0, 0.0, 0.0], vel=[-2.0, 0.0, 0.0],
                             intent={"approaching_diver": 0.9})
        vessel = _make_belief("vessel-1", "vessel", "surface", "human", "cooperative",
                              [100.0, 0.0, 0.0], vel=[0.5, 0.0, 0.0],
                              intent={"transiting_area": 0.7})
        ctx = _make_context([0.0, 0.0, 0.0])
        ranked = triage.rank([vessel, shark], ctx, {})
        assert ranked[0][0].actor_id == "shark-1"
        assert ranked[0][2] == PRIORITY_SAFETY_CRITICAL

    def test_budget_limits_output(self):
        triage = Triage()
        formatter = AlertFormatter(diver_pose=np.zeros(3))
        actors = [
            _make_belief(f"a{i}", "vessel", "surface", "human", "uncooperative",
                         [float(i * 10), 0.0, 0.0])
            for i in range(10)
        ]
        ctx = _make_context([0.0, 0.0, 0.0])
        ranked = triage.rank(actors, ctx, {})
        budget = BandwidthBudget(max_chars=40)
        selected = triage.select_within_budget(ranked, budget, formatter)
        total_chars = sum(len(formatter.format_single(b)) for b, _ in selected)
        assert total_chars <= 40 + 20  # safety_critical_override may slightly exceed

    def test_safety_critical_override_bypasses_budget(self):
        triage = Triage(safety_critical_override=True)
        formatter = AlertFormatter(diver_pose=np.zeros(3))
        shark = _make_belief("shark-1", "shark", "subsurface", "biological", "adversarial",
                             [3.0, 0.0, 0.0], intent={"approaching_diver": 0.95})
        ctx = _make_context([0.0, 0.0, 0.0])
        ranked = triage.rank([shark], ctx, {})
        budget = BandwidthBudget(max_chars=0)  # empty budget
        selected = triage.select_within_budget(ranked, budget, formatter)
        assert any(b.actor_id == "shark-1" for b, _ in selected)


class TestAlertFormatter:
    def test_format_single(self):
        formatter = AlertFormatter(diver_pose=np.zeros(3))
        b = _make_belief("s1", "shark", "subsurface", "biological", "uncooperative",
                         [0.0, 20.0, 0.0], intent={"approaching_diver": 0.9})
        text = formatter.format_single(b)
        assert "SHARK" in text
        assert "20m" in text

    def test_format_alert_combines_fragments(self):
        formatter = AlertFormatter(diver_pose=np.zeros(3))
        b1 = _make_belief("s1", "shark", "subsurface", "biological", "uncooperative",
                          [0.0, 20.0, 0.0], intent={"approaching_diver": 0.9})
        b2 = _make_belief("m1", "mine", "seabed", "passive", "uncooperative",
                          [5.0, 0.0, -10.0])
        alert = formatter.format_alert([(b1, PRIORITY_SAFETY_CRITICAL), (b2, PRIORITY_INFO)],
                                       budget_fraction=0.5)
        assert "|" in alert["text"]
        assert alert["priority"] == PRIORITY_SAFETY_CRITICAL
