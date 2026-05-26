#!/usr/bin/env python3
"""
MANTA World Model – standalone demonstration server.
Drives the world model library directly; no ROS installation required.

Usage
-----
    python3 demo/server.py          # from workspace root
    Open http://localhost:8888

Docker
------
    docker compose --profile demo up demo
"""
import sys, os, json, time, math, threading, queue
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

try:
    from manta_world_model import MultiActorTracker
    from manta_world_model.actor import ActorProfile
    from manta_world_model.intent import IntentModule
except ImportError:
    sys.path.insert(0, os.path.join(_ROOT, "src", "manta_world_model"))
    from manta_world_model import MultiActorTracker
    from manta_world_model.actor import ActorProfile
    from manta_world_model.intent import IntentModule

try:
    from manta_compression.triage import Triage, BandwidthBudget
    from manta_compression.formatter import AlertFormatter
    from manta_compression.scorers import TaskContext
except ImportError:
    sys.path.insert(0, os.path.join(_ROOT, "src", "manta_compression"))
    from manta_compression.triage import Triage, BandwidthBudget
    from manta_compression.formatter import AlertFormatter
    from manta_compression.scorers import TaskContext

# ── Scenario profiles ─────────────────────────────────────────────────────────
_PROFILES = {
    "diver-1":     ActorProfile.from_strings("diver",     "subsurface", "human",      "cooperative"),
    "shark-1":     ActorProfile.from_strings("shark",     "subsurface", "biological", "uncooperative"),
    "shark-2":     ActorProfile.from_strings("shark",     "subsurface", "biological", "uncooperative"),
    "vessel-1":    ActorProfile.from_strings("vessel",    "surface",    "human",      "cooperative"),
    "explosive-1": ActorProfile.from_strings("explosive", "seabed",     "passive",    "uncooperative"),
}

_OBS_HZ_NOMINAL = {
    "diver-1":     1/60,   # ~once per minute in normal ops
    "shark-1":     0.5,
    "shark-2":     0.5,
    "vessel-1":    0.1,    # sparse – offboard relay report
    "explosive-1": 0.05,
}
_OBS_COV = {
    "diver-1":     0.1,
    "shark-1":     2.0,
    "shark-2":     2.0,
    "vessel-1":    8.0,
    "explosive-1": 0.01,
}
_OBS_SOURCE = {
    "diver-1":     "direct",
    "shark-1":     "direct",
    "shark-2":     "direct",
    "vessel-1":    "offboard",
    "explosive-1": "direct",
}

_PRED_FACTOR = 1.8

_INIT_POS = {
    "diver-1":     np.array([14.0,  20.0,  -9.0]),  # entry point, north of op area
    "shark-1":     np.array([62.0,  42.0,  -8.0]),  # well east of op area
    "shark-2":     np.array([-48.0, 55.0, -12.0]),  # well north-west
    "vessel-1":    np.array([60.0,  60.0,   0.0]),
    "explosive-1": np.array([ 2.0,   3.0, -18.0]),
}
_DIVER_TARGET = np.array([2.0, 3.0, -18.0])       # EOD site = explosive position

# Vessel entry vectors — (start_pos, course_deg)
# Each entry brings the vessel on a different bearing through or near the op area
_VESSEL_ENTRIES = [
    (np.array([ 60.0,  60.0,  0.0]),  210.0),  # NE → SW  (the existing first pass)
    (np.array([-75.0,   8.0,  0.0]),   78.0),  # W  → NE
    (np.array([  6.0, -72.0,  0.0]),  355.0),  # S  → N
    (np.array([ 70.0, -38.0,  0.0]),  308.0),  # SE → NW
    (np.array([-62.0,  54.0,  0.0]),  142.0),  # NW → SE
    (np.array([ 22.0,  74.0,  0.0]),  202.0),  # N  → SSW
]

SHARK_SPEED          = 1.2
SHARK_WANDER_ATTRACT = 0.05   # near-pure random walk during diver transit
SHARK_HUNT_ATTRACT   = 0.55   # drawn toward diver once hunting
SHARK_FLEE_ATTRACT   = 0.80   # flee-away weight after deterrence
SHARK_HUNT_PROB      = 0.002  # per-step P(decide to hunt) once diver is on site; ~50s average
SHARK_EOD_RADIUS     = 5.0    # m — diver considered "on site" within this
SONAR_RANGE          = 50.0   # m — max range at which MANTA can observe a shark
DIVER_SPEED          = 0.25   # m/s (~0.49 kt)
SIM_SPEED            = 3.0    # physics time multiplier — display still shows real elapsed time
DIVER_AWARE_RANGE    = 12.0   # m — shark within this triggers diver awareness
DIVER_REACT_T        = 4.0    # s — delay between noticing shark and firing deterrent
DIVER_DETER_T        = 60.0   # s — max deterrence duration (safety cap)
DIVER_SAFE_RANGE     = 20.0   # m — diver stops deterring once shark is this far away
VESSEL_SPEED         = 2.5
VESSEL_OBS_HZ        = 0.1    # offboard update rate for all vessels
VESSEL_OBS_COV       = 8.0
VESSEL_EXPIRE_RADIUS = 115.0  # m from origin — vessel stops being observed beyond this
VESSEL_SPAWN_MIN_T   = 55.0   # real-wall seconds between vessel spawns
VESSEL_SPAWN_MAX_T   = 90.0
OWNSHIP_ORBIT_RADIUS = 25.0  # m
OWNSHIP_SPEED        = 1.5   # m/s (~3 kt)
OWNSHIP_ORBIT_OMEGA  = OWNSHIP_SPEED / OWNSHIP_ORBIT_RADIUS  # rad/s
OWNSHIP_TRAIL_LEN    = 50
SIM_DT               = 0.10
SCENARIO_DURATION    = 600.0
HISTORY_LEN          = 30
DIVER_EMERG_HZ       = 2.0
DIVER_EMERG_RANGE    = 15.0
COMMS_TX_NORMAL      = 60.0  # s between transmissions in normal ops
COMMS_TX_EMERG       = 10.0  # s between transmissions in emergency (floor)
COMMS_BUDGET         = 160   # chars
PORT                 = 8888

# ── Custom vessel intent for demo scale ───────────────────────────────────────
class _VesselAreaIntent(IntentModule):
    OP_RADIUS = 50.0
    HORIZON   = 60.0

    def infer(self, belief, world_state):
        future = belief.projected_pose + belief.velocity * self.HORIZON
        if np.linalg.norm(future[:2]) < self.OP_RADIUS:
            return {"transiting_area": 0.90, "clear": 0.10}
        if np.linalg.norm(belief.projected_pose[:2]) < self.OP_RADIUS:
            return {"transiting_area": 0.95, "clear": 0.05}
        return {"clear": 0.95, "transiting_area": 0.05}

# ── Compression pipeline (module-level, shared across steps) ──────────────────
_TASK_CTX = TaskContext(
    task_id="EOD-001",
    task_type="explosive_disposal",
    target_pose=_INIT_POS["explosive-1"].copy(),
    safety_radius=10.0,
    priority_actor_types=["shark", "explosive"],
    diver_pose=None,
)
_TRIAGE    = Triage()
_FORMATTER = AlertFormatter()

# ── Mutable simulation state ──────────────────────────────────────────────────
def _make_vessel(id_num: int, entry: tuple) -> dict:
    """Build a vessel state dict from an (entry_pos, course_deg) entry."""
    epos, cdeg = entry
    cr = math.radians(cdeg)
    return {
        "id":       f"vessel-{id_num}",
        "pos":      epos.copy(),
        "vel":      np.array([math.sin(cr)*VESSEL_SPEED, math.cos(cr)*VESSEL_SPEED, 0.0]),
        "last_obs": 0.0,
    }

def _new_state():
    cr0 = math.radians(_VESSEL_ENTRIES[0][1])   # vessel-1 initial course
    return {
        "pos": {k: v.copy() for k, v in _INIT_POS.items()},
        "vel": {
            "diver-1":     np.zeros(3),
            "shark-1":     np.zeros(3),
            "shark-2":     np.zeros(3),
            "vessel-1":    np.array([math.sin(cr0)*VESSEL_SPEED, math.cos(cr0)*VESSEL_SPEED, 0.0]),
            "explosive-1": np.zeros(3),
        },
        "last_obs":     {k: 0.0 for k in _PROFILES},
        "tracker":      MultiActorTracker(
            staleness_threshold=120.0,
            process_noise=0.1,
            intent_module_overrides={"vessel-1": _VesselAreaIntent()},
        ),
        "history":       {k: [] for k in _PROFILES},
        "start":         time.time(),
        "diver_emerg":   False,
        "shark_state":   {"shark-1": "wander", "shark-2": "wander"},
        # diver–shark interaction
        "diver_state":   "transit",
        "diver_aware_t": 0.0,
        "deter_shark":   None,
        "shark_deter_t": {"shark-1": 0.0, "shark-2": 0.0},
        # vessel pool — vessel-1 is static in _INIT_POS; extras are spawned dynamically
        "extra_vessels":      [],
        "next_vessel_id":     2,
        "vessel_entry_idx":   1,       # idx into _VESSEL_ENTRIES for next spawn
        "next_vessel_wall_t": 0.0,     # abs wall-time; initialised on first _run() tick
        # ownship orbit
        "ownship_angle": math.pi * 1.5,
        "ownship_hist":  [],
        # compression state
        "known_state":  {},
        "last_tx_wall": 0.0,
        "last_alert":   None,
        "alert_fresh":  False,
    }

_S = _new_state()
_rng = np.random.default_rng(42)

# ── SSE subscriber registry ───────────────────────────────────────────────────
_subs: list[queue.Queue] = []
_subs_lock = threading.Lock()

def _broadcast(payload: dict):
    frame = ("data: " + json.dumps(payload) + "\n\n").encode()
    with _subs_lock:
        dead = []
        for q in _subs:
            try:
                q.put_nowait(frame)
            except queue.Full:
                dead.append(q)
        for d in dead:
            _subs.remove(d)

# ── Physics ───────────────────────────────────────────────────────────────────
def _step_shark(pos: np.ndarray, diver: np.ndarray, dt: float, attraction: float):
    """
    attraction > 0 : drawn toward diver with that weight
    attraction < 0 : fleeing away from diver with |attraction| weight, 50 % faster
    """
    to = diver - pos
    d  = np.linalg.norm(to)
    unit_to = to / d if d > 0.1 else np.zeros(3)
    rn = _rng.standard_normal(3); rn[2] *= 0.05
    n  = np.linalg.norm(rn)
    rn = rn / n if n > 1e-9 else rn
    if attraction >= 0:
        direction = attraction * unit_to + (1 - attraction) * rn
        speed = SHARK_SPEED
    else:
        w = -attraction                                 # positive weight
        direction = w * (-unit_to) + (1 - w) * rn      # bias away from diver
        speed = SHARK_SPEED * 1.5                       # flee faster
    dn = np.linalg.norm(direction)
    if dn > 1e-9: direction /= dn
    vel = direction * speed
    return pos + vel * dt, vel

def _run():
    global _S
    while True:
        t0 = time.time()
        S = _S

        if t0 - S["start"] > SCENARIO_DURATION:
            _S = _new_state()
            time.sleep(SIM_DT)
            continue

        pos, vel = S["pos"], S["vel"]
        eff_dt = SIM_DT * SIM_SPEED   # simulated seconds per real step

        now = time.time()

        # ── Diver state machine ───────────────────────────────────────────────
        diver_at_eod = bool(np.linalg.norm(pos["diver-1"] - _DIVER_TARGET) < SHARK_EOD_RADIUS)

        if S["diver_state"] == "transit" and diver_at_eod:
            S["diver_state"] = "working"

        if S["diver_state"] == "working":
            # Check whether a hunting shark has entered awareness range
            for sid in ("shark-1", "shark-2"):
                if S["shark_state"][sid] == "hunt":
                    d = float(np.linalg.norm(pos[sid][:2] - pos["diver-1"][:2]))
                    if d < DIVER_AWARE_RANGE and S["deter_shark"] is None:
                        S["diver_state"]   = "aware"
                        S["diver_aware_t"] = now
                        S["deter_shark"]   = sid
                        break

        if S["diver_state"] == "aware":
            if now - S["diver_aware_t"] >= DIVER_REACT_T:
                # Diver fires deterrent — shark is startled and flees
                S["diver_state"] = "deterring"
                dsid = S["deter_shark"]
                if dsid:
                    S["shark_state"][dsid]   = "deterred"
                    S["shark_deter_t"][dsid] = now

        if S["diver_state"] == "deterring":
            dsid = S["deter_shark"]
            shark_dist = (float(np.linalg.norm(pos[dsid][:2] - pos["diver-1"][:2]))
                          if dsid else 999.0)
            if (shark_dist >= DIVER_SAFE_RANGE
                    or now - S["diver_aware_t"] >= DIVER_REACT_T + DIVER_DETER_T):
                S["diver_state"] = "working"
                S["deter_shark"] = None

        # Deterred sharks eventually stop fleeing and return to wander
        for sid in ("shark-1", "shark-2"):
            if S["shark_state"][sid] == "deterred":
                if now - S["shark_deter_t"][sid] >= DIVER_DETER_T:
                    S["shark_state"][sid] = "wander"

        # Sharks may decide to hunt once the diver is on site (and only if not already
        # in a special state)
        for sid in ("shark-1", "shark-2"):
            if not diver_at_eod:
                if S["shark_state"][sid] not in ("deterred",):
                    S["shark_state"][sid] = "wander"
            elif S["shark_state"][sid] == "wander" and _rng.random() < SHARK_HUNT_PROB:
                S["shark_state"][sid] = "hunt"

        # ── Diver movement ────────────────────────────────────────────────────
        # Diver stops while managing a shark encounter
        if S["diver_state"] in ("aware", "deterring"):
            dv = np.zeros(3)
        else:
            to_tgt = _DIVER_TARGET - pos["diver-1"]
            dist   = float(np.linalg.norm(to_tgt))
            dv     = (to_tgt / dist) * DIVER_SPEED if dist > 0.5 else np.zeros(3)
        pos["diver-1"] = pos["diver-1"] + dv * eff_dt
        vel["diver-1"] = dv

        # ── Shark movement ────────────────────────────────────────────────────
        for sid in ("shark-1", "shark-2"):
            st = S["shark_state"][sid]
            if   st == "hunt":     attract =  SHARK_HUNT_ATTRACT
            elif st == "deterred": attract = -SHARK_FLEE_ATTRACT
            else:                  attract =  SHARK_WANDER_ATTRACT
            pos[sid], vel[sid] = _step_shark(pos[sid], pos["diver-1"], eff_dt, attract)

        # ── Vessel pool ───────────────────────────────────────────────────────
        # vessel-1 (static, starts at scenario open)
        pos["vessel-1"] = pos["vessel-1"] + vel["vessel-1"] * eff_dt

        # Spawn additional vessels on a random schedule
        if S["next_vessel_wall_t"] == 0.0:
            # First init (can't use _rng before it is created; safe here)
            S["next_vessel_wall_t"] = now + _rng.uniform(VESSEL_SPAWN_MIN_T,
                                                         VESSEL_SPAWN_MAX_T)
        elif now >= S["next_vessel_wall_t"]:
            idx = S["vessel_entry_idx"] % len(_VESSEL_ENTRIES)
            S["extra_vessels"].append(_make_vessel(S["next_vessel_id"],
                                                   _VESSEL_ENTRIES[idx]))
            S["next_vessel_id"]    += 1
            S["vessel_entry_idx"]  += 1
            S["next_vessel_wall_t"] = now + _rng.uniform(VESSEL_SPAWN_MIN_T,
                                                         VESSEL_SPAWN_MAX_T)

        # Move extra vessels; expire once they leave the op area
        _vprof = ActorProfile.from_strings("vessel", "surface", "human", "cooperative")
        alive = []
        for v in S["extra_vessels"]:
            v["pos"] = v["pos"] + v["vel"] * eff_dt
            if float(np.linalg.norm(v["pos"][:2])) <= VESSEL_EXPIRE_RADIUS:
                alive.append(v)
                if now - v["last_obs"] >= 1.0 / VESSEL_OBS_HZ:
                    S["tracker"].update(v["id"], _vprof, v["pos"], v["vel"],
                                        np.eye(3) * VESSEL_OBS_COV, now)
                    v["last_obs"] = now
        S["extra_vessels"] = alive

        # Ownship – orbit around diver at OWNSHIP_SPEED
        a = S["ownship_angle"]
        dp = pos["diver-1"]
        ox  = dp[0] + OWNSHIP_ORBIT_RADIUS * math.cos(a)
        oy  = dp[1] + OWNSHIP_ORBIT_RADIUS * math.sin(a)
        oz  = dp[2] + 3.0   # 3 m shallower than diver
        ovx = -OWNSHIP_ORBIT_OMEGA * OWNSHIP_ORBIT_RADIUS * math.sin(a)
        ovy =  OWNSHIP_ORBIT_OMEGA * OWNSHIP_ORBIT_RADIUS * math.cos(a)
        S["ownship_angle"] += OWNSHIP_ORBIT_OMEGA * eff_dt
        oh = S["ownship_hist"]
        oh.append([round(ox, 1), round(oy, 1)])
        if len(oh) > OWNSHIP_TRAIL_LEN:
            oh.pop(0)

        # Emergency: diver is in active shark encounter
        S["diver_emerg"] = bool(S["diver_state"] in ("aware", "deterring"))

        # Publish observations  (now = time.time() was set above in state machine)
        ownship_pos = np.array([ox, oy, oz])
        for aid, profile in _PROFILES.items():
            # vessel-1: stop observing once it has left the operational area
            if aid == "vessel-1" and float(np.linalg.norm(pos[aid][:2])) > VESSEL_EXPIRE_RADIUS:
                continue
            # sharks: only observable within MANTA's sonar range
            if aid.startswith("shark") and float(np.linalg.norm(pos[aid] - ownship_pos)) > SONAR_RANGE:
                continue
            if aid == "diver-1":
                hz = DIVER_EMERG_HZ if S["diver_emerg"] else _OBS_HZ_NOMINAL["diver-1"]
            else:
                hz = _OBS_HZ_NOMINAL[aid]
            if now - S["last_obs"][aid] >= 1.0 / hz:
                S["tracker"].update(aid, profile, pos[aid], vel[aid],
                                    np.eye(3) * _OBS_COV[aid], now)
                S["last_obs"][aid] = now

        beliefs = S["tracker"].project_all(now=now)

        # ── Compression pipeline ──────────────────────────────────────────────
        diver_b = next((b for b in beliefs if b.actor_id == "diver-1"), None)
        if diver_b is not None:
            _FORMATTER.update_diver_pose(diver_b.projected_pose)
            _TASK_CTX.diver_pose = diver_b.projected_pose.copy()

        # Triage everything except the diver herself
        ranked_beliefs = [b for b in beliefs if b.actor_id != "diver-1"]
        ranked = _TRIAGE.rank(ranked_beliefs, _TASK_CTX, S["known_state"])

        budget = BandwidthBudget(COMMS_BUDGET)
        selected = _TRIAGE.select_within_budget(ranked, budget, _FORMATTER)
        selected_ids = {b.actor_id for b, _ in selected}

        # Build pending message (what would go out right now)
        pending_frags = [_FORMATTER.format_single(b) for b, _ in selected]
        pending_text = " | ".join(pending_frags)

        ranked_display = [
            {
                "id":       b.actor_id,
                "score":    round(s, 3),
                "text":     _FORMATTER.format_single(b),
                "included": b.actor_id in selected_ids,
                "priority": p,
            }
            for b, s, p in ranked
        ]

        # Transmit on schedule; emergency floor is 10 s, not sub-second
        tx_interval = COMMS_TX_EMERG if S["diver_emerg"] else COMMS_TX_NORMAL
        if now - S["last_tx_wall"] >= tx_interval and selected:
            alert = _FORMATTER.format_alert(
                selected, budget.fraction,
                selected[0][0].actor_id if selected else ""
            )
            alert["chars_used"] = budget.used
            alert["tx_t"]       = round(now - S["start"], 1)
            S["last_alert"]  = alert
            S["last_tx_wall"] = now
            S["alert_fresh"]  = True
            for b, _ in selected:
                S["known_state"][b.actor_id] = b
        else:
            S["alert_fresh"] = False

        comms = {
            # Last actually-transmitted message
            "last_text":     S["last_alert"]["text"]       if S["last_alert"] else "",
            "last_priority": S["last_alert"]["priority"]   if S["last_alert"] else 0,
            "last_tx_t":     S["last_alert"]["tx_t"]       if S["last_alert"] else -1.0,
            "last_chars":    S["last_alert"]["chars_used"] if S["last_alert"] else 0,
            # Pending / live triage
            "pending_text":  pending_text,
            "pending_chars": budget.used,
            "budget_total":  COMMS_BUDGET,
            "ranked":        ranked_display,
            "fresh":         S["alert_fresh"],
        }

        # ── Actor payload ─────────────────────────────────────────────────────
        actors = []
        for b in beliefs:
            aid = b.actor_id
            # Dynamic vessel IDs (vessel-2, vessel-3, …) fall back to vessel defaults
            if aid.startswith("vessel"):
                obs_hz = VESSEL_OBS_HZ
            else:
                obs_hz = _OBS_HZ_NOMINAL.get(aid, 0.5)
            predicted = b.staleness_seconds > (_PRED_FACTOR / obs_hz)

            # Lazily initialise history for dynamically spawned actors
            if aid not in S["history"]:
                S["history"][aid] = []
            hist = S["history"][aid]
            hist.append([round(float(b.projected_pose[0]), 1),
                         round(float(b.projected_pose[1]), 1)])
            if len(hist) > HISTORY_LEN:
                hist.pop(0)
            # Eigendecompose 2-D covariance for oriented uncertainty ellipse
            c2 = b.projected_covariance[:2, :2]
            try:
                vals, vecs = np.linalg.eigh(c2)   # ascending eigenvalues
                cov_a  = round(float(np.sqrt(abs(vals[1]))), 3)  # semi-major (m)
                cov_b  = round(float(np.sqrt(abs(vals[0]))), 3)  # semi-minor (m)
                # angle of major axis in world frame (negate in JS for y-flip)
                cov_ang = round(float(np.arctan2(vecs[1, 1], vecs[0, 1])), 4)
            except Exception:
                cov_a = cov_b = round(float(np.sqrt(max(c2[0, 0], 1e-6))), 3)
                cov_ang = 0.0

            actors.append({
                "id":         aid,
                "label":      b.profile.label,
                "domain":     b.profile.domain.value,
                "source":     "offboard" if aid.startswith("vessel") else _OBS_SOURCE.get(aid, "direct"),
                "predicted":  predicted,
                "emergency":  S["diver_emerg"] if aid == "diver-1" else False,
                "x":          round(float(b.projected_pose[0]), 2),
                "y":          round(float(b.projected_pose[1]), 2),
                "z":          round(float(b.projected_pose[2]), 2),
                "vx":         round(float(b.velocity[0]), 3),
                "vy":         round(float(b.velocity[1]), 3),
                "cov":        max(cov_a, cov_b),   # scalar fallback
                "cov_a":      cov_a,
                "cov_b":      cov_b,
                "cov_angle":  cov_ang,
                "staleness":  round(b.staleness_seconds, 1),
                "confidence": round(b.confidence, 3),
                "intent":     {k: round(v, 3) for k, v in
                               sorted(b.intent.items(), key=lambda x: -x[1])},
                "history":    list(hist),
                # diver-state and shark-state for frontend badges / effects
                "diver_state":  S["diver_state"] if aid == "diver-1" else None,
                "shark_state":  S["shark_state"].get(aid) if aid.startswith("shark") else None,
            })

        ownship_dict = {
            "x": round(ox, 2), "y": round(oy, 2), "z": round(oz, 1),
            "vx": round(ovx, 3), "vy": round(ovy, 3),
            "history": list(oh),
        }

        _broadcast({
            "t":           round(now - S["start"], 1),
            "emerg":       S["diver_emerg"],
            "diver_state": S["diver_state"],
            "ownship":     ownship_dict,
            "actors":      actors,
            "comms":       comms,
        })
        time.sleep(max(0.0, SIM_DT - (time.time() - t0)))

# ── Embedded dashboard ────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MANTA World Model</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#080f1c;color:#90a4ae;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── header ── */
#hdr{display:flex;align-items:center;gap:14px;padding:9px 18px;border-bottom:1px solid #162035;background:#050c18;flex-shrink:0}
#hdr h1{font-size:15px;font-weight:700;color:#e0e0e0;letter-spacing:3px;text-transform:uppercase}
.sep{color:#1e3a5f}
.pill{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:2px 9px;font-size:11px;color:#546e7a}
.pill b{color:#90caf9}
#emerg-pill{display:none;background:#b71c1c;border:1px solid #ef5350;border-radius:10px;padding:2px 9px;font-size:11px;font-weight:700;color:#ffcdd2;animation:blink .7s step-start infinite}
@keyframes blink{50%{opacity:.4}}
#live-dot{margin-left:auto;font-size:11px}

/* ── layout ── */
#main{display:flex;flex:1;overflow:hidden}
#cw{flex:1;background:#040b16;display:flex;align-items:center;justify-content:center;position:relative}
canvas{display:block}

/* ── legend ── */
#legend{position:absolute;bottom:12px;left:14px;background:rgba(4,11,22,.88);border:1px solid #162035;border-radius:4px;padding:8px 11px;font-size:10px;line-height:1.9}
.lr{display:flex;align-items:center;gap:6px}
.ld{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.lc{color:#546e7a}
.lsep{border-top:1px solid #0d1b2a;margin:4px 0}
.ltag{font-size:9px;color:#1e3a5f;padding:1px 4px;border:1px solid #1e3a5f;border-radius:2px}

/* ── scale bar ── */
#scalebar{position:absolute;bottom:14px;right:16px;font-size:9px;color:#1e3a5f}

/* ── sidebar ── */
#sb{width:310px;border-left:1px solid #162035;overflow-y:auto;background:#050c18;padding:10px;flex-shrink:0}
#sb h2{font-size:10px;letter-spacing:2.5px;text-transform:uppercase;color:#37474f;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #0d1b2a}
#sb::-webkit-scrollbar{width:3px}
#sb::-webkit-scrollbar-track{background:transparent}
#sb::-webkit-scrollbar-thumb{background:#162035;border-radius:2px}

/* ── actor card ── */
.ac{background:#08111e;border:1px solid #0d1b2a;border-left:3px solid #1e3a5f;border-radius:3px;padding:8px 9px;margin-bottom:7px;transition:border-left-color .35s,box-shadow .35s}
.ac.th{border-left-color:#ef5350!important;box-shadow:0 0 14px rgba(239,83,80,.2)}
.ac.tw{border-left-color:#ffb300!important;box-shadow:0 0 8px rgba(255,179,0,.12)}
.ah{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.aid{font-size:12px;font-weight:700;font-family:'Courier New',monospace}
.az{font-size:9px;color:#37474f;font-family:'Courier New',monospace}
.am{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:5px}
.bdg{font-size:9px;padding:1px 5px;border-radius:2px;background:#0d1b2a;color:#546e7a}
.bdg-pred{font-size:9px;padding:1px 5px;border-radius:2px;background:#1a1200;color:#ffb300;border:1px solid #3d2f00}
.bdg-offb{font-size:9px;padding:1px 5px;border-radius:2px;background:#0d1825;color:#4fc3f7;border:1px solid #0d3a5f}
.bdg-obs{font-size:9px;padding:1px 5px;border-radius:2px;background:#001a0d;color:#69f0ae;border:1px solid #003d1a}
.bdg-emerg{font-size:9px;padding:1px 5px;border-radius:2px;background:#3d0000;color:#ff8a80;border:1px solid #7f0000;animation:blink .7s step-start infinite}
.bdg-aware{font-size:9px;padding:1px 5px;border-radius:2px;background:#1a1200;color:#ffb300;border:1px solid #3d2f00;animation:blink .5s step-start infinite}
.bdg-deter{font-size:9px;padding:1px 5px;border-radius:2px;background:#001428;color:#42a5f5;border:1px solid #0d3a5f}
.bdg-work{font-size:9px;padding:1px 5px;border-radius:2px;background:#001a0d;color:#69f0ae;border:1px solid #003d1a}
.bdg-flee{font-size:9px;padding:1px 5px;border-radius:2px;background:#0d001a;color:#ce93d8;border:1px solid #3d005f}
/* intent bars */
.il{margin-top:2px}
.ir{display:grid;grid-template-columns:110px 1fr 36px;align-items:center;gap:4px;margin-bottom:2px}
.ik{font-size:9px;color:#546e7a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ibg{height:4px;background:#0d1b2a;border-radius:2px;overflow:hidden}
.ibf{height:100%;border-radius:2px;transition:width .4s ease}
.ip{font-size:9px;color:#37474f;text-align:right;font-family:'Courier New',monospace}
/* staleness */
.stl{display:flex;align-items:center;gap:5px;margin-bottom:4px}
.stl-bar{flex:1;height:3px;background:#0d1b2a;border-radius:2px;overflow:hidden}
.stl-fill{height:100%;border-radius:2px;transition:width .4s ease,background .4s ease}
.stl-label{font-size:8px;font-family:'Courier New',monospace;white-space:nowrap}
/* footer */
.af{display:flex;gap:10px;margin-top:5px;font-size:9px;color:#37474f;font-family:'Courier New',monospace}
.af b{color:#546e7a}
.cb{height:2px;background:#0d1b2a;border-radius:1px;margin-top:4px;overflow:hidden}
.cf{height:100%;background:linear-gradient(90deg,#0288d1,#00e5ff);border-radius:1px;transition:width .4s ease}

/* ── comms panel ── */
#comms-section{margin-top:14px;padding-top:12px;border-top:2px solid #0d1b2a}
.comms-sub{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#1e3a5f;margin:8px 0 4px}
.tx-meta{display:flex;gap:7px;align-items:center;margin-bottom:5px}
.pri-badge{font-size:9px;padding:1px 6px;border-radius:2px;font-weight:700;flex-shrink:0}
.pri-0{background:#001020;color:#4fc3f7;border:1px solid #0d3a5f}
.pri-1{background:#1a1200;color:#ffb300;border:1px solid #3d2f00}
.pri-2{background:#200000;color:#ef5350;border:1px solid #5f0000;animation:blink 1s step-start infinite}
.tx-time{font-size:9px;color:#37474f;font-family:'Courier New',monospace}
/* terminal message box */
.msg-box{background:#020810;border:1px solid #0d1b2a;border-radius:3px;padding:7px 9px;font-family:'Courier New',monospace;font-size:10px;color:#b0bec5;line-height:1.6;min-height:32px;word-break:break-all;margin-bottom:5px}
.msg-box.dim{color:#37474f}
.msg-box.flash{border-color:#69f0ae;animation:txflash 1.2s ease-out forwards}
@keyframes txflash{
  0%{border-color:#69f0ae;box-shadow:0 0 10px rgba(105,240,174,.5)}
  60%{border-color:#162035;box-shadow:none}
  100%{border-color:#0d1b2a;box-shadow:none}
}
/* budget bar */
.budget-row{display:flex;align-items:center;gap:6px;margin-bottom:6px}
.budget-label{font-size:8px;color:#37474f;font-family:'Courier New',monospace;white-space:nowrap}
.budget-bar{flex:1;height:5px;background:#0d1b2a;border-radius:3px;overflow:hidden}
.budget-fill{height:100%;border-radius:3px;transition:width .3s ease,background .3s ease}
.budget-chars{font-size:8px;font-family:'Courier New',monospace;color:#37474f;white-space:nowrap}
/* triage list */
.triage-list{margin-top:2px}
.triage-row{display:grid;grid-template-columns:10px 78px 32px 1fr;align-items:baseline;gap:4px;padding:3px 0;border-bottom:1px solid #050e1a;font-size:9px;font-family:'Courier New',monospace}
.ti-ok{color:#69f0ae}
.ti-no{color:#1e3a5f}
.t-id{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#546e7a}
.t-id.ok{color:#90a4ae}
.t-score{text-align:right;color:#37474f}
.t-score.ok{color:#546e7a}
.t-txt{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#1e3a5f}
.t-txt.ok{color:#546e7a}
/* ── TX toast ── */
#tx-toast{position:absolute;bottom:62px;left:50%;transform:translateX(-50%);
  background:rgba(3,10,21,.94);border:1px solid #1e3a5f;border-radius:4px;
  padding:7px 16px;font-family:'Courier New',monospace;font-size:11px;
  text-align:center;min-width:190px;max-width:360px;word-break:break-word;
  pointer-events:none;opacity:0;transition:opacity .25s ease;z-index:20;
  line-height:1.5}
</style>
</head>
<body>
<div id="hdr">
  <h1>MANTA</h1>
  <span class="sep">·</span>
  <span style="font-size:11px;color:#37474f">Multi-Actor World Model</span>
  <span class="sep">·</span>
  <div class="pill">T+ <b id="st">0.0</b> s</div>
  <div class="pill"><b id="ac">0</b> actors</div>
  <div id="emerg-pill">⚠ DIVER EMERGENCY</div>
  <!-- SPEED_PILL -->
  <div id="live-dot" class="pill"><b style="color:#69f0ae">● LIVE</b></div>
</div>

<div id="main">
  <div id="cw">
    <canvas id="map" width="620" height="554"></canvas>
    <div id="tx-toast"></div>
    <div id="legend">
      <div class="lr"><div class="ld" style="background:#00e5ff"></div><span class="lc">Diver</span></div>
      <div class="lr"><div class="ld" style="background:#ff7043"></div><span class="lc">Shark</span></div>
      <div class="lr"><div class="ld" style="background:#69f0ae"></div><span class="lc">Vessel</span></div>
      <div class="lr"><div class="ld" style="background:#ef5350"></div><span class="lc">Explosive</span></div>
      <div class="lr"><div class="ld" style="background:#b0bec5;border-radius:2px"></div><span class="lc">Ownship (Manta)</span></div>
      <div class="lsep"></div>
      <div class="lr"><span class="lc">Dashed ellipse = </span><span class="ltag">OBSERVED</span></div>
      <div class="lr"><span class="lc">Solid / pulsing = </span><span class="ltag">PREDICTED</span></div>
      <div class="lr" style="margin-top:2px"><span style="color:#1e3a5f;font-size:9px">Ellipse grows · fades as prediction ages</span></div>
      <div class="lr"><span style="color:#1e3a5f;font-size:9px">Flash + line = correction on new obs</span></div>
    </div>
    <div id="scalebar">——————<br>40 m</div>
  </div>
  <div id="sb">
    <h2>Actor Beliefs</h2>
    <div id="cards"></div>

    <!-- ── Diver Comms panel ── -->
    <div id="comms-section">
      <h2>Diver Comms Channel</h2>

      <div class="comms-sub">Last transmitted</div>
      <div class="tx-meta">
        <span id="comms-pri" class="pri-badge pri-0">INFO</span>
        <span class="tx-time">T+ <span id="comms-tx-t">—</span></span>
        <span id="comms-chars-last" class="tx-time" style="margin-left:auto"></span>
      </div>
      <div class="msg-box dim" id="comms-msg">— awaiting first transmission —</div>

      <div class="comms-sub" style="margin-top:8px">Pending (next transmission)</div>
      <div class="msg-box dim" id="comms-pending">—</div>
      <div class="budget-row">
        <span class="budget-label">Budget</span>
        <div class="budget-bar"><div class="budget-fill" id="comms-bfill" style="width:0%"></div></div>
        <span class="budget-chars" id="comms-bchars">0 / 160</span>
      </div>

      <div class="comms-sub">Triage ranking</div>
      <div id="comms-triage" class="triage-list"></div>
    </div>
  </div>
</div>

<script>
const canvas = document.getElementById('map');
const ctx    = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const CX = W / 2, CY = H / 2;
const SCALE = 4;

function wx(x){ return CX + x * SCALE; }
function wy(y){ return CY - y * SCALE; }

const COL = { diver:'#00e5ff', shark:'#ff7043', vessel:'#69f0ae', explosive:'#ef5350' };
const THREAT_SET  = new Set(['approaching_diver','coordinated_approach','inbound']);
const TRANSIT_SET = new Set(['transiting_area']);

function threatLevel(a){
  if(a.label==='explosive') return 'p';
  const e=Object.entries(a.intent); if(!e.length) return 'n';
  e.sort((x,y)=>y[1]-x[1]);
  const [l,p]=e[0];
  if(THREAT_SET.has(l))  return p>.7?'h':p>.35?'m':'n';
  if(TRANSIT_SET.has(l)) return p>.8?'m':'n';
  return 'n';
}

function iColor(label,prob){
  if(THREAT_SET.has(label))  return prob>.6?'#ef5350':prob>.3?'#ffb300':'#42a5f5';
  if(TRANSIT_SET.has(label)) return prob>.5?'#ffb300':'#42a5f5';
  return '#42a5f5';
}

/* ── background ── */
function drawBg(){
  ctx.fillStyle='#030a15'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#0b1825'; ctx.lineWidth=1; ctx.setLineDash([]);
  for(let g=-80;g<=80;g+=20){
    ctx.beginPath(); ctx.moveTo(wx(g),0); ctx.lineTo(wx(g),H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,wy(g)); ctx.lineTo(W,wy(g)); ctx.stroke();
  }
  ctx.fillStyle='#162035'; ctx.font='9px monospace'; ctx.textAlign='center';
  for(let g=-60;g<=60;g+=20){
    if(g===0) continue;
    ctx.fillText(g+'m', wx(g), wy(-72)+11);
    ctx.textAlign='right'; ctx.fillText(g+'m', wx(-74)+14, wy(g)+4);
    ctx.textAlign='center';
  }
  ctx.fillStyle='#1e3a5f'; ctx.font='bold 10px sans-serif';
  ctx.textAlign='left'; ctx.fillText('N↑', wx(-72)+2, wy(67));
  ctx.beginPath(); ctx.arc(wx(0),wy(0),50*SCALE,0,Math.PI*2);
  ctx.strokeStyle='#162035'; ctx.lineWidth=1; ctx.setLineDash([4,4]); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='rgba(0,229,255,.018)'; ctx.fill();
}

/* ── ownship ── */
function drawOwnship(o){
  if(!o) return;
  const ox=wx(o.x), oy=wy(o.y), c='#b0bec5';

  // orbit trail
  if(o.history && o.history.length > 1){
    for(let i=1;i<o.history.length;i++){
      const alpha=Math.floor((i/o.history.length)*45).toString(16).padStart(2,'0');
      ctx.beginPath();
      ctx.moveTo(wx(o.history[i-1][0]), wy(o.history[i-1][1]));
      ctx.lineTo(wx(o.history[i][0]),   wy(o.history[i][1]));
      ctx.strokeStyle=c+alpha; ctx.lineWidth=1.2; ctx.setLineDash([]); ctx.stroke();
    }
  }

  // AUV hull oriented along velocity (canvas y is inverted)
  const heading = Math.atan2(-(o.vy||0), o.vx||0);
  ctx.save(); ctx.translate(ox,oy); ctx.rotate(heading);
  ctx.beginPath(); ctx.ellipse(0,0,16,6,0,0,Math.PI*2);
  ctx.fillStyle='rgba(176,190,197,.15)'; ctx.fill();
  ctx.strokeStyle=c; ctx.lineWidth=1.5; ctx.setLineDash([]); ctx.stroke();
  // bow stub
  ctx.beginPath(); ctx.moveTo(14,0); ctx.lineTo(21,0);
  ctx.strokeStyle=c; ctx.lineWidth=2; ctx.stroke();
  // forward sensor arc
  ctx.beginPath(); ctx.arc(0,0,30,-.65,.65);
  ctx.strokeStyle=c+'25'; ctx.lineWidth=1; ctx.setLineDash([2,4]); ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  ctx.font='10px monospace'; ctx.textAlign='left'; ctx.fillStyle=c+'cc';
  ctx.fillText('MANTA', ox+22, oy-4);
  ctx.font='9px monospace'; ctx.fillStyle='#37474f';
  ctx.fillText('z='+o.z.toFixed(0)+'m  ~3kt', ox+22, oy+7);
}

/* ── trail ── */
function drawTrail(a){
  if(!a.history||a.history.length<2) return;
  const c=COL[a.label]||'#fff';
  for(let i=1;i<a.history.length;i++){
    const alpha=Math.floor((i/a.history.length)*80).toString(16).padStart(2,'0');
    ctx.beginPath();
    ctx.moveTo(wx(a.history[i-1][0]),wy(a.history[i-1][1]));
    ctx.lineTo(wx(a.history[i][0]),wy(a.history[i][1]));
    ctx.strokeStyle=c+alpha; ctx.lineWidth=1.5; ctx.setLineDash([]); ctx.stroke();
  }
}

/* ── uncertainty ellipse ── */
function drawUnc(a){
  const c=COL[a.label]||'#fff';
  // Ellipse semi-axes from Kalman covariance eigendecomposition (metres → pixels)
  const M  = SCALE * 4.0;
  const ra = Math.max(10, (a.cov_a || a.cov) * M);
  const rb = Math.max( 6, (a.cov_b || a.cov * 0.65) * M);
  const ang = -(a.cov_angle || 0);   // negate for canvas y-flip

  if(a.predicted){
    // Fade fill and stroke progressively as the prediction ages
    const sf = Math.min(1, a.staleness / 90);   // 0 = just-turned-predicted, 1 = 90s+ stale
    const fillA = Math.round((0.22 - sf*0.13)*255).toString(16).padStart(2,'0');
    const ringA = Math.round((0.60 - sf*0.32)*255).toString(16).padStart(2,'0');
    ctx.beginPath(); ctx.ellipse(wx(a.x),wy(a.y),ra,rb,ang,0,Math.PI*2);
    ctx.fillStyle  =c+fillA; ctx.fill();
    ctx.strokeStyle=c+ringA; ctx.lineWidth=1.5; ctx.setLineDash([]); ctx.stroke();
    // Pulsing outer ring — also fades with age
    const pulse=0.5+0.5*Math.sin(Date.now()/500);
    const pA=Math.round((0.07+pulse*0.10)*(1-sf*0.55)*255).toString(16).padStart(2,'0');
    ctx.beginPath(); ctx.ellipse(wx(a.x),wy(a.y),ra+4+pulse*3,rb+2+pulse*2,ang,0,Math.PI*2);
    ctx.strokeStyle=c+pA; ctx.lineWidth=1; ctx.setLineDash([]); ctx.stroke();
  } else {
    ctx.beginPath(); ctx.ellipse(wx(a.x),wy(a.y),ra,rb,ang,0,Math.PI*2);
    ctx.fillStyle  =c+'15'; ctx.fill();
    ctx.strokeStyle=c+'45'; ctx.lineWidth=1; ctx.setLineDash([2,3]); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.setLineDash([]);
}

/* ── velocity projection ── */
function drawVel(a){
  const spd=Math.sqrt(a.vx**2+a.vy**2);
  if(spd<0.05) return;
  const c=COL[a.label]||'#fff';
  const tx=wx(a.x+a.vx*5), ty=wy(a.y+a.vy*5);
  const ox=wx(a.x), oy=wy(a.y);
  ctx.beginPath(); ctx.moveTo(ox,oy); ctx.lineTo(tx,ty);
  ctx.strokeStyle=c+'55'; ctx.lineWidth=1.5; ctx.setLineDash([3,3]); ctx.stroke();
  ctx.setLineDash([]);
  const ang=Math.atan2(ty-oy,tx-ox);
  ctx.beginPath();
  ctx.moveTo(tx,ty);
  ctx.lineTo(tx-7*Math.cos(ang-.4),ty-7*Math.sin(ang-.4));
  ctx.lineTo(tx-7*Math.cos(ang+.4),ty-7*Math.sin(ang+.4));
  ctx.closePath(); ctx.fillStyle=c+'55'; ctx.fill();
}

/* ── icon ── */
function drawIcon(a){
  const ox=wx(a.x), oy=wy(a.y);
  const c=COL[a.label]||'#fff';
  const tl=threatLevel(a);
  const dc=tl==='h'?'#ef5350':tl==='m'?'#ffb300':c;
  const dash=a.predicted?[3,3]:[];

  // Ghost-fade: icons become translucent as the prediction ages
  const sf = a.predicted ? Math.min(1, a.staleness / 90) : 0;
  const iA = Math.round((1.0 - sf * 0.65) * 255).toString(16).padStart(2,'0');

  ctx.save(); ctx.translate(ox,oy); ctx.setLineDash(dash);

  if(a.label==='diver'){
    ctx.beginPath(); ctx.arc(0,0,9,0,Math.PI*2);
    ctx.fillStyle=a.predicted?'transparent':c+'22'; ctx.fill();
    ctx.strokeStyle=dc+iA; ctx.lineWidth=2.5; ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(-5,0); ctx.lineTo(5,0); ctx.moveTo(0,-5); ctx.lineTo(0,5);
    ctx.strokeStyle=dc+iA; ctx.lineWidth=2; ctx.stroke();
  } else if(a.label==='shark'){
    const va=Math.atan2(-a.vy,a.vx);
    ctx.rotate(va-Math.PI/2);
    ctx.beginPath(); ctx.moveTo(0,-11); ctx.lineTo(-6,7); ctx.lineTo(0,3); ctx.lineTo(6,7);
    ctx.closePath();
    ctx.fillStyle=a.predicted?'transparent':dc+iA;
    ctx.strokeStyle=dc+iA; ctx.lineWidth=1.5;
    a.predicted?ctx.stroke():ctx.fill();
  } else if(a.label==='vessel'){
    const va=Math.atan2(-a.vy,a.vx);
    ctx.rotate(va);
    ctx.beginPath();
    ctx.moveTo(13,0); ctx.lineTo(7,-5); ctx.lineTo(-10,-5);
    ctx.lineTo(-10,5); ctx.lineTo(7,5); ctx.closePath();
    ctx.fillStyle=a.predicted?'transparent':c+'2a'; ctx.fill();
    ctx.strokeStyle=dc+iA; ctx.lineWidth=2; ctx.stroke();
  } else if(a.label==='explosive'){
    ctx.beginPath(); ctx.arc(0,0,8,0,Math.PI*2);
    ctx.strokeStyle=dc+iA; ctx.lineWidth=2; ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(-5,-5); ctx.lineTo(5,5); ctx.moveTo(5,-5); ctx.lineTo(-5,5);
    ctx.strokeStyle=dc+iA; ctx.lineWidth=2.5; ctx.stroke();
  }

  ctx.restore(); ctx.setLineDash([]);
  ctx.font='10px monospace'; ctx.textAlign='left'; ctx.fillStyle=c+iA;
  ctx.fillText(a.id, ox+13, oy-4);
  ctx.font='9px monospace'; ctx.fillStyle='#2a4060';
  ctx.fillText('z='+a.z.toFixed(0)+'m', ox+13, oy+6);
  if(a.predicted){
    // Show how long ago the last observation was; colour shifts orange → red with age
    const staleStr = a.staleness >= 60
      ? (a.staleness/60).toFixed(1)+'m ago'
      : Math.round(a.staleness)+'s ago';
    const predCol = sf > 0.55 ? '#ef5350' : '#ffb300';
    ctx.font='8px monospace'; ctx.fillStyle=predCol+iA;
    ctx.fillText('PRED ' + staleStr, ox+13, oy+15);
  }
  const ie=Object.entries(a.intent).sort((x,y)=>y[1]-x[1]);
  if(ie.length&&ie[0][1]>.45){
    const [il,ip]=ie[0];
    ctx.font='9px sans-serif'; ctx.fillStyle=iColor(il,ip);
    ctx.fillText('▸ '+il, ox+13, oy+(a.predicted?24:16));
  }
}

/* ── TX beam + toast ── */
let _txFlash = null;   // {ox,oy,dx,dy,color,t}  — ownship→diver transmission beam
const TX_DUR = 2600;   // ms total beam lifetime

function drawTxBeam(){
  if(!_txFlash) return;
  const age = Date.now() - _txFlash.t;
  if(age > TX_DUR){ _txFlash = null; return; }
  const frac  = age / TX_DUR;
  const alpha = Math.pow(1 - frac, 1.4);
  const ox = wx(_txFlash.ox), oy = wy(_txFlash.oy);
  const dx = wx(_txFlash.dx), dy = wy(_txFlash.dy);
  const c  = _txFlash.color;

  // Animated dashed beam (dashes scroll from UUV toward diver)
  ctx.save();
  ctx.beginPath(); ctx.moveTo(ox,oy); ctx.lineTo(dx,dy);
  ctx.strokeStyle = c + Math.round(alpha * 170).toString(16).padStart(2,'0');
  ctx.lineWidth   = 1.8;
  ctx.setLineDash([9, 5]);
  ctx.lineDashOffset = -((age / 1000) * 28);  // scrolling dash animation
  ctx.stroke(); ctx.restore(); ctx.setLineDash([]);

  // Ping dot travelling from UUV to diver (arrives in first 550 ms)
  const pingFrac = Math.min(1, age / 550);
  const px = ox + (dx - ox) * pingFrac;
  const py = oy + (dy - oy) * pingFrac;
  ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI*2);
  ctx.fillStyle = c + Math.round(alpha * 200).toString(16).padStart(2,'0'); ctx.fill();

  // Expanding ring at diver end once ping arrives
  if(age > 520){
    const ringAge = age - 520;
    const ringFrac = Math.min(1, ringAge / 800);
    const ringA = (1 - ringFrac) * alpha;
    ctx.beginPath(); ctx.arc(dx, dy, 12 + ringFrac * 28, 0, Math.PI*2);
    ctx.strokeStyle = c + Math.round(ringA * 210).toString(16).padStart(2,'0');
    ctx.lineWidth = 2; ctx.setLineDash([]); ctx.stroke();
  }
}

const PRI_COLORS = ['#42a5f5','#ffb300','#ef5350'];
const PRI_NAMES  = ['INFO','WARNING','SAFETY CRITICAL'];

function showTxToast(text, priority){
  const el = document.getElementById('tx-toast');
  const c  = PRI_COLORS[priority] || PRI_COLORS[0];
  el.style.borderColor = c;
  el.style.color       = c;
  el.innerHTML = `<span style="font-size:8px;opacity:.6;letter-spacing:1.5px">`+
    `TX → DIVER · ${PRI_NAMES[priority]||'INFO'}</span><br>${text}`;
  el.style.opacity = '1';
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.style.opacity = '0'; }, 3200);
}

/* ── diver awareness / deterrence effect ── */
// diverActor: the actor object for diver-1
// diverState: top-level diver_state string from server
let _diverState = 'transit';
function drawDiverEffect(diverActor){
  if(!diverActor) return;
  const ds = diverActor.diver_state || _diverState;
  if(ds !== 'aware' && ds !== 'deterring') return;
  const ox = wx(diverActor.x), oy = wy(diverActor.y);
  const now = Date.now();

  if(ds === 'aware'){
    // Pulsing amber warning ring — diver has spotted the shark
    const pulse = 0.5 + 0.5 * Math.sin(now / 280);
    const r = 18 + pulse * 7;
    ctx.beginPath(); ctx.arc(ox, oy, r, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffb300' + Math.round((0.55 + pulse * 0.30) * 255).toString(16).padStart(2,'0');
    ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([]);
    // Label
    ctx.font = 'bold 9px monospace'; ctx.textAlign = 'center';
    ctx.fillStyle = '#ffb300cc';
    ctx.fillText('⚠ SHARK NEARBY', ox, oy - 26);
    ctx.textAlign = 'left';
  } else {
    // Deterring: two expanding concentric rings (acoustic deterrent pulse)
    const period = 1100; // ms per cycle
    for(let offset = 0; offset < 2; offset++){
      const phase = ((now + offset * period / 2) % period) / period;  // 0..1
      const r = 16 + phase * 52;
      const alpha = (1 - phase) * (offset === 0 ? 0.85 : 0.55);
      ctx.beginPath(); ctx.arc(ox, oy, r, 0, Math.PI * 2);
      ctx.strokeStyle = '#42a5f5' + Math.round(alpha * 255).toString(16).padStart(2,'0');
      ctx.lineWidth = offset === 0 ? 2.5 : 1.5;
      ctx.setLineDash([]); ctx.stroke();
    }
    // Label
    ctx.font = 'bold 9px monospace'; ctx.textAlign = 'center';
    ctx.fillStyle = '#42a5f5bb';
    ctx.fillText('◉ DETERRING', ox, oy - 26);
    ctx.textAlign = 'left';
  }
}

/* ── prediction-correction flash ── */
// When a predicted actor receives a new observation the icon snaps to the real
// position.  We record the ghost (old predicted location) and draw a brief
// expanding ring + dashed correction line to the new observed position.
const _prevActor   = {};   // id → {predicted, x, y}
const _corrections = [];   // [{px,py,ox,oy,color,t}]
const _CORR_DUR    = 1800; // ms

function drawCorrections(){
  const now = Date.now();
  let i = _corrections.length;
  while(i--){
    const cr = _corrections[i];
    const age = now - cr.t;
    if(age > _CORR_DUR){ _corrections.splice(i,1); continue; }
    const frac  = age / _CORR_DUR;
    const fadeA = 1 - frac;

    // Expanding ghost ring at old predicted position
    const r = 10 + frac * 28;
    ctx.beginPath(); ctx.arc(wx(cr.px), wy(cr.py), r, 0, Math.PI*2);
    ctx.strokeStyle = cr.color + Math.round(fadeA * 170).toString(16).padStart(2,'0');
    ctx.lineWidth = 1.5; ctx.setLineDash([2,3]); ctx.stroke(); ctx.setLineDash([]);

    // Dashed correction line (fades out first, over first 55% of animation)
    if(frac < 0.55){
      const lineA = 1 - frac / 0.55;
      ctx.beginPath();
      ctx.moveTo(wx(cr.px), wy(cr.py));
      ctx.lineTo(wx(cr.ox), wy(cr.oy));
      ctx.strokeStyle = cr.color + Math.round(lineA * 195).toString(16).padStart(2,'0');
      ctx.lineWidth = 1.5; ctx.setLineDash([5,3]); ctx.stroke(); ctx.setLineDash([]);
      // Arrow head at observed position
      const ang = Math.atan2(wy(cr.oy)-wy(cr.py), wx(cr.ox)-wx(cr.px));
      ctx.beginPath();
      ctx.moveTo(wx(cr.ox), wy(cr.oy));
      ctx.lineTo(wx(cr.ox)-8*Math.cos(ang-.45), wy(cr.oy)-8*Math.sin(ang-.45));
      ctx.lineTo(wx(cr.ox)-8*Math.cos(ang+.45), wy(cr.oy)-8*Math.sin(ang+.45));
      ctx.closePath();
      ctx.fillStyle = cr.color + Math.round(lineA * 195).toString(16).padStart(2,'0');
      ctx.fill();
    }

    // Brief bright flash ring at the new observed position (first 30%)
    if(frac < 0.30){
      const flashA = 1 - frac / 0.30;
      ctx.beginPath(); ctx.arc(wx(cr.ox), wy(cr.oy), 18 - frac*10, 0, Math.PI*2);
      ctx.strokeStyle = cr.color + Math.round(flashA * 230).toString(16).padStart(2,'0');
      ctx.lineWidth = 2.5; ctx.setLineDash([]); ctx.stroke();
    }
  }
  ctx.setLineDash([]);
}

function render(actors, ownship){
  const diverActor = actors.find(a => a.id === 'diver-1');
  drawBg();
  drawOwnship(ownship);
  drawTxBeam();                 // acoustic comms beam (below uncertainty rings)
  for(const a of actors) drawTrail(a);
  for(const a of actors) drawUnc(a);
  drawDiverEffect(diverActor);  // awareness / deterrence rings (below icons)
  drawCorrections();
  for(const a of actors) drawVel(a);
  for(const a of actors) drawIcon(a);
}

/* ── sidebar: actor cards ── */
function updateSidebar(actors){
  const sortOrder={h:0,m:1,n:2,p:3};
  const sorted=[...actors].sort((a,b)=>sortOrder[threatLevel(a)]-sortOrder[threatLevel(b)]);
  for(const a of sorted){
    const tid='card-'+a.id;
    let card=document.getElementById(tid);
    if(!card){
      card=document.createElement('div');
      card.id=tid;
      document.getElementById('cards').appendChild(card);
    }
    const c=COL[a.label]||'#fff';
    const tl=threatLevel(a);
    card.className='ac'+(tl==='h'?' th':tl==='m'?' tw':'');
    card.style.borderLeftColor=c;
    const srcBadge=a.source==='offboard'?'<span class="bdg-offb">OFFBOARD RPT</span>':'';
    const stateBadge=a.predicted?'<span class="bdg-pred">PREDICTED</span>':'<span class="bdg-obs">OBSERVED</span>';
    const emergBadge=a.emergency?'<span class="bdg-emerg">EMERGENCY OPS</span>':'';
    // Diver-state badge
    let diverStateBadge='';
    if(a.diver_state==='transit')    diverStateBadge='';
    else if(a.diver_state==='working')   diverStateBadge='<span class="bdg-work">ON SITE</span>';
    else if(a.diver_state==='aware')     diverStateBadge='<span class="bdg-aware">⚠ SHARK NEARBY</span>';
    else if(a.diver_state==='deterring') diverStateBadge='<span class="bdg-deter">◉ DETERRING</span>';
    // Shark-state badge
    let sharkStateBadge='';
    if(a.shark_state==='hunt')      sharkStateBadge='<span class="bdg-emerg">⟶ HUNTING</span>';
    else if(a.shark_state==='deterred') sharkStateBadge='<span class="bdg-flee">⟵ DETERRED</span>';
    const staleMax=120;
    const stalePct=Math.min(100,(a.staleness/staleMax)*100).toFixed(0);
    const staleColor=a.predicted?'#ffb300':'#37474f';
    const intents=Object.entries(a.intent).sort((x,y)=>y[1]-x[1]);
    const intentHTML=intents.length
      ?intents.map(([l,p])=>`
          <div class="ir">
            <span class="ik">${l}</span>
            <div class="ibg"><div class="ibf" style="width:${(p*100).toFixed(0)}%;background:${iColor(l,p)}"></div></div>
            <span class="ip">${p.toFixed(3)}</span>
          </div>`).join('')
      :'<span style="font-size:9px;color:#1e3a5f">passive / no intent model</span>';
    const spd=Math.sqrt(a.vx**2+a.vy**2).toFixed(2);
    card.innerHTML=`
      <div class="ah">
        <span class="aid" style="color:${c}">${a.id}</span>
        <span class="az">${a.z.toFixed(1)} m</span>
      </div>
      <div class="am">
        <span class="bdg">${a.label}</span>
        <span class="bdg">${a.domain}</span>
        ${stateBadge}${srcBadge}${emergBadge}${diverStateBadge}${sharkStateBadge}
      </div>
      <div class="stl">
        <span class="stl-label" style="color:${staleColor}">age ${a.staleness.toFixed(0)}s</span>
        <div class="stl-bar"><div class="stl-fill" style="width:${stalePct}%;background:${staleColor}"></div></div>
      </div>
      <div class="il">${intentHTML}</div>
      <div class="cb"><div class="cf" style="width:${(a.confidence*100).toFixed(0)}%"></div></div>
      <div class="af">
        <span>conf <b>${a.confidence.toFixed(3)}</b></span>
        <span>spd <b>${spd} m/s</b></span>
      </div>`;
  }
  const ids=new Set(actors.map(a=>a.id));
  for(const card of document.querySelectorAll('.ac')){
    if(!ids.has(card.id.replace('card-',''))){ card.remove(); }
  }
}

/* ── sidebar: diver comms ── */
const PRI_LABEL = ['INFO', 'WARNING', 'SAFETY CRITICAL'];
const PRI_CLASS = ['pri-0', 'pri-1', 'pri-2'];

function updateComms(c){
  if(!c) return;

  // Last transmitted
  const msgBox = document.getElementById('comms-msg');
  if(c.last_text){
    msgBox.textContent = c.last_text;
    msgBox.classList.remove('dim');
    if(c.fresh){
      msgBox.classList.remove('flash');
      void msgBox.offsetWidth;   // reflow to restart animation
      msgBox.classList.add('flash');
    }
  } else {
    msgBox.textContent = '— awaiting first transmission —';
    msgBox.classList.add('dim');
  }

  const pri = (c.last_priority !== undefined && c.last_text) ? c.last_priority : -1;
  const priEl = document.getElementById('comms-pri');
  if(pri >= 0){
    priEl.textContent  = PRI_LABEL[pri];
    priEl.className    = 'pri-badge ' + PRI_CLASS[pri];
  }
  document.getElementById('comms-tx-t').textContent =
    c.last_tx_t >= 0 ? c.last_tx_t.toFixed(0)+'s' : '—';
  document.getElementById('comms-chars-last').textContent =
    c.last_chars ? c.last_chars+' / '+c.budget_total+' ch' : '';

  // Pending message
  const pendBox = document.getElementById('comms-pending');
  if(c.pending_text){
    pendBox.textContent = c.pending_text;
    pendBox.classList.remove('dim');
  } else {
    pendBox.textContent = '—';
    pendBox.classList.add('dim');
  }

  // Budget bar (pending)
  const pct = c.budget_total > 0 ? (c.pending_chars / c.budget_total * 100) : 0;
  const bColor = pct > 90 ? '#ef5350' : pct > 70 ? '#ffb300' : '#42a5f5';
  document.getElementById('comms-bfill').style.width  = pct.toFixed(0)+'%';
  document.getElementById('comms-bfill').style.background = bColor;
  document.getElementById('comms-bchars').textContent =
    c.pending_chars + ' / ' + c.budget_total;

  // Triage ranking
  const list = document.getElementById('comms-triage');
  list.innerHTML = '';
  for(const row of (c.ranked || [])){
    const div = document.createElement('div');
    div.className = 'triage-row';
    div.innerHTML =
      `<span class="${row.included?'ti-ok':'ti-no'}">${row.included?'✓':'✗'}</span>`+
      `<span class="t-id ${row.included?'ok':''}">${row.id}</span>`+
      `<span class="t-score ${row.included?'ok':''}">${row.score.toFixed(2)}</span>`+
      `<span class="t-txt ${row.included?'ok':''}">${row.text}</span>`;
    list.appendChild(div);
  }
}

/* ── SSE ── */
const sse = new EventSource('/stream');
sse.onmessage = e => {
  const s = JSON.parse(e.data);
  document.getElementById('st').textContent = s.t.toFixed(1);
  document.getElementById('ac').textContent = s.actors.length;
  document.getElementById('emerg-pill').style.display = s.emerg ? 'block' : 'none';

  _diverState = s.diver_state || 'transit';

  // TX beam + toast on fresh transmission
  if(s.comms && s.comms.fresh && s.comms.last_text){
    const da = s.actors.find(a => a.id === 'diver-1');
    const pri = s.comms.last_priority || 0;
    _txFlash = {
      ox: s.ownship.x, oy: s.ownship.y,
      dx: da ? da.x : 0, dy: da ? da.y : 0,
      color: PRI_COLORS[pri] || PRI_COLORS[0],
      t: Date.now(),
    };
    showTxToast(s.comms.last_text, pri);
  }

  // Clear stale prev-state on scenario reset (t rewinds to near 0)
  if(s.t < 1.0){
    Object.keys(_prevActor).forEach(k => delete _prevActor[k]);
    _corrections.length = 0;
  }

  // Detect prediction → observation transitions and spawn a correction flash
  for(const a of s.actors){
    const prev = _prevActor[a.id];
    if(prev && prev.predicted && !a.predicted){
      const dist = Math.hypot(a.x - prev.x, a.y - prev.y);
      if(dist > 0.3){   // only flash if there's a real positional correction
        _corrections.push({
          px: prev.x, py: prev.y,
          ox: a.x,    oy: a.y,
          color: COL[a.label] || '#fff',
          t: Date.now(),
        });
      }
    }
    _prevActor[a.id] = {predicted: a.predicted, x: a.x, y: a.y};
  }

  render(s.actors, s.ownship);
  updateSidebar(s.actors);
  updateComms(s.comms);
};
sse.onerror = () => {
  document.getElementById('live-dot').innerHTML = '<b style="color:#ef5350">● OFFLINE</b>';
};

drawBg();
</script>
</body>
</html>
"""

# Inject dynamic values into the HTML template
if SIM_SPEED != 1.0:
    _pill = (f'<div class="pill" style="color:#ce93d8">'
             f'&#x26A1; <b>{SIM_SPEED:.0f}&times;</b> WARP</div>')
    HTML = HTML.replace('<!-- SPEED_PILL -->', _pill)
else:
    HTML = HTML.replace('<!-- SPEED_PILL -->', '')

# ── HTTP handler ───────────────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            q: queue.Queue = queue.Queue(maxsize=8)
            with _subs_lock:
                _subs.append(q)
            try:
                while True:
                    self.wfile.write(q.get(timeout=30))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError, queue.Empty):
                pass
            finally:
                with _subs_lock:
                    if q in _subs:
                        _subs.remove(q)

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    threading.Thread(target=_run, daemon=True).start()

    print(f"\n  MANTA World Model — Demo Server")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  URL      : http://localhost:{PORT}")
    print(f"  Scenario : EOD dive · two sharks · vessel transit")
    print(f"  Duration : {int(SCENARIO_DURATION)}s (~10 min) then auto-reset")
    print(f"  Diver TX : {int(COMMS_TX_NORMAL)}s normally; {int(COMMS_TX_EMERG)}s min in emergency")
    print(f"  Budget   : {COMMS_BUDGET} chars/message")
    print(f"  ─────────────────────────────────────────────────\n")

    try:
        HTTPServer(("0.0.0.0", PORT), _Handler).serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
