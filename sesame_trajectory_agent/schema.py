"""Strict proposal validation. The agent cannot change task goals or acceptance gates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

LEGS = {"right_front": ("R1", "R3"), "right_rear": ("R2", "R4"),
        "left_front": ("L1", "L3"), "left_rear": ("L2", "L4")}


@dataclass(frozen=True)
class Proposal:
    source_clip: int = 0
    start_time_s: float = 0.25
    tempo: float = 1.0
    amplitude: float = 1.0
    smoothing_window_s: float = 0.0
    height_offset_m: float = 0.0
    phase_offsets_s: dict = field(default_factory=lambda: {k: 0.0 for k in LEGS})
    joint_offsets_rad: dict = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self):
        return asdict(self)


SCALAR_PARAMETERS = ("start_time_s", "tempo", "amplitude", "smoothing_window_s", "height_offset_m")
PARAMETER_RULES = {"numeric_values":"finite; no amplitude, joint-offset, phase-offset or height clipping",
    "start_time_s":"nonnegative; finite source window must contain every requested sample",
    "tempo":"positive; source time = start_time_s + tempo * output_time + leg offset",
    "smoothing_window_s":"optional, nonnegative; 0 disables smoothing; cannot exceed source duration",
    "amplitude":"finite, no artificial range; negative values invert variation about the source mean",
    "result_checks":"configured execution and contact checks; no joint-position-limit or foot-tracking-error acceptance condition"}


def number(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a number")
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name}: must be finite and in [{lo}, {hi}]")
    return float(value)


def parse_proposal(raw, joint_names, clips):
    if not isinstance(raw, dict):
        raise ValueError("Each candidate must be a JSON object")
    if set(raw) - set(Proposal.__dataclass_fields__):
        raise ValueError(f"Unknown proposal fields: {sorted(set(raw)-set(Proposal.__dataclass_fields__))}")
    p = Proposal().to_dict() | raw
    if type(p["source_clip"]) is not int or not 0 <= p["source_clip"] < clips:
        raise ValueError("source_clip is outside the training reference library")
    for k in SCALAR_PARAMETERS:
        p[k] = number(p[k], k, -math.inf, math.inf)
    if p["tempo"] <= 0 or p["start_time_s"] < 0 or p["smoothing_window_s"] < 0:
        raise ValueError("tempo must be positive; start_time_s and smoothing_window_s must be nonnegative")
    for key, allowed in [("phase_offsets_s", set(LEGS)),
                         ("joint_offsets_rad", set(joint_names))]:
        values = p[key]
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"Invalid keys in {key}; allowed: {sorted(allowed)}")
        p[key] = {k: number(v, f"{key}.{k}", -math.inf, math.inf) for k, v in values.items()}
    if not isinstance(p["rationale"], str) or len(p["rationale"]) > 1000:
        raise ValueError("rationale must be a string with at most 1000 characters")
    return Proposal(**p)


def proposal_key(proposal):
    """Identify identical motion parameters, independent of rationale or JSON order."""
    import json
    values=proposal.to_dict()
    values.pop("rationale")
    values["phase_offsets_s"]={k:values["phase_offsets_s"].get(k,0.) for k in LEGS}
    values["joint_offsets_rad"]={k:v for k,v in values["joint_offsets_rad"].items() if v!=0}
    return json.dumps(values,sort_keys=True,separators=(",",":"))


def validate_config(c):
    required = {"source", "seed", "rounds", "candidates_per_round", "duration_s",
                "fps", "target_speed_m_s", "robust_trials", "gates", "api"}
    if not required <= set(c) or set(c)-required-{"physics_dt_s","diagnostic_only_metrics","mean_speed_measurement","execution_feedback"}:
        raise ValueError(f"Config requires {sorted(required)}; physics_dt_s and diagnostic_only_metrics are optional")
    from .decisions import validate_diagnostic_only
    validate_diagnostic_only(c.get("diagnostic_only_metrics",[]))
    if c.get("mean_speed_measurement","control_samples") not in ("control_samples","displacement"):
        raise ValueError("Invalid mean speed measurement")
    if "execution_feedback" in c:
        feedback=c["execution_feedback"]
        expected_feedback={"position_gain","velocity_gain","heading_gain","lateral_gain","yaw_rate_gain","velocity_filter_alpha"}
        if not isinstance(feedback,dict) or set(feedback)!=expected_feedback:
            raise ValueError("Invalid execution feedback parameters")
        for key,value in feedback.items():number(value,key,-100.,100.)
        number(feedback["velocity_filter_alpha"],"velocity_filter_alpha",.001,1.)
    if c.get("physics_dt_s",.002) not in (.002,.001,.0005,.00025):
        raise ValueError("Unsupported physics_dt_s")
    for key, lo, hi in [("seed", 0, 2**31-1), ("rounds", 1, 100),
                        ("candidates_per_round", 1, 64), ("robust_trials", 0, 32)]:
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError(f"Invalid config {key}")
    if c["fps"] != 50:
        raise ValueError("This backend requires 50 Hz; resampling must be explicit")
    number(c["duration_s"], "duration_s", .2, 6.)
    if abs(c["duration_s"] * 50 - round(c["duration_s"] * 50)) > 1e-7:
        raise ValueError("duration_s must be a whole number of control steps")
    number(c["target_speed_m_s"], "target_speed_m_s", .01, .3)
    expected = {"joint_velocity_max_rad_s", "joint_acceleration_max_rad_s2",
                "foot_surface_min_z_m", "root_height_min_m", "tilt_max_deg",
                "velocity_rmse_max_m_s", "mean_speed_error_max_m_s",
                "heading_rms_max_deg", "heading_max_deg", "lateral_displacement_max_m",
                "joint_tracking_rmse_max_rad", "root_tracking_rmse_max_m",
                "contact_foot_speed_rms_max_m_s",
                "nonfoot_contact_fraction_max", "torque_near_stall_fraction_max",
                "action_rate_rms_max_per_s"}
    if not isinstance(c["gates"], dict) or set(c["gates"]) != expected:
        raise ValueError(f"Gate keys must be exactly {sorted(expected)}")
    for k, v in c["gates"].items():
        number(v, k, -.01 if k == "foot_surface_min_z_m" else 0., 10000.)
    for k in ("nonfoot_contact_fraction_max", "torque_near_stall_fraction_max"):
        number(c["gates"][k], k, 0., 1.)
    if not isinstance(c["source"], str) or not c["source"]:
        raise ValueError("source is required")
    if not isinstance(c["api"], dict) or set(c["api"]) != {"base_url", "model", "api_key_env"}:
        raise ValueError("api must contain base_url, model and api_key_env")
    if not all(isinstance(v, str) for v in c["api"].values()):
        raise ValueError("API settings must be strings; do not put a secret in this file")
    return c
