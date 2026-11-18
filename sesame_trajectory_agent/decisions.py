"""Metadata-only rules for the original, assistant-selected research filter.

These numerical thresholds are not course requirements or the paper's success
criterion. Rejection means only that this experimental filter did not pass;
it is not evidence that a trajectory cannot be learned by a policy.
Current configs exempt the two user-selected diagnostic metrics. Archived
configs without the exemption field preserve their original strict behavior.
"""
import math


METRIC_GATES = {
    "velocity_rmse_m_s": "velocity_rmse_max_m_s",
    "mean_speed_error_m_s": "mean_speed_error_max_m_s",
    "heading_rms_deg": "heading_rms_max_deg",
    "heading_max_deg": "heading_max_deg",
    "lateral_displacement_max_m": "lateral_displacement_max_m",
    "joint_tracking_rmse_rad": "joint_tracking_rmse_max_rad",
    "root_tracking_rmse_m": "root_tracking_rmse_max_m",
    "contact_foot_speed_rms_m_s": "contact_foot_speed_rms_max_m_s",
    "nonfoot_contact_fraction": "nonfoot_contact_fraction_max",
    "torque_near_stall_fraction": "torque_near_stall_fraction_max",
    "action_rate_rms_per_s": "action_rate_rms_max_per_s",
}

DIAGNOSTIC_ONLY_ALLOWED = frozenset({"velocity_rmse_m_s", "contact_foot_speed_rms_m_s"})


def validate_diagnostic_only(value):
    if (not isinstance(value,(list,tuple)) or any(not isinstance(x,str) for x in value)
            or len(set(value))!=len(value) or set(value)-DIAGNOSTIC_ONLY_ALLOWED):
        raise ValueError("diagnostic_only_metrics may contain only the two user-exempted metrics")
    return frozenset(value)


def decide(metrics, gates, diagnostic_only_metrics=()):
    # Absent field preserves archived protocols. Only the trusted run config,
    # never a generated proposal, controls which recorded metrics gate admission.
    diagnostic_only=validate_diagnostic_only(diagnostic_only_metrics)
    reasons = []
    if metrics.get("finite") is not True:
        reasons.append("nonfinite_simulation_state")
    if metrics.get("completed") is not True:
        reasons.append("incomplete_rollout")
    if metrics.get("fell") is not False:
        reasons.append("fall_height_or_tilt")
    for metric, limit in METRIC_GATES.items():
        if metric in diagnostic_only:
            continue
        value = metrics.get(metric)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            reasons.append(f"missing_or_invalid:{metric}")
        elif value > gates[limit]:
            reasons.append(f"exceeds:{metric}")
    return reasons
