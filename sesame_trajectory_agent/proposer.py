"""Explicit offline heuristic for pipeline testing; this is not an LLM result."""
import random
from .schema import Proposal, LEGS


def offline_proposals(request):
    rng = random.Random(request["seed"] + request["round"])
    feedback = [r for r in request["feedback"] if r.get("proposal") and r.get("nominal_metrics")]
    best = min(feedback, key=lambda r: r["nominal_metrics"].get("velocity_rmse_m_s", 1e6)
               if r["nominal_metrics"].get("velocity_rmse_m_s") is not None else 1e6, default=None)
    proposals = []
    for i in range(request["candidate_count"]):
        if request["round"] == 0 and i == 0:
            p = Proposal(rationale="Offline unmodified seed template; no LLM was called").to_dict()
        elif best is not None and i == 0:
            p = dict(best["proposal"])
            speed = best["nominal_metrics"].get("mean_forward_speed_m_s") or 0.
            p["tempo"] += .05 if speed < request["task"]["target_speed_m_s"] else -.05
            p["rationale"] = "Offline heuristic: adjust tempo using previous measured speed; not an LLM decision"
        else:
            p = Proposal(source_clip=rng.randrange(request["source_clips"]),
                         tempo=rng.uniform(.85, 1.1), amplitude=rng.uniform(.85, 1.1),
                         phase_offsets_s={k: rng.uniform(-.04, .04) for k in LEGS},
                         rationale="Offline bounded random proposal; no LLM was called").to_dict()
        proposals.append(p)
    return {"candidates": proposals}
