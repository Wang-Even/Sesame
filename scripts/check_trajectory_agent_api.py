#!/usr/bin/env python3
"""One live API/schema test, without local numerical runtime or simulation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sesame_trajectory_agent.api_client import ApiError, call_model_with_metadata
from sesame_trajectory_agent.schema import PARAMETER_RULES, LEGS, Proposal, validate_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/trajectory_agent.json")
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Choose a new output directory")
    config = validate_config(json.loads(a.config.read_text()))
    source = Path(config["source"])
    if not source.is_absolute(): source = ROOT/source
    names = json.loads((source/"motion_names.json").read_text())
    joints = json.loads((source/"joint_names.json").read_text())
    request = {"instruction":"Connectivity/schema check: propose one conservative reference variation; no simulation is performed in this check.",
        "round":0,"seed":config["seed"],"candidate_count":1,
        "task":{k:config[k] for k in ("duration_s","fps","target_speed_m_s")},
        "gates":config["gates"],"robot":{"joint_names":joints},
        "source_clips":len(names),"source_names":names,"parameter_rules":PARAMETER_RULES,
        "legs":LEGS,
        "response_example":{"candidates":[Proposal().to_dict()]},"feedback":[]}
    a.output.mkdir(parents=True)
    def save(name,value):
        (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    save("request.json",request)
    try:
        response,metadata = call_model_with_metadata(request,config["api"])
    except ApiError as e:
        save("result.json",{"status":"failed","error":str(e),"requested_model":config["api"]["model"]})
        p.exit(1,str(e)+"\n")
    save("response.json",response)
    save("proposals.json",{"rounds":[response]})
    metadata.update(status="passed",checked_at=datetime.now(timezone.utc).isoformat(),
                    local_gpu_used=False,simulation_performed=False)
    save("result.json",metadata)
    print(json.dumps(metadata,ensure_ascii=False,allow_nan=False))


if __name__ == "__main__":
    main()
