"""DeepSeek JSON proposals. Credentials never enter prompts or saved responses."""
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .schema import parse_proposal

API_IMPLEMENTED = True
LOCAL_CREDENTIALS = Path(__file__).resolve().parents[1]/".runtime/credentials/deepseek.env"


class ApiError(RuntimeError):
    """Sanitized API failure suitable for the runner's status log."""


def load_api_environment(api_config, path=None):
    """Environment takes priority; otherwise parse a local KEY=value file.

    Never sources shell code. The default file is outside source snapshots
    and covered by /.runtime/ ignore. No credential is returned or logged.
    """
    name = api_config.get("api_key_env", "")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ApiError("api_key_env must name an environment variable, not contain a key")
    if os.environ.get(name):
        return
    path = Path(path) if path is not None else LOCAL_CREDENTIALS
    if not path.is_file():
        raise ApiError("API credential missing: set the configured environment variable or local credential file")
    if path.stat().st_mode & 0o077:
        raise ApiError("Local credential file must have owner-only permissions (chmod 600)")
    if path.stat().st_size > 16384:
        raise ApiError("Local credential file is unexpectedly large")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not value or any(ch.isspace() for ch in value):
                raise ApiError("Local credential value is empty or contains whitespace")
            os.environ[name] = value
            return
    raise ApiError("Configured API-key environment variable was not found in the local credential file")


def completion_url(api_config):
    base = api_config.get("base_url", "").rstrip("/")
    url = urllib.parse.urlsplit(base)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ApiError("base_url must be an HTTPS endpoint without credentials, query or fragment")
    if not api_config.get("model"):
        raise ApiError("API model is missing")
    return base if url.path.endswith("/chat/completions") else base+"/chat/completions"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ApiError("API redirected the request; verify base_url")


def _parse_json(text):
    def invalid_constant(_):
        raise ValueError("Non-finite JSON constant")
    return json.loads(text,parse_constant=invalid_constant)


def _post(url, payload, api_key):
    request = urllib.request.Request(url,
        data=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(),
        headers={"Authorization":"Bearer "+api_key,"Content-Type":"application/json"},method="POST")
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request,timeout=45) as response:
            raw = response.read(2*1024*1024+1)
            if len(raw)>2*1024*1024:
                raise ApiError("API response exceeded the size limit")
            data = _parse_json(raw.decode("utf-8"))
            status = response.status
    except urllib.error.HTTPError as e:
        hints = {400:"invalid request",401:"authentication failed",402:"insufficient balance",
                 403:"access denied",404:"endpoint or model not found",429:"rate limit"}
        raise ApiError(f"DeepSeek HTTP {e.code}: {hints.get(e.code,'provider error')}") from None
    except (urllib.error.URLError,TimeoutError,OSError):
        raise ApiError("DeepSeek network/TLS failure or timeout; no automatic retry was made") from None
    except (ValueError,UnicodeError):
        raise ApiError("DeepSeek returned invalid response JSON") from None
    if not isinstance(data,dict):
        raise ApiError("DeepSeek returned an unexpected response envelope")
    return data,status


def call_model_with_metadata(request: dict, api_config: dict, reasoning_effort="none"):
    """One non-streaming request; no silent retries or offline substitution."""
    url = completion_url(api_config)
    load_api_environment(api_config)
    count = request.get("candidate_count")
    if reasoning_effort not in ("none","low","high"):
        raise ValueError("Unsupported reasoning effort")
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("candidate_count must be an integer in [1,64]")
    payload = {
        "model":api_config["model"],"stream":False,
        "thinking":{"type":"disabled" if reasoning_effort=="none" else "enabled"},
        "max_tokens":min(32768,max(1024 if reasoning_effort=="none" else 8192,count*800)),
        "response_format":{"type":"json_object"},
        "messages":[
            {"role":"system","content":
             "You propose parameters for a quadruped reference-trajectory generator. "
             "Return ONLY a JSON object with the single key candidates, containing exactly "
             "candidate_count objects following response_example and parameter_rules. "
             "There are no artificial amplitude, phase, joint-offset or height caps; physical result checks still apply. "
             "Use only the named legs and joints. Use measured feedback when provided. "
             "Never repeat tested_parameters: change a parameter to address actual failed metrics. "
             "Task, gates, controller, budget and schema are fixed. Do not invent validation "
             "results, claim physical success, include code, or change acceptance thresholds. "
             "Keep rationale short. A proposal is not an executed trajectory."},
            {"role":"user","content":json.dumps(request,ensure_ascii=False,allow_nan=False)},
        ],
    }
    if reasoning_effort=="none":payload["temperature"]=.2
    else:payload["reasoning_effort"]=reasoning_effort
    started = time.monotonic()
    envelope,status = _post(url,payload,os.environ[api_config["api_key_env"]])
    choices = envelope.get("choices")
    if not isinstance(choices,list) or not choices or not isinstance(choices[0],dict):
        raise ApiError("API response has no completion choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ApiError("API output did not finish normally; truncated/tool-call output is not accepted")
    message = choice.get("message")
    content = message.get("content") if isinstance(message,dict) else None
    if not isinstance(content,str) or not content.strip():
        raise ApiError("API returned blank JSON content")
    try:
        result = _parse_json(content)
    except (ValueError,TypeError):
        raise ApiError("API proposal is not valid JSON") from None
    if not isinstance(result,dict) or set(result)!={"candidates"} or not isinstance(result["candidates"],list) or len(result["candidates"])!=count:
        raise ApiError("API proposal schema or candidate count is invalid")
    try:
        for item in result["candidates"]:
            parse_proposal(item,request["robot"]["joint_names"],request["source_clips"])
    except (ValueError,KeyError,TypeError):
        raise ApiError("API proposal violates the allowed parameters or robot joint names") from None
    usage = envelope.get("usage",{})
    usage = {k:v for k,v in usage.items() if type(v) is int} if isinstance(usage,dict) else {}
    metadata = {"http_status":status,"requested_model":api_config["model"],
                "response_model":envelope.get("model"),"finish_reason":choice["finish_reason"],
                "usage":usage,"elapsed_s":round(time.monotonic()-started,3),
                "thinking":payload["thinking"]["type"],"reasoning_effort":reasoning_effort,
                "candidate_count":count,"schema_valid":True}
    return result,metadata


def call_model(request: dict, api_config: dict) -> dict:
    result,_ = call_model_with_metadata(request,api_config)
    return result


def call_json_role(role, instructions, request, api_config, max_tokens=8192):
    """A distinct planning/coding/diagnosis call; credentials stay in transport.

    Unlike the parameter proposer this accepts role-specific JSON. The caller
    validates the schema and never treats a model's success claim as evidence.
    """
    url = completion_url(api_config)
    load_api_environment(api_config)
    payload = {"model":api_config["model"], "stream":False,
        "thinking":{"type":"disabled"}, "temperature":.2,
        "max_tokens":max_tokens, "response_format":{"type":"json_object"},
        "messages":[{"role":"system","content":instructions},
                    {"role":"user","content":json.dumps(request,ensure_ascii=False,allow_nan=False)}]}
    started = time.monotonic()
    envelope,status = _post(url,payload,os.environ[api_config["api_key_env"]])
    choices = envelope.get("choices", [])
    if not choices or choices[0].get("finish_reason") != "stop":
        raise ApiError("Role completion missing or truncated; no offline substitution")
    try:
        result = _parse_json(choices[0]["message"]["content"])
    except (ValueError,TypeError,KeyError):
        raise ApiError("Role completion is not valid JSON") from None
    if not isinstance(result,dict):
        raise ApiError("Role completion must be a JSON object")
    usage = envelope.get("usage", {})
    return result, {"role":role,"requested_model":api_config["model"],
        "response_model":envelope.get("model"),"http_status":status,
        "elapsed_s":round(time.monotonic()-started,3),"finish_reason":"stop",
        "usage":{k:v for k,v in usage.items() if type(v) is int}}
