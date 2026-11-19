"""Optional pre-submit content inspection for the synthetic runtime."""

import os
from urllib.parse import urlsplit

import httpx

MODES = ("offline", "fixture-allow", "fixture-block", "fixture-pattern", "live")
DEFAULT = "https://service.api.aisecurity.paloaltonetworks.com"
SUFFIX = ".api.aisecurity.paloaltonetworks.com"
PATH = "/v1/scan/sync/request"
# Each list is read in order. The first name is this project's own. The others are
# names that Prisma AIRS tooling commonly uses, so an existing setup works unchanged.
KEYS = ("AIRS_API_KEY", "PANW_AI_SEC_API_KEY", "PRISMA_AIRS_API_KEY")
PROFILES = (
    "AIRS_PROFILE",
    "AIRS_PROFILE_NAME",
    "PRISMA_AIRS_AI_PROFILE",
    "PRISMA_AIRS_PROFILE_NAME",
    "PRISMA_AIRS_PROFILE",
)
ENDPOINTS = (
    "AIRS_API_ENDPOINT",
    "PANW_AI_SEC_API_ENDPOINT",
    "PRISMA_AIRS_API_ENDPOINT",
)
# fixture-pattern stands in for a data pattern on out-of-case material. It matches
# this synthetic marker and says nothing about what a real profile would detect.
FIXTURE_PATTERN = "case B"


def first(names):
    return next((value for name in names if (value := os.getenv(name))), None)


def endpoint():
    """The API key only ever goes to a Prisma AIRS scan host over TLS."""
    base = (first(ENDPOINTS) or DEFAULT).strip().rstrip("/")
    host = urlsplit(base).hostname or ""
    if base.lower() != f"https://{host}" or not host.endswith(SUFFIX):
        return None
    return f"https://{host}{PATH}"


def unavailable(reason):
    return {"source": "live", "action": "unavailable", "reason": reason}


def inspect(text, mode="offline", post=None, *, tr_id=None, session=None):
    """Return source, action and reason. Only a real verdict yields allow or block."""
    if mode == "offline":
        return {"source": "offline", "action": "unavailable", "reason": "offline_mode"}
    if mode in ("fixture-allow", "fixture-block"):
        action = mode.split("-")[1]
        return {"source": "simulated", "action": action, "reason": "fixed_fixture"}
    if mode == "fixture-pattern":
        hit = FIXTURE_PATTERN in text
        return {
            "source": "simulated",
            "action": "block" if hit else "allow",
            "reason": "pattern_match" if hit else "no_pattern_match",
        }
    if mode != "live":
        raise ValueError("unknown AIRS mode")
    token, profile, url = first(KEYS), first(PROFILES), endpoint()
    if not (token and profile):
        return unavailable("missing_credentials")
    if url is None:
        return unavailable("endpoint_rejected")
    body = {
        "ai_profile": {"profile_name": profile},
        "contents": [{"response": text}],
        "metadata": {"app_name": "agent-behavior-monitor"},
    }
    if tr_id:
        body["tr_id"] = tr_id
    if session:
        body["session_id"] = session
    try:
        send = post or httpx.post
        response = send(url, timeout=2.0, headers={"x-pan-token": token}, json=body)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException:
        return unavailable("timeout")
    except httpx.HTTPStatusError as error:
        return unavailable(f"http_{error.response.status_code}")
    except (httpx.HTTPError, ValueError):
        return unavailable("transport_error")
    if not isinstance(data, dict) or data.get("action") not in ("allow", "block"):
        return unavailable("unusable_response")
    verdict = {"source": "live", "action": data["action"], "reason": "verdict"}
    for key in ("category", "scan_id", "report_id"):
        if isinstance(data.get(key), str):
            verdict[key] = data[key]
    return verdict
