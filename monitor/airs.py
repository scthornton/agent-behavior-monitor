"""Optional pre-submit response inspection for the synthetic runtime."""
import os
import httpx

URL = "https://service.api.aisecurity.paloaltonetworks.com/v1/scan/sync/request"


def inspect(text, mode="offline", post=httpx.post):
    if mode == "offline":
        return {"source": "offline", "action": "unavailable"}
    if mode in ("fixture-allow", "fixture-block"):
        return {"source": "simulated", "action": mode.split("-")[1]}
    if mode != "live":
        raise ValueError("unknown AIRS mode")
    action = "unavailable"
    token, profile = os.getenv("AIRS_API_KEY"), os.getenv("AIRS_PROFILE")
    if token and profile:
        try:
            response = post(URL, timeout=2.0, headers={"x-pan-token": token},
                            json={"ai_profile": {"profile_name": profile},
                                  "contents": [{"response": text}]})
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("action") in ("allow", "block"):
                action = data["action"]
        except (httpx.HTTPError, ValueError):
            pass
    return {"source": "live", "action": action}
