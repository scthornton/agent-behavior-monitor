"""Local pre-dispatch scoring endpoint, shared with the replay host."""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .contracts import WINDOW, Request
from .experiment import Bundle
from .models import NAMES

MAX_BODY = 1_000_000  # bytes. A full 1,024-event history is about a quarter of this.


class Scorer:
    def __init__(self, bundle, timeout=2.0):
        self.bundle = bundle
        self.timeout = timeout
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.slot = BoundedSemaphore(1)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def compare(self, history, p):
        return {k: self.bundle.one(k, history, p) for k in NAMES}

    def bounded(self, history, p):
        """Score one request at a time. Waiting and scoring share one deadline."""
        deadline = time.monotonic() + self.timeout
        if not self.slot.acquire(timeout=self.timeout):
            return None, "scorer_busy"
        future = self.pool.submit(self.compare, history, p)
        future.add_done_callback(lambda _: self.slot.release())
        try:
            return future.result(timeout=max(0.0, deadline - time.monotonic())), None
        except TimeoutError:
            return None, "scorer_timeout"
        except Exception:  # noqa: BLE001 - any scorer fault must hold the call
            return None, "scorer_error"

    def decide(self, request: Request):
        p = request.proposed.model_dump()
        history = [e.model_dump() for e in request.history[-WINDOW:]]
        cfg = self.bundle.config
        response = {
            "model_version": cfg["version"],
            "score": None,
            "scores": {},
            "thresholds": cfg["thresholds"],
            "evidence": history,
        }
        reason = "history_incomplete"
        if request.complete:
            values, reason = self.bounded(history, p)
            if values is not None:
                response.update(score=values["gru"], scores=values)
                reason = (
                    "behavior_threshold"
                    if values["gru"] >= cfg["thresholds"]["gru"]
                    else "below_threshold"
                )
        disposition = "allow" if reason == "below_threshold" else "review"
        if p["permission"] == "deny":
            reason, disposition = "permission_denied", "deny"
        return dict(**response, disposition=disposition, reason=reason)


class BodyLimit:
    """Refuse an oversized body before it is parsed."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.limit:
            reply = JSONResponse({"detail": "request body too large"}, 413)
            return await reply(scope, receive, send)
        seen = 0

        async def guarded():
            # A chunked body declares no length, so count it as it arrives.
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise HTTPException(413, "request body too large")
            return message

        await self.app(scope, guarded, send)


def create_app(out):
    scorer = Scorer(Bundle(out / "main"))

    @asynccontextmanager
    async def lifespan(app):
        yield
        scorer.close()

    app = FastAPI(title="Local behavior monitor", version="1", lifespan=lifespan)
    app.add_middleware(BodyLimit, limit=MAX_BODY)

    @app.exception_handler(RequestValidationError)
    async def invalid(_, error):
        # Say what is wrong without echoing the request. A body can hold megabytes of
        # content or a NaN, and a NaN cannot even be written back as JSON.
        detail = [
            {"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]}
            for e in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.post("/score")
    def score(request: Request):
        return scorer.decide(request)

    return app
