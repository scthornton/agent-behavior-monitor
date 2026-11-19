"""Local pre-dispatch scoring endpoint, shared with the replay host."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager
from threading import BoundedSemaphore

from fastapi import FastAPI

from .contracts import WINDOW, Request
from .experiment import Bundle
from .models import NAMES


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
            reason = "scorer_busy"
            if self.slot.acquire(blocking=False):
                future = self.pool.submit(self.compare, history, p)
                future.add_done_callback(lambda _: self.slot.release())
                try:
                    values = future.result(timeout=self.timeout)
                except TimeoutError:
                    reason = "scorer_timeout"
                else:
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


def create_app(out):
    scorer = Scorer(Bundle(out / "main"))

    @asynccontextmanager
    async def lifespan(app):
        yield
        scorer.close()

    app = FastAPI(title="Local behavior monitor", version="1", lifespan=lifespan)

    @app.post("/score")
    def score(request: Request):
        return scorer.decide(request)

    return app
