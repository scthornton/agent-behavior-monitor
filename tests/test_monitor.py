import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from monitor import airs
from monitor.airs import inspect
from monitor.contracts import Request
from monitor.experiment import Bundle, calibrate, sessions, train
from monitor.fixtures import FAMILIES, generate, plan, replay
from monitor.models import FIELDS, Encoder, examples, fields, rule
from monitor.service import MAX_BODY, Scorer, create_app
from monitor.workflow import Call, Workspace


def request_at_upload(case):
    ws, labels = replay("catalog_check", 0, "scope", case, "same", "tenant")
    return Request(history=ws.history[:-1], proposed=ws.events[-2]), ws, labels


@pytest.fixture(autouse=True)
def no_live_credentials(monkeypatch):
    """Tests never see real credentials or the network, whatever the shell exports."""
    for name in list(os.environ):
        if name.startswith(("AIRS_", "PANW_", "PRISMA_AIRS_")):
            monkeypatch.delenv(name)

    def blocked(*args, **kwargs):
        raise AssertionError("tests must not reach the network")

    monkeypatch.setattr(airs.httpx, "post", blocked)


@pytest.fixture(scope="session")
def run_dir(tmp_path_factory):
    """Use MONITOR_RUN when it is set. Otherwise generate and train a private run."""
    if os.environ.get("MONITOR_RUN"):
        return Path(os.environ["MONITOR_RUN"])
    out = tmp_path_factory.mktemp("run")
    generate(out)
    train(out)
    return out


@pytest.fixture(scope="session")
def corpus(run_dir, tmp_path_factory):
    """Generation is deterministic, so a folder that ships models only can rebuild it."""
    if (run_dir / "sessions.jsonl").exists():
        return run_dir
    out = tmp_path_factory.mktemp("corpus")
    generate(out)
    return out


@pytest.fixture(scope="session")
def bundle(run_dir):
    return Bundle(run_dir / "main")


def test_controlled_upload_and_ambiguity(bundle):
    clean, a, la = request_at_upload(1)
    risky, b, lb = request_at_upload(0)
    excluded, c, lc = request_at_upload(2)
    assert clean.proposed == risky.proposed == excluded.proposed
    assert risky == excluded
    assert not any(la) and not any(lc) and lb == [0] * (len(lb) - 1) + [1]
    assert a.portal == c.portal != b.portal
    assert "amber" in b.portal[0][1] and "amber" not in c.portal[0][1]
    diffs = [i for i, (x, y) in enumerate(zip(clean.history, risky.history)) if x != y]
    assert len(diffs) == 1
    changed = risky.history[diffs[0]]
    assert (changed.tool, changed.state, changed.scope) == (
        "read",
        "completed",
        "related_lookup",
    )
    # Substitute that one observed event and rescore. This is a paired input
    # intervention. It supplies no attribution about arbitrary real workflows.
    modified = list(clean.history)
    modified[diffs[0]] = changed

    def score(kind, history):
        return bundle.one(
            kind, [e.model_dump() for e in history], clean.proposed.model_dump()
        )

    assert score("current", risky.history) == score("current", clean.history)
    assert score("gru", modified) == score("gru", risky.history)
    assert score("gru", modified) != score("gru", clean.history)


@pytest.mark.parametrize("field", ["session", "tenant"])
def test_isolation(field):
    req, _, _ = request_at_upload(1)
    foreign = req.history[0].model_copy(update={field: "foreign"})
    with pytest.raises(ValidationError):
        Request(proposed=req.proposed, history=[foreign])
    left, right = Workspace("A", "same"), Workspace("B", "same")
    left.step(Call("read", "sealed"))
    assert not right.cache and not right.events and not right.portal


def test_future_terminal_and_hidden_fields(corpus):
    req, _, _ = request_at_upload(1)
    with pytest.raises(ValidationError):
        Request(
            proposed=req.proposed,
            history=[req.history[0].model_copy(update={"seq": req.proposed.seq + 1})],
        )
    with pytest.raises(ValidationError):
        Request(proposed=req.history[-1], history=[])
    with pytest.raises(ValidationError):
        Request.model_validate({**req.model_dump(), "scenario": "secret"})
    sample = sessions(corpus, "test", "main")[0]
    rows = deepcopy(list(examples([sample])))
    i = len(rows) // 2
    sample["events"][2 * i + 1]["state"] = "failed"
    sample["events"][-1]["scope"] = "outside"
    assert list(examples([sample]))[: i + 1] == rows[: i + 1]
    p = rows[i][1]
    assert fields(
        dict(p, session="other", tenant="other", seq=99, label=0, text="private")
    ) == fields(p)
    assert FIELDS == (
        "tool",
        "operation",
        "classification",
        "destination",
        "scope",
        "permission",
        "state",
    )


def test_denial_before_execution():
    w = Workspace("t", "s")

    def broken(_):
        raise AssertionError("permission gate must run first")

    _, terminal, _ = w.step(Call("read", "locked"), broken)
    assert terminal.state == "denied" and not w.cache
    assert [e.state for e in w.events] == ["proposed", "denied"]


def staged_workspace():
    w = Workspace("t", "s")
    for call in (Call("read", "cover"), Call("summarize"), Call("stage")):
        w.step(call)
    return w


def test_unavailable_airs_never_means_allow():
    w = staged_workspace()
    _, end, v = w.step(Call("submit"), airs="live")
    assert v == {
        "source": "live",
        "action": "unavailable",
        "reason": "missing_credentials",
    }
    assert end.state == "held" and not w.portal
    w = staged_workspace()
    _, end, v = w.step(Call("submit"), airs="fixture-block")
    assert v["source"] == "simulated" and end.state == "denied" and not w.portal
    w = staged_workspace()
    _, end, v = w.step(Call("submit"))
    assert v == {"source": "offline", "action": "unavailable", "reason": "offline_mode"}
    assert end.state == "completed" and len(w.portal) == 1
    # A mistyped mode is refused before anything is recorded.
    w = staged_workspace()
    recorded = len(w.events)
    with pytest.raises(ValueError):
        w.step(Call("submit"), airs="fixture")
    assert len(w.events) == recorded


def test_live_wire_with_transport_double(monkeypatch):
    sent = []

    def reply(status, payload):
        def post(url, *, json, headers, timeout):
            sent.append((url, json, headers))
            return httpx.Response(
                status, json=payload, request=httpx.Request("POST", url)
            )

        return post

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("fixture")

    block = reply(200, {"action": "block", "category": "malicious", "scan_id": "s-1"})
    # Without a profile nothing is sent, and the reason says why.
    monkeypatch.setenv("PANW_AI_SEC_API_KEY", "fixture")
    assert inspect("synthetic", "live", block)["reason"] == "missing_credentials"
    monkeypatch.setenv("PRISMA_AIRS_AI_PROFILE", "sample")
    verdict = inspect("synthetic", "live", block, tr_id="s-29", session="s")
    assert verdict == {
        "source": "live",
        "action": "block",
        "reason": "verdict",
        "category": "malicious",
        "scan_id": "s-1",
    }
    assert sent == [
        (
            "https://service.api.aisecurity.paloaltonetworks.com/v1/scan/sync/request",
            {
                "ai_profile": {"profile_name": "sample"},
                "contents": [{"response": "synthetic"}],
                "metadata": {"app_name": "agent-behavior-monitor"},
                "tr_id": "s-29",
                "session_id": "s",
            },
            {"x-pan-token": "fixture"},
        )
    ]
    # Every failure is unavailable, with a reason a person can act on.
    assert inspect("synthetic", "live", timeout)["reason"] == "timeout"
    assert inspect("synthetic", "live", reply(401, {}))["reason"] == "http_401"
    odd = reply(200, {"action": "maybe"})
    assert inspect("synthetic", "live", odd) == airs.unavailable("unusable_response")
    # A regional scan host is accepted. Any other host never receives the key.
    monkeypatch.setenv("AIRS_API_ENDPOINT", "https://service-de" + airs.SUFFIX)
    assert inspect("synthetic", "live", block)["action"] == "block"
    assert sent[-1][0].startswith("https://service-de.api.aisecurity.")
    for bad in (
        "https://example.com",
        "http://service" + airs.SUFFIX,
        "https://x.test/" + airs.SUFFIX,
    ):
        monkeypatch.setenv("AIRS_API_ENDPOINT", bad)
        before = len(sent)
        assert inspect("synthetic", "live", block)["reason"] == "endpoint_rejected"
        assert len(sent) == before


def test_timeout_holds_and_denial_wins(monkeypatch):
    b = SimpleNamespace(
        config={"version": "stub", "thresholds": {"gru": 0.5, "rule": 0.00001}}
    )
    sc = Scorer(b, timeout=0.001)
    monkeypatch.setattr(sc, "compare", lambda *args: time.sleep(0.05))
    try:
        w = Workspace("t", "s")
        q = Request(proposed=w.proposal(Call("read", "cover")), history=[])
        d = sc.decide(q)
        assert (d["disposition"], d["reason"], d["score"]) == (
            "review",
            "scorer_timeout",
            None,
        )
        _, end, _ = w.step(Call("read", "cover"), sc.decide)
        assert end.state == "held" and not w.cache
        denied = Request(proposed=w.proposal(Call("read", "locked")), history=[])
        assert sc.decide(denied)["disposition"] == "deny"
    finally:
        sc.close()


def test_http_decision_boundary(run_dir):
    req, _, _ = request_at_upload(1)
    with TestClient(create_app(run_dir)) as api:
        body = req.model_dump()
        ok = api.post("/score", json=body)
        assert ok.status_code == 200 and ok.json()["disposition"] == "allow"
        assert ok.json()["model_version"] and set(ok.json()["scores"]) == {
            "rule",
            "gru",
            "shuffled",
            "current",
        }
        bad = api.post("/score", json=dict(body, history=body["history"][1:]))
        assert (
            bad.json()["reason"] == "history_incomplete" and bad.json()["score"] is None
        )
        # Malformed and oversized bodies are refused cleanly and never echoed back.
        text, headers = json.dumps(body), {"Content-Type": "application/json"}
        nan = api.post(
            "/score", content=text.replace('"seq": 29', '"seq": NaN'), headers=headers
        )
        assert nan.status_code == 422 and "NaN" not in nan.text
        quoted = dict(body, proposed=dict(body["proposed"], seq="29"))
        assert api.post("/score", json=quoted).status_code == 422
        reordered = api.post("/score", json=dict(body, history=body["history"][::-1]))
        assert reordered.status_code == 422 and "tenant" not in reordered.text
        padding = b" " * (MAX_BODY + 1)
        assert api.post("/score", content=padding, headers=headers).status_code == 413
        chunks = (b" " * 65536 for _ in range(MAX_BODY // 65536 + 2))
        assert api.post("/score", content=chunks, headers=headers).status_code == 413
        body["history"][0]["tenant"] = "wrong"
        assert api.post("/score", json=body).status_code == 422


def test_split_and_thresholds(run_dir, corpus):
    manifest = json.loads((corpus / "split.json").read_text())
    by_group = {}
    assert len({s["session"] for s in manifest}) == len(manifest)
    for s in manifest:
        assert by_group.setdefault(s["group"], s["split"]) == s["split"]
    # A full run has every fold. The shipped reference folder has the main fold only.
    trained = [f for f in ("main", *FAMILIES) if (run_dir / f / "config.json").exists()]
    assert "main" in trained
    assert len(trained) == 4 or not (run_dir / "sessions.jsonl").exists()
    for fold in ("main", *FAMILIES):
        fit, val, test = [
            sessions(corpus, split, fold) for split in ("train", "validation", "test")
        ]
        if fold != "main":
            assert all(s["family"] != fold for s in fit + val)
            assert all(s["family"] == fold for s in test)
        if fold not in trained:
            continue
        cfg = json.loads((run_dir / fold / "config.json").read_text())
        assert cfg["vocab"] == Encoder.fit(fit).vocab
        assert all(v["benign_per_100"] <= 10 for v in cfg["validation"].values())
    assert calibrate(np.array([1.0] * 10), [{"first": None, "labels": [0]}] * 10) > 1.0


def test_ablation_erases_order(bundle):
    req, _, _ = request_at_upload(0)
    history = [e.model_dump() for e in req.history[-12:]]
    p = req.proposed.model_dump()
    for kind in ("current", "shuffled", "gru"):
        left, right = [bundle.enc.encode(h, p, kind) for h in (history, history[::-1])]
        assert np.array_equal(left, right) == (kind != "gru")


def test_full_trace_overlap_and_first_decisions(corpus):
    traces = {}
    for split in ("train", "validation", "test"):
        unique = set()
        for s in sessions(corpus, split, "main"):
            steps = [i for i, label in enumerate(s["labels"]) if label]
            assert s["first"] == (steps[0] if steps else None)
            if s["case"] == 0:
                expected = {"scope": "submit", "retry": "read", "change": "stage"}
                assert s["events"][steps[0] * 2]["tool"] == expected[s["family"]]
            else:
                assert not steps
            unique.add(tuple(fields(e) for e in s["events"]))
        assert all(unique.isdisjoint(other) for other in traces.values())
        traces[split] = unique


def test_decision_client_faults_hold():
    def broken(_):
        raise RuntimeError("scorer exploded")

    def slow(_):
        raise httpx.ReadTimeout("fixture")

    replies = (broken, slow, lambda _: {"score": 0.0}, lambda _: {"disposition": "OK"})
    for client in replies:
        w = Workspace("t", "s")
        _, end, _ = w.step(Call("read", "cover"), client)
        assert end.state == "held" and not w.cache
        # The stream stays well formed, and the hold stops the session.
        _, end, _ = w.step(Call("read", "cover"))
        assert end.state == "held" and [e.seq for e in w.events] == [1, 2, 3, 4]


def test_overlong_session_holds_without_scoring(monkeypatch):
    monkeypatch.setattr("monitor.workflow.MAX_HISTORY", 2)
    seen = []

    def client(request):
        seen.append(request)
        return {"disposition": "allow"}

    w = Workspace("t", "s")
    states = [w.step(Call("search"), client)[1].state for _ in range(3)]
    request, end, _ = w.step(Call("search"), client)
    assert states == ["completed"] * 3 and len(seen) == 3
    assert request is None and end.state == "held"


def test_concurrent_requests_wait_for_the_scorer(bundle):
    req, _, _ = request_at_upload(1)
    sc = Scorer(bundle)
    try:
        with ThreadPoolExecutor(8) as pool:
            decisions = list(pool.map(lambda _: sc.decide(req), range(160)))
    finally:
        sc.close()
    outcomes = {(d["disposition"], d["reason"]) for d in decisions}
    assert outcomes == {("allow", "below_threshold")}


def test_session_evidence_does_not_expire(bundle):
    sc = Scorer(bundle)
    try:
        # Fourteen routine lookups push the restricted read out of the 12-event window.
        w = Workspace("t", "padded")
        prefix = [Call("read", "cover"), Call("read", "related")]
        suffix = [Call("summarize", "include_related"), Call("stage")]
        for call in [*prefix, *[Call("search")] * 14, *suffix]:
            w.step(call)
        request = Request(proposed=w.proposal(Call("submit")), history=w.history)
        history, p = [e.model_dump() for e in w.history], request.proposed.model_dump()
        assert rule(history, p) == 0.0 and rule(history, p, window=None) == 0.8
        d = sc.decide(request)
        assert (d["disposition"], d["reason"]) == ("review", "rule_threshold")
        _, end, _ = w.step(Call("submit"), sc.decide)
        assert end.state == "held" and not w.portal
        # An approved amendment that scrolled out of the window is still approved.
        w = Workspace("t", "amended", amendment=True)
        prefix = [Call("read", "cover"), Call("search", "amend")]
        for call in [*prefix, *[Call("search")] * 14, Call("summarize")]:
            w.step(call)
        history = [e.model_dump() for e in w.history]
        p = w.proposal(Call("stage", "bulletin")).model_dump()
        assert rule(history, p) == 0.95 and rule(history, p, window=None) == 0.0
    finally:
        sc.close()


def test_hold_stops_the_session(bundle):
    sc = Scorer(bundle)
    try:
        w = Workspace("demo", "latched")
        for call in plan("catalog_check", 0, "scope", 0):
            _, end, _ = w.step(call, sc.decide)
        assert end.tool == "submit" and end.state == "held"
        # Padding and resubmitting used to slide the evidence out and release the upload.
        for call in [*[Call("search")] * 8, Call("submit")]:
            _, end, _ = w.step(call, sc.decide)
            assert end.state == "held"
        assert not w.portal
        # A host without its own latch still gets a hold from the service.
        request = Request(proposed=w.proposal(Call("submit")), history=w.history)
        d = sc.decide(request)
        assert (d["disposition"], d["reason"]) == ("review", "session_held")
    finally:
        sc.close()


def test_content_check_separates_what_metadata_cannot():
    # A rule-only decision client keeps this independent of any trained weights.
    stub = SimpleNamespace(
        config={"version": "stub", "thresholds": {"gru": 0.5, "rule": 0.00001}},
        one=lambda kind, history, p: rule(history, p) if kind == "rule" else 0.0,
    )
    sc = Scorer(stub)
    outcomes = {}
    try:
        for case in (1, 0, 2):  # case material only, outside included, outside excluded
            w = Workspace("demo", f"case-{case}")
            for call in plan("catalog_check", 0, "scope", case):
                _, end, verdict = w.step(call, sc.decide, "fixture-pattern")
                if w.stopped:
                    break
            outcomes[case] = (end.tool, end.state, verdict["action"], len(w.portal))
    finally:
        sc.close()
    # Both sessions that read the outside record look identical to every detector.
    # The content check blocks the one that carries it. The other stays held.
    assert outcomes == {
        1: ("submit", "completed", "allow", 1),
        0: ("submit", "denied", "block", 0),
        2: ("submit", "held", "allow", 0),
    }
