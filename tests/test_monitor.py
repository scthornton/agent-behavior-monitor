import json
import os
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from monitor.airs import inspect
from monitor.contracts import Request
from monitor.experiment import Bundle, calibrate, sessions, train
from monitor.fixtures import FAMILIES, generate, replay
from monitor.models import FIELDS, Encoder, examples, fields
from monitor.service import Scorer, create_app
from monitor.workflow import Call, Workspace


def request_at_upload(case):
    ws, labels = replay("catalog_check", 0, "scope", case, "same", "tenant")
    return Request(history=ws.history[:-1], proposed=ws.events[-2]), ws, labels


@pytest.fixture(autouse=True)
def no_live_credentials(monkeypatch):
    """Tests never see real Prisma AIRS credentials, whatever the shell exports."""
    for name in list(os.environ):
        if name.startswith(("AIRS_", "PANW_", "PRISMA_AIRS_")):
            monkeypatch.delenv(name)


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


def test_unavailable_airs_never_means_allow(monkeypatch):
    monkeypatch.delenv("AIRS_API_KEY", raising=False)
    w = Workspace("t", "s")
    for call in (Call("read", "cover"), Call("summarize"), Call("stage")):
        w.step(call)
    _, end, v = w.step(Call("submit"), airs="live")
    assert (
        v == {"source": "live", "action": "unavailable"}
        and end.state == "held"
        and not w.portal
    )
    _, end, v = w.step(Call("submit"), airs="fixture-block")
    assert v["source"] == "simulated" and end.state == "denied" and not w.portal
    _, end, v = w.step(Call("submit"))
    assert (
        v == {"source": "offline", "action": "unavailable"} and end.state == "completed"
    )
    assert len(w.portal) == 1


def test_live_wire_with_transport_double(monkeypatch):
    monkeypatch.setenv("AIRS_API_KEY", "fixture")
    monkeypatch.setenv("AIRS_PROFILE", "sample")

    def fake(url, *, json, headers, timeout):
        assert url.endswith("/v1/scan/sync/request") and headers == {
            "x-pan-token": "fixture"
        }
        assert json == {
            "ai_profile": {"profile_name": "sample"},
            "contents": [{"response": "synthetic"}],
        }
        return httpx.Response(
            200, json={"action": "block"}, request=httpx.Request("POST", url)
        )

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("fixture")

    assert inspect("synthetic", "live", fake)["action"] == "block"
    assert inspect("synthetic", "live", timeout)["action"] == "unavailable"


def test_timeout_holds_and_denial_wins(monkeypatch):
    b = SimpleNamespace(config={"version": "stub", "thresholds": {"gru": 0.5}})
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
