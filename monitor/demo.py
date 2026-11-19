"""Export one interactive offline timeline, backed by real replay and model scores."""

import json
from pathlib import Path

from .experiment import Bundle
from .fixtures import plan, save
from .service import Scorer
from .workflow import Workspace

CASES = [
    ("Case material only", 1),
    ("Outside material included", 0),
    ("Outside material excluded", 2),
]
LAYOUT = "catalog_check"


def demo(out, airs):
    scorer = Scorer(Bundle(out / "main"))
    runs = []
    try:
        for idx, (title, case) in enumerate(CASES):
            ws = Workspace("demo", f"replay-{idx + 1}")
            calls = plan(LAYOUT, 0, "scope", case)
            trace = []
            # Shadow timeline allows consistent comparisons even if an earlier
            # proposal would have been held. Never bypass a permission denial.
            for call in calls:
                request, terminal, _ = ws.step(call)
                decision = scorer.decide(request)
                trace.append(
                    {
                        "proposed": request.proposed.model_dump(),
                        "decision": decision,
                        "reference_outcome": terminal.state,
                    }
                )
            if idx == 0:
                save(out / "upload-request.json", request.model_dump())
            guarded = Workspace("demo", ws.session)
            for call in calls:
                _, terminal, verdict = guarded.step(call, scorer.decide, airs)
                if terminal.state in ("denied", "held"):
                    break
            runs.append(
                {
                    "title": title,
                    "steps": trace,
                    "sources": list(ws.staged_sources),
                    "guarded_last": terminal.model_dump(),
                    "airs": verdict,
                    "airs_mode": airs,
                    "submissions": len(guarded.portal),
                }
            )
            print(
                title,
                trace[-1]["decision"]["scores"],
                "guarded",
                terminal.state,
                "content",
                verdict["action"],
                "portal entries",
                len(guarded.portal),
            )
    finally:
        scorer.close()
    save(out / "demo.json", runs)
    page = Path(__file__).with_name("timeline.html").read_text()
    data = json.dumps(runs, ensure_ascii=True).replace("<", "\\u003c")
    (out / "demo.html").write_text(page.replace("/*REPLAYS*/[]", data))
