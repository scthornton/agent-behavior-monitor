# Agent behavior monitor

A local TensorFlow experiment asks whether an agent's recent tool calls help review an approved case upload before it runs. The rules and GRU caught 72/72 risky sessions at the first consequential proposal on held-out workflows. Both held 24/288 benign sessions. The current-call classifier caught 0/72 at the same validation budget. The rules caught eight later risky uploads that the GRU missed.

**Open the saved demo** in a browser. It works offline without Python, a server or an account. It contains one timeline with three sessions, actual scores and an expandable record of what reached the in-memory portal. [Five-minute demo](reference/demo.html)

[Design and research](docs/design.md) | [Findings and failure examples](docs/findings.md) | [Results for all four folds](reference/results.json) | [Session split manifest](reference/split.json)

## Five-minute walkthrough

1. **First minute.** Open the HTML file and choose **Case material only**. Click **Jump to upload**. Inspect the proposed event. The tool is submit, the operation is upload, the destination is the approved case portal, and the permission service allows it. Point out the three detector scores and their cutoffs. The fourth row is the order ablation.
2. **Second minute.** Choose **Outside material included** at the same decision. Event 16 now records a completed restricted read under a related-case lookup grant. Events 20 and 22 record summary preparation and staging. The current-call score stays at 0.45554. The GRU moves from 0.02817 to 0.34111 and crosses its validation cutoff of 0.31668. The rules also request review. Move backward in the timeline to see the evidence available at earlier decisions.
3. **Third minute.** Choose **Outside material excluded**. The restricted read and every other model input match the previous session. All four scores match too. Expand **Reveal fixture truth and enforced replay outcome**. This summary used only approved case material. The two review holds cost one legitimate submission and caught one risky submission.
4. **Fourth minute.** Distinguish the reference replay from enforcement. Reference capture executes permitted synthetic calls in memory so every detector sees the same trajectory. The separate guarded replay stops at its first denial or review. Only the clean case reaches its portal. A permission denial always prevents dispatch.
5. **Fifth minute.** Show the missed later upload and family holdouts in the findings. The rules were cheaper and caught more labeled decisions. The small corpus demonstrates what history can reveal under these task definitions. No customer traffic or live AIRS verdict contributed to these results.

## Setup and clean reproduction

Use CPython 3.12 and uv. The verified machine ran Ubuntu on an ARM64 NVIDIA DGX Spark. The experiment uses its CPU. Network access is needed to install dependencies. Training, evaluation, replay and local scoring then work offline. Exact resolved packages are in requirements.txt.

```bash
cd /home/plucky/agent-behavior-monitor
uv venv --python /usr/bin/python3 .venv
uv pip sync --python .venv/bin/python requirements.txt
test ! -e run-release
.venv/bin/python -m monitor generate --out run-release
.venv/bin/python -m monitor train --out run-release
.venv/bin/python -m monitor evaluate --out run-release
.venv/bin/python -m monitor demo --out run-release
MONITOR_RUN=run-release .venv/bin/python -m pytest -q
```

Use your checkout path on another machine. For a fresh run in this workspace, choose an unused output name in all commands. The release verification started with a fresh run-release output path. Generation writes sessions.jsonl and split.json. Training writes each fold's vocabulary, validation results, cutoffs, weights and model version. Evaluation writes results.json. The demo writes demo.html, demo.json and a score request. Open the generated page directly.

```bash
xdg-open run-release/demo.html
```

The reference folder ships the main trained weights, scores and demo. To serve the shipped models immediately, use reference as the output argument to serve.

## Exercise the public scoring interface

Start the local service in one terminal.

```bash
.venv/bin/python -m monitor serve --out run-release --port 8765
```

In another terminal, post the synthetic case and measure warm sequential HTTP requests.

```bash
curl --fail --silent --show-error -H 'Content-Type: application/json' \
  --data-binary @run-release/upload-request.json http://127.0.0.1:8765/score
.venv/bin/python scripts/bench_http.py --out run-release
```

POST /score accepts a proposed event and the full list of earlier terminal events for that tenant and session. A successful response contains score (ordered GRU), scores for all four detectors, model_version, thresholds, disposition, reason and the observed evidence. A score is a ranking number. A score of 0.34 has no established interpretation as a probability of disclosure.

Event numbers start at one and alternate between proposals and terminal outcomes. A gap yields review with a null score. A cross-session event, a future outcome or an invalid wire schema yields HTTP 422. Denials take precedence over scores. A busy or timed-out scorer yields review. The local host and policy registry are trusted. The endpoint binds to loopback and accepts host-supplied metadata. Exposing it to untrusted clients would require authenticated identity binding, authoritative permission lookup and durable replay protection.

## Optional AIRS demo

The content inspection point is after local authorization and behavioral policy allow a submit, immediately before the in-memory portal receives its staged synthetic text. The default offline mode records an unavailable verdict. To show a simulated block without modifying the reference timeline, use a separate output folder.

```bash
mkdir -p run-sim
cp -r reference/main run-sim/
.venv/bin/python -m monitor demo --out run-sim --airs fixture-block
```

For an explicitly live US-region run, supply AIRS_API_KEY and AIRS_PROFILE in the process environment and pass `--airs live` to the demo. This sends the staged synthetic response to the documented US endpoint. A block prevents submission. Transport errors, missing credentials and unusable responses return unavailable and hold the upload. The live path was verified against a transport double. Tenant authentication, profile behavior, latency and detection quality remain unverified. [Runtime API use cases](https://pan.dev/prisma-airs/api/airuntimesecurity/usecases/)

## Implementation map

| File | Responsibility |
| --- | --- |
| monitor/contracts.py | Versioned metadata allowlist, tenant and session boundaries, order checks |
| monitor/workflow.py | Tool registry, permission check, synthetic workspace, dispatch and capture |
| monitor/fixtures.py | Workflow layouts, paired variants and private oracle |
| monitor/models.py | Train-only vocabularies, matched inputs, history rule and Keras models |
| monitor/experiment.py | Grouped training, validation budgets, folds, test metrics and CPU timing |
| monitor/service.py | Bounded scoring worker, failure handling and FastAPI endpoint |
| monitor/airs.py | Optional content verdict, independent of behavioral features |
| monitor/demo.py | Reference and guarded replay, single-file browser timeline |

The seed is 1741. Five training layouts, two validation layouts and three test layouts each have eight variations, three families and five matched cases per family. The manifest records every session's grouping and split. Generation is deterministic. Three clean runs on this host produced identical model versions, decisions and metrics apart from timing.

## Verification on the Spark

Twelve tests passed. A headless Chromium run opened the offline HTML, switched all three sessions, moved to the previous and next proposals and revealed the enforced outcome. A loopback client exercised the public endpoint with the saved upload and 260 sampled requests, including warmups. All 5,016 held-out decisions also had matching alert classifications in batch evaluation and single-request inference for every method. The adapter tests used synthetic responses and missing credentials.
