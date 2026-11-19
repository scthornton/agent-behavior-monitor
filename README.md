# Agent behavior monitor

This prototype checks an AI agent's recent tool calls before it lets the next one run. The demo follows an agent preparing a case summary for upload. The portal is approved in every session. What changes is which records the agent read along the way.

## Try the demo

Download [demo.html](reference/demo.html) using GitHub's **Download raw file** button, then open it in a browser. The demo runs offline without installing Python.

Choose **Case material only** and click **Jump to upload**. The monitor allows the upload. Switch to **Outside material included** at the same step. The agent also read a restricted record from another case, so the monitor holds the upload for review.

Now choose **Outside material excluded**, shown below. The agent left the restricted material out of its summary. The monitor sees the same read and asks for review again. Tool history can tell a reviewer which record to check. It can't tell them what made it into the summary.

![The upload is held for review after a restricted read at event 16. The detector that sees only the upload stays below its cutoff.](docs/demo.png)

## What the test showed

The test used 360 synthetic sessions covering out-of-case uploads, repeated access attempts after a denial, and unapproved task changes. During tuning, each detector could flag at most 10 out of every 100 benign sessions.

| Detector | Risky sessions caught at the first risky action | Benign sessions flagged |
| --- | ---: | ---: |
| Rules using recent calls | 72/72 | 24/288 |
| Classifier using only the current call | 0/72 | 0/288 |
| TensorFlow sequence model (GRU), shipped seed | 72/72 | 24/288 |
| The same GRU across 12 training seeds | 48 to 72 of 72, mean 59 | 8 to 32 of 288 |

Earlier calls matter: the detector that sees only the current call catches nothing. The rules are the dependable way to use them here. The GRU matched the rules on one training seed out of twelve. It missed eight later uploads the rules caught, and it missed every first risky action of a risk type that was left out of its training. Shuffling the earlier calls did not hurt it, so call order added nothing measurable.

Two limits apply to every detector. The 24 benign sessions flagged are the ones that read the outside record and left it out. They look identical to the risky ones, so catching that risk means reviewing both. And the comparison reads only the last 12 calls, which a longer session can outrun. The scoring service closes that gap: it also checks the rule against the whole session, and a session that has been held stays held.

[Full results and failure examples](docs/findings.md) · [Experiment design](docs/design.md) · [Seed and budget results](reference/robustness.json)

## Run it locally

From the repository root, set up Python 3.12 with uv.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip sync requirements.txt
```

Start the scoring API with the included models. In a second terminal, run the curl command from the repository root to score a sample upload.

```bash
python -m monitor serve --out reference --port 8765
```

```bash
curl -sS -H 'Content-Type: application/json' \
  --data-binary @reference/upload-request.json http://127.0.0.1:8765/score
```

To reproduce the experiment on CPU, run the following in the activated environment. Open `run/demo.html` for the resulting replay.

```bash
python -m monitor generate --out run
python -m monitor train --out run
python -m monitor evaluate --out run
python -m monitor robustness --out run
python -m monitor demo --out run
MONITOR_RUN=run python -m pytest -q
```

The `robustness` step retrains the two sequence models under twelve seeds and re-calibrates the shipped models at other alert budgets. It takes a few minutes. Without `MONITOR_RUN`, `python -m pytest -q` generates the sessions and trains every fold in a temporary folder first, which takes about a minute on a laptop CPU.

## Adding a content check

Tool history cannot separate the last two demo sessions. A content check can. The monitor asks for one on every upload it would allow or hold for review. A block always wins, and an allow never releases a hold. To see this offline, rebuild the demo with a simulated check that looks for the synthetic out-of-case marker. Then open `run/demo.html` and expand the last section for each session.

```bash
python -m monitor demo --out run --airs fixture-pattern
```

| Session | Tool history says | Simulated content check | Outcome |
| --- | --- | --- | --- |
| Case material only | allow | allow | uploaded |
| Outside material included | review | block | denied |
| Outside material excluded | review | allow | held for review |

`--airs live` sends the staged synthetic summary to the Prisma AIRS scan API instead. It reads an API key and a security profile name from `AIRS_API_KEY` and `AIRS_PROFILE`. The `PANW_AI_SEC_API_KEY` and `PRISMA_AIRS_*` names also work, and `AIRS_API_ENDPOINT` selects a regional scan host. If the check cannot return a verdict, the upload is held and the page shows the reason. The live path has been tested with simulated responses only.

The experiment uses synthetic records and an in-memory portal. It runs without an LLM account, GPU or Prisma AIRS credentials. [AIRS integration details](docs/design.md#runtime-path-and-trust)
