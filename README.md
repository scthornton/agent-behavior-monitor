# Agent behavior monitor

This prototype checks an AI agent's recent tool calls before it lets the next one run. The demo follows an agent preparing a case summary for upload. The portal is approved in every session. What changes is which records the agent read along the way.

## Try the demo

Download [demo.html](reference/demo.html) using GitHub's **Download raw file** button, then open it in a browser. The demo runs offline without installing Python.

Choose **Case material only** and click **Jump to upload**. The monitor allows the upload. Switch to **Outside material included** at the same step. The agent also read a restricted record from another case, so the monitor holds the upload for review.

Now choose **Outside material excluded**, shown below. The agent left the restricted material out of its summary. The monitor sees the same read and asks for review again. Tool history can tell a reviewer which record to check. It can't tell them what made it into the summary.

![The upload is held for review after a restricted read at event 16. The detector that sees only the upload stays below its cutoff.](docs/demo.png)

## What worked

The test used 360 synthetic sessions covering out-of-case uploads, repeated access attempts after a denial, and unapproved task changes. During tuning, each detector could flag at most 10 out of every 100 benign sessions.

| Detector | Risky sessions caught at the first risky action | Benign sessions flagged |
| --- | ---: | ---: |
| Rules using recent calls | 72/72 | 24/288 |
| Classifier using only the current call | 0/72 | 0/288 |
| TensorFlow sequence model (GRU) | 72/72 | 24/288 |

The rules also caught eight later uploads the GRU missed. Shuffling the earlier calls preserved the GRU's first-action detection rate. When a whole type of risk was left out of training, the GRU missed every first risky action of that type. [Full results and failure examples](docs/findings.md) · [Experiment design](docs/design.md)

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
python -m monitor demo --out run
MONITOR_RUN=run python -m pytest -q
```

The experiment uses synthetic records and an in-memory portal. It runs without an LLM account, GPU or Prisma AIRS credentials. The optional AIRS adapter was tested with simulated responses. [AIRS integration details](docs/design.md#runtime-path-and-trust)
