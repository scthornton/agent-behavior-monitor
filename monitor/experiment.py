"""Fit and freeze all folds before evaluation reads the held-out decisions."""

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .fixtures import FAMILIES, SEED, save
from .models import NAMES, Encoder, examples, matrix, predict, rule, tf, train_model

BUDGET = 0.10


def sessions(out, split, fold):
    with (out / "sessions.jsonl").open() as src:
        records = (json.loads(line) for line in src)
        return [
            s
            for s in records
            if s["split"] == split
            and (fold == "main" or (s["family"] == fold) == (split == "test"))
        ]


def scores(kind, model, enc, rows):
    if kind == "rule":
        return np.array([rule(h, p) for h, p, _ in rows])
    return predict(model, matrix(enc, rows, kind)).astype(float).round(5)


def by_session(values, traces):
    offset = 0
    for s in traces:
        n = len(s["labels"])
        yield s, values[offset : offset + n]
        offset += n
    assert offset == len(values)


def calibrate(values, val):
    peaks = [max(v) for s, v in by_session(values, val) if s["first"] is None]
    # Scores are rounded to five decimals everywhere. Move one unit past the
    # disallowed order statistic, including when all scores are tied at zero.
    peaks.sort(reverse=True)
    cutoff = round(peaks[int(len(peaks) * BUDGET)] + 0.00001, 5)
    return cutoff


def train(out):
    for fold in ("main", *FAMILIES):
        dest = out / fold
        dest.mkdir(exist_ok=True)
        fit, val = sessions(out, "train", fold), sessions(out, "validation", fold)
        enc = Encoder.fit(fit)
        fit_rows, val_rows = list(examples(fit)), list(examples(val))
        y = np.array([label for _, _, label in fit_rows], np.float32)
        config = {
            "seed": SEED,
            "budget": BUDGET,
            "vocab": enc.vocab,
            "thresholds": {},
            "validation": {},
            "training_sessions": len(fit),
            "validation_sessions": len(val),
        }
        signature = hashlib.sha256()
        for kind in NAMES:
            model = None
            if kind != "rule":
                model = train_model(kind, matrix(enc, fit_rows, kind), y)
                model.save(dest / f"{kind}.keras")
                for w in model.get_weights():
                    signature.update(w.tobytes())
            values = scores(kind, model, enc, val_rows)
            cutoff = calibrate(values, val)
            config["thresholds"][kind] = cutoff
            config["validation"][kind] = metrics(values, val, cutoff)
        signature.update(json.dumps(config, sort_keys=True).encode())
        config["version"] = "v1-" + signature.hexdigest()[:12]
        save(dest / "config.json", config)
        print(f"Froze {fold} {config['version']}", flush=True)


def metrics(values, traces, cutoff):
    tp = fp = fn = tn = flagged = benign = risky = timely = ever = 0
    failures = {}
    families = {}
    for s, v in by_session(values, traces):
        actual = np.array(s["labels"], bool)
        alerts = v >= cutoff
        a = int((alerts & actual).sum())
        b = int((alerts & ~actual).sum())
        c = int((~alerts & actual).sum())
        d = int((~alerts & ~actual).sum())
        tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + d
        first = s["first"]
        sub = families.setdefault(
            s["family"], {"risky": 0, "timely": 0, "benign": 0, "flagged": 0}
        )
        if first is None:
            benign += 1
            flagged += int(alerts.any())
            sub["benign"] += 1
            sub["flagged"] += int(alerts.any())
        else:
            risky += 1
            timely += int(alerts[first])
            ever += int(a > 0)
            sub["risky"] += 1
            sub["timely"] += int(alerts[first])
        for name, mask in (
            ("false_positive", alerts & ~actual),
            ("miss", ~alerts & actual),
        ):
            if mask.any() and name not in failures:
                i = int(np.flatnonzero(mask)[0])
                failures[name] = {
                    "session": s["session"],
                    "family": s["family"],
                    "layout": s["group"],
                    "case": s["case"],
                    "step": i + 1,
                    "score": float(v[i]),
                    "cutoff": cutoff,
                    "proposal": s["events"][i * 2],
                    "prior": s["events"][1 : i * 2 : 2],
                }
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "flagged": flagged,
        "benign": benign,
        "benign_per_100": 100 * flagged / benign,
        "timely": timely,
        "risky": risky,
        "eventually_detected": ever,
        "families": families,
        "errors": failures,
    }


class Bundle:
    def __init__(self, dest):
        self.config = json.loads((dest / "config.json").read_text())
        self.enc = Encoder(self.config["vocab"])
        self.models = {
            k: tf.keras.models.load_model(dest / f"{k}.keras", compile=False)
            for k in NAMES
            if k != "rule"
        }
        self.fast = {
            k: tf.function(m.call, reduce_retracing=True)
            for k, m in self.models.items()
        }

    def one(self, kind, history, proposed):
        if kind == "rule":
            return rule(history, proposed)
        x = self.enc.encode(history, proposed, kind)
        return round(
            float(
                self.fast[kind](tf.convert_to_tensor(x[None]), training=False)
                .numpy()
                .item()
            ),
            5,
        )


def latency(bundle, rows):
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(rows), 260, replace=False)
    data = {}
    for k in NAMES:
        samples = []
        for i in indices:
            h, p, _ = rows[i]
            start = time.perf_counter_ns()
            bundle.one(k, h, p)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        data[k] = dict(zip(("p50_ms", "p95_ms"), np.percentile(samples[20:], [50, 95])))
    try:
        info = json.loads(subprocess.check_output(["lscpu", "-J"], text=True))["lscpu"]
        model = ", ".join(
            dict.fromkeys(row["data"] for row in info if row["field"] == "Model name:")
        )
    except (OSError, KeyError, ValueError, subprocess.SubprocessError):
        model = platform.processor()
    product = Path("/sys/class/dmi/id/product_name")
    return {
        "samples": 240,
        "warmup": 20,
        "cpu": model or platform.machine(),
        "arch": platform.machine(),
        "system": product.read_text().strip()
        if product.exists()
        else platform.system(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "timing": data,
    }


def evaluate(out):
    results = {}
    for fold in ("main", *FAMILIES):
        traces = sessions(out, "test", fold)
        rows = list(examples(traces))
        b = Bundle(out / fold)
        result = {
            "version": b.config["version"],
            "thresholds": b.config["thresholds"],
            "validation": {
                k: v["benign_per_100"] for k, v in b.config["validation"].items()
            },
            "methods": {},
        }
        for k in NAMES:
            values = scores(k, b.models.get(k), b.enc, rows)
            result["methods"][k] = metrics(values, traces, result["thresholds"][k])
        if fold == "main":
            result["latency"] = latency(b, rows)
        results[fold] = result
    save(
        out / "results.json",
        {"seed": SEED, "target_per_100": 100 * BUDGET, "folds": results},
    )
    for k, m in results["main"]["methods"].items():
        print(
            k,
            {
                key: m[key]
                for key in ("timely", "risky", "flagged", "benign", "tp", "fp", "fn")
            },
        )
