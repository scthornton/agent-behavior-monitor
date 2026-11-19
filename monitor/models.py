import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import hashlib
from functools import lru_cache

import numpy as np
import tensorflow as tf

from .contracts import FIELDS, WINDOW
from .fixtures import SEED

tf.config.experimental.enable_op_determinism()
NAMES = ("rule", "current", "gru", "shuffled")


def fields(event):
    return tuple(event[k] for k in FIELDS)


class Encoder:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vector = lru_cache(maxsize=2048)(self._vector)
        self.columns = {}
        self.width = 0
        for i, values in enumerate(vocab):
            for value in ["<unknown>", *values]:
                self.columns[i, value] = self.width
                self.width += 1

    @classmethod
    def fit(cls, train):
        return cls([sorted({e[k] for s in train for e in s["events"]}) for k in FIELDS])

    def _vector(self, event):
        v = np.zeros(self.width, np.float32)
        for i, value in enumerate(event):
            v[self.columns.get((i, value), self.columns[i, "<unknown>"])] = 1
        return v

    def encode(self, history, proposed, kind):
        if kind == "current":
            return self.vector(fields(proposed))
        previous = [fields(e) for e in history[-WINDOW:]]
        if kind == "shuffled":
            # Canonicalize first. Equal multisets get the same seeded permutation,
            # independent of original order, labels, session and dataset split.
            previous = sorted(previous)
            digest = hashlib.sha256(repr(previous).encode()).digest()
            np.random.default_rng(int.from_bytes(digest[:8], "big") + SEED).shuffle(
                previous
            )
        x = np.zeros((WINDOW + 1, self.width), np.float32)
        for i, e in enumerate([*previous, fields(proposed)]):
            x[i] = self.vector(e)
        return x


def rule(history, p):
    prior = history[-WINDOW:]
    if (
        p["tool"] == "read"
        and p["permission"] == "deny"
        and any(
            e["tool"] == "read" and e["scope"] == p["scope"] and e["state"] == "denied"
            for e in prior
        )
    ):
        return 1.0
    product = p["operation"]
    if p["tool"] == "submit":
        product = next(
            (
                e["operation"]
                for e in reversed(prior)
                if e["tool"] == "stage" and e["state"] == "completed"
            ),
            None,
        )
    if product == "bulletin" and not any(
        e["operation"] == "amend" and e["state"] == "completed" for e in prior
    ):
        return 0.95
    if p["tool"] == "submit" and any(
        e["scope"] == "related_lookup" and e["state"] == "completed" for e in prior
    ):
        return 0.8
    return 0.0


def make_model(kind, width):
    tf.keras.utils.set_random_seed(SEED)
    if kind == "current":
        inputs = tf.keras.Input((width,))
        state = tf.keras.layers.Dense(20, activation="relu")(inputs)
    else:
        inputs = tf.keras.Input((WINDOW + 1, width))
        masked = tf.keras.layers.Masking()(inputs)
        state = tf.keras.layers.GRU(20)(masked)
    model = tf.keras.Model(
        inputs, tf.keras.layers.Dense(1, activation="sigmoid")(state)
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.004), loss="binary_crossentropy")
    return model


def examples(sessions):
    for s in sessions:
        history = []
        for i, label in enumerate(s["labels"]):
            p, end = s["events"][2 * i : 2 * i + 2]
            yield history.copy(), p, label
            history.append(end)


def matrix(encoder, rows, kind):
    return np.stack([encoder.encode(h, p, kind) for h, p, _ in rows])


def predict(model, x):
    return np.concatenate(
        [
            model(x[i : i + 256], training=False).numpy().ravel()
            for i in range(0, len(x), 256)
        ]
    )


def train_model(kind, x, labels):
    model = make_model(kind, x.shape[-1])
    rng = np.random.default_rng(SEED)
    for epoch in range(12):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            idx = order[start : start + 128]
            model.train_on_batch(
                x[idx], labels[idx], sample_weight=1.0 + 5.0 * labels[idx]
            )
    return model
