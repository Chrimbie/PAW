#!/usr/bin/env python3
"""PAW 1.0: four-neuron task-offloading regression engine, Python stdlib only.

Research boundary: disk-backed task-state isolation, not general intelligence or
proof that neuron saturation has been eliminated. No hidden larger model.

Structure: input affine neuron -> tanh hidden 1 -> tanh hidden 2 -> linear output.
At every Adam update: load state -> train -> atomically save -> clear vessel.
Each task has independent weights AND Adam moments, persisted as validated JSON.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

import config

FORMAT = "PAW-TASK-V1"
WEIGHTS = 8


class PAWError(Exception):
    """User-visible PAW fault."""


def finite_number(x: object, field: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (float, int)):
        raise PAWError(f"{field} must be a finite number")
    value = float(x)
    if not math.isfinite(value):
        raise PAWError(f"{field} must be finite")
    return value


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def sha(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def task_check(task: str) -> str:
    if not isinstance(task, str) or not (1 <= len(task) <= config.MAX_TASK_NAME_LENGTH):
        raise PAWError("Task name must contain 1 to 100 characters")
    if not task.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in task):
        raise PAWError("Task name cannot be blank or contain control characters")
    return task


def initial_params(seed: int) -> list[float]:
    rng = random.Random(seed)
    # Weight initialization small enough to avoid initial tanh saturation.
    return [rng.uniform(0.5, 1.0) for _ in range(4)] + [0.0] * 4


@dataclass
class Core:
    params: list[float]

    def __post_init__(self) -> None:
        if len(self.params) != WEIGHTS:
            raise PAWError("Core must contain exactly eight scalar parameters")
        self.params = [finite_number(x, "core parameter") for x in self.params]

    def forward(self, x: float) -> tuple[float, tuple[float, float, float]]:
        x = finite_number(x, "input")
        w0, w1, w2, w3, b0, b1, b2, b3 = self.params
        a0 = w0 * x + b0
        h1 = math.tanh(w1 * a0 + b1)
        h2 = math.tanh(w2 * h1 + b2)
        output = w3 * h2 + b3
        if not math.isfinite(output):
            raise PAWError("Non-finite output; refusing prediction or update")
        return output, (a0, h1, h2)

    def loss_and_gradient(self, batch: Sequence[tuple[float, float]]) -> tuple[float, list[float]]:
        if not batch:
            raise PAWError("Empty training batch")
        grads = [0.0] * WEIGHTS
        mse = 0.0
        w0, w1, w2, w3 = self.params[:4]
        n = len(batch)
        for x, y in batch:
            pred, (a0, h1, h2) = self.forward(x)
            err = pred - y
            mse += (err * err) / n
            d = 2.0 * err / n
            # Reverse of the four affine connections (two use tanh).
            grads[3] += d * h2
            grads[7] += d
            d2 = d * w3 * (1.0 - h2 * h2)
            grads[2] += d2 * h1
            grads[6] += d2
            d1 = d2 * w2 * (1.0 - h1 * h1)
            grads[1] += d1 * a0
            grads[5] += d1
            d0 = d1 * w1
            grads[0] += d0 * x
            grads[4] += d0
        if not math.isfinite(mse) or any(not math.isfinite(g) for g in grads):
            raise PAWError("Non-finite training loss/gradient; no state committed")
        return mse, grads

    def hidden_health(self, xs: Sequence[float]) -> dict:
        if not xs:
            raise PAWError("Health measurement requires inputs")
        h1s, h2s = [], []
        saturation = 0
        for x in xs:
            _, (_, h1, h2) = self.forward(x)
            h1s.append(h1)
            h2s.append(h2)
            saturation += abs(h1) >= config.SATURATION_ABS
            saturation += abs(h2) >= config.SATURATION_ABS
        def variance(vals: list[float]) -> float:
            mean = fmean(vals)
            return fmean((v - mean) ** 2 for v in vals)
        v1, v2 = variance(h1s), variance(h2s)
        dead_count = (sum(v < config.DEAD_VARIANCE for v in (v1, v2))
                      if len(xs) >= 8 and len(set(xs)) >= 2 else None)
        return {"saturated_fraction": saturation / (2 * len(xs)),
                "dead_hidden_neurons": dead_count,
                "hidden_variances": [v1, v2],
                "max_absolute_activation": max(map(abs, h1s + h2s)),
                "samples": len(xs)}


@dataclass
class Adam:
    m: list[float]
    v: list[float]
    steps: int

    @classmethod
    def fresh(cls) -> Adam:
        return cls([0.0] * WEIGHTS, [0.0] * WEIGHTS, 0)

    @classmethod
    def load(cls, data: dict) -> Adam:
        if not isinstance(data, dict) or set(data) != {"m", "v", "steps"}:
            raise PAWError("Invalid optimizer state")
        if type(data["steps"]) is not int or data["steps"] < 0:
            raise PAWError("Invalid Adam step counter")
        if any(not isinstance(data[k], list) or len(data[k]) != WEIGHTS for k in ("m", "v")):
            raise PAWError("Adam moments have wrong shape")
        m = [finite_number(n, "Adam m") for n in data["m"]]
        v = [finite_number(n, "Adam v") for n in data["v"]]
        if any(n < 0 for n in v):
            raise PAWError("Adam variance moment cannot be negative")
        return cls(m, v, data["steps"])

    def dump(self) -> dict:
        return {"m": self.m[:], "v": self.v[:], "steps": self.steps}

    def step(self, params: list[float], grads: list[float], lr: float) -> list[float]:
        norm = math.sqrt(sum(g * g for g in grads))
        if norm > config.GRADIENT_CLIP_NORM:
            grads = [g * config.GRADIENT_CLIP_NORM / norm for g in grads]
        self.steps += 1
        t = self.steps
        b1_pow = 1.0 - 0.9 ** t
        b2_pow = 1.0 - 0.999 ** t
        result = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            update = lr * (self.m[i] / b1_pow) / (math.sqrt(self.v[i] / b2_pow) + 1e-8)
            result.append(finite_number(p - update, "updated parameter"))
        return result


def normalized_pairs(rows: Sequence[tuple[float, float]], stats: dict) -> list[tuple[float, float]]:
    return [((x - stats["x_mean"]) / stats["x_scale"],
             (y - stats["y_mean"]) / stats["y_scale"]) for x, y in rows]


def data_stats(rows: Sequence[tuple[float, float]]) -> dict:
    if len(rows) < 2:
        raise PAWError("At least two training rows are required")
    xs, ys = [r[0] for r in rows], [r[1] for r in rows]
    xm, ym = fmean(xs), fmean(ys)
    xscale = math.sqrt(fmean((x - xm) ** 2 for x in xs))
    yscale = math.sqrt(fmean((y - ym) ** 2 for y in ys))
    if not math.isfinite(xscale) or xscale < 1e-12:
        raise PAWError("Input values must vary; input standard deviation is zero")
    if not math.isfinite(yscale):
        raise PAWError("Invalid output variance")
    return {"x_mean": xm, "x_scale": xscale,
            "y_mean": ym, "y_scale": max(yscale, 1e-12)}


def read_csv(path: str | Path) -> list[tuple[float, float]]:
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["x", "y"]:
                raise PAWError("CSV needs exactly two columns, in order: x,y")
            result = []
            for line, item in enumerate(reader, 2):
                if len(result) >= config.MAX_CSV_ROWS:
                    raise PAWError(f"CSV exceeds configured {config.MAX_CSV_ROWS} row limit")
                try:
                    result.append((finite_number(float(item["x"]), f"row {line} x"),
                                   finite_number(float(item["y"]), f"row {line} y")))
                except (ValueError, TypeError, KeyError) as exc:
                    raise PAWError(f"Bad numeric CSV row {line}: {exc}") from exc
    except OSError as exc:
        raise PAWError(f"Cannot read CSV: {exc}") from exc
    if len(result) < 5:
        raise PAWError("CSV must include at least five rows")
    return result


def split_rows(rows: list[tuple[float, float]], seed: int | None = None,
               validation_fraction: float | None = None) -> tuple[list, list]:
    shuffled = rows[:]
    random.Random(config.SEED if seed is None else seed).shuffle(shuffled)
    fraction = config.VALIDATION_FRACTION if validation_fraction is None else validation_fraction
    if not 0 < fraction < 0.5:
        raise PAWError("Validation fraction must be greater than 0 and less than 0.5")
    n_val = max(1, min(len(shuffled) - 2, round(len(shuffled) * fraction)))
    return shuffled[n_val:], shuffled[:n_val]


class Store:
    def __init__(self, root: str | Path = config.STORE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise PAWError("Store root must not be a symlink")

    def file(self, task: str) -> Path:
        return self.root / (hashlib.sha256(task_check(task).encode("utf-8")).hexdigest() + ".json")

    @contextmanager
    def lock(self, task: str):
        path = self.file(task).with_suffix(".lock")
        deadline = time.monotonic() + config.LOCK_TIMEOUT_SECONDS
        fd = None
        while fd is None:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise PAWError(f"Task is locked: {task}. Check for another PAW process. "
                                   f"Only remove {path.name} after confirming it is stale.")
                time.sleep(0.05)
        try:
            os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            yield
        finally:
            # If process crashes, lock remains, protecting against silent overwrite.
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _decode(path: Path, expected_task: str | None) -> dict:
        try:
            if path.stat().st_size > 1_000_000:
                raise PAWError("Task file exceeds 1MB safety limit")
            with path.open("r", encoding="utf-8") as f:
                outer = json.load(f)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PAWError(f"Cannot read task state {path.name}: {exc}") from exc
        if not isinstance(outer, dict) or set(outer) != {"checksum", "payload"}:
            raise PAWError(f"Invalid task envelope: {path.name}")
        payload = outer["payload"]
        if not isinstance(payload, dict) or not isinstance(outer["checksum"], str) or sha(payload) != outer["checksum"]:
            raise PAWError(f"Task checksum mismatch: {path.name}")
        if payload.get("format") != FORMAT or not isinstance(payload.get("task"), str):
            raise PAWError(f"Unsupported task format in {path.name}")
        task_check(payload["task"])
        if expected_task is not None and payload["task"] != expected_task:
            raise PAWError("Task identity mismatch")
        if not isinstance(payload.get("stats"), dict) or set(payload["stats"]) != {"x_mean", "x_scale", "y_mean", "y_scale"}:
            raise PAWError("Invalid normalization values")
        for k, v in payload["stats"].items():
            finite_number(v, k)
        if payload["stats"]["x_scale"] <= 0 or payload["stats"]["y_scale"] <= 0:
            raise PAWError("Normalization scales must be positive")
        if type(payload.get("seed")) is not int:
            raise PAWError("Invalid initialization seed")
        Core(payload.get("params", []))
        Adam.load(payload.get("adam", {}))
        if type(payload.get("updates")) is not int or payload["updates"] < 0:
            raise PAWError("Invalid update counter")
        if payload["updates"] != payload["adam"]["steps"]:
            raise PAWError("Update count does not match optimizer")
        if ("validation_fraction" not in payload
                or not 0 < finite_number(payload["validation_fraction"], "validation fraction") < 0.5):
            raise PAWError("Invalid validation fraction")
        goal = payload.get("run_goal_epochs")
        if goal is not None and (type(goal) is not int or goal < 1):
            raise PAWError("Invalid interrupted-run goal")
        if type(payload.get("epochs_trained")) is not int or payload["epochs_trained"] < 0:
            raise PAWError("Invalid epoch counter")
        if type(payload.get("cursor")) is not int or payload["cursor"] < 0:
            raise PAWError("Invalid resume cursor")
        if (not isinstance(payload.get("train_signature"), str)
                or len(payload["train_signature"]) != 64):
            raise PAWError("Invalid dataset signature")
        if type(payload.get("current_batch_size")) is not int or payload["current_batch_size"] < 1:
            raise PAWError("Invalid training batch size")
        return payload

    def load(self, task: str, recovery: bool = False) -> dict | None:
        path = self.file(task)
        backup = path.with_suffix(".bak")
        if not path.exists():
            if backup.exists():
                if recovery:
                    return self._decode(backup, task)
                raise PAWError("Primary task file is missing but backup exists; run recover TASK")
            return None
        try:
            return self._decode(path, task)
        except PAWError:
            if recovery and backup.exists():
                return self._decode(backup, task)
            raise

    def save(self, task: str, payload: dict) -> None:
        # Caller holds the task lock. Atomic replace: readers never see half a JSON document.
        path = self.file(task)
        if payload["task"] != task:
            raise PAWError("Refusing to save payload under a different task")
        document = canonical({"payload": payload, "checksum": sha(payload)})
        if len(document) > 1_000_000:
            raise PAWError("Task state exceeds 1MB safety limit")
        fd, temp = tempfile.mkstemp(prefix=".paw_tmp_", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(document)
                f.flush()
                os.fsync(f.fileno())
            # Previous valid checkpoint is recoverable in a .bak file.
            if path.exists():
                backup = path.with_suffix(".bak")
                bfd, btmp = tempfile.mkstemp(prefix=".paw_bak_", dir=self.root)
                try:
                    with path.open("rb") as src, os.fdopen(bfd, "wb") as dest:
                        import shutil
                        shutil.copyfileobj(src, dest)
                        dest.flush()
                        os.fsync(dest.fileno())
                    os.replace(btmp, backup)
                finally:
                    if os.path.exists(btmp):
                        os.unlink(btmp)
            os.replace(temp, path)
            # POSIX durability of the directory entry; Windows replacement already atomic.
            if os.name == "posix":
                try:
                    dfd = os.open(self.root, os.O_RDONLY)
                    try:
                        os.fsync(dfd)
                    finally:
                        os.close(dfd)
                except OSError:
                    pass
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def tasks(self) -> list[dict]:
        result = []
        for path in sorted(self.root.glob("*.json")):
            payload = self._decode(path, expected_task=None)
            if self.file(payload["task"]) != path:
                raise PAWError(f"Unexpected filename for task {payload['task']}")
            result.append({"task": payload["task"], "updates": payload["updates"]})
        return sorted(result, key=lambda x: x["task"].lower())

    def recover(self, task: str) -> int:
        with self.lock(task):
            path = self.file(task)
            backup = path.with_suffix(".bak")
            if not backup.exists():
                raise PAWError("No backup exists")
            payload = self._decode(backup, task)
            self.save(task, payload)
            return payload["updates"]


class PAW:
    def __init__(self, store: Store | None = None):
        self.store = store or Store()

    @staticmethod
    def _fresh(task: str, train: list[tuple[float, float]], signature: str) -> dict:
        seed = config.SEED
        return {"format": FORMAT, "task": task, "seed": seed, "params": initial_params(seed),
                "adam": Adam.fresh().dump(), "stats": data_stats(train), "updates": 0,
                "epochs_trained": 0, "cursor": 0, "run_goal_epochs": None,
                "validation_fraction": config.VALIDATION_FRACTION,
                "train_signature": signature,
                "current_batch_size": config.BATCH_SIZE}

    @staticmethod
    def _predict(payload: dict, x: float) -> float:
        stats = payload["stats"]
        normalized_x = (x - stats["x_mean"]) / stats["x_scale"]
        normalized_y = Core(payload["params"]).forward(normalized_x)[0]
        return finite_number(normalized_y * stats["y_scale"] + stats["y_mean"], "prediction")

    def predict(self, task: str, x: float) -> float:
        with self.store.lock(task):
            state = self.store.load(task)
            if state is None:
                raise PAWError(f"Unknown task: {task}")
            return self._predict(state, finite_number(x, "input"))

    def evaluate(self, task: str, rows: list[tuple[float, float]]) -> dict:
        with self.store.lock(task):
            state = self.store.load(task)
            if state is None:
                raise PAWError(f"Unknown task: {task}")
            return self.metrics(state, rows)

    @staticmethod
    def metrics(payload: dict, rows: Sequence[tuple[float, float]]) -> dict:
        if not rows:
            raise PAWError("Cannot evaluate an empty dataset")
        errors = [(PAW._predict(payload, x) - y) ** 2 for x, y in rows]
        metric = finite_number(fmean(errors), "evaluation MSE")
        stats = payload["stats"]
        baseline = finite_number(fmean((stats["y_mean"] - y) ** 2 for _, y in rows), "baseline MSE")
        xs = [(x - stats["x_mean"]) / stats["x_scale"] for x, _ in rows]
        return {"mse": metric, "rmse": math.sqrt(metric), "mean_baseline_mse": baseline,
                "health": Core(payload["params"]).hidden_health(xs)}

    def _log_epoch(self, payload: dict, train_metrics: dict, val_metrics: dict) -> None:
        # Human-readable per-epoch audit log; task checkpoints remain authoritative.
        health = val_metrics["health"]
        line = (f"| {payload['task']!r} | {payload['epochs_trained']} | {payload['updates']} | "
                f"{train_metrics['mse']:.10g} | {val_metrics['mse']:.10g} | "
                f"{health['saturated_fraction']:.6f} | {health['dead_hidden_neurons']} |\n")
        log = self.store.root / "log.md"
        new = not log.exists()
        with log.open("a", encoding="utf-8") as f:
            if new:
                f.write("# PAW v1 Training Log\n\n"
                        "| Task | Epoch | Updates | Train MSE | Validation MSE | Validation saturation fraction | Dead hidden |\n"
                        "|---|---:|---:|---:|---:|---:|---:|\n")
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def train(self, task: str, rows: list[tuple[float, float]], epochs: int | None = None,
              reporter=None) -> dict:
        task_check(task)
        if not rows or len(rows) < 5:
            raise PAWError("At least 5 rows needed")
        if epochs is None:
            epochs = config.EPOCHS_PER_TRAIN
        if type(epochs) is not int or epochs < 1:
            raise PAWError("Training epochs must be a positive integer")
        if config.BATCH_SIZE < 1 or not math.isfinite(config.LEARNING_RATE) or config.LEARNING_RATE <= 0:
            raise PAWError("Invalid batch size or learning rate in config.py")
        signature = sha(rows)
        with self.store.lock(task):
            state = self.store.load(task)
            fresh = state is None
            seed = config.SEED if fresh else state["seed"]
            val_fraction = config.VALIDATION_FRACTION if fresh else state["validation_fraction"]
            train, valid = split_rows(rows, seed, val_fraction)
            if fresh:
                state = self._fresh(task, train, signature)
                self.store.save(task, state)
            active_run = state["run_goal_epochs"] is not None
            if active_run:
                # Restart first completes the PREVIOUS requested run; never adds a second run.
                if signature != state["train_signature"] or config.BATCH_SIZE != state["current_batch_size"]:
                    raise PAWError("Interrupted run requires same CSV and batch size; "
                                   "finish it before changing the dataset")
                if state["cursor"] >= len(train):
                    raise PAWError("Checkpoint cursor exceeds current training data")
                target_epochs = state["run_goal_epochs"]
            else:
                if state["cursor"]:
                    raise PAWError("Checkpoint has an unfinished epoch but no run goal")
                state["train_signature"] = signature
                state["current_batch_size"] = config.BATCH_SIZE
                target_epochs = state["epochs_trained"] + epochs
                state["run_goal_epochs"] = target_epochs
                self.store.save(task, state)
            stats = state["stats"]
            normalized = normalized_pairs(train, stats)
            prev_updates = state["updates"]
            start_epoch = state["epochs_trained"]
            epochs_this_call = max(0, target_epochs - start_epoch)
            while state["epochs_trained"] < target_epochs:
                # Deterministic, restartable permutation for EACH absolute epoch.
                epoch_rows = normalized[:]
                random.Random(state["seed"] + state["epochs_trained"]).shuffle(epoch_rows)
                start = state["cursor"]
                while start < len(epoch_rows):
                    batch = epoch_rows[start:start + config.BATCH_SIZE]
                    # One optimizer update = load, learn, persist, clear.
                    state = self.store.load(task)
                    if state is None:
                        raise PAWError("Task state disappeared mid-training")
                    vessel = Core(state["params"])
                    opt = Adam.load(state["adam"])
                    _, grads = vessel.loss_and_gradient(batch)
                    next_params = opt.step(vessel.params, grads, config.LEARNING_RATE)
                    Core(next_params)  # reject non-finite candidates
                    state["params"] = next_params
                    state["adam"] = opt.dump()
                    state["updates"] = opt.steps
                    next_cursor = start + len(batch)
                    if next_cursor >= len(epoch_rows):
                        state["cursor"] = 0
                        state["epochs_trained"] += 1
                    else:
                        state["cursor"] = next_cursor
                    self.store.save(task, state)
                    vessel.params[:] = initial_params(state["seed"])
                    if vessel.params != initial_params(state["seed"]):
                        raise PAWError("Core did not clear correctly")
                    del vessel, opt
                    start = next_cursor
                completed = state["epochs_trained"] - start_epoch
                train_metrics = self.metrics(state, train)
                val_metrics = self.metrics(state, valid)
                self._log_epoch(state, train_metrics, val_metrics)
                if reporter and (completed == 1 or completed % config.LOG_EVERY_EPOCHS == 0 or completed == epochs_this_call):
                    reporter(completed, epochs_this_call, train_metrics, val_metrics)
            # Mark training run complete so a subsequent train call starts NEW epochs.
            state["run_goal_epochs"] = None
            self.store.save(task, state)
            final_train = self.metrics(state, train)
            final_val = self.metrics(state, valid)
            return {"task": task, "created": fresh, "epochs_completed_this_call": epochs_this_call,
                    "resumed_interrupted_run": active_run,
                    "total_epochs_trained": state["epochs_trained"],
                    "updates_before": prev_updates, "updates_after": state["updates"],
                    "train_rows": len(train), "validation_rows": len(valid),
                    "training": final_train, "validation": final_val,
                    "storage_bytes": self.store.file(task).stat().st_size}


def generate_demo(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("x", "y"))
        for i in range(256):
            x = -2 + 4 * i / 255
            writer.writerow((f"{x:.12g}", f"{2 * x - 1:.12g}"))


def pretty(metrics: dict) -> str:
    health = metrics["health"]
    dead = health["dead_hidden_neurons"]
    dead_label = f"{dead}/2" if dead is not None else "unmeasured (few inputs)"
    return (f"MSE={metrics['mse']:.8g} RMSE={metrics['rmse']:.7g} "
            f"mean-baseline-MSE={metrics['mean_baseline_mse']:.8g} "
            f"saturation={health['saturated_fraction']:.2%} "
            f"dead-hidden={dead_label}")


def usage() -> None:
    print("PAW 1.0: python paw.py [demo | train TASK FILE.csv | predict TASK X | "
          "evaluate TASK FILE.csv | tasks | inspect TASK | recover TASK | delete TASK | test]")
    print("Run with no arguments for an interactive menu. Edit config.py for all settings.")


def cli(args: list[str]) -> int:
    engine = PAW()
    if not args:
        print("PAW v1 | Four-neuron / task-isolated external learning")
        usage()
        print("Commands: demo, train, predict, evaluate, tasks, inspect, recover, delete, quit")
        while True:
            try:
                line = input("PAW> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if line in {"quit", "exit", "q"}:
                return 0
            if not line:
                continue
            # shlex handles paths containing spaces (quote a path with spaces).
            import shlex
            try:
                tokens = shlex.split(line, posix=os.name != "nt")
                if os.name == "nt":
                    tokens = [v[1:-1] if len(v) >= 2 and v[0] == v[-1] == '"' else v
                              for v in tokens]
                cli(tokens)
            except PAWError as exc:
                print("Error:", exc)
    command = args[0].lower()
    if command == "demo" and len(args) == 1:
        path = config.ROOT / "examples" / "linear.csv"
        generate_demo(path)
        print(f"Created example CSV: {path}")
        print("Training task 'demo_linear'...")
        args = ["train", "demo_linear", str(path)]
        command = "train"
    if command == "train" and len(args) == 3:
        task = task_check(args[1])
        rows = read_csv(args[2])
        def report(epoch, epochs, tr, va):
            print(f"Epoch {epoch}/{epochs} | train {pretty(tr)} | validation {pretty(va)}", flush=True)
        summary = engine.train(task, rows, reporter=report)
        print("Training finished:")
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    if command == "predict" and len(args) == 3:
        try:
            x = float(args[2])
        except ValueError as exc:
            raise PAWError("X must be numeric") from exc
        print(f"{engine.predict(args[1], x):.12g}")
        return 0
    if command == "evaluate" and len(args) == 3:
        print(json.dumps(engine.evaluate(args[1], read_csv(args[2])), indent=2))
        return 0
    if command == "tasks" and len(args) == 1:
        tasks = engine.store.tasks()
        print(json.dumps(tasks, indent=2))
        return 0
    if command == "inspect" and len(args) == 2:
        with engine.store.lock(args[1]):
            state = engine.store.load(args[1])
            if state is None:
                raise PAWError(f"Unknown task: {args[1]}")
            print(json.dumps({"task": args[1], "updates": state["updates"],
                              "params": state["params"], "stats": state["stats"],
                              "path": str(engine.store.file(args[1]))}, indent=2))
        return 0
    if command == "recover" and len(args) == 2:
        print(f"Restored backup: {engine.store.recover(args[1])} updates")
        return 0
    if command == "delete" and len(args) == 2:
        task = task_check(args[1])
        confirm = input(f"Permanently delete task '{task}' and backup? Type DELETE {task}: ")
        if confirm != f"DELETE {task}":
            print("Canceled")
            return 0
        with engine.store.lock(task):
            for path in (engine.store.file(task), engine.store.file(task).with_suffix(".bak")):
                if path.exists():
                    path.unlink()
        print("Deleted", task)
        return 0
    if command == "test" and len(args) == 1:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(config.ROOT / "tests"), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    usage()
    raise PAWError("Invalid command or number of arguments")


if __name__ == "__main__":
    try:
        sys.exit(cli(sys.argv[1:]))
    except PAWError as exc:
        print(f"PAW error: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nStopped. Last committed step remains saved.", file=sys.stderr)
        sys.exit(130)
