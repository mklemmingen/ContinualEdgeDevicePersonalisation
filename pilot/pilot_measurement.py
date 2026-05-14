#!/usr/bin/env python3
"""
Pilot measurement script for the Pi 5 paper.

What this does: poke around the Pi 5 and figure out roughly what it costs
- in time and in joules - to do inference, train a tiny CNN with various
chunks of it frozen, run a label-free drift detector, and see how much
the model forgets when you keep training it on shifted data. All while
reading power off the on-chip Renesas DA9091 PMIC. No external power
meter needed, which is the whole point.

Just run it:

    python3 pilot_measurement.py

No flags, no args. If you really want to tweak something edit the
constants near the top. I kept it that way on purpose - one of these
runs takes ~7-8 minutes and I didn't want to forget which knobs I'd
turned between runs.

A background thread polls the PMIC + thermals + CPU clock at 5 Hz the
whole time. Every sample gets tagged with whichever step we're in, all
goes into one fat CSV. Easy to load with pandas later.

The steps, roughly in order:

    Step 0  PMIC self-test. If vcgencmd doesn't answer, bail early
            rather than running for 8 min and writing junk.
    Step 1  Idle baseline (30 s).
    Step 2  Inference: time it, then 30 s sustained for power
              - FP32
              - INT8 via torch.ao.quantization (dynamic). Linear-only,
                so the win is bounded - Conv1d stays FP32.
    Step 3  Trainable-fraction sweep, 5 configs:
              last-layer-only, mu (~2%), ~10%, ~50%, full.
              For each: per-update timing + 30 s sustained.
              The curve I actually care about is energy/update vs
              fraction trained.
    Step 4  SGD vs Adam, both with the mu-training freeze pattern.
              Adam has extra state so I want to know what it costs.
    Step 5  Label-free drift detector sim - UDDA-TC-ish WLV signal
              piped into a sliding CUSUM. 30 s of inference + WLV.
    Step 6  BWT proxy. Train on task 0, eval, train on task 1 (mean
              shifted + label rotation so they actually interfere),
              re-eval task 0. Done for full and mu-training separately.
    Step 7  Idle decay (5 s) so we capture the cooldown tail.

References I'm leaning on:
    BWT:        Lopez-Paz & Ranzato, "Gradient Episodic Memory for
                Continual Learning", NeurIPS 2017 (arXiv:1706.08840).
                BWT = mean over i<T of (R[T,i] - R[i,i]).
                Negative number = forgetting.
    UDDA-TC:    Liu et al., IEEE TCE 2025 (10.1109/TCE.2025.3579882).
                WLV = weighted variance of softmax outputs over a
                sliding window; CUSUM on top.
    Pi 5 PMIC:  Renesas DA9091. 12+ rails via `vcgencmd pmic_read_adc`.
                Caveat: it does NOT measure the 5V main-rail current,
                so anything hanging off USB / a HAT / NVMe / the fan
                is invisible to us. The numbers here are SoC-ish, not
                "wall power".
    QD note:    torch.ao.quantization.quantize_dynamic with qint8 only
                touches Linear. Convs stay FP32.

Outputs (dropped right next to this script):
    pilot_data.csv     ~5 Hz wide CSV, one row per sample. Columns:
                       timestamp_unix, timestamp_iso, phase, step_label,
                       elapsed_in_phase_s, event, total_W, soc_C,
                       cpu_clock_MHz, throttled_flags, then per-rail
                       <rail>_V/_A/_W. Drops straight into pandas.
    pilot_results.json Per-phase summary - means/std/min/max, timings,
                       energy numbers, the ratios I actually want to
                       quote in the paper, BWT, drift-detector cost.

- Marty
"""

# --------------------------------------------------------------------
# Knobs. No CLI flags on purpose - touch these if you really need to.
# --------------------------------------------------------------------

SAMPLE_INTERVAL_S = 0.2         # 5 Hz poller. faster = more I/O, less signal.
IDLE_DURATION_S = 30.0
SUSTAINED_DURATION_S = 30.0
TIMING_ITERATIONS = 200
TIMING_WARMUP = 20
BATCH_SIZE = 32
SEQ_LENGTH = 250
NUM_CLASSES = 5
SEED = 42
THREADS = 4                     # 4x A76 on the Pi 5
SETTLE_S = 2.0                  # throw away the first 2 s of each phase
WLV_WINDOW = 50                 # WLV sliding window length
BWT_TASK_TRAIN_STEPS = 100      # train steps per task in the BWT bit
BWT_TASK_EVAL_BATCHES = 8       # eval batches per task
TASK1_SHIFT_MEAN = 1.0          # mean shift for "task 1" - makes it actually drift

OUT_CSV = "pilot_data.csv"
OUT_JSON = "pilot_results.json"

# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------

import csv
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn

# --------------------------------------------------------------------
# The model
# --------------------------------------------------------------------


class TinyECGNet(nn.Module):
    """Tiny 1D CNN, three blocks: encoder / middle / decoder.
    Param counts are picked so the middle block lands around 2% of
    total. That's the Huang et al. [b3] design point I'm targeting
    for the mu-training comparison."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.middle = nn.Sequential(
            nn.Linear(64 * 4, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        z = self.encoder(x).flatten(1)
        z = self.middle(z)
        return self.decoder(z)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def freeze(m):
    for p in m.parameters():
        p.requires_grad = False


def unfreeze(m):
    for p in m.parameters():
        p.requires_grad = True


# --------------------------------------------------------------------
# PMIC + thermal stuff
# --------------------------------------------------------------------


_PMIC_LINE = re.compile(
    r"^\s*(?P<rail>\S+?)_(?P<suffix>[AV])\s+"
    r"(?P<kind>current|volt)\(\d+\)=(?P<value>-?\d+\.?\d*)[AV]"
)


def parse_pmic_output(text):
    """Eats `vcgencmd pmic_read_adc` text, spits back
    {rail_name: {'volt', 'current', 'power'}}. We compute power
    ourselves (V*I) when current is reported. If the rail only
    reports current we try to guess its nominal voltage from the
    rail name - '3V3_SYS' obviously means 3.3 V, etc."""
    out = {}
    for line in text.splitlines():
        m = _PMIC_LINE.match(line)
        if not m:
            continue
        rail = m.group("rail")
        kind = m.group("kind")
        value = float(m.group("value"))
        out.setdefault(rail, {})[kind] = value

    for rail, d in out.items():
        if "current" not in d:
            continue
        v = d.get("volt")
        if v is None:
            v = _infer_nominal_voltage(rail)
        if v is not None:
            d["volt_used"] = v
            d["power"] = v * d["current"]
    return out


def _infer_nominal_voltage(rail_name):
    m = re.match(r"^(\d+)V(\d+)?", rail_name)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2) or 0)
        return major + minor / 10.0
    if "DDR" in rail_name:
        return 1.1
    return None


def sum_power(parsed):
    return sum(d["power"] for d in parsed.values() if "power" in d)


def _vcgencmd(*args, timeout=2):
    return subprocess.check_output(
        ["vcgencmd"] + list(args),
        stderr=subprocess.DEVNULL, timeout=timeout,
    ).decode("utf-8", errors="replace")


def read_soc_temp_C():
    try:
        out = _vcgencmd("measure_temp")
        m = re.search(r"temp=([\d.]+)", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def read_cpu_clock_MHz():
    try:
        out = _vcgencmd("measure_clock", "arm")
        m = re.search(r"=(\d+)", out)
        return int(m.group(1)) / 1e6 if m else None
    except Exception:
        return None


def read_throttled_flags():
    try:
        out = _vcgencmd("get_throttled")
        m = re.search(r"throttled=0x([0-9a-fA-F]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def check_pmic_access():
    """Sanity-check: can we actually read the PMIC and does the number
    look like a real Pi 5 idle? Returns (ok, parsed, total_W, summary).
    Bail-out point for the script if anything's off."""
    print("=" * 64)
    print(" Step 0: PMIC self-test")
    print("-" * 64)
    try:
        out = _vcgencmd("pmic_read_adc", timeout=5)
    except FileNotFoundError:
        # not on a Pi, or vcgencmd not in PATH. either way - we're done here.
        print("FAIL: `vcgencmd` not found. This script is Pi-5-only.")
        return False, {}, 0.0, ""
    except subprocess.CalledProcessError as e:
        print(f"FAIL: vcgencmd exited with error: {e}")
        return False, {}, 0.0, ""
    except subprocess.TimeoutExpired:
        print("FAIL: vcgencmd timed out after 5s.")
        return False, {}, 0.0, ""

    parsed = parse_pmic_output(out)
    rails = sorted(parsed.keys())
    if not rails:
        print("FAIL: pmic_read_adc returned no parseable rails.")
        print("Raw output:")
        print(out)
        return False, {}, 0.0, ""

    total = sum_power(parsed)
    ext5v = parsed.get("EXT5V", {}).get("volt")
    temp = read_soc_temp_C()
    clock = read_cpu_clock_MHz()
    throttled = read_throttled_flags()

    print(f"OK: {len(rails)} PMIC rails accessible.")
    if ext5v is not None:
        ok_v = 4.5 <= ext5v <= 5.5
        print(f"    EXT5V_V        = {ext5v:.3f} V "
              f"{'(plausible)' if ok_v else '(OUT OF RANGE!)'}")
    print(f"    summed power   = {total:.3f} W")
    if temp is not None:
        print(f"    SoC temperature = {temp:.1f} C")
    if clock is not None:
        print(f"    CPU clock       = {clock:.0f} MHz")
    if throttled is not None:
        bits = int(throttled, 16)
        notes = []
        if bits & 0x1:        notes.append("under-voltage NOW")
        if bits & 0x2:        notes.append("freq-cap NOW")
        if bits & 0x4:        notes.append("throttled NOW")
        if bits & 0x10000:    notes.append("under-voltage HAS-OCCURRED")
        if bits & 0x40000:    notes.append("throttled HAS-OCCURRED")
        flag_str = ", ".join(notes) if notes else "no throttling flags"
        print(f"    throttled bits  = 0x{throttled} ({flag_str})")
    print(f"    rails: {', '.join(rails)}")

    # Sanity range. A Pi 5 idling is usually around 2-4 W; under load
    # I see ~6-9 W. Anything outside ~0.5-30 W and I want to know.
    if total < 0.5 or total > 30.0:
        print(f"WARN: total power {total:.2f} W outside 0.5-30 W; "
              "PMIC reading may be off.")

    summary = (f"{len(rails)} rails, total {total:.2f} W, "
               f"EXT5V {ext5v:.2f} V" if ext5v else
               f"{len(rails)} rails, total {total:.2f} W")
    print()
    return True, parsed, total, summary


# --------------------------------------------------------------------
# Background poller. Runs in its own thread, writes one CSV row per
# tick. Has to be a thread because the work in main() blocks - if I
# did this in the main loop, I'd miss samples during heavy training.
# --------------------------------------------------------------------


class TelemetryPoller(threading.Thread):
    def __init__(self, csv_path, sample_interval_s):
        super().__init__(daemon=True, name="TelemetryPoller")
        self.csv_path = csv_path
        self.sample_interval_s = sample_interval_s
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._phase = "init"
        self._step_label = ""
        self._phase_start_t = time.time()
        self._pending_event = None
        self._rails = None
        self._csv_file = None
        self._csv_writer = None
        self.n_samples = 0
        self.n_errors = 0

    def set_phase(self, phase, step_label="", event_marker=None):
        with self._lock:
            self._phase = phase
            self._step_label = step_label
            self._phase_start_t = time.time()
            self._pending_event = event_marker

    def mark_event(self, event_marker):
        with self._lock:
            self._pending_event = event_marker

    def run(self):
        self._sample()
        while not self.stop_event.wait(self.sample_interval_s):
            self._sample()

    def _sample(self):
        t = time.time()
        try:
            out = _vcgencmd("pmic_read_adc", timeout=2)
        except Exception:
            self.n_errors += 1
            return
        parsed = parse_pmic_output(out)
        total_w = sum_power(parsed)
        soc_c = read_soc_temp_C()
        cpu_mhz = read_cpu_clock_MHz()
        throttled = read_throttled_flags()

        with self._lock:
            phase = self._phase
            step_label = self._step_label
            event = self._pending_event or ""
            self._pending_event = None
            phase_start = self._phase_start_t

        if self._csv_writer is None:
            self._rails = sorted(parsed.keys())
            self._csv_file = open(self.csv_path, "w", newline="")
            cols = ["timestamp_unix", "timestamp_iso", "phase",
                    "step_label", "elapsed_in_phase_s", "event",
                    "total_W", "soc_C", "cpu_clock_MHz",
                    "throttled_flags"]
            for r in self._rails:
                cols.extend([f"{r}_V", f"{r}_A", f"{r}_W"])
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(cols)

        elapsed = t - phase_start
        iso = datetime.datetime.fromtimestamp(t).isoformat(
            timespec="milliseconds")
        row = [f"{t:.3f}", iso, phase, step_label, f"{elapsed:.3f}",
               event, f"{total_w:.4f}",
               f"{soc_c:.2f}" if soc_c is not None else "",
               f"{cpu_mhz:.1f}" if cpu_mhz is not None else "",
               throttled or ""]
        for r in self._rails:
            d = parsed.get(r, {})
            row.append(f"{d['volt']:.4f}" if "volt" in d else "")
            row.append(f"{d['current']:.6f}" if "current" in d else "")
            row.append(f"{d['power']:.4f}" if "power" in d else "")
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        self.n_samples += 1

    def stop(self):
        self.stop_event.set()
        self.join(timeout=5)
        if self._csv_file is not None:
            self._csv_file.close()


# --------------------------------------------------------------------
# Phase helpers - the little run-the-thing-for-N-seconds loops
# --------------------------------------------------------------------


def banner(label, duration_s=None):
    bar = "#" * 64
    print()
    print(bar)
    print(f"# PHASE: {label}")
    if duration_s is not None:
        print(f"# DURATION: {duration_s:.0f} s")
    print(f"# WALL CLOCK START: {time.strftime('%H:%M:%S')} "
          f"(unix {time.time():.3f})")
    print(bar, flush=True)


def banner_end(label, n_ops=None, extra=None):
    bar = "#" * 64
    print(bar)
    print(f"# PHASE END:   {label}")
    print(f"# WALL CLOCK END: {time.strftime('%H:%M:%S')} "
          f"(unix {time.time():.3f})")
    if n_ops is not None:
        print(f"# OPERATIONS COMPLETED: {n_ops}")
    if extra:
        print(f"# {extra}")
    print(bar, flush=True)
    print()


def run_idle(duration_s):
    banner("IDLE BASELINE", duration_s)
    t0 = time.time()
    t_end = time.perf_counter() + duration_s
    while True:
        rem = t_end - time.perf_counter()
        if rem <= 0:
            break
        sys.stdout.write(f"\r  IDLE remaining: {rem:5.1f} s   ")
        sys.stdout.flush()
        time.sleep(min(0.25, rem))
    print()
    t1 = time.time()
    banner_end("IDLE")
    return t0, t1


def run_sustained_inference(label, model, x, duration_s):
    banner(label, duration_s)
    n = 0
    last = 0.0
    t0 = time.time()
    pc0 = time.perf_counter()
    pc_end = pc0 + duration_s
    model.eval()
    with torch.no_grad():
        while True:
            now = time.perf_counter()
            rem = pc_end - now
            if rem <= 0:
                break
            model(x)
            n += 1
            if now - last >= 0.25:
                sys.stdout.write(
                    f"\r  {label} remaining: {rem:5.1f} s "
                    f" inferences: {n}    ")
                sys.stdout.flush()
                last = now
    print()
    t1 = time.time()
    banner_end(label, n_ops=n)
    return t0, t1, n


def run_sustained_training(label, model, x, y, opt, loss_fn, duration_s):
    banner(label, duration_s)
    n = 0
    last = 0.0
    t0 = time.time()
    pc0 = time.perf_counter()
    pc_end = pc0 + duration_s
    while True:
        now = time.perf_counter()
        rem = pc_end - now
        if rem <= 0:
            break
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
        n += 1
        if now - last >= 0.25:
            sys.stdout.write(
                f"\r  {label} remaining: {rem:5.1f} s "
                f" updates: {n}    ")
            sys.stdout.flush()
            last = now
    print()
    t1 = time.time()
    banner_end(label, n_ops=n)
    return t0, t1, n


def measure_per_update(model, x, y, opt, loss_fn, warmup, n_iter):
    for _ in range(warmup):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    pc0 = time.perf_counter()
    for _ in range(n_iter):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    return (time.perf_counter() - pc0) / n_iter


def measure_inference_latency(model, x, warmup, n_iter):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        pc0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        return (time.perf_counter() - pc0) / n_iter


# --------------------------------------------------------------------
# Trainable-fraction sweep. Picks which params get requires_grad=True.
# --------------------------------------------------------------------


def configure_trainable(model, config_name):
    """Switch on requires_grad for a particular freeze pattern.
    Returns how many params ended up trainable so the caller can
    log it."""
    unfreeze(model)
    if config_name == "last_layer":
        # Classic transfer-learning style: freeze everything, train
        # only the final classifier layer.
        freeze(model.encoder)
        freeze(model.middle)
        for layer in list(model.decoder.children())[:-1]:
            freeze(layer)
    elif config_name == "mu_2pct":
        # The mu-training default from the survey - only train the
        # middle bottleneck block. ~2% of total params.
        freeze(model.encoder)
        freeze(model.decoder)
    elif config_name == "mu_10pct":
        # Middle + decoder. Loosely "head-tuning".
        freeze(model.encoder)
    elif config_name == "half":
        # Freeze the first half of the encoder, train the rest.
        # Rough proxy for "freeze the cheap front-end features".
        encoder_layers = list(model.encoder.children())
        for layer in encoder_layers[:len(encoder_layers) // 2]:
            freeze(layer)
    elif config_name == "full":
        # Nothing frozen. Baseline for "if you trained everything".
        pass
    else:
        raise ValueError(f"Unknown trainable config: {config_name}")
    return trainable_params(model)


# --------------------------------------------------------------------
# WLV drift detector (UDDA-TC-ish). I'm not trying to *evaluate* the
# detector here, I just want to know what it costs to run.
# --------------------------------------------------------------------


class WLVDetector:
    """Weighted variance of softmax outputs over a sliding window,
    pushed into a CUSUM accumulator. Approximation of the WLV signal
    from Liu et al. [b6] - good enough for measuring per-decision
    cost, which is all I need from it here."""

    def __init__(self, window_size=WLV_WINDOW, cusum_threshold=5.0):
        self.window = deque(maxlen=window_size)
        self.window_size = window_size
        self.threshold = cusum_threshold
        self.cusum = 0.0
        self.baseline = None
        self.detections = 0

    def update(self, logits):
        # Mean softmax across the batch -> one C-vector per call.
        probs = torch.softmax(logits, dim=-1).mean(dim=0)
        self.window.append(probs.detach().numpy())
        if len(self.window) < self.window_size:
            # Need a full window before WLV means anything.
            return False
        # WLV itself: per-class variance across the window, weighted
        # by the empirical class prior (just the per-class mean across
        # the same window - cheap, no labels needed). Sum it up.
        import numpy as np
        arr = np.stack(self.window)  # (W, C)
        prior = arr.mean(axis=0)  # (C,)
        wlv = float((arr.var(axis=0) * prior).sum())
        if self.baseline is None:
            # First full window sets the baseline. Could do something
            # fancier (rolling) but this is enough for the cost number.
            self.baseline = wlv
            return False
        # Standard one-sided CUSUM. Reset on detection so we can
        # actually count multiple events in a single phase.
        self.cusum = max(0.0, self.cusum + (wlv - self.baseline))
        if self.cusum > self.threshold:
            self.detections += 1
            self.cusum = 0.0
            return True
        return False


def run_wlv_drift_detection(label, model, x, duration_s):
    """Same as the sustained-inference loop, but every call also goes
    through the WLV+CUSUM detector. The diff in power/timing vs
    plain inference is what I'll quote as the detector overhead."""
    banner(label, duration_s)
    detector = WLVDetector()
    n = 0
    last = 0.0
    t0 = time.time()
    pc0 = time.perf_counter()
    pc_end = pc0 + duration_s
    model.eval()
    with torch.no_grad():
        while True:
            now = time.perf_counter()
            rem = pc_end - now
            if rem <= 0:
                break
            logits = model(x)
            detector.update(logits)
            n += 1
            if now - last >= 0.25:
                sys.stdout.write(
                    f"\r  {label} remaining: {rem:5.1f} s "
                    f" inferences+WLV: {n}  detections: "
                    f"{detector.detections}    ")
                sys.stdout.flush()
                last = now
    print()
    t1 = time.time()
    banner_end(label, n_ops=n,
               extra=f"WLV detections: {detector.detections}")
    return t0, t1, n, detector.detections


# --------------------------------------------------------------------
# BWT proxy. Synthetic two-task experiment to *see* forgetting.
# This is not a "real" CL benchmark - I'm using it to show the
# direction of the effect and roughly how mu compares to full.
# --------------------------------------------------------------------


def make_task_data(n_batches, batch_size, seq_length, num_classes,
                   shift_mean, label_rotation, seed):
    """Build deterministic (x, y) batches.

    Two things matter here:
    1. The model has to actually be able to learn it (otherwise BWT
       is just noise on noise). So labels are a deterministic
       function of the per-sample input mean.
    2. Task 0 and task 1 have to *interfere* with each other,
       otherwise there's no forgetting to measure. I do this by
       rotating the label assignment for task 1 - same inputs would
       map to different classes."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, 1, seq_length, generator=g) + shift_mean
        means = x.mean(dim=(1, 2))  # (B,)
        # Bin per-sample input mean into [0, num_classes) using a
        # fixed -2..+2 range. shift_mean is subtracted back out so
        # task 1's binning is consistent with task 0's.
        bins = ((means - shift_mean + 2.0) * num_classes / 4.0
                ).clamp(0, num_classes - 1).long()
        y = (bins + label_rotation) % num_classes
        batches.append((x, y))
    return batches


def evaluate_accuracy(model, batches):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in batches:
            preds = model(xb).argmax(dim=-1)
            correct += int((preds == yb).sum())
            total += yb.numel()
    return correct / total if total > 0 else 0.0


def train_on_task(model, batches, opt, loss_fn, n_steps):
    model.train()
    for step in range(n_steps):
        xb, yb = batches[step % len(batches)]
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()


def run_bwt_proxy(label, freeze_config, train_steps, eval_batches):
    """Train on task 0, eval, train on task 1, re-eval task 0.
    BWT = R[1,0] - R[0,0]. Negative means we forgot."""
    banner(label)
    t0 = time.time()
    # Fresh model + apply the freeze pattern we're testing.
    model = TinyECGNet()
    n_trainable = configure_trainable(model, freeze_config)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.05)
    # Pre-build all the task data so timing isn't polluted by data gen.
    task0_train = make_task_data(train_steps, BATCH_SIZE, SEQ_LENGTH,
                                  NUM_CLASSES, shift_mean=0.0,
                                  label_rotation=0, seed=100)
    task0_eval = make_task_data(eval_batches, BATCH_SIZE, SEQ_LENGTH,
                                 NUM_CLASSES, shift_mean=0.0,
                                 label_rotation=0, seed=200)
    task1_train = make_task_data(train_steps, BATCH_SIZE, SEQ_LENGTH,
                                  NUM_CLASSES,
                                  shift_mean=TASK1_SHIFT_MEAN,
                                  label_rotation=1, seed=300)
    # Phase A: learn task 0, lock in the baseline accuracy.
    train_on_task(model, task0_train, opt, loss_fn, train_steps)
    acc_00 = evaluate_accuracy(model, task0_eval)
    # Phase B: keep training on the shifted task. Re-eval on task 0
    # to see what we lost.
    train_on_task(model, task1_train, opt, loss_fn, train_steps)
    acc_10 = evaluate_accuracy(model, task0_eval)
    bwt = acc_10 - acc_00
    t1 = time.time()
    banner_end(label,
               extra=f"R[0,0]={acc_00:.3f}  R[1,0]={acc_10:.3f}  "
                     f"BWT={bwt:+.3f}  trainable_params={n_trainable}")
    return {
        "freeze_config": freeze_config,
        "trainable_params": n_trainable,
        "acc_task0_after_task0": acc_00,
        "acc_task0_after_task1": acc_10,
        "bwt_proxy": bwt,
        "t_start_unix": t0,
        "t_end_unix": t1,
    }


# --------------------------------------------------------------------
# Post-run number crunching. Read the CSV back, build per-phase stats.
# --------------------------------------------------------------------


def per_phase_stats(csv_path, settle_s):
    """Walk the CSV, group rows by (phase, step_label), throw away
    the first `settle_s` seconds of each (power doesn't snap to a new
    steady-state instantly - especially after a frequency change),
    and compute mean/std/min/max/n for power and SoC temp."""
    by_key = {}
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                elapsed = float(row["elapsed_in_phase_s"])
                total = float(row["total_W"])
            except (KeyError, ValueError):
                continue
            if elapsed < settle_s:
                continue
            key = (row["phase"], row.get("step_label", ""))
            try:
                soc_c = float(row.get("soc_C") or "nan")
                if soc_c != soc_c:
                    soc_c = None
            except ValueError:
                soc_c = None
            d = by_key.setdefault(key, {"powers": [], "temps": []})
            d["powers"].append(total)
            if soc_c is not None:
                d["temps"].append(soc_c)

    out = {}
    for (phase, step_label), d in by_key.items():
        powers = d["powers"]
        if not powers:
            continue
        n = len(powers)
        mean = sum(powers) / n
        std = (sum((v - mean) ** 2 for v in powers) / n) ** 0.5 \
              if n > 1 else 0.0
        entry = {
            "phase": phase,
            "step_label": step_label,
            "n_samples": n,
            "power_W_mean": mean,
            "power_W_std": std,
            "power_W_min": min(powers),
            "power_W_max": max(powers),
        }
        temps = d["temps"]
        if temps:
            tn = len(temps)
            tm = sum(temps) / tn
            ts = (sum((v - tm) ** 2 for v in temps) / tn) ** 0.5 \
                 if tn > 1 else 0.0
            entry["temperature_C_mean"] = tm
            entry["temperature_C_std"] = ts
            entry["temperature_C_min"] = min(temps)
            entry["temperature_C_max"] = max(temps)
        # Keep step_label separated in the key so the sweep configs
        # don't collapse into one "training" bucket. Empty step_label
        # gets bare "phase" so the simple ones (idle) stay readable.
        full_key = phase if not step_label else f"{phase}/{step_label}"
        out[full_key] = entry
    return out


# --------------------------------------------------------------------
# Main. Glues the steps together.
# --------------------------------------------------------------------


def main():
    start_wall_t = time.time()
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)

    csv_path = Path(__file__).parent / OUT_CSV
    json_path = Path(__file__).parent / OUT_JSON

    # Step 0 - bail early if the PMIC isn't talking to us.
    ok, _, _, pmic_summary = check_pmic_access()
    if not ok:
        print("Aborting. Fix PMIC access first.")
        sys.exit(1)

    # One shared synthetic batch reused across phases. We don't care
    # about training to convergence here - just want a realistic
    # forward/backward shape on the CPU.
    x = torch.randn(BATCH_SIZE, 1, SEQ_LENGTH)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    loss_fn = nn.CrossEntropyLoss()

    ref = TinyECGNet()
    n_total = count_params(ref)
    n_encoder = count_params(ref.encoder)
    n_middle = count_params(ref.middle)
    n_decoder = count_params(ref.decoder)

    print("=" * 64)
    print(" Pilot configuration (no CLI args; constants in script)")
    print("-" * 64)
    print(f"  Platform:     {platform.machine()} on {platform.node()}")
    print(f"  Total params: {n_total:,}")
    print(f"    Encoder:    {n_encoder:6,}  "
          f"({n_encoder/n_total*100:5.2f}%)")
    print(f"    Middle:     {n_middle:6,}  "
          f"({n_middle/n_total*100:5.2f}%)  <-- mu-training default")
    print(f"    Decoder:    {n_decoder:6,}  "
          f"({n_decoder/n_total*100:5.2f}%)")
    print(f"  Batch x seq:  {BATCH_SIZE} x {SEQ_LENGTH}")
    print(f"  Threads:      {THREADS}")
    print(f"  Sample rate:  {1.0/SAMPLE_INTERVAL_S:.1f} Hz "
          f"(every {SAMPLE_INTERVAL_S*1000:.0f} ms)")
    print(f"  Wide CSV:     {csv_path}")
    print(f"  Summary JSON: {json_path}")
    print("=" * 64)

    # Kick off the poller. Sleep ~1.5 s so the CSV header is written
    # and a couple of "setup" samples land before we move on - makes
    # the first phase boundary cleaner in pandas later.
    poller = TelemetryPoller(csv_path, SAMPLE_INTERVAL_S)
    poller.set_phase("setup", event_marker="run_start")
    poller.start()
    time.sleep(1.5)

    # Idle baseline. Everything downstream gets compared to this.
    poller.set_phase("idle", event_marker="idle_start")
    idle_t0, idle_t1 = run_idle(IDLE_DURATION_S)
    poller.mark_event("idle_end")

    # FP32 inference: micro-timing, then a 30 s sustained loop for power.
    poller.set_phase("timing", "fp32", event_marker="fp32_timing_start")
    inf_fp32 = TinyECGNet()
    inf_fp32_per_call_s = measure_inference_latency(
        inf_fp32, x, TIMING_WARMUP, TIMING_ITERATIONS)
    poller.set_phase("inference", "fp32",
                     event_marker="fp32_sustained_start")
    fp32_t0, fp32_t1, n_fp32 = run_sustained_inference(
        "INFERENCE FP32 SUSTAINED", inf_fp32, x, SUSTAINED_DURATION_S)
    poller.mark_event("fp32_sustained_end")

    # INT8 dynamic quant. Wrapped in try/except because if the torch
    # build doesn't support qnnpack/quantize_dynamic I want the rest
    # of the run to keep going rather than die here.
    int8_supported = True
    inf_int8_per_call_s = None
    int8_t0 = int8_t1 = None
    n_int8 = None
    try:
        from torch.ao.quantization import quantize_dynamic
        inf_int8 = quantize_dynamic(
            TinyECGNet().eval(), {nn.Linear}, dtype=torch.qint8)
        poller.set_phase("timing", "int8",
                         event_marker="int8_timing_start")
        inf_int8_per_call_s = measure_inference_latency(
            inf_int8, x, TIMING_WARMUP, TIMING_ITERATIONS)
        poller.set_phase("inference", "int8",
                         event_marker="int8_sustained_start")
        int8_t0, int8_t1, n_int8 = run_sustained_inference(
            "INFERENCE INT8-DYNAMIC SUSTAINED", inf_int8, x,
            SUSTAINED_DURATION_S)
        poller.mark_event("int8_sustained_end")
    except Exception as e:
        int8_supported = False
        print(f"\n[INT8 quantisation skipped: {type(e).__name__}: {e}]")

    # The main event: trainable-fraction sweep.
    sweep_configs = ["last_layer", "mu_2pct", "mu_10pct", "half", "full"]
    sweep_results = []
    for cfg in sweep_configs:
        m = TinyECGNet()
        n_trainable = configure_trainable(m, cfg)
        opt = torch.optim.SGD(
            [p for p in m.parameters() if p.requires_grad], lr=0.01)
        # Defensive skip - shouldn't happen with the current model
        # because there's always a Linear in the decoder, but if
        # someone swaps the architecture and forgets, better to skip
        # than to feed a 0-param optimizer.
        if n_trainable == 0:
            print(f"[skip {cfg}: 0 trainable params]")
            continue
        poller.set_phase("timing", f"train_{cfg}",
                         event_marker=f"{cfg}_timing_start")
        per_update_s = measure_per_update(
            m, x, y, opt, loss_fn, TIMING_WARMUP, TIMING_ITERATIONS)
        poller.set_phase("training", f"train_{cfg}",
                         event_marker=f"{cfg}_sustained_start")
        st_t0, st_t1, n_st = run_sustained_training(
            f"TRAIN-{cfg.upper()} SUSTAINED",
            m, x, y, opt, loss_fn, SUSTAINED_DURATION_S)
        poller.mark_event(f"{cfg}_sustained_end")
        sweep_results.append({
            "config": cfg,
            "trainable_params": n_trainable,
            "trainable_fraction": n_trainable / n_total,
            "per_update_ms": per_update_s * 1000,
            "n_sustained_updates": n_st,
            "t_start_unix": st_t0,
            "t_end_unix": st_t1,
        })

    # SGD vs Adam, both with the mu-training freeze. Adam keeps moment
    # estimates so it's heavier per-step - I want a number on that.
    optimizer_results = []
    for opt_name, opt_class, opt_kwargs in [
        ("sgd", torch.optim.SGD, dict(lr=0.01)),
        ("adam", torch.optim.Adam, dict(lr=0.001)),
    ]:
        m = TinyECGNet()
        configure_trainable(m, "mu_2pct")
        opt = opt_class(
            [p for p in m.parameters() if p.requires_grad],
            **opt_kwargs)
        poller.set_phase("training", f"mu_opt_{opt_name}",
                         event_marker=f"opt_{opt_name}_start")
        per_update_s = measure_per_update(
            m, x, y, opt, loss_fn, TIMING_WARMUP, TIMING_ITERATIONS)
        ot_t0, ot_t1, n_ot = run_sustained_training(
            f"MU-TRAINING ({opt_name.upper()}) SUSTAINED",
            m, x, y, opt, loss_fn, SUSTAINED_DURATION_S)
        poller.mark_event(f"opt_{opt_name}_end")
        optimizer_results.append({
            "optimizer": opt_name,
            "per_update_ms": per_update_s * 1000,
            "n_sustained_updates": n_ot,
            "t_start_unix": ot_t0,
            "t_end_unix": ot_t1,
        })

    # WLV drift detector. Same forward pass as plain inference, plus
    # the detector update. The interesting number is the delta.
    drift_model = TinyECGNet().eval()
    poller.set_phase("drift_detection", "wlv_cusum",
                     event_marker="wlv_start")
    wlv_t0, wlv_t1, n_wlv, n_detections = run_wlv_drift_detection(
        "WLV+CUSUM DRIFT DETECTION SUSTAINED",
        drift_model, x, SUSTAINED_DURATION_S)
    poller.mark_event("wlv_end")

    # BWT proxy. Run it twice - full network then mu-training -
    # so we have something concrete to compare.
    poller.set_phase("bwt", "full",
                     event_marker="bwt_full_start")
    bwt_full = run_bwt_proxy(
        "BWT PROXY (full-network)", "full",
        BWT_TASK_TRAIN_STEPS, BWT_TASK_EVAL_BATCHES)
    poller.set_phase("bwt", "mu_2pct",
                     event_marker="bwt_mu_start")
    bwt_mu = run_bwt_proxy(
        "BWT PROXY (mu-training, ~2% trainable)", "mu_2pct",
        BWT_TASK_TRAIN_STEPS, BWT_TASK_EVAL_BATCHES)
    poller.mark_event("bwt_end")

    # Tail end: idle for 5 s so the cooldown shows up in the CSV.
    poller.set_phase("teardown", event_marker="run_end")
    time.sleep(5.0)
    poller.stop()

    total_runtime_s = time.time() - start_wall_t
    print(f"\nTelemetry poller stopped. {poller.n_samples} samples "
          f"written, {poller.n_errors} read errors.")
    print(f"Total wall-clock runtime: {total_runtime_s:.1f} s "
          f"({total_runtime_s/60:.2f} min).")

    # ----------------------------------------------------------------
    # Crunch the CSV we just wrote.
    # ----------------------------------------------------------------
    phase_stats = per_phase_stats(csv_path, SETTLE_S)

    def get_w(key):
        return phase_stats.get(key, {}).get("power_W_mean")

    idle_w = get_w("idle")
    fp32_inf_w = get_w("inference/fp32")
    int8_inf_w = get_w("inference/int8") if int8_supported else None

    # E (mJ) = t (s) * P (W) * 1000. Trivial, but the None-handling
    # keeps the JSON clean when something upstream failed.
    def energy_mJ(seconds, watts):
        if seconds is None or watts is None:
            return None
        return seconds * watts * 1000

    # Marginal-over-idle power. clamped to 0 because measurement
    # noise sometimes makes a quiet phase look slightly below idle.
    def margin(w):
        if w is None or idle_w is None:
            return None
        return max(0.0, w - idle_w)

    fp32_inf_energy_mJ = energy_mJ(inf_fp32_per_call_s, fp32_inf_w)
    fp32_inf_marginal_mJ = energy_mJ(inf_fp32_per_call_s,
                                      margin(fp32_inf_w))
    int8_inf_energy_mJ = energy_mJ(inf_int8_per_call_s, int8_inf_w)
    int8_inf_marginal_mJ = energy_mJ(inf_int8_per_call_s,
                                      margin(int8_inf_w))

    # Bolt the measured sustained power onto each sweep entry and
    # turn it into an energy-per-update number. Two flavours: gross
    # (raw sustained * time) and marginal (subtracting idle).
    for r in sweep_results:
        key = f"training/train_{r['config']}"
        w = get_w(key)
        r["sustained_power_W"] = w
        r["energy_mJ_per_update_gross"] = energy_mJ(
            r["per_update_ms"] / 1000.0, w)
        r["energy_mJ_per_update_marginal"] = energy_mJ(
            r["per_update_ms"] / 1000.0, margin(w))

    for r in optimizer_results:
        key = f"training/mu_opt_{r['optimizer']}"
        w = get_w(key)
        r["sustained_power_W"] = w
        r["energy_mJ_per_update_gross"] = energy_mJ(
            r["per_update_ms"] / 1000.0, w)
        r["energy_mJ_per_update_marginal"] = energy_mJ(
            r["per_update_ms"] / 1000.0, margin(w))

    drift_w = get_w("drift_detection/wlv_cusum")
    drift_per_decision_s = (wlv_t1 - wlv_t0) / max(1, n_wlv)

    # The mu_2pct vs full comparison is the thing I actually want
    # in the paper's headline table. Pluck them out by name.
    mu_entry = next(
        (r for r in sweep_results if r["config"] == "mu_2pct"), None)
    full_entry = next(
        (r for r in sweep_results if r["config"] == "full"), None)
    headline_ratios = {}
    if mu_entry and full_entry:
        if mu_entry["per_update_ms"] > 0:
            headline_ratios["full_over_mu_walltime"] = (
                full_entry["per_update_ms"] / mu_entry["per_update_ms"])
        if (mu_entry.get("energy_mJ_per_update_marginal") or 0) > 0:
            headline_ratios["full_over_mu_energy_marginal"] = (
                full_entry["energy_mJ_per_update_marginal"]
                / mu_entry["energy_mJ_per_update_marginal"])

    try:
        load_avg = os.getloadavg()
    except (AttributeError, OSError):
        load_avg = None

    results = {
        "platform": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "load_avg_5min": load_avg[1] if load_avg else None,
            "pmic_self_test_summary": pmic_summary,
        },
        "config": {
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "idle_duration_s": IDLE_DURATION_S,
            "sustained_duration_s": SUSTAINED_DURATION_S,
            "timing_iterations": TIMING_ITERATIONS,
            "timing_warmup": TIMING_WARMUP,
            "batch_size": BATCH_SIZE,
            "seq_length": SEQ_LENGTH,
            "num_classes": NUM_CLASSES,
            "seed": SEED,
            "threads": THREADS,
            "settle_s": SETTLE_S,
            "wlv_window": WLV_WINDOW,
            "bwt_task_train_steps": BWT_TASK_TRAIN_STEPS,
            "bwt_task_eval_batches": BWT_TASK_EVAL_BATCHES,
            "task1_shift_mean": TASK1_SHIFT_MEAN,
        },
        "model": {
            "n_total_params": n_total,
            "n_encoder_params": n_encoder,
            "n_middle_params": n_middle,
            "n_decoder_params": n_decoder,
            "middle_fraction_of_total": n_middle / n_total,
        },
        "phase_power_stats": phase_stats,
        "phase_timestamps_unix": {
            "idle": [idle_t0, idle_t1],
            "inference_fp32": [fp32_t0, fp32_t1],
            "inference_int8": ([int8_t0, int8_t1]
                                if int8_supported else None),
            "drift_detection_wlv": [wlv_t0, wlv_t1],
            "bwt_full": [bwt_full["t_start_unix"],
                          bwt_full["t_end_unix"]],
            "bwt_mu_2pct": [bwt_mu["t_start_unix"],
                             bwt_mu["t_end_unix"]],
        },
        "inference_fp32": {
            "per_call_ms": inf_fp32_per_call_s * 1000,
            "sustained_power_W": fp32_inf_w,
            "energy_mJ_per_call_gross": fp32_inf_energy_mJ,
            "energy_mJ_per_call_marginal": fp32_inf_marginal_mJ,
            "n_sustained_inferences": n_fp32,
        },
        "inference_int8_dynamic": {
            "supported": int8_supported,
            "per_call_ms": (inf_int8_per_call_s * 1000
                             if inf_int8_per_call_s else None),
            "sustained_power_W": int8_inf_w,
            "energy_mJ_per_call_gross": int8_inf_energy_mJ,
            "energy_mJ_per_call_marginal": int8_inf_marginal_mJ,
            "n_sustained_inferences": n_int8,
            "note": ("quantize_dynamic with qint8 only touches Linear "
                     "layers - this model is mostly Conv1d, so don't "
                     "expect the textbook 4x speedup. The Linear-layer "
                     "share is what bounds the win here."),
        },
        "trainable_fraction_sweep": sweep_results,
        "optimizer_comparison_mu_2pct": optimizer_results,
        "drift_detector_wlv": {
            "n_inferences_with_wlv": n_wlv,
            "n_detections": n_detections,
            "per_decision_ms": drift_per_decision_s * 1000,
            "sustained_power_W": drift_w,
            "energy_mJ_per_decision": energy_mJ(
                drift_per_decision_s, drift_w),
            "energy_mJ_per_decision_marginal": energy_mJ(
                drift_per_decision_s, margin(drift_w)),
            "wlv_window_size": WLV_WINDOW,
            "note": ("My approximation of the UDDA-TC WLV signal: "
                     "per-class variance of softmax outputs across a "
                     "sliding window, weighted by the empirical class "
                     "prior, then summed and fed to CUSUM. Good enough "
                     "to put a number on the cost; not a real detector "
                     "eval."),
        },
        "bwt_proxy": {
            "task1_shift_mean": TASK1_SHIFT_MEAN,
            "full_network": bwt_full,
            "mu_2pct": bwt_mu,
            "note": ("Two synthetic tasks. Task 0 is N(0,1) with one "
                     "label mapping, task 1 is N(shift,1) with a rotated "
                     "label mapping (so they interfere). Train on 0, "
                     "eval, train on 1, re-eval 0. BWT = R[1,0] - R[0,0]; "
                     "negative means we forgot. Batch counts are small "
                     "so individual accuracies are noisy - the point is "
                     "the *relative* comparison between full and mu. "
                     "Not a stand-in for a real biosignal CL benchmark - "
                     "see the open-targets section of the paper."),
        },
        "headline_ratios_mu_2pct_vs_full": headline_ratios,
        "power": {
            "method": "pi5_pmic_vcgencmd",
            "idle_W": idle_w,
            "csv_path": str(csv_path),
            "rails_summed": ("Sum of every PMIC rail that reports "
                             "current. Does NOT include the 5V main rail, "
                             "so anything on USB / NVMe / a HAT / the fan "
                             "is invisible here."),
        },
        "poller": {
            "samples_written": poller.n_samples,
            "read_errors": poller.n_errors,
            "sample_rate_Hz": 1.0 / SAMPLE_INTERVAL_S,
        },
        "runtime": {
            "total_wall_clock_s": total_runtime_s,
        },
    }
    json_path.write_text(json.dumps(results, indent=2))

    # ----------------------------------------------------------------
    # Console summary - the bit I actually read after the run.
    # ----------------------------------------------------------------
    print()
    print("=" * 64)
    print(" SUMMARY")
    print("-" * 64)
    print(f"  Idle baseline:          {idle_w:.3f} W")
    if fp32_inf_w is not None:
        print(f"  Inference FP32 sustained: "
              f"{fp32_inf_w:.3f} W   "
              f"({inf_fp32_per_call_s*1000:.3f} ms/call, "
              f"{fp32_inf_marginal_mJ:.3f} mJ/call marginal)")
    if int8_supported and int8_inf_w is not None:
        print(f"  Inference INT8 sustained: "
              f"{int8_inf_w:.3f} W   "
              f"({inf_int8_per_call_s*1000:.3f} ms/call, "
              f"{int8_inf_marginal_mJ:.3f} mJ/call marginal)")
    print()
    print(" Trainable-fraction sweep:")
    print(f"   {'config':12s} {'#trainable':>10s} {'frac%':>7s} "
          f"{'ms/upd':>9s} {'W':>7s} {'mJ/upd marg':>13s}")
    for r in sweep_results:
        print(f"   {r['config']:12s} {r['trainable_params']:10d} "
              f"{r['trainable_fraction']*100:7.2f} "
              f"{r['per_update_ms']:9.3f} "
              f"{(r['sustained_power_W'] or 0):7.3f} "
              f"{(r['energy_mJ_per_update_marginal'] or 0):13.3f}")
    print()
    print(" Optimizer comparison (mu-training, ~2% trainable):")
    for r in optimizer_results:
        print(f"   {r['optimizer']:6s}  "
              f"{r['per_update_ms']:7.3f} ms/upd   "
              f"{(r['sustained_power_W'] or 0):.3f} W   "
              f"{(r['energy_mJ_per_update_marginal'] or 0):.3f} "
              f"mJ/upd marginal")
    print()
    print(f" WLV+CUSUM drift detector ({n_wlv} inferences, "
          f"{n_detections} detections):")
    print(f"   per-decision: {drift_per_decision_s*1000:.3f} ms   "
          f"{(drift_w or 0):.3f} W   "
          f"{(results['drift_detector_wlv']['energy_mJ_per_decision_marginal'] or 0):.3f} "
          f"mJ marginal")
    print()
    print(" BWT proxy (synthetic two-task forgetting):")
    print(f"   full-network: R[0,0]={bwt_full['acc_task0_after_task0']:.3f}"
          f"  R[1,0]={bwt_full['acc_task0_after_task1']:.3f}  "
          f"BWT={bwt_full['bwt_proxy']:+.3f}")
    print(f"   mu-training : R[0,0]={bwt_mu['acc_task0_after_task0']:.3f}"
          f"  R[1,0]={bwt_mu['acc_task0_after_task1']:.3f}  "
          f"BWT={bwt_mu['bwt_proxy']:+.3f}")
    print()
    if headline_ratios:
        print(" Headline ratios (mu_2pct vs full):")
        for k, v in headline_ratios.items():
            print(f"   {k:32s}  {v:.2f}x")
        print()
    print(f" Wide CSV (1 row per ~{SAMPLE_INTERVAL_S*1000:.0f} ms): "
          f"{csv_path}")
    print(f" Summary JSON (paper-ready):     {json_path}")
    print(f" Total runtime: {total_runtime_s:.1f} s "
          f"({total_runtime_s/60:.2f} min)")
    print("=" * 64)


if __name__ == "__main__":
    main()
