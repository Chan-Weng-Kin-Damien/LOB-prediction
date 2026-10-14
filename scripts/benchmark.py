#!/usr/bin/env python
"""Per-tick inference latency of a submission, measured rather than asserted.

The scorer drives ``PredictionModel.predict`` one tick at a time on a single CPU
thread, so that is exactly what this measures. It loads the submission's *own*
``solution.py`` rather than reconstructing the ensemble here, so the number
covers whatever that file actually does -- shared history buffers, per-target
weights and all -- and cannot drift away from what ships.

    python scripts/benchmark.py
    python scripts/benchmark.py --submission submission_safe_3model
    python scripts/benchmark.py --iterations 5000

Individual ``.onnx`` files in the submission directory are timed separately, with
their context length read from the graph's input shape. Latency is
hardware-specific; the header line records the machine.

Run this on an otherwise idle machine. Single-threaded ONNX still competes for
cache and memory bandwidth, so background load inflates the result -- the old
three-model ensemble measured 1085 us idle and 2135 us while an evaluation job
ran on the same host. The script warns when it detects a busy machine. A useful
cross-check is to compare a single component against its recorded figure before
trusting an ensemble number measured under load.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lobkit.data import RAW_DIM, SEQUENCE_LENGTH  # noqa: E402
from lobkit.runtime import WindowOnnxModel  # noqa: E402


def describe_host() -> str:
    parts = [platform.platform(), platform.machine()]
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
        )
        if cpu.returncode == 0 and cpu.stdout.strip():
            parts.append(cpu.stdout.strip())
    except Exception:
        pass
    try:
        import onnxruntime as ort

        parts.append(f"onnxruntime {ort.__version__}")
    except ImportError:
        pass
    return " | ".join(parts)


def context_len_of(model_path: Path) -> int:
    """Read the fixed context length straight off the ONNX input shape."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    return int(shape[1])


def load_prediction_model(submission: Path):
    """Instantiate the submission's own ``PredictionModel``."""
    spec = importlib.util.spec_from_file_location(
        f"_bench_{submission.name}", submission / "solution.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PredictionModel()


def summarise(samples_us: np.ndarray) -> dict:
    return {
        "mean": float(samples_us.mean()),
        "p50": float(np.percentile(samples_us, 50)),
        "p95": float(np.percentile(samples_us, 95)),
        "p99": float(np.percentile(samples_us, 99)),
        "min": float(samples_us.min()),
        "max": float(samples_us.max()),
    }


def render(label: str, stats: dict) -> str:
    return (
        f"{label:<38} {stats['mean']:>9.1f} {stats['p50']:>9.1f} "
        f"{stats['p95']:>9.1f} {stats['p99']:>9.1f} {stats['min']:>9.1f}"
    )


def time_loop(step, states: np.ndarray, warmup: int, iterations: int) -> np.ndarray:
    """Fill history, warm the graphs, then time steady state."""
    for i in range(SEQUENCE_LENGTH):
        step(i, states[i], False)
    for i in range(warmup):
        step(SEQUENCE_LENGTH + i, states[i], True)
    samples = np.empty(iterations)
    base = SEQUENCE_LENGTH + warmup
    for i in range(iterations):
        state = states[i]
        t0 = time.perf_counter()
        step(base + i, state, True)
        samples[i] = (time.perf_counter() - t0) * 1e6
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--submission", default=str(ROOT / "submission_v2"))
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "benchmark.json"))
    args = parser.parse_args()

    submission = Path(args.submission)
    if not (submission / "solution.py").exists():
        print(f"no solution.py in {submission}", file=sys.stderr)
        return 1

    print(describe_host())
    print(f"submission: {submission.name}")
    print("single intra-op / inter-op thread, CPUExecutionProvider, all graph optimisations")

    load1 = os.getloadavg()[0]
    ncpu = os.cpu_count() or 1
    busy = load1 > 0.3 * ncpu
    if busy:
        print(
            f"\n!! WARNING: 1-minute load average is {load1:.1f} on {ncpu} cores. Latency may be\n"
            f"!! inflated by contention. Cross-check one component against its recorded\n"
            f"!! figure before trusting these numbers, or re-run when the machine is quiet."
        )
    print()

    rng = np.random.default_rng(0)
    n = args.iterations + args.warmup + SEQUENCE_LENGTH
    states = rng.standard_normal((n, RAW_DIM)).astype(np.float32)

    print("=== per predicted tick, microseconds ===")
    print(f"{'component':<38} {'mean':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'min':>9}")

    results = {}

    # --- each ONNX graph on its own ------------------------------------------
    for model_path in sorted(submission.glob("*.onnx")):
        ctx = context_len_of(model_path)
        model = WindowOnnxModel(model_path, ctx)
        samples = time_loop(
            lambda seq, state, need: model.predict(1, state, need),
            states,
            args.warmup,
            args.iterations,
        )
        key = model_path.stem
        results[key] = summarise(samples)
        print(render(f"{key} (ctx {ctx})", results[key]))

    # --- the submission exactly as it ships ----------------------------------
    from utils import DataPoint  # noqa: E402

    model = load_prediction_model(submission)
    samples = time_loop(
        lambda seq, state, need: model.predict(DataPoint(1, seq, need, state)),
        states,
        args.warmup,
        args.iterations,
    )
    results["submission"] = summarise(samples)
    print(render(f"SUBMISSION {submission.name} (as shipped)", results["submission"]))

    # --- the cheap paths -----------------------------------------------------
    samples = np.empty(args.iterations)
    for i in range(args.iterations):
        state = states[i]
        t0 = time.perf_counter()
        model.predict(DataPoint(1, 90000 + i, False, state))
        samples[i] = (time.perf_counter() - t0) * 1e6
    results["non_prediction_tick"] = summarise(samples)
    print(render("non-prediction tick (append only)", results["non_prediction_tick"]))

    history = [states[i] for i in range(128)]
    samples = np.empty(args.iterations)
    for i in range(args.iterations):
        t0 = time.perf_counter()
        np.asarray(history[-128:], dtype=np.float32)
        samples[i] = (time.perf_counter() - t0) * 1e6
    results["window_materialise"] = summarise(samples)
    print(render("  of which: window materialise", results["window_materialise"]))

    sub = results["submission"]["mean"]
    win = results["window_materialise"]["mean"]
    print(f"\nthroughput: {1e6 / sub:,.0f} predicted ticks/sec, single-threaded")
    print(
        f"window construction is {win / sub * 100:.2f}% of a step -- the ONNX graphs dominate,\n"
        f"so replacing the Python history buffer with a ring buffer would recover at most "
        f"{win:.0f} us."
    )
    print(
        f"a full 1000-step sequence costs "
        f"{(901 * sub + 99 * results['non_prediction_tick']['mean']) / 1000:.0f} ms end to end."
    )

    results["_host"] = describe_host()
    results["_submission"] = submission.name
    results["_iterations"] = args.iterations
    results["_loadavg_1min"] = load1
    results["_contended"] = bool(busy)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
