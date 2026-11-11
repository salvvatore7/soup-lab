"""Run command variants in an interleaved order and compare them.

Interleaving (A B B A, then A B B A again) keeps thermal throttling, a busy
background process or a warming disk cache from landing on one variant only.
Every run is a fresh process. Wall time is always measured, and with --gpu the
peak GPU memory is sampled from nvidia-smi while the run is alive.

Example
    python ablate.py --rounds 2 --gpu \
        --variant buffers2="soup train --config sft.yaml --set training.stream_buffers=2" \
        --variant buffers3="soup train --config sft.yaml --set training.stream_buffers=3" \
        --out ablation.json

Stdlib only, no torch needed.
"""
import argparse
import json
import shlex
import statistics
import subprocess
import threading
import time


def gpu_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return max(int(x) for x in out.split() if x.strip().isdigit())
    except Exception:
        return None


class GpuPeak(threading.Thread):
    def __init__(self, every=0.2):
        super().__init__(daemon=True)
        self.every, self.peak, self._halt = every, None, threading.Event()

    def run(self):
        while not self._halt.is_set():
            used = gpu_used_mib()
            if used is not None and (self.peak is None or used > self.peak):
                self.peak = used
            self._halt.wait(self.every)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)


def run_once(cmd, sample_gpu):
    sampler = GpuPeak() if sample_gpu else None
    if sampler:
        sampler.start()
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    if sampler:
        sampler.stop()
    return {"seconds": round(seconds, 3), "returncode": proc.returncode,
            "gpu_peak_mib": sampler.peak if sampler else None,
            "stderr_tail": proc.stderr[-300:] if proc.returncode else ""}


def schedule(names, rounds):
    """Forward order on even rounds, reversed on odd ones, so two rounds of two variants give A B B A."""
    order = []
    for i in range(rounds):
        order.extend(names if i % 2 == 0 else list(reversed(names)))
    return order


def summarise(runs):
    ok = [r["seconds"] for r in runs if r["returncode"] == 0]
    peaks = [r["gpu_peak_mib"] for r in runs if r["gpu_peak_mib"] is not None]
    out = {"runs": len(runs), "failed": len(runs) - len(ok)}
    if ok:
        med = statistics.median(ok)
        out.update(median_s=round(med, 3), min_s=min(ok), max_s=max(ok),
                   spread_pct=round(100 * (max(ok) - min(ok)) / med, 1) if med else 0.0)
    if peaks:
        out["gpu_peak_mib"] = max(peaks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True, metavar="NAME=COMMAND")
    ap.add_argument("--rounds", type=int, default=2, help="2 rounds of 2 variants is A B B A")
    ap.add_argument("--gpu", action="store_true", help="sample peak GPU memory via nvidia-smi")
    ap.add_argument("--warmup", action="store_true", help="one untimed run of each variant first")
    ap.add_argument("--out", help="write the full result as JSON")
    args = ap.parse_args()

    variants = {}
    for item in args.variant:
        name, _, command = item.partition("=")
        if not command:
            ap.error(f"--variant needs NAME=COMMAND, got {item!r}")
        variants[name] = shlex.split(command)

    if args.warmup:
        for name, cmd in variants.items():
            run_once(cmd, False)

    results = {name: [] for name in variants}
    for name in schedule(list(variants), args.rounds):
        results[name].append(run_once(variants[name], args.gpu))

    summary = {name: summarise(runs) for name, runs in results.items()}
    for name, s in summary.items():
        print(name, json.dumps(s))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"order": schedule(list(variants), args.rounds), "summary": summary, "runs": results}, fh, indent=2)


if __name__ == "__main__":
    main()
