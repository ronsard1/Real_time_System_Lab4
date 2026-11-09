"""
pipeline.py - multi-rate environmental monitoring pipeline with timing-compliance reporting (Mini Project 4.2).

One gateway loop services several sensor streams with different required periods. Every period each stream
releases a sampling job (release r, deadline r + T). When the processor is free, the scheduling rule picks ONE
due job; the clock then advances by that read's simulated processing time (log-normal with a 3 % chance of a
long tail). Classification per job:
    on-time  serviced within 10 % of its period after release
    late     serviced later than that but before the next release
    missed   not serviced before the next release (the sample is dropped)
Rules: FP  fixed priority, shortest period first (rate monotonic, like CAN: lowest ID = highest priority)
       EDF earliest deadline first among due jobs
       RR  round robin over the streams that have a due job
Each stream has a threshold alert; the detection time is logged next to the true crossing time.

    python pipeline.py        -> results_4_2.txt, figures/
"""
import heapq
import math
import os
import random
import statistics
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
OUT = []
RUN = 60.0
ON_TIME_FRAC = 0.10
TAIL_P, TAIL_X = 0.03, (4.0, 10.0)


@dataclass
class Stream:
    name: str
    period: float           # required sampling period (s)
    proc_ms: float          # median processing time per read (ms)
    threshold: float
    signal: callable
    unit: str
    next_release: float = 0.0
    counts: dict = field(default_factory=lambda: {"required": 0, "on-time": 0, "late": 0, "missed": 0})
    lateness: list = field(default_factory=list)
    samples: list = field(default_factory=list)       # (t, value, class)
    missed_t: list = field(default_factory=list)
    alerts: list = field(default_factory=list)        # (true crossing, detection)
    above: bool = False


def make_streams(fast=False):
    s = [
        Stream("air quality", 0.2, 45, 80.0, lambda t: 55 + 30 * math.sin(2 * math.pi * t / 23 + 1.0), "AQI"),
        Stream("humidity", 0.5, 30, 70.0, lambda t: 60 + 15 * math.sin(2 * math.pi * t / 31 + 0.3), "%"),
        Stream("temperature", 1.0, 30, 30.0, lambda t: 26 + 6 * math.sin(2 * math.pi * t / 37 - 0.5), "degC"),
    ]
    if fast:
        s.insert(0, Stream("vibration", 0.05, 12, 6.0, lambda t: 4 + 2.5 * math.sin(2 * math.pi * t / 7), "mm/s"))
    return s


def true_crossings(st):
    out, prev = [], st.signal(0) > st.threshold
    for i in range(1, int(RUN / 0.001)):
        t = i * 0.001
        cur = st.signal(t) > st.threshold
        if cur and not prev:
            out.append(t)
        prev = cur
    return out


def run(rule, fast=False, seed=1):
    rng = random.Random(seed)
    streams = make_streams(fast)
    due = []                                   # heap of (release time, stream index) for the next releases
    for i, st in enumerate(streams):
        heapq.heappush(due, (0.0, i))
    pending = {}                               # stream index -> release time of its current due job
    t, rr = 0.0, 0
    while t < RUN:
        while due and due[0][0] <= t:          # move released jobs into the pending set
            r, i = heapq.heappop(due)
            st = streams[i]
            if i in pending:                   # previous job never serviced before this release: missed
                st.counts["missed"] += 1
                st.missed_t.append(pending[i])
            st.counts["required"] += 1
            pending[i] = r
            heapq.heappush(due, (r + st.period, i))
        if not pending:
            t = due[0][0]                      # idle until the next release
            continue
        if rule == "FP":
            i = min(pending, key=lambda k: streams[k].period)
        elif rule == "EDF":
            i = min(pending, key=lambda k: pending[k] + streams[k].period)
        else:                                  # RR: next stream in cyclic order that has a due job
            order = [(rr + k) % len(streams) for k in range(len(streams))]
            i = next(k for k in order if k in pending)
            rr = (i + 1) % len(streams)
        st, r = streams[i], pending.pop(i)
        lat = t - r
        cls = "on-time" if lat <= ON_TIME_FRAC * st.period else "late"
        st.counts[cls] += 1
        st.lateness.append(lat)
        v = st.signal(t)                       # noise-free reading: alert latency then reflects timing only
        st.samples.append((t, v, cls))
        if v > st.threshold and not st.above:
            st.above = True
            st.alerts.append(t)
        elif v < st.threshold - 1.0:
            st.above = False
        proc = rng.lognormvariate(math.log(st.proc_ms), 0.25) / 1000
        if rng.random() < TAIL_P:
            proc *= rng.uniform(*TAIL_X)       # occasional long tail (overrun)
        t += proc
    return streams


def alert_latency(st):
    lat = []
    for tc in true_crossings(st):
        d = next((a for a in st.alerts if a >= tc - 0.5), None)
        if d is not None and d - tc < 5:
            lat.append(d - tc)
    return lat


def say(s=""):
    print(s)
    OUT.append(s)


def plot(streams, tag, title):
    fig, axes = plt.subplots(len(streams), 1, figsize=(10, 1.55 * len(streams) + 0.6), sharex=True)
    for ax, st in zip(axes, streams):
        tt = [i * 0.05 for i in range(int(RUN / 0.05))]
        ax.plot(tt, [st.signal(x) for x in tt], color="#7f7f7f", lw=0.8)
        ax.axhline(st.threshold, color="black", ls=":", lw=0.8)
        ok = [(a, b) for a, b, c in st.samples if c == "on-time"]
        late = [(a, b) for a, b, c in st.samples if c == "late"]
        ax.plot([a for a, _ in ok], [b for _, b in ok], ".", ms=2, color="#2ca02c", label="on-time")
        ax.plot([a for a, _ in late], [b for _, b in late], "o", ms=3, mfc="none", color="#ff7f0e", label="late")
        if st.missed_t:
            ax.plot(st.missed_t, [st.signal(x) for x in st.missed_t], "x", ms=4, color="#d62728", label="missed")
        for a in st.alerts:
            ax.axvline(a, color="#d62728", lw=0.5, alpha=0.5)
        ax.set_ylabel(f"{st.name}\n({st.unit})", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6, ncol=3, loc="upper right")
    axes[0].set_title(title + " (red lines = alerts)", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"signals_{tag}.png"), dpi=130)
    plt.close(fig)


def main():
    configs = [(rule, fast) for fast in (False, True) for rule in ("FP", "EDF", "RR")]
    say(f"Run {RUN:.0f} s; on-time = serviced within {ON_TIME_FRAC:.0%} of the period after release; "
        f"tail: {TAIL_P:.0%} of reads take {TAIL_X[0]:.0f}-{TAIL_X[1]:.0f} x longer")
    say("Streams (period, median processing): air quality 200 ms / 45 ms, humidity 500 ms / 30 ms, temperature 1 s / 30 ms;")
    say("load 2 adds vibration 50 ms / 12 ms. Utilisation (median processing / period): load 1 = 0.32, load 2 = 0.56;")
    say("including the log-normal spread and the long tail about 0.39 and 0.69. Reads are not preemptible.")
    say("=" * 112)
    say(f"{'config':16s}{'stream':13s}{'required':>9s}{'on-time':>9s}{'late':>7s}{'missed':>8s}{'on-time %':>10s}"
        f"{'mean late ms':>13s}{'max late ms':>12s}{'alerts':>7s}{'alert lat ms':>13s}")
    summary = {}
    for rule, fast in configs:
        tag = f"{rule}_{'load2' if fast else 'load1'}"
        streams = run(rule, fast)
        for st in streams:
            c = st.counts
            al = alert_latency(st)
            summary[(tag, st.name)] = c
            say(f"{tag:16s}{st.name:13s}{c['required']:>9d}{c['on-time']:>9d}{c['late']:>7d}{c['missed']:>8d}"
                f"{100 * c['on-time'] / c['required']:>9.1f}%{statistics.mean(st.lateness) * 1000:>13.1f}"
                f"{max(st.lateness) * 1000:>12.1f}{len(st.alerts):>7d}"
                f"{(statistics.mean(al) * 1000 if al else float('nan')):>13.0f}")
        say("")
        plot(streams, tag, f"{rule} scheduling, {'4 streams (load 2)' if fast else '3 streams (load 1)'}")

    # bar chart: on-time / late / missed per stream for every configuration
    tags = [f"{r}_{l}" for l in ("load1", "load2") for r in ("FP", "EDF", "RR")]
    names = ["vibration", "air quality", "humidity", "temperature"]
    fig, axes = plt.subplots(1, len(tags), figsize=(12, 3.0), sharey=True)
    for ax, tag in zip(axes, tags):
        present = [n for n in names if (tag, n) in summary]
        bottom = [0] * len(present)
        for cls, col in (("on-time", "#2ca02c"), ("late", "#ff7f0e"), ("missed", "#d62728")):
            vals = [100 * summary[(tag, n)][cls] / summary[(tag, n)]["required"] for n in present]
            ax.bar(range(len(present)), vals, bottom=bottom, color=col, label=cls)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in present], fontsize=6.5)
        ax.set_title(tag.replace("_", ", "), fontsize=8)
    axes[0].set_ylabel("% of required samples")
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "compliance_bars.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(HERE, "results_4_2.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
