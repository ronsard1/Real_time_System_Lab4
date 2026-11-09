"""
edge_cloud_sim.py - edge versus cloud latency for a repeated real-time decision task (Mini Project 4.3).

A vibration reading arrives every 50 ms (20 Hz); each needs one anomaly decision before a 100 ms deadline.
  edge   acquisition + local inference on a small processor: ~42 ms, low spread, no network
  cloud  upload + remote inference (~6 ms on a big server) + download; each network leg is log-normal
         (median 14 ms) and, with a probability that grows with congestion, hit by a Pareto-tailed spike
         (queueing, retransmission) - the genuinely long tail of a wide-area network
  hybrid 90 % of readings are clear-cut and decided at the edge; the 10 % "hard" readings need the larger
         cloud model. Hybrid-A sends them after the edge pre-screen and waits for the answer. Hybrid-B
         (deadline-aware) waits for the cloud only until a timeout and otherwise acts on the edge decision.
Both architectures see the SAME event sequence (same random hard/easy flag per event).

    python edge_cloud_sim.py     -> results_4_3.txt, figures/
"""
import math
import os
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
OUT = []
N = 20000                 # decision events (1000 s at 20 Hz)
DEADLINE = 100.0          # ms
BASE_CONGESTION = 0.2
HARD_FRACTION = 0.10
SCREEN_MS = 12.0          # edge pre-screen that flags hard readings (hybrid)
TIMEOUT_MS = 85.0         # hybrid-B: give up on the cloud after this, act on the edge decision


def edge_latency(rng):
    """Local inference on constrained hardware: longer compute, tiny spread, ~1 ms local transfer."""
    return 1.0 + max(30.0, rng.gauss(41.0, 2.5))


def network_leg(rng, congestion):
    """One wide-area leg: log-normal body plus a Pareto spike whose probability and size grow with congestion."""
    base = rng.lognormvariate(math.log(14.0 * (1 + 1.5 * congestion)), 0.25)
    if rng.random() < 0.005 + 0.08 * congestion:
        base += 20.0 * (1 + congestion) * rng.paretovariate(1.6)
    return base


def cloud_latency(rng, congestion):
    return network_leg(rng, congestion) + rng.gauss(6.0, 1.0) + network_leg(rng, congestion)


def run(congestion, seed=7):
    rng_e, rng_c, rng_h = random.Random(seed), random.Random(seed + 1), random.Random(seed + 2)
    edge, cloud, hyb_a, hyb_b, fallback = [], [], [], [], 0
    for _ in range(N):
        e = edge_latency(rng_e)
        c = cloud_latency(rng_c, congestion)
        hard = rng_h.random() < HARD_FRACTION
        edge.append(e)
        cloud.append(c)
        if hard:
            hyb_a.append(SCREEN_MS + c)
            if SCREEN_MS + c <= TIMEOUT_MS:
                hyb_b.append(SCREEN_MS + c)
            else:                                     # cloud too slow: decide locally (edge model already ran)
                hyb_b.append(max(TIMEOUT_MS, e))
                fallback += 1
        else:
            hyb_a.append(e)
            hyb_b.append(e)
    return dict(edge=edge, cloud=cloud, hybrid_A=hyb_a, hybrid_B=hyb_b, fallback=fallback)


def summary(x):
    q = statistics.quantiles(x, n=1000, method="inclusive")
    return dict(mean=statistics.mean(x), p95=q[949], p99=q[989], worst=max(x),
                comp=100 * sum(v <= DEADLINE for v in x) / len(x))


def say(s=""):
    print(s)
    OUT.append(s)


def main():
    res = run(BASE_CONGESTION)
    say(f"{N} decision events at 20 Hz, deadline {DEADLINE:.0f} ms, baseline congestion {BASE_CONGESTION}, "
        f"hard readings {HARD_FRACTION:.0%}, hybrid-B timeout {TIMEOUT_MS:.0f} ms")
    say("=" * 86)
    say(f"{'architecture':26s}{'mean ms':>9s}{'p95 ms':>9s}{'p99 ms':>9s}{'worst ms':>10s}{'compliance':>12s}")
    labels = {"edge": "edge", "cloud": "cloud (congestion 0.2)", "hybrid_A": "hybrid-A (cloud for hard)",
              "hybrid_B": "hybrid-B (deadline-aware)"}
    for k, lab in labels.items():
        s = summary(res[k])
        say(f"{lab:26s}{s['mean']:>9.1f}{s['p95']:>9.1f}{s['p99']:>9.1f}{s['worst']:>10.1f}{s['comp']:>11.2f}%")
    say(f"  hybrid-B fell back to the edge decision for {res['fallback']} of "
        f"{int(N * HARD_FRACTION)} hard readings ({100 * res['fallback'] / (N * HARD_FRACTION):.1f}%)")

    say("=" * 86)
    say("Congestion sweep (compliance % / p99 ms):")
    levels = [i / 10 for i in range(11)]
    sweep = {k: [] for k in labels}
    for cg in levels:
        r = run(cg)
        line = f"  congestion {cg:.1f}:"
        for k in labels:
            s = summary(r[k])
            sweep[k].append(s["comp"])
            line += f"  {k} {s['comp']:6.2f}% / {s['p99']:6.0f}"
        say(line)

    # figure 1: empirical CDFs with the deadline
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    cols = {"edge": "#2ca02c", "cloud": "#1f77b4", "hybrid_A": "#ff7f0e", "hybrid_B": "#9467bd"}
    for k in labels:
        x = sorted(res[k])
        y = [i / len(x) for i in range(1, len(x) + 1)]
        ax[0].plot(x, y, color=cols[k], label=labels[k], lw=1.2)
        ax[1].plot(x, [1 - v + 1e-6 for v in y], color=cols[k], lw=1.2)
    for a in ax:
        a.axvline(DEADLINE, color="#d62728", ls="--", lw=1)
        a.set_xlabel("end-to-end decision latency (ms)")
        a.grid(alpha=0.3)
    ax[0].set_xlim(0, 160)
    ax[0].set_ylabel("fraction of decisions <= latency")
    ax[0].set_title("Empirical CDF (deadline 100 ms dashed)", fontsize=9)
    ax[0].legend(fontsize=7, loc="lower right")
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")
    ax[1].set_ylabel("fraction slower than latency")
    ax[1].set_title("Tail on log axes: the cloud's long tail", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_latency_cdf.png"), dpi=140)
    plt.close(fig)

    # figure 2: compliance versus congestion
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for k in labels:
        ax.plot(levels, sweep[k], "o-" if k != "edge" else "-", color=cols[k], label=labels[k].split(" (")[0], ms=4)
    ax.axhline(99.9, color="black", ls=":", lw=0.8)
    ax.text(0.01, 99.95, "99.9 % target", fontsize=7)
    ax.set_xlabel("cloud congestion level")
    ax.set_ylabel("deadline compliance (%)")
    ax.set_ylim(min(min(v) for v in sweep.values()) - 2, 100.5)
    ax.set_title("Deadline compliance as cloud congestion rises (edge is unaffected)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_compliance_vs_congestion.png"), dpi=140)
    plt.close(fig)
    with open(os.path.join(HERE, "results_4_3.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
