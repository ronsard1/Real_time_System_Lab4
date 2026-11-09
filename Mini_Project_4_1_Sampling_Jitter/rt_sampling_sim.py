"""
rt_sampling_sim.py - sampling jitter and processing overruns in a fixed-period sensing-and-control loop
(Mini Project 4.1). Pure Python: random + statistics, matplotlib for the figures.

Loop model (PLC scan cycle, Section 4.5): cycle k starts T after the previous start, or immediately when
the previous computation overran (start_k = max(start_{k-1} + T, finish_{k-1})). The input is sampled at
start_k + jitter_k (jitter uniform in [-J, J]); the computation then takes c_k, usually 20-50 % of T but,
with probability p, 120-250 % of T (a WCET violation). An overrun therefore pushes every later cycle back.
The controller acts when the computation of the sample that first exceeds the threshold ends.

    python rt_sampling_sim.py            -> results_4_1.txt, figures/
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

T_NOMINAL = 0.05          # s (20 Hz)
RUN = 20.0                # s
AMP, PERIOD = 1.0, 1.0    # true signal: 1.0 * sin(2 pi t / 1 s) + small continuous noise
SEEDS = range(30)         # every table row pools 30 independent runs
THRESHOLD = 0.5
C_NORMAL = (0.2, 0.5)     # computation time, fraction of T_NOMINAL
C_OVERRUN = (1.2, 2.5)
SEED = 4

_rng = random.Random(99)
_NOISE = [(0.03 * _rng.random(), 2 * math.pi * _rng.uniform(1.5, 6.0), 2 * math.pi * _rng.random()) for _ in range(6)]


def true_signal(t):
    return AMP * math.sin(2 * math.pi * t / PERIOD) + sum(a * math.sin(w * t + p) for a, w, p in _NOISE)


def true_crossings():
    """Upward crossings of the threshold by the noiseless sine (the events the controller must catch)."""
    t0 = PERIOD * math.asin(THRESHOLD / AMP) / (2 * math.pi)
    return [t0 + k * PERIOD for k in range(int(RUN / PERIOD) + 1) if t0 + k * PERIOD < RUN]


def simulate(J=0.0, p=0.0, T=T_NOMINAL, abs_jitter=None, abs_over=None, seed=SEED, controller="threshold"):
    """Return per-cycle sample times, values, computation times and actuation events."""
    rng = random.Random(seed)
    n = int(round(RUN / T))
    jbound = abs_jitter if abs_jitter is not None else J * T
    s, v, c, fin = [], [], [], []
    start, prev_fin = 0.0, 0.0
    for k in range(n):
        if k:
            start = max(s_start + T, prev_fin)
        s_start = start
        jit = rng.uniform(-jbound, jbound)
        ck_norm = rng.uniform(*C_NORMAL) * T_NOMINAL          # same draws in every run (fair comparison)
        over = rng.random() < p
        ck_over = (rng.uniform(*abs_over) if abs_over else rng.uniform(*C_OVERRUN) * T)
        t_sample = max(start + jit, prev_fin)
        ck = ck_over if over else ck_norm
        s.append(t_sample)
        v.append(true_signal(t_sample))
        c.append(ck)
        prev_fin = t_sample + ck
        fin.append(prev_fin)
    act = []
    if controller == "threshold":
        for k in range(1, n):                                  # crossing from below -> actuation at end of computation
            if v[k - 1] < THRESHOLD <= v[k]:
                act.append(fin[k])
    return dict(t=s, v=v, c=c, fin=fin, act=act, T=T, n=n)


def p_output(run, grid):
    """Lab task: P-type controller u = Kp (r - y), held from the end of each computation (zero-order hold)."""
    kp, r = 2.0, 0.0
    out, j = [], 0
    for g in grid:
        while j + 1 < len(run["fin"]) and run["fin"][j + 1] <= g:
            j += 1
        out.append(kp * (r - run["v"][j]) if run["fin"][0] <= g else 0.0)
    return out


def stats(run, ideal):
    T = run["T"]
    iv = [b - a for a, b in zip(run["t"], run["t"][1:])]
    over = [ck - T for ck in run["c"] if ck > T]
    delays, late_cycles = [], 0
    for tau in true_crossings():                      # noise can trigger slightly before tau: match in a window
        win = lambda x: tau - 0.1 <= x <= tau + PERIOD / 2
        a_i = next((x for x in ideal["act"] if win(x)), None)
        a_r = next((x for x in run["act"] if win(x)), None)
        if a_i is None or a_r is None:
            continue
        delays.append(a_r - a_i)
        late_cycles += (a_r - a_i) > T
    span = run["t"][-1] - run["t"][0]
    return dict(mean_iv=statistics.mean(iv), sd_iv=statistics.pstdev(iv), max_over=max(over) if over else 0.0,
                pct_missed=100 * sum(x > 1.05 * T for x in iv) / len(iv), pct_15=100 * sum(x > 1.5 * T for x in iv) / len(iv),
                mean_d=statistics.mean(delays) if delays else float("nan"), max_d=max(delays) if delays else float("nan"),
                late=late_cycles, n_cross=len(delays), drift=span - (run["n"] - 1) * T)


def pooled(J=0.0, p=0.0, **kw):
    """Run the configuration for every seed; pool intervals, overruns and actuation delays."""
    iv, over, delays, late, drift, perr = [], [], [], 0, [], []
    for sd in SEEDS:
        ideal = simulate(0, 0, seed=sd, T=kw.get("T", T_NOMINAL))
        run = simulate(J, p, seed=sd, **kw)
        T = run["T"]
        iv += [b - a for a, b in zip(run["t"], run["t"][1:])]
        over += [ck - T for ck in run["c"] if ck > T]
        for tau in true_crossings():
            win = lambda x: tau - 0.1 <= x <= tau + PERIOD / 2
            a_i = next((x for x in ideal["act"] if win(x)), None)
            a_r = next((x for x in run["act"] if win(x)), None)
            if a_i is not None and a_r is not None:
                delays.append(a_r - a_i)
                late += (a_r - a_i) > T
        drift.append(run["t"][-1] - run["t"][0] - (run["n"] - 1) * T)
        perr.append(p_quality(run, ideal))
    T = kw.get("T", T_NOMINAL)
    return dict(mean_iv=statistics.mean(iv), sd_iv=statistics.pstdev(iv), max_over=max(over) if over else 0.0,
                pct_missed=100 * sum(x > 1.05 * T for x in iv) / len(iv), pct_15=100 * sum(x > 1.5 * T for x in iv) / len(iv),
                mean_d=statistics.mean(delays), max_d=max(delays), late=late, n_cross=len(delays),
                drift=statistics.mean(drift), perr=statistics.mean(perr))


def p_quality(run, ideal):
    grid = [i * 0.005 for i in range(int(RUN / 0.005))]
    a, b = p_output(run, grid), p_output(ideal, grid)
    return 100 * math.sqrt(statistics.mean([(x - y) ** 2 for x, y in zip(a, b)])) / (2.0 * AMP)


def say(s=""):
    print(s)
    OUT.append(s)


def figures(name, run, ideal, title):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=False)
    tt = [i * 0.005 for i in range(int(6.0 / 0.005))]
    ax[0].plot(tt, [true_signal(x) for x in tt], color="#7f7f7f", lw=1, label="true signal")
    ki = [k for k, x in enumerate(ideal["t"]) if x <= 6.0]
    kr = [k for k, x in enumerate(run["t"]) if x <= 6.0]
    ax[0].plot([ideal["t"][k] for k in ki], [ideal["v"][k] for k in ki], "o", ms=3.5, mfc="none", color="#1f77b4", label="ideal samples")
    ax[0].plot([run["t"][k] for k in kr], [run["v"][k] for k in kr], "x", ms=4, color="#d62728", label="jittered / overrun samples")
    ax[0].axhline(THRESHOLD, ls=":", color="black", lw=0.8)
    ax[0].set_xlim(0, 6)
    ax[0].set_ylabel("signal")
    ax[0].set_xlabel("time (s), first 6 s")
    ax[0].legend(fontsize=7, loc="lower left", ncol=3)
    ax[0].set_title(title, fontsize=9)
    T = run["T"]
    dev = [(b - a - T) * 1000 for a, b in zip(run["t"], run["t"][1:])]
    ax[1].plot(range(1, len(dev) + 1), dev, lw=0.7, color="#d62728")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_xlabel("cycle index")
    ax[1].set_ylabel("interval - T (ms)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig1_{name}.png"), dpi=140)
    plt.close(fig)

    st = stats(run, ideal)
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.eventplot([ideal["act"], run["act"]], lineoffsets=[1, 0], linelengths=0.6, colors=["#1f77b4", "#d62728"])
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["ideal", "jitter/overrun"])
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, max(run["t"][-1], RUN))
    ax.set_title(f"{title}: actuation instants - mean delay {st['mean_d'] * 1000:.1f} ms, max {st['max_d'] * 1000:.1f} ms", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig2_{name}.png"), dpi=140)
    plt.close(fig)


def main():
    ideal = simulate(0, 0)
    check = stats(simulate(0, 0, seed=SEED), ideal)
    say(f"T_nominal {T_NOMINAL * 1000:.0f} ms, run {RUN:.0f} s, {ideal['n']} cycles, threshold {THRESHOLD}, "
        f"{len(true_crossings())} true upward crossings per run, {len(SEEDS)} runs per configuration")
    say(f"Check J = 0, p = 0: mean actuation delay {check['mean_d'] * 1000:.3f} ms, max {check['max_d'] * 1000:.3f} ms (must be 0)")
    say("=" * 118)
    hdr = (f"{'config':22s}{'mean iv ms':>11s}{'sd iv ms':>9s}{'max over ms':>12s}{'% > 1.05T':>10s}{'% > 1.5T':>9s}"
           f"{'mean dly ms':>12s}{'max dly ms':>11s}{'late>T':>8s}{'drift s':>8s}{'P err %':>8s}")
    say(hdr)
    configs = [("baseline", 0, 0)] + [(f"jitter J={j:.1f}T", j, 0) for j in (0.1, 0.2, 0.3)] + \
              [(f"overrun p={p:.2f}", 0, p) for p in (0.06, 0.12, 0.18)] + \
              [(f"J={j:.1f}T, p={p:.2f}", j, p) for j, p in ((0.1, 0.06), (0.2, 0.12), (0.3, 0.18))]
    rows = {}
    for name, j, p in configs:
        st = pooled(j, p)
        pe = st["perr"]
        rows[name] = (st, pe)
        say(f"{name:22s}{st['mean_iv'] * 1000:>11.2f}{st['sd_iv'] * 1000:>9.2f}{st['max_over'] * 1000:>12.1f}{st['pct_missed']:>10.1f}"
            f"{st['pct_15']:>9.1f}{st['mean_d'] * 1000:>12.1f}{st['max_d'] * 1000:>11.1f}{100 * st['late'] / st['n_cross']:>7.1f}%"
            f"{st['drift']:>8.2f}{pe:>8.1f}")
    say("  30 runs x 20 crossings per row. iv = inter-sample interval; over = computation beyond T; dly = actuation delay")
    say("  versus the ideal loop for the same crossing; late>T = crossings acted on more than one period late; drift =")
    say("  extra time the 400 cycles took (schedule lag); P err = RMS deviation of a P-controller output (Kp = 2) from")
    say("  the ideal loop, % of the output range")

    # equal average disturbance: mean |jitter| = J/2 * T;  mean overrun excess per cycle = p * 0.85 T
    say("=" * 118)
    say("Equal average disturbance per cycle (mean |jitter| = J T / 2 versus mean overrun excess = 0.85 p T):")
    for (j, p) in ((0.1, 0.06), (0.2, 0.12), (0.3, 0.18)):
        a, b = rows[f"jitter J={j:.1f}T"][0], rows[f"overrun p={p:.2f}"][0]
        say(f"  {j * 50:4.1f}% vs {p * 85:4.1f}% of T: jitter mean delay {a['mean_d'] * 1000:6.1f} ms (max {a['max_d'] * 1000:5.1f}), "
            f"overrun mean delay {b['mean_d'] * 1000:6.1f} ms (max {b['max_d'] * 1000:5.1f})")

    # lab task: halve T with absolute disturbance fixed
    say("=" * 118)
    say("Halving T_nominal with absolute jitter (15 ms) and overrun times (60-125 ms) fixed, p = 0.06:")
    for T in (0.05, 0.025):
        st = pooled(p=0.06, T=T, abs_jitter=0.015, abs_over=(0.06, 0.125))
        say(f"  T = {T * 1000:4.1f} ms: jitter bound = {0.015 / T * 100:4.0f}% of T, sd interval {st['sd_iv'] * 1000:5.2f} ms "
            f"= {st['sd_iv'] / T * 100:4.1f}% of T, % > 1.5T {st['pct_15']:5.1f}, mean actuation delay {st['mean_d'] * 1000:6.1f} ms, "
            f"drift {st['drift']:.2f} s")

    figures("jitter_J0.3", simulate(0.3, 0), ideal, "Jitter only, J = 0.3 T")
    figures("overrun_p0.12", simulate(0, 0.12), ideal, "Overruns only, p = 0.12")
    figures("combined_J0.2_p0.12", simulate(0.2, 0.12), ideal, "Jitter J = 0.2 T and overruns p = 0.12")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    xs = [5, 10, 15]
    ax.plot(xs, [rows[f"jitter J={j:.1f}T"][0]["mean_d"] * 1000 for j in (0.1, 0.2, 0.3)], "o-", label="jitter only")
    ax.plot([5.1, 10.2, 15.3], [rows[f"overrun p={p:.2f}"][0]["mean_d"] * 1000 for p in (0.06, 0.12, 0.18)], "s-",
            color="#d62728", label="overruns only")
    ax.set_xlabel("average timing disturbance per cycle (% of T)")
    ax.set_ylabel("mean actuation delay (ms)")
    ax.set_title("Same average disturbance, very different damage", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_jitter_vs_overrun.png"), dpi=140)
    plt.close(fig)
    with open(os.path.join(HERE, "results_4_1.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
