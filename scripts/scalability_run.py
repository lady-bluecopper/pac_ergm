"""
Scalability experiment for the PAC-ERGM: fitting time and time per simulated network as the
network grows, with the sparse implementation (src/pac/sparse.py).

From a large directed network (default: SNAP soc-Slashdot0902 in data/slashdot/), induced
subgraphs of growing size (about --sizes edges) are drawn by breadth-first exploration from a
random node. On each subgraph the PAC-ERGM with --terms is calibrated (band), fitted
(Robbins-Monro, fixed number of iterations) and used to simulate --n-networks networks.
Chain lengths grow with the number of edges m: burn-in = burnin_sweeps * m steps, interval
between two samples = interval_sweeps * m steps. The cost of one MCMC step is also measured on
the same chain with nu = 0 (a plain ERGM chain), to show what the band adds.

    python scripts/scalability_run.py                                    # full experiment
    python scripts/scalability_run.py --sizes 100 1000 --replicates 1 --n-iter 20   # quick check
    nohup python scripts/scalability_run.py > scalability.log 2>&1 &     # on a server

Writes output/scalability/run_<timestamp>/:
    setup.json             arguments and machine (CPU model, cores, Python)
    run_<size>_r<k>.json   timings, estimates, statistics of the simulated networks (one per run)
    sub_<size>_r<k>.csv    the subgraph's edge list (from,to; 0-based)
    sim_<size>_r<k>.csv.gz the simulated networks' edge lists (net,from,to; 0-based, same nodes)
    summary.csv            one row per run
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.estimation import run_pac
from src.sparse import (
    as_adjacency, bernoulli_theta, build_sparse_spec, calibrate_band_sparse,
    compute_mple_sparse, compute_stats_sparse, run_mcmc_inner_sparse,
)


# ---------------------------------------------------------------------------
#   Data and subgraphs
# ---------------------------------------------------------------------------
def load_edgelist(path):
    """SNAP edge list (FromNodeId ToNodeId, '#' comments) -> binary csr, no self-loops."""
    E = np.loadtxt(path, dtype=np.int64, comments="#")
    n = int(E.max()) + 1
    return as_adjacency(sp.csr_matrix((np.ones(len(E), dtype=np.int8), (E[:, 0], E[:, 1])),
                                      shape=(n, n)))


def bfs_subgraph(A, target, rng, max_tries=100):
    """Induced subgraph on the first nodes reached by a breadth-first exploration (edges in
    either direction) from a random node, stopped as soon as it has >= target edges."""
    n = A.shape[0]
    AT = A.T.tocsr()
    U = (A + AT).tocsr()                                       # exploration ignores direction
    candidates = np.flatnonzero(np.diff(U.indptr) > 0)
    for _ in range(max_tries):
        start = int(rng.choice(candidates))
        in_s = np.zeros(n, dtype=bool)
        in_s[start] = True
        order, head, m = [start], 0, 0
        while head < len(order) and m < target:
            x = order[head]; head += 1
            for y in U.indices[U.indptr[x]:U.indptr[x + 1]]:
                if in_s[y]:
                    continue
                in_s[y] = True
                order.append(int(y))
                m += int(in_s[A.indices[A.indptr[y]:A.indptr[y + 1]]].sum())     # y -> S
                m += int(in_s[AT.indices[AT.indptr[y]:AT.indptr[y + 1]]].sum())   # S -> y
                if m >= target:
                    break
        if m >= target:
            nodes = np.asarray(order)
            return A[nodes][:, nodes].tocsr(), nodes
    raise RuntimeError(f"no component with {target} edges found in {max_tries} tries")


# ---------------------------------------------------------------------------
#   One run: calibrate, fit, simulate
# ---------------------------------------------------------------------------
def make_cfg(args, m, seed):
    """The estimation settings of the thesis, with chain lengths proportional to m."""
    return OmegaConf.create({
        "seed": seed, "verbose": bool(args.verbose),
        "proposal": {"kind": "neighborhood", "p_nbhd": 0.5},
        "estimation": {"warm_start": "none", "nu_init": 10.0, "nu_step": 2.5, "nu_decay": 0.002,
                       "penalty": "logistic", "tau_factor": 16, "theta_update": "rm_diag",
                       "theta_decay": 0.001},
        "pac": {"norm": "l2", "delta": 0.1, "n_bootstrap": args.n_bootstrap},
        "control": {"n_iter": args.n_iter, "burnin": int(args.burnin_sweeps * m),
                    "interval": max(1, int(round(args.interval_sweeps * m))),
                    "mcmc_samples": args.samples, "step_size": 0.3,
                    "polyak_window": args.window, "n_chains": args.chains,
                    "n_jobs": args.n_jobs, "coverage_agg": "mean"},
    })


def time_per_step(A, theta, nu, g_obs, eps, scale, spec, steps, seed):
    """Microseconds per MCMC step of one chain, exact band (nu = 0 is the plain ERGM chain)."""
    t0 = time.perf_counter()
    run_mcmc_inner_sparse(A, theta, nu, g_obs, eps, "l2", steps, spec, np.random.default_rng(seed),
                          proposal="neighborhood", p_nbhd=0.5, scale=scale, penalty="hard")
    return (time.perf_counter() - t0) / steps * 1e6


def one_run(A_full, size, rep, args, out_dir):
    seed = int(np.random.SeedSequence([args.seed, size, rep]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    res = {"target_edges": size, "replicate": rep, "seed": seed, "terms": args.terms, "times_s": {}}
    T = res["times_s"]

    t0 = time.perf_counter()
    A, nodes = bfs_subgraph(A_full, size, rng)
    T["subsample"] = time.perf_counter() - t0
    n, m = A.shape[0], int(A.nnz)
    res.update(n=n, m=m)
    r, c = A.nonzero()
    np.savetxt(out_dir / f"sub_{size}_r{rep}.csv", np.column_stack([r, c]), fmt="%d",
               delimiter=",", header="from,to", comments="")
    print(f"\n=== size {size} rep {rep}: n={n} m={m}", flush=True)

    nets = pac_run(A, size, rep, seed, args, res)

    sim = [np.column_stack([np.full(net.nnz, k), *net.nonzero()]) for k, net in enumerate(nets)]
    np.savetxt(out_dir / f"sim_{size}_r{rep}.csv.gz", np.vstack(sim), fmt="%d",
               delimiter=",", header="net,from,to", comments="")
    with open(out_dir / f"run_{size}_r{rep}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


def pac_run(A, size, rep, seed, args, res):
    """Calibrate, fit and simulate the PAC-ERGM on the subgraph A; results go into res.
    Returns the simulated networks (csr)."""
    T = res["times_s"]
    n, m = A.shape[0], int(A.nnz)
    spec = build_sparse_spec(args.terms, n)
    g_obs = compute_stats_sparse(A, spec)
    cfg = make_cfg(args, m, seed)
    res["config"] = OmegaConf.to_container(cfg)

    t0 = time.perf_counter()
    calibrate_band_sparse(cfg, A, g_obs, spec)
    T["calibration"] = time.perf_counter() - t0
    eps, scale = float(cfg.pac.epsilon), np.asarray(cfg.pac.norm_scale)

    t0 = time.perf_counter()
    if args.start == "mple":            # all the edges + a weighted sample of the non-edges
        theta0 = compute_mple_sparse(A, spec, max_dyads=args.mple_dyads,
                                     rng=np.random.default_rng(seed + 2))
    else:
        theta0 = bernoulli_theta(A, spec)
    T["start"] = time.perf_counter() - t0
    res["theta_start"] = [float(x) for x in theta0]

    t0 = time.perf_counter()
    theta_hat, _, hist, _ = run_pac(cfg, A, g_obs, spec, theta_init=theta0,
                                    inner=run_mcmc_inner_sparse)
    T["fit"] = time.perf_counter() - t0
    T["fit_per_iteration"] = T["fit"] / args.n_iter
    W = min(args.window, args.n_iter)
    nu_hat = float(np.median(hist["nu"][-W:]))

    # --- simulation from the fitted model (exact band), as in the GoF of the thesis
    g_burn = int(args.burnin_sweeps * m)
    g_gap = max(1, int(round(args.gen_interval_sweeps * m)))
    kw = dict(proposal="neighborhood", p_nbhd=0.5, scale=scale, interval=1, penalty="hard")
    sim_rng = np.random.default_rng(seed + 1)
    t0 = time.perf_counter()
    _, _, net = run_mcmc_inner_sparse(A, theta_hat, nu_hat, g_obs, eps, "l2", 1, spec, sim_rng,
                                      burnin=g_burn - 1, **kw)
    T["simulation_burnin"] = time.perf_counter() - t0
    g_sim, in_band, nets = [], [], []
    t0 = time.perf_counter()
    for _ in range(args.n_networks):
        g, ib, net = run_mcmc_inner_sparse(net, theta_hat, nu_hat, g_obs, eps, "l2", 1, spec,
                                           sim_rng, burnin=g_gap - 1, **kw)
        g_sim.append(g[0].tolist()); in_band.append(int(ib[0])); nets.append(net)
    T["per_network"] = (time.perf_counter() - t0) / args.n_networks

    # --- cost of one MCMC step, with and without the band
    steps = int(min(args.step_sweeps * m, args.max_step_probe))
    res["us_per_step"] = {"ergm_chain_nu0": time_per_step(A, theta_hat, 0.0, g_obs, eps, scale, spec, steps, seed),
                          "pac_chain": time_per_step(A, theta_hat, nu_hat, g_obs, eps, scale, spec, steps, seed)}

    res.update(
        labels=list(spec.labels), g_obs=g_obs.tolist(), epsilon=eps, scale=scale.tolist(),
        theta_hat=[float(x) for x in theta_hat], nu_hat=nu_hat,
        coverage_last_window=float(np.mean(hist["coverage"][-W:])),
        steps_per_chain_per_iteration=int(cfg.control.burnin + cfg.control.mcmc_samples * cfg.control.interval),
        simulated_g=g_sim, simulated_in_band=in_band,
        history={"nu": [float(x) for x in hist["nu"]], "coverage": [float(x) for x in hist["coverage"]],
                 "theta": [[float(v) for v in th] for th in hist["theta"]]},
    )
    print(f"--- size {size} rep {rep}: pac fit {T['fit']:.1f}s ({T['fit_per_iteration']:.2f}s/iter), "
          f"per network {T['per_network']:.2f}s, us/step ERGM chain {res['us_per_step']['ergm_chain_nu0']:.1f} "
          f"PAC chain {res['us_per_step']['pac_chain']:.1f}", flush=True)
    return nets


def cpu_model():
    """CPU model name; platform.processor() is often just the architecture (e.g. 'amd64')."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform.startswith("freebsd") or sys.platform == "darwin":
            key = "machdep.cpu.brand_string" if sys.platform == "darwin" else "hw.model"
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "slashdot" / "soc-Slashdot0902.txt.gz"))
    ap.add_argument("--terms", default="edges + mutual + gwesp(0.5)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1_000, 10_000, 100_000])
    ap.add_argument("--replicates", type=int, default=1, help="subgraphs per size (different start nodes)")
    ap.add_argument("--n-iter", type=int, default=300, help="Robbins-Monro iterations")
    ap.add_argument("--window", type=int, default=100, help="Polyak-Ruppert window W")
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--samples", type=int, default=100, help="samples per chain per iteration")
    ap.add_argument("--burnin-sweeps", type=float, default=10.0, help="burn-in = this * m steps")
    ap.add_argument("--interval-sweeps", type=float, default=0.1, help="interval = this * m steps")
    ap.add_argument("--start", choices=["mple", "bernoulli"], default="mple", help="starting theta")
    ap.add_argument("--mple-dyads", type=int, default=1_000_000,
                    help="non-edges in the MPLE: all if fewer, otherwise a weighted sample of this size")
    ap.add_argument("--n-bootstrap", type=int, default=200, help="networks for the band calibration")
    ap.add_argument("--n-networks", type=int, default=10, help="networks simulated from the fit")
    ap.add_argument("--gen-interval-sweeps", type=float, default=1.0, help="steps between two simulated networks / m")
    ap.add_argument("--step-sweeps", type=float, default=5.0, help="steps (/ m) of the per-step timing")
    ap.add_argument("--max-step-probe", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print every Robbins-Monro iteration")
    ap.add_argument("--out", default=str(ROOT / "output" / "scalability"))
    args = ap.parse_args()

    out_dir = Path(args.out) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "setup.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "platform": platform.platform(), "processor": platform.processor(),
                   "cpu_model": cpu_model(), "cpu_count": os.cpu_count(), "python": sys.version}, f, indent=1)

    t0 = time.perf_counter()
    A_full = load_edgelist(args.data)
    print(f"loaded {args.data}: n={A_full.shape[0]} m={A_full.nnz} ({time.perf_counter() - t0:.1f}s)")

    rows = []
    for size in args.sizes:
        for rep in range(args.replicates):
            r = one_run(A_full, size, rep, args, out_dir)
            rows.append(summary_row(r))
            with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
                w.writeheader(); w.writerows(rows)
    print("\nDONE ->", out_dir)


SUMMARY_FIELDS = [
    "target_edges", "replicate", "n", "m", "t_subsample", "t_calibration", "t_start", "t_fit",
    "t_fit_per_iteration", "t_simulation_burnin", "t_per_network", "us_step_nu0", "us_step",
    "nu_hat", "coverage", "epsilon",
]


def summary_row(r):
    """One row of summary.csv: the timings and the estimate of one run."""
    def rnd(x, d=3):
        return None if x is None else round(float(x), d)
    T, us = r["times_s"], r["us_per_step"]
    return {"target_edges": r["target_edges"], "replicate": r["replicate"], "n": r["n"], "m": r["m"],
            **{f"t_{k}": rnd(v) for k, v in T.items()},
            "us_step_nu0": rnd(us["ergm_chain_nu0"], 2), "us_step": rnd(us["pac_chain"], 2),
            "nu_hat": rnd(r["nu_hat"]), "coverage": rnd(r["coverage_last_window"]),
            "epsilon": r["epsilon"]}


if __name__ == "__main__":
    main()
