"""
Epsilon-sensitivity experiment for a fitted PAC-ERGM.

Starting from a definitive PAC-ERGM fit, re-estimate the model with

    epsilon_k = k * epsilon_0

for a set of radius multipliers k.  The definitive fit (k = 1) is not
re-estimated; it is evaluated with the same GoF procedure as the other
radii.  Every re-estimation starts from the definitive theta and nu.

The experiment follows:
  * k in {0.5, 0.75, 1.5, 3, 4};
  * 200 Robbins-Monro iterations;
  * burn-in 300,000 at every iteration;
  * theta = Polyak-Ruppert mean over the last 50 iterations;
  * nu = median over the last 50 iterations;
  * for k > 1, the theta step is scaled as a0 / k^2;
  * GoF uses the exact (hard) band and the settings of the definitive run.

Usage:

    python scripts/epsilon_sensitivity_run.py \
        --base-run output/run_2026..._spec3

Quick check:

    python scripts/epsilon_sensitivity_run.py \
        --base-run output/run_2026..._spec3 \
        --multipliers 0.5 1.5 \
        --n-iter 5 --burnin 1000 --window 5

Writes output/epsilon_sensitivity/run_<timestamp>/:
    setup.json                 experiment settings
    run_k<k>.json              estimates and evaluation criteria for each radius
    run_k<k>.npz               trajectories, GoF samples, simulated networks
    summary.csv                one row per radius, including k = 1
    theta_by_radius.csv        fitted theta for every radius
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor
from omegaconf import OmegaConf
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.estimation import run_pac
from src.gof import gof_sample_stats
from src.model import build_spec, compute_stats, load_data
from src.sampler import run_mcmc_inner


# ---------------------------------------------------------------------------
#   Paths and definitive run
# ---------------------------------------------------------------------------

def local_path(path):
    """Map data paths saved on another machine back into this repository."""
    p = Path(str(path)).expanduser()
    if p.exists():
        return str(p.resolve())

    q = str(path).replace("\\", "/")
    marker = "/data/"
    i = q.find(marker)
    if i >= 0:
        return str((ROOT / q[i + 1:]).resolve())

    # Config files in this project historically used paths relative to notebooks/.
    if not p.is_absolute():
        return str((ROOT / "notebooks" / p).resolve())
    return str(p)


def load_base_run(base_run):
    """Load the definitive fit, model and observed statistics."""
    run = Path(base_run).expanduser().resolve()
    cfg_path = run / "config_used.yaml"
    summary_path = run / "results_summary.json"
    if not cfg_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"{run} must contain config_used.yaml and results_summary.json"
        )

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    cfg.data.edges_path = local_path(cfg.data.edges_path)
    if cfg.data.get("nodes_path"):
        cfg.data.nodes_path = local_path(cfg.data.nodes_path)
    if cfg.data.get("edgecov"):
        cfg.data.edgecov = {k: local_path(v) for k, v in dict(cfg.data.edgecov).items()}

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    adj, nodes, ecov = load_data(cfg.data)
    spec = build_spec(nodes, ecov, cfg.model, directed=bool(cfg.data.is_directed))
    labels = list(spec.labels)
    g_obs = compute_stats(adj, spec)

    theta0 = theta_from_summary(summary, labels)
    nu0, coverage0 = final_window_from_history(run, cfg, summary)

    eps0 = float(cfg.pac.epsilon)
    scale = np.asarray(list(cfg.pac.norm_scale), dtype=float)

    return run, cfg, summary, adj, spec, labels, g_obs, theta0, nu0, coverage0, eps0, scale


def theta_from_summary(summary, labels):
    for key in ("theta_final_polyak", "theta_final", "theta_hat"):
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, dict):
            return np.asarray([value[label] for label in labels], dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.shape == (len(labels),):
            return arr
    raise KeyError(
        "results_summary.json does not contain theta_final_polyak, theta_final, or theta_hat"
    )


def final_window_from_history(run, cfg, summary):
    """Use the same final-window nu convention as the sensitivity notebook when possible."""
    hist_path = run / "history.csv"
    W = int(cfg.control.get("polyak_window", 50))
    if hist_path.exists():
        h = pd.read_csv(hist_path)
        if "nu" in h and len(h):
            nu0 = float(np.median(h["nu"].to_numpy(float)[-W:]))
            coverage0 = (
                float(np.mean(h["coverage"].to_numpy(float)[-W:]))
                if "coverage" in h
                else None
            )
            return nu0, coverage0

    for key in ("nu_final", "nu_hat"):
        if key in summary:
            nu0 = float(summary[key])
            break
    else:
        raise KeyError("results_summary.json does not contain nu_final or nu_hat")

    coverage0 = None
    for key in ("coverage_last_window", "coverage_last", "coverage_final_window"):
        if key in summary:
            coverage0 = float(summary[key])
            break
    return nu0, coverage0


def copy_cfg(cfg):
    c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(c, False)
    return c


# ---------------------------------------------------------------------------
#   GoF and evaluation criteria
# ---------------------------------------------------------------------------

def net_chain(A0, theta, nu, eps, seed, n_nets, burnin, gap,
              g_obs, norm, spec, proposal, p_nbhd, scale, penalty):
    """One long chain: burn in once, then retain n_nets networks gap steps apart."""
    rng = np.random.default_rng(seed)
    A = A0
    nets = []
    for j in range(n_nets):
        _, _, A = run_mcmc_inner(
            A, theta, nu, g_obs, eps, norm,
            n_samples=1, spec=spec, rng=rng,
            burnin=burnin if j == 0 else 0,
            proposal=proposal, p_nbhd=p_nbhd,
            scale=scale, interval=gap, penalty=penalty,
        )
        nets.append(A.copy())
    return nets


def simulate_fit(cfg, adj, g_obs, spec, theta, nu, eps, scale):
    """GoF statistics plus degree/ESP networks, using the definitive-run GoF settings."""
    c = copy_cfg(cfg)
    c.pac.epsilon = float(eps)

    n_chains = int(c.gof.net_chains)
    n_networks = int(c.gof.n_networks)
    per_chain = int(np.ceil(n_networks / n_chains))

    pool = get_reusable_executor(max_workers=n_chains)
    seeds = np.random.SeedSequence(int(c.gof.net_seed)).spawn(n_chains)
    futures = [
        pool.submit(
            net_chain,
            adj, np.asarray(theta, float), float(nu), float(eps), seed,
            per_chain, int(c.gof.net_burnin), int(c.gof.net_gap),
            g_obs, str(c.pac.norm), spec,
            str(c.proposal.kind), float(c.proposal.p_nbhd),
            scale, str(c.gof.penalty),
        )
        for seed in seeds
    ]

    G, _ = gof_sample_stats(
        c, adj, g_obs, spec, np.asarray(theta, float), float(nu)
    )
    nets = [A for future in futures for A in future.result()][:n_networks]
    return np.asarray(G, float), nets


def energy_score(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(X) < 2:
        return float("nan")
    return float(
        np.linalg.norm(X - y, axis=1).mean()
        - 0.5 * pdist(X, metric="euclidean").mean()
    )


def capped_hist(values, maxd):
    values = np.asarray(values, dtype=int)
    return np.bincount(
        np.minimum(values, maxd), minlength=maxd + 1
    )[:maxd + 1]


def network_counts(A, maxd):
    A = np.asarray(A)
    S = A @ A
    return (
        capped_hist(A.sum(axis=0), maxd),
        capped_hist(A.sum(axis=1), maxd),
        capped_hist(S[A == 1], maxd),
    )


def criteria(G, nets, g_obs, scale, eps, eps0, obs_counts, maxd):
    """In-model and out-of-model criteria used in the sensitivity table."""
    G = np.asarray(G, dtype=float)
    Z = (G - g_obs) / scale
    d = np.linalg.norm(Z, axis=1)

    counts = [network_counts(A, maxd) for A in nets]
    C_in = np.asarray([c[0] for c in counts])
    C_out = np.asarray([c[1] for c in counts])
    C_esp = np.asarray([c[2] for c in counts])

    edges = G[:, 0]
    return {
        "R": float(np.linalg.norm(Z.mean(axis=0))),
        "pi_own": float(np.mean(d <= eps)),
        "pi_base": float(np.mean(d <= eps0)),
        "ES": energy_score(Z, np.zeros(Z.shape[1])),
        "ES_indeg": energy_score(C_in, obs_counts[0]),
        "ES_outdeg": energy_score(C_out, obs_counts[1]),
        "ES_esp": energy_score(C_esp, obs_counts[2]),
        # Kept as sanity checks for sparse/degenerate excursions.
        "edges_median": float(np.median(edges)),
        "edges_q05": float(np.quantile(edges, 0.05)),
        "edges_q95": float(np.quantile(edges, 0.95)),
        "frac_edges_over_2x": float(np.mean(edges > 2 * g_obs[0])),
        "d_median": float(np.median(d)),
    }


# ---------------------------------------------------------------------------
#   One radius
# ---------------------------------------------------------------------------

def radius_tag(k):
    return f"k{k:g}"


def save_radius(run_dir, result, arrays):
    tag = radius_tag(result["k"])
    with open(run_dir / f"run_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(run_dir / f"run_{tag}.npz", **arrays)


def evaluate_base(k, cfg, adj, g_obs, spec, theta0, nu0, coverage0,
                  eps0, scale, obs_counts, maxd, step0):
    """k = 1: use the definitive estimate, but evaluate it with the same code."""
    t0 = time.perf_counter()
    G, nets = simulate_fit(cfg, adj, g_obs, spec, theta0, nu0, eps0, scale)
    result = {
        "k": float(k),
        "epsilon": float(eps0),
        "theta": theta0.tolist(),
        "nu_hat": float(nu0),
        "coverage_estimation": coverage0,
        "step_size": float(step0),
        "seconds_fit": 0.0,
        "seconds_total": time.perf_counter() - t0,
        **criteria(G, nets, g_obs, scale, eps0, eps0, obs_counts, maxd),
    }
    arrays = {
        "gof_samples": G,
        "networks": np.asarray(nets, dtype=np.int8),
    }
    return result, arrays


def estimate_radius(k, base_cfg, adj, g_obs, spec, theta0, nu0,
                    eps0, scale, obs_counts, maxd, args):
    eps = float(k * eps0)
    cfg = copy_cfg(base_cfg)
    cfg.pac.epsilon = eps
    cfg.control.n_iter = int(args.n_iter)
    cfg.control.burnin = int(args.burnin)
    cfg.control.polyak_window = min(int(args.window), int(args.n_iter))
    if args.n_jobs is not None:
        cfg.control.n_jobs = int(args.n_jobs)

    cfg.estimation.nu_init = float(nu0)
    step0 = float(base_cfg.control.step_size)
    cfg.control.step_size = step0 / max(float(k), 1.0) ** 2

    # Optional Hydra-style overrides without re-composing the configuration.
    if args.set:
        overrides = OmegaConf.from_dotlist(args.set)
        cfg = OmegaConf.merge(cfg, overrides)
        OmegaConf.set_struct(cfg, False)

    print(
        f"\n{'=' * 78}\n"
        f"k = {k:g} | epsilon = {eps:.4f} | "
        f"step = {float(cfg.control.step_size):.4g}\n"
        f"{'=' * 78}",
        flush=True,
    )

    t_all = time.perf_counter()
    t0 = time.perf_counter()
    theta_hat, _, hist, _ = run_pac(
        cfg, adj, g_obs, spec, theta_init=theta0.copy()
    )
    fit_seconds = time.perf_counter() - t0

    W = min(int(cfg.control.polyak_window), len(hist["coverage"]))
    nu_hist = np.asarray(hist["nu"][1:], dtype=float)
    nu_hat = float(np.median(nu_hist[-W:]))
    coverage_estimation = float(np.mean(np.asarray(hist["coverage"], float)[-W:]))

    G, nets = simulate_fit(
        cfg, adj, g_obs, spec, theta_hat, nu_hat, eps, scale
    )

    result = {
        "k": float(k),
        "epsilon": eps,
        "theta": np.asarray(theta_hat, float).tolist(),
        "nu_hat": nu_hat,
        "coverage_estimation": coverage_estimation,
        "nu_window_mean": float(np.mean(nu_hist[-W:])),
        "nu_window_sd": float(np.std(nu_hist[-W:])),
        "step_size": float(cfg.control.step_size),
        "seconds_fit": fit_seconds,
        "seconds_total": time.perf_counter() - t_all,
        **criteria(G, nets, g_obs, scale, eps, eps0, obs_counts, maxd),
    }
    arrays = {
        "nu_hist": nu_hist,
        "coverage_hist": np.asarray(hist["coverage"], dtype=float),
        "theta_hist": np.asarray(hist["theta"], dtype=float),
        "gof_samples": G,
        "networks": np.asarray(nets, dtype=np.int8),
    }
    return result, arrays


# ---------------------------------------------------------------------------
#   Summary files
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "k", "epsilon", "step_size", "nu_hat", "coverage_estimation",
    "pi_own", "pi_base", "R", "ES", "ES_indeg", "ES_outdeg", "ES_esp",
    "edges_median", "edges_q05", "edges_q95", "frac_edges_over_2x",
    "d_median", "seconds_fit", "seconds_total",
]


def write_summary(run_dir, results, labels):
    rows = []
    for k in sorted(results):
        r = results[k]
        rows.append({name: r.get(name) for name in SUMMARY_FIELDS})

    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    with open(run_dir / "theta_by_radius.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        ks = sorted(results)
        w.writerow(["term", *[f"k={k:g}" for k in ks]])
        for j, label in enumerate(labels):
            w.writerow([label, *[results[k]["theta"][j] for k in ks]])


# ---------------------------------------------------------------------------
#   CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--base-run", required=True,
        help="definitive PAC run containing config_used.yaml and results_summary.json",
    )
    ap.add_argument(
        "--multipliers", type=float, nargs="+",
        default=[0.5, 0.75, 1.5, 3.0, 4.0],
        help="radius multipliers k; k=1 is always added as the definitive fit",
    )
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="override control.n_jobs for the re-estimations")
    ap.add_argument(
        "--out", default=str(ROOT / "output" / "epsilon_sensitivity"),
        help="root output directory",
    )
    args = ap.parse_args()

    (
        base_run, base_cfg, base_summary, adj, spec, labels, g_obs,
        theta0, nu0, coverage0, eps0, scale,
    ) = load_base_run(args.base_run)

    obs_counts = network_counts(adj, int(base_cfg.ergm.maxd))
    step0 = float(base_cfg.control.step_size)

    run_dir = Path(args.out).expanduser() / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    multipliers = sorted(set(float(k) for k in args.multipliers) | {1.0})
    if any(k <= 0 for k in multipliers):
        raise SystemExit("all radius multipliers must be > 0")

    setup = {
        "base_run": str(base_run),
        "multipliers": multipliers,
        "n_iter": int(base_cfg.control.n_iter),
        "burnin": int(base_cfg.control.burnin),
        "window": int(base_cfg.control.polyak_window),
        "max_degree": int(base_cfg.ergm.maxd),
        "n_jobs": args.n_jobs,
        "epsilon_base": eps0,
        "scale": scale.tolist(),
        "labels": labels,
        "theta_start": theta0.tolist(),
        "nu_start": nu0,
    }
    with open(run_dir / "setup.json", "w", encoding="utf-8") as f:
        json.dump(setup, f, indent=1)

    print("base run:", base_run)
    print("labels:", labels)
    print(f"epsilon_0 = {eps0:.4f} | nu_0 = {nu0:.4f}")
    print("radii:", {k: round(k * eps0, 4) for k in multipliers})
    print("output:", run_dir, flush=True)

    results = {}
    for k in multipliers:
        if np.isclose(k, 1.0):
            result, arrays = evaluate_base(
                k, base_cfg, adj, g_obs, spec, theta0, nu0, coverage0,
                eps0, scale, obs_counts, int(base_cfg.ergm.maxd), step0,
            )
        else:
            result, arrays = estimate_radius(
                k, base_cfg, adj, g_obs, spec, theta0, nu0,
                eps0, scale, obs_counts, int(base_cfg.ergm.maxd), args,
            )

        results[k] = result
        save_radius(run_dir, result, arrays)
        write_summary(run_dir, results, labels)

        print(
            f"--- k={k:g}: nu={result['nu_hat']:.3f} | "
            f"pi_k={result['pi_own']:.3f} | pi_1={result['pi_base']:.3f} | "
            f"R={result['R']:.3f} | ES={result['ES']:.3f} | "
            f"total={result['seconds_total'] / 60:.1f} min",
            flush=True,
        )

    write_summary(run_dir, results, labels)
    print("\nDONE ->", run_dir)


if __name__ == "__main__":
    main()
