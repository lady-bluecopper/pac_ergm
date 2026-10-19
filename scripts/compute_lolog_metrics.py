"""
In-model criteria of LOLOG on its OWN statistics, for the final runs of Specifications 2 and 3
(office University 2005), written next to thesis_metrics.json as lolog_own_metrics.json.

    python scripts/lolog_own_metrics.py                 # both specifications
    python scripts/lolog_own_metrics.py --spec 3        # one of them
    python scripts/lolog_own_metrics.py --spec 2 --run output/run_...   # another saved run

Why: thesis_metrics.json evaluates LOLOG on the statistics of the PAC-ERGM, but lolog's gwesp
counts CYCLIC shared partners (ergm type ITP), not the transitive ones (OTP), and in
Specification 3 LOLOG has out/in 2- and 3-stars instead of gwodeg/gwideg. Here LOLOG is
evaluated on the statistics of its own model:
    Spec 2: edges, mutual, gwesp ITP, edgecov use/team/floor/metric
    Spec 3: edges, mutual, gwesp ITP, out-stars 2 and 3, in-stars 2 and 3, the four edgecov
The band is recalibrated exactly as calibrate_band does (same dyad-independent null, same
bootstrap graphs, same seed) on these statistics; as a check, the same bootstrap gives back
the epsilon of the run on the PAC-ERGM statistics. R, coverage and ES are then computed on the
networks simulated by LOLOG in the run (arrays.npz).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.special import comb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir
from src import build_spec, compute_stats, load_data

SPECS = {
    2: {"config": "config_office_uni2005_pac",
        "run": ROOT / "output" / "run_20260927_044435_PAC_office_uni2005_final",
        "stars": False},
    3: {"config": "config_office_uni2005_gwdeg_pac",
        "run": ROOT / "output" / "run_20260928_020722_PAC_office_uni2005_gwdeg_final",
        "stars": True},
}
AL = 0.5            # gwesp decay, the same in ergm, lolog and the PAC-ERGM


def load(config_name):
    """Config, data and PAC spec as in the notebooks (paths in the config are relative to notebooks/)."""
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose(config_name=config_name)
    OmegaConf.set_struct(cfg, False)
    nb = ROOT / "notebooks"
    cfg.data.edges_path = str((nb / cfg.data.edges_path).resolve())
    cfg.data.nodes_path = str((nb / cfg.data.nodes_path).resolve())
    cfg.data.edgecov = {k: str((nb / v).resolve()) for k, v in dict(cfg.data.edgecov).items()}
    adj, nodes, ecov = load_data(cfg.data)
    return cfg, adj, build_spec(nodes, ecov, cfg.model, directed=True), \
        {k: np.asarray(v, float) for k, v in dict(ecov).items()}


def lolog_statistics(C, stars):
    labels = ["edges", "mutual", "gwesp_ITP"]
    labels += ["out_star2", "out_star3", "in_star2", "in_star3"] if stars else []
    labels += [f"edgecov_{k}" for k in ("use", "team", "floor", "metric")]

    def stats(A):
        A = np.asarray(A, float)
        sp = (A @ A).T[A == 1]                     # cyclic partners of i->j: k with j->k, k->i
        g = [A.sum(), np.sum(A * A.T) / 2, np.exp(AL) * np.sum(1 - (1 - np.exp(-AL)) ** sp)]
        if stars:
            od, idg = A.sum(1), A.sum(0)
            g += [comb(od, 2).sum(), comb(od, 3).sum(), comb(idg, 2).sum(), comb(idg, 3).sum()]
        g += [(A * C[k]).sum() for k in ("use", "team", "floor", "metric")]
        return np.array(g)
    return labels, stats


def bootstrap(cfg, adj, spec, stats):
    """The null of calibrate_band, step by step: logistic fit of the dyad-independent terms,
    then B independent-dyad graphs; returns the PAC and the LOLOG statistics of each graph."""
    n = adj.shape[0]
    di = [t for t in spec.terms if t.dyad_independent]
    base = np.zeros_like(adj); X, y = [], []
    for u in range(n):
        for v in range(n):
            if u != v:
                X.append([x for t in di for x in t.delta(base, u, v, 1)]); y.append(int(adj[u, v]))
    X = np.asarray(X, float); y = np.asarray(y)
    beta = minimize(lambda b: np.sum(np.logaddexp(0, X @ b) - y * (X @ b)), np.zeros(X.shape[1]),
                    jac=lambda b: X.T @ (1 / (1 + np.exp(-(X @ b))) - y), method="BFGS",
                    options={"gtol": 1e-10, "maxiter": 20000}).x
    P = np.zeros((n, n)); P[~np.eye(n, dtype=bool)] = X @ beta
    P = 1 / (1 + np.exp(-P)); np.fill_diagonal(P, 0)
    rng = np.random.default_rng(cfg.seed)
    B = int(cfg.pac.n_bootstrap)
    G_pac, G_lol = [], []
    for _ in range(B):
        Ab = (rng.uniform(size=(n, n)) < P).astype(np.int64); np.fill_diagonal(Ab, 0)
        G_pac.append(compute_stats(Ab, spec)); G_lol.append(stats(Ab))
    return np.array(G_pac), np.array(G_lol)


def band(G, delta):
    s = np.maximum(G.std(axis=0), 1e-6)
    d = np.linalg.norm((G - G.mean(axis=0)) / s, axis=1)
    return s, float(round(np.quantile(d, 1 - delta), 2))


def one_spec(k, run):
    S = SPECS[k]
    run = Path(run or S["run"]).resolve()
    cfg, adj, spec, C = load(S["config"])
    labels, stats = lolog_statistics(C, S["stars"])
    delta = float(cfg.pac.delta)

    G_pac, G_lol = bootstrap(cfg, adj, spec, stats)
    _, eps_pac = band(G_pac, delta)
    s_lol, eps_lol = band(G_lol, delta)
    eps_run = float(OmegaConf.load(run / "config_used.yaml").pac.epsilon)

    obs = stats(adj)
    nets = np.load(run / "arrays.npz", allow_pickle=True)["lolog_networks"]
    Z = (np.array([stats(A) for A in nets]) - obs) / s_lol
    d = np.linalg.norm(Z, axis=1)
    out = {
        "specification": k, "run": run.name,
        "description": "LOLOG evaluated on the statistics of its own model, band recalibrated "
                       "as calibrate_band does (scripts/lolog_own_metrics.py)",
        "statistics": labels,
        "observed": dict(zip(labels, obs.tolist())),
        "band": {"epsilon": eps_lol, "delta": delta, "scale": dict(zip(labels, s_lol.tolist())),
                 "n_bootstrap": int(cfg.pac.n_bootstrap), "seed": int(cfg.seed)},
        "check_epsilon_on_pac_statistics": {"recomputed": eps_pac, "saved_in_run": eps_run},
        "LOLOG": {"R": float(np.linalg.norm(Z.mean(axis=0))), "pi_A": float(np.mean(d <= eps_lol)),
                  "ES": float(d.mean() - pdist(Z).mean() / 2), "n_sim": int(len(Z)),
                  "mean_z": dict(zip(labels, np.round(Z.mean(axis=0), 3).tolist()))},
    }
    with open(run / "lolog_own_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    L = out["LOLOG"]
    print(f"Spec {k} ({run.name}): band check eps {eps_pac} (run {eps_run}) | LOLOG band eps {eps_lol}\n"
          f"  LOLOG on its own statistics: R = {L['R']:.3f}, pi(A) = {L['pi_A']:.3f}, ES = {L['ES']:.3f} "
          f"({L['n_sim']} networks) -> {run / 'lolog_own_metrics.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=int, choices=[2, 3], nargs="+", default=[2, 3])
    ap.add_argument("--run", default=None, help="run folder (only with a single --spec)")
    args = ap.parse_args()
    if args.run and len(args.spec) > 1:
        ap.error("--run needs a single --spec")
    for k in args.spec:
        one_spec(k, args.run)


if __name__ == "__main__":
    main()
