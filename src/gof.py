"""
gof.py — goodness of fit of an estimated PAC-MaxEnt model.

    gof_sample_stats      sample the model's own statistics (in-model check)
    gof_sample_networks   sample whole networks, for out-of-model checks
    plot_stat_gof         observed statistics against their sampled distributions
    plot_degree_esp_gof   in-degree, out-degree and ESP distributions:
                          observed network against the simulated ones
"""
import numpy as np

from .model import _esp_S
from .sampler import run_mcmc_inner

# ---------------------------------------------------------------------------
#  Goodness of fit: sampling + plots (read cfg.gof)
# ---------------------------------------------------------------------------
def gof_sample_stats(cfg, adj, g_obs, spec, theta, nu):
    """Sample the sufficient statistics from the estimated model (cfg.gof), hard band,
    and return (g_samples, coverage)."""
    rng = np.random.default_rng(int(cfg.gof.seed))
    g_samples, in_band, _ = run_mcmc_inner(
        adj, theta, nu, g_obs, float(cfg.pac.epsilon), str(cfg.pac.norm),
        n_samples=int(cfg.gof.n_samples), spec=spec, rng=rng,
        burnin=int(cfg.gof.burnin), proposal=str(cfg.proposal.kind),
        p_nbhd=float(cfg.proposal.p_nbhd), scale=cfg.pac.norm_scale,
        interval=int(cfg.gof.interval), penalty=str(cfg.gof.penalty),
    )
    return g_samples, float(np.mean(in_band))


def gof_sample_networks(cfg, adj, g_obs, spec, theta, nu):
    """Sample cfg.gof.n_networks networks from the estimated model (one per call,
    restarting from the observed graph) for the degree/ESP GoF."""
    rng = np.random.default_rng(int(cfg.gof.net_seed))
    nets = []
    for _ in range(int(cfg.gof.n_networks)):
        _, _, A = run_mcmc_inner(
            adj, theta, nu, g_obs, float(cfg.pac.epsilon), str(cfg.pac.norm),
            n_samples=1, spec=spec, rng=rng, burnin=int(cfg.gof.net_burnin),
            proposal=str(cfg.proposal.kind), p_nbhd=float(cfg.proposal.p_nbhd),
            scale=cfg.pac.norm_scale, interval=int(cfg.gof.net_gap), penalty=str(cfg.gof.penalty),
        )
        nets.append(A.copy())
    return nets


def esp_distribution(A, esp_type="OTP", max_sp=None):
    """ESP distribution: number of edges with exactly s shared partners."""
    S = _esp_S(A, esp_type)
    r, c = np.nonzero(A)
    vals = S[r, c].astype(int)
    if max_sp is None:
        max_sp = int(vals.max()) if vals.size else 0
    return np.bincount(vals, minlength=max_sp + 1)[:max_sp + 1]
