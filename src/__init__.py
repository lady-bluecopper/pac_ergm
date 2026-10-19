"""
pac — PAC-MaxEnt for network data.

    model        the model: network loading, statistics registry, spec
    sampler      MCMC sampler for the PAC-MaxEnt distribution
    estimation   MPLE warm start, band calibration, PAC outer loop
    gof          goodness of fit

A typical run:

    from src.pac import load_data, build_spec, compute_stats, calibrate_band, run_pac

    adj, nodes, ecov = load_data(cfg.data)
    spec  = build_spec(nodes, ecov, cfg.model, directed=bool(cfg.data.is_directed))
    g_obs = compute_stats(adj, spec)
    calibrate_band(cfg, adj, g_obs, spec)
    theta, nu, history, _ = run_pac(cfg, adj, g_obs, spec)

To add a statistic, subclass StatTerm and decorate it with @register("name").
"""
from .model import (
    REGISTRY, register, StatTerm,
    load_data, build_spec, compute_stats, compute_delta_g,
)
from .sampler import run_mcmc_inner
from .estimation import compute_mple, calibrate_band, run_pac, theta_std_errors
from .gof import (
    gof_sample_stats, gof_sample_networks, esp_distribution
)