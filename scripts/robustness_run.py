"""
Repeat PAC-ERGM fits across random seeds from a single robustness config.

Usage
-----
    python scripts/robustness_run.py conf/robustness.yaml

The robustness config specifies:
  * the model specifications to run;
  * the number of independent repetitions;
  * optional stage-specific overrides (needed by the two-stage Spec. 1 run).

For each specification, the script loads the ordinary model config, resolves its
paths, computes the MPLE once, and fixes the band once before the repetitions.
If pac.epsilon and pac.norm_scale are already supplied, they are used as-is;
otherwise the band is calibrated once using the seed in the ordinary model
config. Only the estimation and GoF seeds change between repetitions.

Each run writes
    <out>/spec<k>_seed<s>.json
    <out>/spec<k>_seed<s>_history.npz
"""

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import (
    build_spec,
    calibrate_band,
    compute_mple,
    compute_stats,
    load_data,
    run_pac,
)
from src.gof import gof_sample_stats


def clone_cfg(cfg):
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(out, False)
    return out


def flatten_updates(d, prefix=""):
    """Accept either nested YAML mappings or dotted OmegaConf keys."""
    out = {}
    for key, value in dict(d or {}).items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_updates(value, full))
        else:
            out[full] = value
    return out


def with_updates(cfg, updates):
    out = clone_cfg(cfg)
    for key, value in flatten_updates(updates).items():
        OmegaConf.update(out, key, value, force_add=True)
    return out


def resolve_path(path, base_dir):
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def resolve_data_paths(cfg, config_path):
    """Resolve paths in cfg.data relative to the ordinary model config file."""
    base = config_path.parent
    cfg.data.edges_path = str(resolve_path(cfg.data.edges_path, base))
    if cfg.data.get("nodes_path"):
        cfg.data.nodes_path = str(resolve_path(cfg.data.nodes_path, base))
    if cfg.data.get("edgecov"):
        cfg.data.edgecov = {
            name: str(resolve_path(path, base))
            for name, path in dict(cfg.data.edgecov).items()
        }


def set_fixed_band(cfg, spec_cfg):
    """Use an explicitly supplied band, if present in the robustness config."""
    band = spec_cfg.get("band")
    if not band:
        return False
    if band.get("epsilon") is None or band.get("norm_scale") is None:
        raise ValueError("band requires both epsilon and norm_scale")
    cfg.pac.epsilon = float(band.epsilon)
    cfg.pac.norm_scale = [float(x) for x in band.norm_scale]
    return True


def ensure_band(cfg, adj, g_obs, spec, spec_cfg):
    """Fix one band for all repetitions of a specification."""
    if set_fixed_band(cfg, spec_cfg):
        source = "robustness config"
    elif cfg.pac.get("epsilon") is not None and cfg.pac.get("norm_scale") is not None:
        source = "model config"
    else:
        # Important: this happens once, before the random-seed repetitions.
        # The calibration uses cfg.seed from the ordinary model config.
        calibrate_band(cfg, adj, g_obs, spec)
        source = f"one calibration with base seed {cfg.seed}"

    scale = np.asarray(list(cfg.pac.norm_scale), dtype=float)
    eps = float(cfg.pac.epsilon)
    if len(scale) != len(g_obs):
        raise ValueError(
            f"norm_scale has length {len(scale)}, but the model has {len(g_obs)} statistics"
        )
    return eps, scale, source


def nu_estimate(nu_hist, W, rule=None):
    """Final nu summary. Default: median over the Polyak window."""
    nu_hist = np.asarray(nu_hist, dtype=float)
    rule = rule or {}
    method = str(rule.get("method", "median_window"))

    if method == "median_window":
        return float(np.median(nu_hist[-W:]))
    if method == "mean_window":
        return float(np.mean(nu_hist[-W:]))
    if method == "mean_last":
        n = int(rule.get("n", 20))
        return float(np.mean(nu_hist[-n:]))
    if method == "median_last":
        n = int(rule.get("n", 20))
        return float(np.median(nu_hist[-n:]))
    raise ValueError(
        f"unknown nu summary method '{method}'; use median_window, mean_window, "
        "mean_last, or median_last"
    )


def criteria(g_samples, g_obs, scale, max_n=5000):
    """R and energy score on standardized sufficient statistics."""
    Z = (np.asarray(g_samples, dtype=float) - g_obs) / scale
    Zs = Z
    if len(Z) > max_n:
        idx = np.random.default_rng(0).choice(len(Z), max_n, replace=False)
        Zs = Z[idx]
    return {
        "R": float(np.linalg.norm(Z.mean(axis=0))),
        "ES": float(np.linalg.norm(Zs, axis=1).mean() - pdist(Zs).mean() / 2),
    }


def stages_from_config(spec_cfg):
    stages = spec_cfg.get("stages")
    if not stages:
        return [OmegaConf.create({"name": "fit", "overrides": {}})]
    return list(stages)


def run_one(prepared, seed, out_dir, save_history=True, run_gof=True):
    spec_id = prepared["id"]
    name = prepared["name"]
    base_cfg = prepared["cfg"]
    adj = prepared["adj"]
    spec = prepared["spec"]
    labels = prepared["labels"]
    g_obs = prepared["g_obs"]
    scale = prepared["scale"]
    theta_mple = prepared["theta_mple"]
    spec_cfg = prepared["spec_cfg"]

    common = {"seed": int(seed), "gof.seed": int(seed)}
    stages = stages_from_config(spec_cfg)

    out = {
        "specification": spec_id,
        "name": name,
        "seed": int(seed),
        "labels": labels,
        "theta_start": theta_mple.tolist(),
        "band": {
            "epsilon": float(base_cfg.pac.epsilon),
            "norm_scale": scale.tolist(),
            "delta": float(base_cfg.pac.delta),
        },
        "times_s": {},
        "stages": [],
    }
    hist_save = {}
    theta_start = theta_mple.copy()
    final_cfg = None
    t_all = time.perf_counter()

    for stage_idx, stage in enumerate(stages):
        stage_name = str(stage.get("name", f"stage{stage_idx + 1}"))
        updates = {}
        updates.update(common)
        updates.update(flatten_updates(stage.get("overrides", {})))
        c = with_updates(base_cfg, updates)

        start_from = str(
            stage.get("start_from", "mple" if stage_idx == 0 else "previous_last")
        )

        if start_from == "mple":
            theta_stage = theta_mple.copy()
        elif start_from == "previous_last":
            theta_stage = theta_start.copy()
        elif start_from == "config":
            theta_stage = np.asarray(list(c.estimation.theta_init), dtype=float)
        else:
            raise ValueError(
                f"{name}/{stage_name}: unknown start_from '{start_from}' "
                "(use mple, previous_last, or config)"
            )

        print(
            f"\n=== {name} | seed {seed} | {stage_name}: "
            f"{c.control.n_iter} iterations, burn-in {c.control.burnin}, "
            f"W={c.control.polyak_window}, nu_init={c.estimation.nu_init}",
            flush=True,
        )

        t0 = time.perf_counter()
        theta_hat, _, hist, _ = run_pac(
            c,
            adj,
            g_obs,
            spec,
            theta_init=theta_stage,
        )
        t_fit = time.perf_counter() - t0

        W = int(c.control.polyak_window)
        th = np.asarray(hist["theta"], dtype=float)
        nu_h = np.asarray(hist["nu"], dtype=float)
        cov_h = np.asarray(hist["coverage"], dtype=float)

        rule = stage.get("nu_summary", spec_cfg.get("nu_summary", {}))
        nu_hat = nu_estimate(nu_h, W, rule)

        stage_out = {
            "name": stage_name,
            "n_iter": int(c.control.n_iter),
            "W": W,
            "theta_hat": np.asarray(theta_hat, dtype=float).tolist(),
            "nu_hat": nu_hat,
            "coverage_window": float(cov_h[-W:].mean()),
            "t_fit": t_fit,
            "t_per_iteration": t_fit / int(c.control.n_iter),
        }
        out["stages"].append(stage_out)

        tag = stage_name.replace(" ", "")
        hist_save.update(
            {
                f"{tag}_theta": th,
                f"{tag}_nu": nu_h,
                f"{tag}_coverage": cov_h,
            }
        )

        print(
            f"--- {stage_name}: {t_fit:.0f}s | "
            f"window concentration {stage_out['coverage_window']:.3f}",
            flush=True,
        )

        theta_start = th[-1].copy()
        if stage.get("round_start") is not None:
            theta_start = np.round(theta_start, int(stage.round_start))

        final_cfg = c

    last = out["stages"][-1]
    out.update(
        theta_hat=last["theta_hat"],
        nu_hat=last["nu_hat"],
        coverage_window=last["coverage_window"],
    )

    if run_gof:
        t0 = time.perf_counter()
        g_gof, gof_cov = gof_sample_stats(
            final_cfg,
            adj,
            g_obs,
            spec,
            np.asarray(out["theta_hat"], dtype=float),
            float(out["nu_hat"]),
        )
        out["times_s"]["gof"] = time.perf_counter() - t0

        out["gof"] = {
            "n_samples": int(len(g_gof)),
            "coverage": float(gof_cov),
            **criteria(g_gof, g_obs, scale),
        }

        hist_save["gof_samples"] = g_gof

        print(
            f"--- GoF: concentration {gof_cov:.3f} | "
            f"R {out['gof']['R']:.3f} | "
            f"ES {out['gof']['ES']:.3f} "
            f"({out['times_s']['gof']:.0f}s)",
            flush=True,
        )

    out["times_s"]["total"] = time.perf_counter() - t_all

    stem = out_dir / f"spec{spec_id}_seed{seed}"

    with open(f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if save_history:
        np.savez_compressed(
            f"{stem}_history.npz",
            **hist_save,
        )

    print(
        f"theta_hat = "
        f"{dict(zip(labels, np.round(out['theta_hat'], 4)))} | "
        f"nu_hat = {out['nu_hat']:.3f}\n"
        f"written {stem}.json",
        flush=True,
    )

    return out


def prepare_spec(spec_cfg, robustness_cfg_path):
    spec_id = int(spec_cfg.id)
    name = str(spec_cfg.get("name", f"spec{spec_id}"))

    config_path = resolve_path(
        spec_cfg.config,
        robustness_cfg_path.parent,
    )

    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    resolve_data_paths(cfg, config_path)

    cfg = with_updates(
        cfg,
        spec_cfg.get("overrides", {}),
    )

    adj, nodes, ecov = load_data(cfg.data)
    spec = build_spec(
        nodes,
        ecov,
        cfg.model,
        directed=bool(cfg.data.is_directed),
    )

    labels = list(spec.labels)
    g_obs = compute_stats(adj, spec)

    theta_mple = np.asarray(
        compute_mple(adj, spec),
        dtype=float,
    )

    _, scale, _ = ensure_band(
        cfg,
        adj,
        g_obs,
        spec,
        spec_cfg,
    )

    return {
        "id": spec_id,
        "name": name,
        "cfg": cfg,
        "adj": adj,
        "spec": spec,
        "labels": labels,
        "g_obs": np.asarray(g_obs, dtype=float),
        "scale": scale,
        "theta_mple": theta_mple,
        "spec_cfg": spec_cfg,
        "config_path": config_path,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="robustness YAML config",)

    args = ap.parse_args()

    robustness_cfg_path = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(robustness_cfg_path)

    out_dir = resolve_path(
        cfg.get("out_dir", "../output/robustness"),
        robustness_cfg_path.parent,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = list(cfg.specifications)
    run_gof = bool(cfg.get("run_gof", True))
    save_history = bool(cfg.get("save_history", True))
    seeds = np.arange(cfg.n_tests)

    for spec_cfg in specs:
        prepared = prepare_spec(spec_cfg, robustness_cfg_path)
        for seed in seeds:
            run_one(
                prepared,
                seed,
                out_dir,
                save_history=save_history,
                run_gof=run_gof,
            )


if __name__ == "__main__":
    main()
