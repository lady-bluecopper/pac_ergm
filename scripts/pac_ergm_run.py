"""Command-line runner for PAC-ERGM estimation.

Usage:

    python scripts/pac_ergm_run.py --config-name config_spec_2

Hydra-style overrides can follow the config name, e.g.::

    python scripts/pac_ergm_run.py -cn config_spec_2 control.n_iter=20

Each specification writes to the canonical output folder consumed directly by
``notebooks/Plots.ipynb`` (for example ``output/office_uni2005_spec2``), with
the exact configuration, full trajectories, GoF samples/networks, metrics,
coefficient table and raw arrays.
"""
import re
import sys
from pathlib import Path

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_data, build_spec, compute_stats, calibrate_band, run_pac
from src.evaluation import evaluate_and_save


def resolve(p):
    """Config paths are written relative to notebooks/ -> make them absolute."""
    p = Path(p)
    return str(p if p.is_absolute() else (ROOT / "notebooks" / p).resolve())


def specification_from_config(config_name):
    """config_spec_1 -> 1, etc."""
    m = re.search(r"spec[_-]?(\d+)$", Path(config_name).stem)
    if not m:
        raise SystemExit(
            "Cannot infer the specification number from config name "
            f"'{config_name}'. Expected a name such as config_spec_1."
        )
    return int(m.group(1))


def main():
    argv = sys.argv[1:]
    config_name = None
    overrides = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--config-name", "-cn"):
            if i + 1 >= len(argv):
                raise SystemExit(f"{argv[i]} requires a configuration name")
            config_name = argv[i + 1]
            i += 2
        else:
            overrides.append(argv[i])
            i += 1

    if config_name is None:
        raise SystemExit("Missing required argument: --config-name <config_name>")
    specification = specification_from_config(config_name)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3")
    cfg = compose(config_name=config_name, overrides=overrides)
    cfg.data.edges_path = resolve(cfg.data.edges_path)
    cfg.data.nodes_path = resolve(cfg.data.nodes_path)
    if cfg.data.get("edgecov", None):
        cfg.data.edgecov = {k: resolve(v) for k, v in dict(cfg.data.edgecov).items()}
    OmegaConf.set_struct(cfg, False)  # band calibration adds epsilon / norm_scale

    adj, nodes, ecov = load_data(cfg.data)
    spec = build_spec(nodes, ecov, cfg.model, directed=bool(cfg.data.is_directed))
    g_obs = compute_stats(adj, spec)

    calibration = calibrate_band(cfg, adj, g_obs, spec)

    # Specifications 1 and 3 use an explicit starting value saved in the config.
    # When theta_init is absent, run_pac follows cfg.estimation.warm_start (e.g. MPLE).
    theta_init_cfg = cfg.estimation.get("theta_init", None)
    theta_init = (
        None if theta_init_cfg is None
        else np.asarray(theta_init_cfg, dtype=float)
    )
    theta_final, _, history, final_samples = run_pac(
        cfg, adj, g_obs, spec, theta_init=theta_init
    )

    # theta_final is the Polyak--Ruppert mean returned by run_pac.
    # For nu, the paper reports the median over the same final window.
    W = min(int(cfg.control.get("polyak_window", 30)), len(history["nu"]))
    nu_final = float(np.median(np.asarray(history["nu"], dtype=float)[-W:]))

    run_names = {
        1: "ukFaculty_spec1",
        2: "office_uni2005_spec2",
        3: "office_uni2005_spec3",
    }
    if specification not in run_names:
        raise SystemExit(f"No canonical output folder is defined for specification {specification}")
    run_dir = Path(resolve(cfg.out_dir)) / run_names[specification]
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config_used.yaml")

    evaluate_and_save(
        cfg=cfg,
        adj=adj,
        nodes=nodes,
        edgecov=ecov,
        spec=spec,
        g_obs=g_obs,
        theta_final=theta_final,
        nu_final=nu_final,
        history=history,
        final_samples=final_samples,
        run_dir=run_dir,
        project_root=ROOT,
        specification=specification,
        calibration_info=calibration,
    )

    print("\nDONE ->", run_dir)


if __name__ == "__main__":
    main()
