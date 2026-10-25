# BaCon ERGM

Code for **BaCon ERGM** (Band-Concentrated Exponential Random Graph Models), a chance-constrained maximum-entropy extension of ERGMs designed to keep the observed network typical under the fitted model.

The repository contains the three experimental specifications used in the paper, the BaCon estimation and evaluation pipeline, the standard ERGM and LOLOG baselines, and the sensitivity/scalability experiments.

## Repository structure

```text
conf/
    config_spec_1.yaml
    config_spec_2.yaml
    config_spec_3.yaml

src/
    model.py          network loading, statistics, model specification
    sampler.py        dense MCMC sampler
    estimation.py     MPLE, band calibration, Robbins--Monro estimation
    evaluation.py     post-fit GoF, metrics, standard errors, serialization
    gof.py            GoF helpers
    sparse.py         sparse implementation used by the scalability experiment

scripts/
    pac_ergm_run.py               main BaCon ERGM runner
    epsilon_sensitivity_run.py    sensitivity to the band radius
    scalability_run.py            sparse scalability experiment
    robustness_run.py             repeated-seed robustness experiment
    lolog_office_baseline.R       LOLOG baseline for Specification 2
    lolog_office_stars_baseline.R LOLOG baseline for Specification 3
    compute_lolog_metrics.py      standalone/legacy LOLOG metric checker

notebooks/
    ERGM_estimation.ipynb    standard ERGM baselines
    Plots.ipynb              paper figures and tables from saved outputs
```

## Experimental specifications

The three configs in `conf/` define the models and all run settings.

- **Specification 1 — `ukFaculty`**: `edges + nodematch(GroupC) + nodefactor(GroupC, base=3) + gwesp(0.25)`.
- **Specification 2 — `office_uni2005`**: `edges + mutual + gwesp(0.5) + edgecov(use) + edgecov(team) + edgecov(floor) + edgecov(metric)`.
- **Specification 3 — `office_uni2005` with degree terms**: Specification 2 + `gwodeg(0.5) + gwideg(0.5)`.

## Python setup

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## BaCon ERGM workflow

Run a specification from the repository root by passing its config explicitly:

```bash
python scripts/pac_ergm_run.py --config-name config_spec_1
python scripts/pac_ergm_run.py --config-name config_spec_2
python scripts/pac_ergm_run.py --config-name config_spec_3
```

Hydra-style overrides can follow the config name without modifying the YAML file, for example:

```bash
python scripts/pac_ergm_run.py -cn config_spec_2 control.n_iter=20 control.burnin=10000
```

For each run, `pac_ergm_run.py`:

1. loads the data and model from the selected config;
2. calibrates the band, including `epsilon` and the statistic-specific scale vector;
3. runs the joint Robbins--Monro estimation of `theta` and `nu`;
4. reports `theta` as the Polyak--Ruppert average over the final window and `nu` as the median over that window;
5. calls `src/evaluation.py` to run the hard-band GoF and save the results.

The three specifications write to fixed output directories:

```text
output/ukFaculty_spec1/
output/office_uni2005_spec2/
output/office_uni2005_spec3/
```

The main files produced there are:

```text
config_used.yaml        exact resolved configuration used by the run
history.csv             full theta, nu, coverage, residual and g_bar trajectories
results_summary.json    final estimates and run summary
thesis_metrics.json     in-model evaluation metrics
coefficients.csv        BaCon estimates and standard errors, plus ERGM values when available
arrays.npz              raw trajectories, GoF samples/networks and baseline arrays
nodes_used.csv          node table used by the fit
```

For Specifications 2 and 3, when the LOLOG baseline is available, the run also writes:

```text
lolog_coefficients.csv
lolog_own_metrics.json
```

## Standard ERGM baseline

The standard ERGM comparison is kept in R through one notebook:

```text
notebooks/ERGM_estimation.ipynb
```

Set `spec_no` to `1`, `2`, or `3`. 
The notebook reads the corresponding BaCon config for the data/model definition and writes to:

```text
output/run_ERGM_spec_1/
output/run_ERGM_spec_2/
output/run_ERGM_spec_3/
```

`src/evaluation.py` loads these files automatically when they are present. For a complete BaCon-vs-ERGM comparison in the saved BaCon artifacts, run the relevant ERGM baseline before `pac_ergm_run.py`.

The ERGM notebook requires the R packages `ergm` and `network`.

## LOLOG baseline

Specifications 2 and 3 are also compared with LOLOG:

- `scripts/lolog_office_baseline.R` — Specification 2;
- `scripts/lolog_office_stars_baseline.R` — Specification 3.

During BaCon post-estimation evaluation, `src/evaluation.py` checks for the corresponding LOLOG output. If it is missing and `Rscript` is available, the appropriate R script is run automatically. If R/LOLOG is unavailable, the BaCon run is still saved without the LOLOG comparison.

The reported LOLOG in-model metrics are computed on the sufficient statistics of the LOLOG specification itself. In particular, LOLOG uses cyclic directed GWESP, and Specification 3 uses in/out stars rather than the BaCon/ERGM geometrically weighted degree terms.

The LOLOG scripts require the R packages `lolog` and `network`.


## Epsilon-sensitivity experiment

The band-radius sensitivity analysis starts from an already fitted BaCon run. The paper experiment uses Specification 3:

```bash
python scripts/epsilon_sensitivity_run.py \
    --base-run output/office_uni2005_spec3
```

The default experiment re-estimates the model for radius multipliers around the calibrated radius and writes a timestamped directory under:

```text
output/epsilon_sensitivity/
```

with `summary.csv`, per-radius JSON/NPZ files, and the fitted `theta` values.

## Scalability experiment

The sparse scalability experiment uses induced subgraphs of the Slashdot network:

```bash
python scripts/scalability_run.py
```

Results are written under `output/scalability/run_<timestamp>/`. The runner records fitting and simulation times together with the network size and fitted BaCon quantities.
