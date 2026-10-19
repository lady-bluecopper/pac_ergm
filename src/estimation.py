"""
estimation.py — PAC-MaxEnt estimation: warm start, band calibration, outer loop.

    compute_mple     MPLE warm start for theta (logistic regression on change-stats)
    calibrate_band   band radius epsilon and per-statistic scale, by parametric
                     bootstrap under the model's dyad-independent null
    run_pac          joint estimate of (theta, nu): theta by Robbins-Monro on the
                     moment gap g_obs - g_bar, nu by Robbins-Monro on the coverage gap
"""
import numpy as np
from joblib.externals.loky import get_reusable_executor

from .model import compute_stats, compute_delta_g
from .sampler import compute_norm, run_mcmc_inner

# ---------------------------------------------------------------------------
#  MPLE = logistic regression on the change-stats of the n(n-1) ordered pairs
# ---------------------------------------------------------------------------
def compute_mple(adj, spec):
    from sklearn.linear_model import LogisticRegression
    n = adj.shape[0]
    base = adj.copy()
    X, y = [], []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            cur = int(base[u, v])
            base[u, v] = 0                                  # conditions on the rest ("add" direction)
            X.append(compute_delta_g(base, u, v, spec))
            base[u, v] = cur
            y.append(cur)
    X = np.asarray(X); y = np.asarray(y)
    clf = LogisticRegression(fit_intercept=False, C=1e12, solver="lbfgs", max_iter=5000)
    clf.fit(X, y)
    theta = clf.coef_[0]
    print(f"MPLE theta = {dict(zip(spec.labels, np.round(theta, 4)))}")
    return theta

# ---------------------------------------------------------------------------
#  Update of theta (Robbins-Monro) and nu
# ---------------------------------------------------------------------------
def update_theta_rm(theta, g_obs, g_bar, samples_g, step_size):
    """Simple Robbins-Monro: theta += alpha * (g_obs - g_bar)."""
    return theta + step_size * (g_obs - g_bar)


def update_theta_rm_cov(theta, g_obs, g_bar, samples_g, step_size):
    """RM preconditioned by the covariance of g (useful with heterogeneous scales)."""
    residual = g_obs - g_bar
    k = len(theta)
    if samples_g.shape[0] > k + 1:
        cov = np.cov(samples_g.T)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov += 1e-6 * np.eye(k)
        try:
            update = np.linalg.solve(cov, residual)
        except np.linalg.LinAlgError:
            update = residual
    else:
        update = residual
    return theta + step_size * update


def make_theta_rm_diag(scale):
    """Diagonal RM (diagonal natural gradient): theta += step * (g_obs - g_bar) / scale^2.
       Stable with correlated / heterogeneously-scaled statistics."""
    s2 = np.asarray(scale, dtype=float) ** 2
    def update(theta, g_obs, g_bar, samples_g, step_size):
        return theta + step_size * (g_obs - g_bar) / s2
    return update

def make_theta_update(cfg, norm_scale):
    """Build the theta-update function from cfg.estimation.theta_update:
    'rm_diag' -> diagonal RM (needs norm_scale), 'rm' -> plain RM, 'rm_cov' -> covariance-preconditioned RM."""
    kind = str(cfg.estimation.theta_update)
    if kind == "rm_diag":
        if norm_scale is None:
            raise ValueError("theta_update='rm_diag' requires norm_scale (run the band calibration first)")
        return make_theta_rm_diag(norm_scale)
    if kind == "rm":
        return update_theta_rm
    if kind == "rm_cov":
        return update_theta_rm_cov
    raise ValueError(f"unknown theta_update '{kind}' (use 'rm_diag', 'rm', or 'rm_cov')")


#################################################
# BAND construction
#################################################
def calibrate_band(cfg, adj, g_obs, spec):
    """Band (A) calibration: parametric bootstrap under the DYAD-INDEPENDENT null
    built from the spec's dyad-independent terms. With `edges` alone this reduces
    to a homogeneous Bernoulli at the observed density; with nodematch/nodefactor/
    edgecov it is the covariate-aware null. Sets cfg.pac.norm_scale and cfg.pac.epsilon."""
    from sklearn.linear_model import LogisticRegression
    n = adj.shape[0]; k = len(spec); delta = float(cfg.pac.delta)
    B = int(cfg.pac.n_bootstrap)

    di = [t for t in spec.terms if t.dyad_independent]
    if not di:
        raise ValueError("no dyad-independent term in the model: cannot build the null")

    # 1) design matrix = change-stats of the dyad-independent terms.
    #    They ignore the rest of the graph, so the base graph is irrelevant.
    base = np.zeros_like(adj)
    X, y = [], []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            X.append([x for t in di for x in t.delta(base, u, v, 1)])
            y.append(int(adj[u, v]))
    X = np.asarray(X, dtype=float); y = np.asarray(y)

    # 2) null probability matrix P[u,v]
    beta = LogisticRegression(fit_intercept=False, C=1e12,
                              max_iter=5000, tol=1e-10).fit(X, y).coef_[0]
    L = X @ beta
    P, idx = np.zeros((n, n)), 0
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            P[u, v] = L[idx]; idx += 1
    P = 1.0 / (1.0 + np.exp(-P)); np.fill_diagonal(P, 0.0)

    # 3) parametric bootstrap under the null
    rng_b = np.random.default_rng(cfg.seed)
    gs = np.zeros((B, k))
    for b in range(B):
        Ab = (rng_b.uniform(size=(n, n)) < P).astype(np.int64)
        np.fill_diagonal(Ab, 0)
        gs[b] = compute_stats(Ab, spec)

    # 4) scale (per-stat std) and epsilon ((1-delta)-quantile of the fluctuation)
    scale    = np.maximum(gs.std(axis=0), 1e-6)
    mean_ref = gs.mean(axis=0)
    dists    = np.array([compute_norm(gs[b] - mean_ref, cfg.pac.norm, scale) for b in range(B)])
    eps_emp  = float(round(np.quantile(dists, 1 - delta), 2))

    cfg.pac.norm_scale = [float(x) for x in scale]
    cfg.pac.epsilon    = eps_emp
    return {"epsilon": eps_emp, "scale": scale, "mean_ref": mean_ref, "dists": dists,
            "p_hat": float(adj.sum() / (n * (n - 1)))}


# ---------------------------------------------------------------------------
# PAC outer loop: joint estimate of (theta, nu)
# ---------------------------------------------------------------------------
def run_pac(cfg, adj, g_obs, spec, theta_init=None, inner=None) -> tuple:
    """
    Joint estimate of (theta, nu) for PAC-MaxEnt, OFFICE version.
    Everything that defines the run is read from cfg (proposal / estimation / pac / control);
    theta_init is built from cfg (MPLE warm start) when not passed.
    Every chain restarts from the observed graph at each outer iteration.
    g_bar is aggregated with the median of the per-chain means; the coverage that drives
    nu with the median (default) or, control.coverage_agg = "mean", with the mean, so
    that a chain which has left the band counts in the nu update.
    The chains of one iteration are independent given (theta, nu), so they run in
    parallel on control.n_jobs processes (default 1 = one after the other). Each chain
    has its own random stream spawned from cfg.seed: the result does not depend on n_jobs.
    inner is the MCMC sampler of one chain: sampler.run_mcmc_inner (default, dense adj) or
    sparse.run_mcmc_inner_sparse (adj a scipy.sparse matrix, spec from build_sparse_spec).
    """
    inner = run_mcmc_inner if inner is None else inner
    seed_seq = np.random.SeedSequence(cfg.seed if cfg.seed is not None else None)
    stat_names = spec
    k = len(stat_names)

    # --- run definition (from cfg) ---
    proposal    = str(cfg.proposal.kind)
    p_nbhd      = float(cfg.proposal.p_nbhd)
    est         = cfg.estimation
    nu_step     = float(est.nu_step)
    nu_decay    = float(est.nu_decay)
    nu_max      = float(est.get("nu_max", 500.0))
    penalty     = str(est.penalty)
    theta_decay = float(est.get("theta_decay", 1.0))

    epsilon     = float(cfg.pac.epsilon)
    norm_type   = str(cfg.pac.norm)
    _ns         = cfg.pac.get("norm_scale", None)
    norm_scale  = None if _ns is None else np.asarray(list(_ns), dtype=float)
    target_coverage = 1.0 - float(cfg.pac.delta)
    tau = (epsilon / float(est.tau_factor)) if penalty == "logistic" else None

    n_iter      = int(cfg.control.n_iter)
    interval    = int(cfg.control.interval)
    M           = int(cfg.control.mcmc_samples)
    burnin      = int(cfg.control.burnin)
    step_size_0 = float(cfg.control.step_size)
    n_chains    = int(cfg.control.n_chains)
    n_jobs      = int(cfg.control.get("n_jobs", 1))
    executor    = get_reusable_executor(max_workers=n_jobs) if n_jobs > 1 else None
    coverage_agg = str(cfg.control.get("coverage_agg", "median"))   # "median" | "mean"
    if coverage_agg not in ("median", "mean"):
        raise ValueError(f"unknown control.coverage_agg '{coverage_agg}' (use 'median' or 'mean')")

    # --- warm start (MPLE), built from cfg when not provided ---
    if theta_init is None and str(est.get("warm_start", "none")) == "mple":
        theta_init = compute_mple(adj, spec)
    theta = np.zeros(k) if theta_init is None else np.asarray(theta_init, dtype=float)
    nu = float(est.nu_init)

    theta_update_fn = make_theta_update(cfg, norm_scale)

    history = {"theta": [theta.copy()], "nu": [nu], "g_bar": [], "coverage": [], "residual": [],
               "g_sd": [], "g_bar_chains": [], "coverage_chains": []}
    final_samples = (np.empty((0, k)), np.empty(0, dtype=np.int8))

    print(f"PAC | {k} stats: {stat_names}")
    print(f"g_obs = {np.round(g_obs, 2)} | epsilon={epsilon} delta={cfg.pac.delta} "
          f"norm={norm_type} n_chains={n_chains} n_jobs={n_jobs} coverage_agg={coverage_agg}")
    print(f"  penalty: {'LOGISTIC tau=%.4f' % tau if penalty == 'logistic' else 'hard 1_A'}")
    print("=" * 70)

    for t in range(n_iter):
        alpha_t = step_size_0 / (1.0 + theta_decay * t)
        eta_t = nu_step / (1.0 + nu_decay * t)
        g_bars, p_bars, all_g, all_ib = [], [], [], []
        chain_rngs = [np.random.default_rng(s) for s in seed_seq.spawn(n_chains)]
        chain_args = (adj, theta, nu, g_obs, epsilon, norm_type, M, spec)
        chain_kw = dict(burnin=burnin, proposal=proposal, p_nbhd=p_nbhd, scale=norm_scale,
                        interval=interval, penalty=penalty, tau=tau)
        if executor is None:
            chains = [inner(*chain_args, r, **chain_kw) for r in chain_rngs]
        else:
            futures = [executor.submit(inner, *chain_args, r, **chain_kw)
                       for r in chain_rngs]
            chains = [f.result() for f in futures]
        for sg, sib, _ in chains:
            g_bars.append(np.mean(sg, axis=0)); p_bars.append(float(np.mean(sib)))
            all_g.append(sg); all_ib.append(sib)

        g_bar = np.median(g_bars, axis=0)
        # coverage: the median ignores the chains that left the band until they are half
        # of them; the mean is the fraction of samples in A, i.e. the coverage itself.
        p_bar = float(np.mean(p_bars) if coverage_agg == "mean" else np.median(p_bars))
        samples_g = np.concatenate(all_g, axis=0); samples_ib = np.concatenate(all_ib, axis=0)

        if p_bar >= 0.5:      # invece di 0.5
            theta = theta_update_fn(theta, g_obs, g_bar, samples_g, alpha_t)

        nu = float(np.clip(nu + eta_t * (target_coverage - p_bar), 0.0, nu_max))

        # Residual in the SAME metric as the band: each statistic divided by its
        # calibration scale, so the number is comparable with epsilon and across models.
        residual = compute_norm(g_obs - g_bar, norm_type, norm_scale)
        history["theta"].append(theta.copy()); history["nu"].append(nu)
        history["g_bar"].append(g_bar.copy()); history["coverage"].append(p_bar)
        history["residual"].append(residual)
        # dispersion of this iteration: spread of the sampled statistics (pooled over
        # chains), and the per-chain means / coverages behind the two medians above.
        history["g_sd"].append(samples_g.std(axis=0))
        history["g_bar_chains"].append(np.array(g_bars))
        history["coverage_chains"].append(np.array(p_bars))
        final_samples = (samples_g, samples_ib)

        if cfg.verbose:
            print(f"iter {t+1:3d}/{n_iter} | "
                  + " ".join(f"theta[{s}]={theta[i]:+.3f}" for i, s in enumerate(stat_names))
                  + f" | residual={residual:.2f} | coverage={p_bar:.3f} | nu={nu:.4f}")

    polyak_window = int(cfg.control.get('polyak_window', 30))
    polyak_window = min(polyak_window, len(history['theta']))
    theta_polyak  = np.mean(np.array(history['theta'][-polyak_window:]), axis=0)
    print(f"Polyak avg (last {polyak_window}) = {np.round(theta_polyak, 4)} | nu = {nu:.4f}")
    return theta_polyak, nu, history, final_samples

def theta_std_errors(g_samples):
    """Statistical standard errors of theta.

    For an exponential family the Fisher information is the covariance of the
    sufficient statistics under the fitted model, so Var(theta) = Cov(g)^-1.
    This is the same quantity ergm calls the 'model' source of its standard
    errors; the Monte-Carlo source is estimated separately, from the spread of
    theta over the Polyak window.
    """
    C = np.cov(np.asarray(g_samples, dtype=float), rowvar=False)
    return np.sqrt(np.diag(np.linalg.inv(C)))