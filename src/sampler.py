"""
sampler.py — MCMC sampler for the PAC-MaxEnt distribution.

The target is  p(G) ∝ exp( theta · g(G) + nu · s(G) ),  where s(G) is the membership
of G in the band  A = { G : ||g(G) - g_obs|| <= epsilon }  — the hard indicator 1_A,
or its smooth logistic version, used during estimation to help mixing.
"""
import numpy as np

from .model import compute_stats, compute_delta_g

# ---------------------------------------------------------------------------
# PURE helpers for the PAC band 
# ---------------------------------------------------------------------------
def compute_norm(diff, norm, scale=None):
    d = np.asarray(diff, dtype=float)
    if scale is not None:
        d = d / np.asarray(scale, dtype=float)
    if norm == "l2":     return float(np.linalg.norm(d, ord=2))
    elif norm == "l1":   return float(np.linalg.norm(d, ord=1))
    elif norm == "linf": return float(np.max(np.abs(d)))
    else: raise ValueError(f"Unknown norm '{norm}'. Use 'l2','l1','linf'.")

def _sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)

def logistic_membership(d, epsilon, tau):
    return _sigmoid((epsilon - d) / tau)

# ---------------------------------------------------------------------------
#    DIRECTED sampler on the PAC distribution.
# ---------------------------------------------------------------------------
def _tnt_log_h(n_edges, n_dyads, adding):
    """log q(reverse)/q(forward) for the TNT proposal, including the empty/full cases."""
    def p_tie(m):
        if m == n_dyads:
            return 1.0
        if m == 0:
            return 0.0
        return 0.5

    def p_nonedge(m):
        if m == 0:
            return 1.0
        if m == n_dyads:
            return 0.0
        return 0.5

    m = int(n_edges)
    if adding:
        q_fwd = p_nonedge(m) / (n_dyads - m)
        q_rev = p_tie(m + 1) / (m + 1)
    else:
        q_fwd = p_tie(m) / m
        q_rev = p_nonedge(m - 1) / (n_dyads - m + 1)
    return float(np.log(q_rev) - np.log(q_fwd))


def _neighborhood_log_h(adj, u, v):
    """
    log q(reverse)/q(forward) for the transitive branch i->k->j.

    Toggling u->v does not change the paths u->k->v (with no loops), but
    it changes the denominator outdeg(u). So the ratio is d_out(u) before /
    d_out(u) after.
    """
    d_before = int(adj[u].sum())
    d_after = d_before + (1 if adj[u, v] == 0 else -1)
    if d_before <= 0 or d_after <= 0:
        raise ValueError("neighborhood proposal with invalid out-degree")
    return float(np.log(d_before) - np.log(d_after))


def run_mcmc_inner(
    adj_init, theta, nu, g_obs, epsilon, norm, n_samples,
    spec, rng, burnin=0, proposal="tnt", p_nbhd=0.5, scale=None,
    interval=1, penalty="hard", tau=None,
):
    if interval < 1:
        raise ValueError("interval must be >= 1")
    adj = adj_init.copy()
    n   = adj.shape[0]
    D   = n * (n - 1)                                          # ORDERED pairs

    use_logistic = (penalty == "logistic")
    if use_logistic and (tau is None or tau <= 0):
        raise ValueError("penalty='logistic' richiede tau>0")

    g_current       = compute_stats(adj, spec)
    d_current       = compute_norm(g_current - g_obs, norm, scale)
    in_band_current = int(d_current <= epsilon)
    s_current       = logistic_membership(d_current, epsilon, tau) if use_logistic else 0.0

    k = len(spec)
    samples_g       = np.zeros((n_samples, k))
    samples_in_band = np.zeros(n_samples, dtype=np.int8)

    if proposal in ("tnt", "neighborhood"):
        ii, jj = np.nonzero(adj)
        edges  = [(int(a), int(b)) for a, b in zip(ii, jj)]
        pos    = {e: t for t, e in enumerate(edges)}

    def rand_ordered_nonedge():
        while True:
            i = int(rng.integers(0, n)); j = int(rng.integers(0, n))
            if i != j and adj[i, j] == 0:
                return i, j

    def record(step):
        if step >= burnin:
            kk = step - burnin + 1
            if kk % interval == 0:
                m = kk // interval - 1
                samples_g[m]       = g_current
                samples_in_band[m] = in_band_current

    total = burnin + interval * n_samples
    for step in range(total):
        null_move = False

        if proposal == "tnt":
            n_e = len(edges)
            tie = (rng.uniform() < 0.5 and n_e > 0) or (n_e == D)
            if tie:
                i, j = edges[int(rng.integers(0, n_e))]
            else:
                i, j = rand_ordered_nonedge()
            log_h = _tnt_log_h(n_e, D, adding=(adj[i, j] == 0))

        elif proposal == "neighborhood":                       # directed transitive move i->k->j
            n_e = len(edges)
            if rng.uniform() < p_nbhd:
                i = int(rng.integers(0, n))
                outs_i = np.nonzero(adj[i])[0]
                if outs_i.size == 0:
                    null_move = True
                else:
                    kk_ = int(outs_i[rng.integers(0, outs_i.size)])
                    outs_k = np.nonzero(adj[kk_])[0]
                    outs_k = outs_k[outs_k != i]               # j != i
                    if outs_k.size == 0:
                        null_move = True
                    else:
                        j = int(outs_k[rng.integers(0, outs_k.size)])
                if not null_move:
                    log_h = _neighborhood_log_h(adj, i, j)
            else:
                tie = (rng.uniform() < 0.5 and n_e > 0) or (n_e == D)
                if tie:
                    i, j = edges[int(rng.integers(0, n_e))]
                else:
                    i, j = rand_ordered_nonedge()
                log_h = _tnt_log_h(n_e, D, adding=(adj[i, j] == 0))
        else:                                                  # uniform over an ordered pair
            i = int(rng.integers(0, n)); j = int(rng.integers(0, n))
            if i == j:
                null_move = True
            log_h = 0.0

        if null_move:
            record(step); continue

        delta_g      = compute_delta_g(adj, i, j, spec)
        g_proposed   = g_current + delta_g
        d_prop       = compute_norm(g_proposed - g_obs, norm, scale)
        in_band_prop = int(d_prop <= epsilon)

        if use_logistic:
            s_prop       = logistic_membership(d_prop, epsilon, tau)
            penalty_term = nu * (s_prop - s_current)
        else:
            penalty_term = nu * (in_band_prop - in_band_current)

        log_r = float(theta @ delta_g) + penalty_term + log_h
        if np.log(rng.uniform()) < log_r:
            adding = (adj[i, j] == 0)
            adj[i, j] = 1 - adj[i, j]                          # DIRECTED: only [i,j]
            g_current, d_current, in_band_current = g_proposed, d_prop, in_band_prop
            if use_logistic:
                s_current = s_prop
            if proposal in ("tnt", "neighborhood"):
                if adding:
                    pos[(i, j)] = len(edges); edges.append((i, j))
                else:
                    t = pos.pop((i, j)); last = edges.pop()
                    if t < len(edges):
                        edges[t] = last; pos[last] = t

        record(step)

    return samples_g, samples_in_band, adj
