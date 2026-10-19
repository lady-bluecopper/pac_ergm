"""
sparse.py — PAC-MaxEnt on large sparse networks, with scipy.sparse.

The same model and the same MCMC as model.py + sampler.py, written on a sparse
representation, so that memory and the cost of one MCMC step do not grow with N^2:

    csr_matrix         operations on the whole network: statistics, reference model
    SparseDigraph      the state of a chain: two lil_matrix, A and its transpose, whose
                       internal row lists are the sorted out- and in-neighbours of each
                       node, read and updated in place (one toggle = a few list edits)

Only the terms of the scalability experiment: edges, mutual, gwesp (OTP), gwodeg, gwideg.
On the same network, statistics, change statistics and chains coincide with the dense
version (tests/test_sparse.py): run_mcmc_inner_sparse draws the same random numbers in
the same order as sampler.run_mcmc_inner.

    from src.pac.sparse import (as_adjacency, build_sparse_spec, compute_stats_sparse,
                                calibrate_band_sparse, bernoulli_theta, run_mcmc_inner_sparse)

    A     = as_adjacency(edges_csr)                 # binary csr, no self-loops
    spec  = build_sparse_spec("edges + mutual + gwesp(0.5)", A.shape[0])
    g_obs = compute_stats_sparse(A, spec)
    calibrate_band_sparse(cfg, A, g_obs, spec)
    run_pac(cfg, A, g_obs, spec, theta_init=bernoulli_theta(A, spec), inner=run_mcmc_inner_sparse)
"""
import bisect

import numpy as np
import scipy.sparse as sp

from .model import StatSpec, StatTerm, _h, _parse_term
from .sampler import compute_norm, logistic_membership, _tnt_log_h


# ---------------------------------------------------------------------------
#   Sparse adjacency: whole-network form (csr) and chain state (two lil)
# ---------------------------------------------------------------------------
def as_adjacency(A):
    """Any 0/1 adjacency (dense array or scipy.sparse) -> binary csr_matrix, no self-loops."""
    C = sp.coo_matrix(A)
    keep = (C.data != 0) & (C.row != C.col)
    M = sp.csr_matrix((np.ones(int(keep.sum()), dtype=np.int8), (C.row[keep], C.col[keep])),
                      shape=C.shape)
    M.sum_duplicates()
    M.data[:] = 1
    return M


def _common(a, b):
    """Sorted common elements of two sorted lists: scan the shorter, bisect the longer."""
    if len(a) > len(b):
        a, b = b, a
    out, nb = [], len(b)
    for x in a:
        k = bisect.bisect_left(b, x)
        if k < nb and b[k] == x:
            out.append(x)
    return out


def _n_common(a, b):
    """Number of common elements of two sorted lists."""
    if len(a) > len(b):
        a, b = b, a
    cnt, nb = 0, len(b)
    for x in a:
        k = bisect.bisect_left(b, x)
        if k < nb and b[k] == x:
            cnt += 1
    return cnt


class SparseDigraph:
    """Directed graph as two lil_matrix: A (row u = out-neighbours of u) and A^T
    (row v = in-neighbours of v). Rows are sorted lists, edited in place on a toggle."""

    def __init__(self, A):
        A = A.tocsr() if isinstance(A, SparseDigraph) else as_adjacency(A)
        self.n = A.shape[0]
        self.A, self.AT = A.tolil(), A.T.tolil()
        self.out, self.inn = self.A.rows, self.AT.rows

    @staticmethod
    def _flip(M, a, b):
        row, dat = M.rows[a], M.data[a]
        k = bisect.bisect_left(row, b)
        if k < len(row) and row[k] == b:
            row.pop(k); dat.pop(k)
        else:
            row.insert(k, b); dat.insert(k, 1)

    def has(self, u, v):
        row = self.out[u]
        k = bisect.bisect_left(row, v)
        return k < len(row) and row[k] == v

    def toggle(self, u, v):
        self._flip(self.A, u, v)
        self._flip(self.AT, v, u)

    def outdeg(self, u):
        return len(self.out[u])

    def indeg(self, v):
        return len(self.inn[v])

    def edges(self):
        """Edges in row-major order, the order of np.nonzero on the dense matrix."""
        return [(u, v) for u in range(self.n) for v in self.out[u]]

    def tocsr(self):
        return self.A.tocsr()


def _csr(A):
    return A.tocsr() if isinstance(A, SparseDigraph) else as_adjacency(A)


# ---------------------------------------------------------------------------
#   STATISTICS: value on a csr matrix, change-statistic on a SparseDigraph
# ---------------------------------------------------------------------------
SPARSE_REGISTRY = {}


def register_sparse(name):
    def deco(cls):
        cls.name = name
        SPARSE_REGISTRY[name] = cls
        return cls
    return deco


@register_sparse("edges")
class SparseEdges(StatTerm):
    dyad_independent = True

    def value(self, A):
        return [float(A.nnz)]

    def delta(self, G, u, v, sign):
        return [float(sign)]


@register_sparse("mutual")
class SparseMutual(StatTerm):
    def value(self, A):
        return [0.5 * float(A.multiply(A.T).sum())]

    def delta(self, G, u, v, sign):
        return [float(sign) * (1.0 if G.has(v, u) else 0.0)]


@register_sparse("gwesp")
class SparseGwesp(StatTerm):
    def setup(self):
        self.decay = self._param("decay", pos=0, cast=float)
        if self.decay is None:
            raise ValueError("gwesp needs a decay: write e.g. gwesp(0.5)")
        self.esp_type = self._param("esp_type", pos=1, default="OTP", cast=str)
        if self.esp_type != "OTP":
            raise ValueError("the sparse gwesp is implemented for OTP only")

    def value(self, A):
        A32 = A.astype(np.int32)
        S = (A32 @ A32).multiply(A32).tocsr()          # shared partners, on the edges only
        return [float(np.sum(_h(S.data.astype(float), self.decay)))]

    def delta(self, G, u, v, sign):
        return [_gwesp_delta_otp_sparse(G, u, v, self.decay, sign)]


class _SparseGWDegree(StatTerm):
    def setup(self):
        self.decay = self._param("decay", pos=0, cast=float)
        if self.decay is None:
            raise ValueError(f"{self.name} needs a decay: write e.g. {self.name}(0.5)")

    def _degrees(self, A):
        raise NotImplementedError

    def _toggled_degree(self, G, u, v):
        raise NotImplementedError

    def value(self, A):
        return [float(np.sum(_h(self._degrees(A).astype(float), self.decay)))]

    def delta(self, G, u, v, sign):
        d = float(self._toggled_degree(G, u, v))
        return [_h(d + sign, self.decay) - _h(d, self.decay)]


@register_sparse("gwodeg")
class SparseGWOutDegree(_SparseGWDegree):
    def _degrees(self, A):
        return np.diff(A.indptr)

    def _toggled_degree(self, G, u, v):
        return G.outdeg(u)


@register_sparse("gwideg")
class SparseGWInDegree(_SparseGWDegree):
    def _degrees(self, A):
        return np.bincount(A.indices, minlength=A.shape[0])

    def _toggled_degree(self, G, u, v):
        return G.indeg(v)


def _gwesp_delta_otp_sparse(G, u, v, alpha, sign):
    """model._gwesp_delta_otp on neighbour lists: same three parts, same summation order."""
    ea = np.exp(alpha); b = 1.0 - np.exp(-alpha)
    def h(s): return ea * (1.0 - b ** s)

    # Part 1 — the edge (u->v): s = #{k: u->k->v}, invariant to toggling (u,v)
    s_uv  = float(_n_common(G.out[u], G.inn[v]))
    part1 = sign * h(s_uv)

    # Part 2 — base edges (u->j) that gain/lose v as an OTP partner (u->v->j)
    Js = _common(G.out[u], G.out[v])
    part2 = 0.0
    if Js:
        s_uj  = np.array([_n_common(G.out[u], G.inn[j]) for j in Js], dtype=float)
        part2 = float(np.sum(h(s_uj + sign) - h(s_uj)))

    # Part 3 — base edges (i->v) that gain/lose u as an OTP partner (i->u->v)
    Is = _common(G.inn[v], G.inn[u])
    part3 = 0.0
    if Is:
        s_iv  = np.array([_n_common(G.out[i], G.inn[v]) for i in Is], dtype=float)
        part3 = float(np.sum(h(s_iv + sign) - h(s_iv)))

    return part1 + part2 + part3


def build_sparse_spec(statistic_terms, n):
    """StatSpec of sparse terms from an ergm-like string, e.g. 'edges + mutual + gwesp(0.5)'."""
    terms = []
    for token in str(statistic_terms).split("+"):
        if not token.strip():
            continue
        kind, args, kwargs = _parse_term(token)
        if kind not in SPARSE_REGISTRY:
            raise ValueError(f"'{kind}' has no sparse version (available: {sorted(SPARSE_REGISTRY)})")
        terms.append(SPARSE_REGISTRY[kind](args, kwargs, None))
    if not terms:
        raise ValueError("statistic_terms is empty")
    return StatSpec(n=n, directed=True, terms=terms)


def compute_stats_sparse(A, spec):
    """g(G) on a sparse adjacency (csr, any scipy.sparse, dense array or SparseDigraph)."""
    A = _csr(A)
    out = []
    for term in spec.terms:
        out.extend(term.value(A))
    return np.asarray(out, dtype=float)


def compute_delta_g_sparse(G, u, v, spec):
    """g(G') - g(G) for toggling the ordered pair (u,v) of a SparseDigraph."""
    sign = -1 if G.has(u, v) else 1
    out = []
    for term in spec.terms:
        out.extend(term.delta(G, u, v, sign))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
#   Reference model, band calibration and starting point for large networks
# ---------------------------------------------------------------------------
def bernoulli_digraph(n, p, rng):
    """Directed Bernoulli graph G(n, p) without self-loops, drawn directly in sparse form:
    the number of edges is Binomial(n(n-1), p), the edges a uniform subset of the pairs."""
    N = n * (n - 1)
    m = int(rng.binomial(N, p))
    codes = np.empty(0, dtype=np.int64)
    while codes.size < m:
        extra = rng.integers(0, N, size=int(1.1 * (m - codes.size)) + 16)
        codes = np.unique(np.concatenate([codes, extra]))
    if codes.size > m:
        codes = rng.choice(codes, size=m, replace=False)
    r = codes // (n - 1)
    c = codes % (n - 1)
    c = c + (c >= r)                                    # skip the diagonal
    return sp.csr_matrix((np.ones(m, dtype=np.int8), (r, c)), shape=(n, n))


def _check_bernoulli_null(spec):
    di = [t.name for t in spec.terms if t.dyad_independent]
    if di != ["edges"]:
        raise NotImplementedError("the sparse reference model is the Bernoulli graph: the only "
                                  f"dyad-independent term must be 'edges' (got {di})")


def calibrate_band_sparse(cfg, A, g_obs, spec):
    """calibrate_band for large networks: the dyad-independent null of a spec whose only
    dyad-independent term is 'edges' is the Bernoulli graph at the observed density, drawn
    in sparse form. Same scale and epsilon as calibrate_band (B = cfg.pac.n_bootstrap)."""
    _check_bernoulli_null(spec)
    A = _csr(A)
    n = A.shape[0]; delta = float(cfg.pac.delta); B = int(cfg.pac.n_bootstrap)
    p_hat = A.nnz / (n * (n - 1))

    rng_b = np.random.default_rng(cfg.seed)
    gs = np.array([compute_stats_sparse(bernoulli_digraph(n, p_hat, rng_b), spec) for _ in range(B)])

    scale    = np.maximum(gs.std(axis=0), 1e-6)
    mean_ref = gs.mean(axis=0)
    dists    = np.array([compute_norm(gs[b] - mean_ref, cfg.pac.norm, scale) for b in range(B)])
    eps_emp  = float(round(np.quantile(dists, 1 - delta), 2))

    cfg.pac.norm_scale = [float(x) for x in scale]
    cfg.pac.epsilon    = eps_emp
    return {"epsilon": eps_emp, "scale": scale, "mean_ref": mean_ref, "dists": dists, "p_hat": p_hat}


def mple_design_sparse(A, spec, max_dyads=None, rng=None):
    """Design of the MPLE: the change statistics of every ordered pair (u,v), computed with
    y_uv = 0 and the rest of the network fixed, as in estimation.compute_mple. All the edges
    enter; the non-edges too if they are at most max_dyads (or max_dyads is None), otherwise a
    uniform sample of max_dyads of them, weighted by #non-edges / #sampled (as ergm's
    MPLE.maxNumDyads). Returns X, y, w."""
    G = SparseDigraph(A)
    n = G.n
    edges = G.edges()
    X1 = []
    for u, v in edges:
        G.toggle(u, v)                                  # condition on y_uv = 0
        X1.append(compute_delta_g_sparse(G, u, v, spec))
        G.toggle(u, v)
    n_non = n * (n - 1) - len(edges)
    if max_dyads is None or n_non <= max_dyads:
        non = [(u, v) for u in range(n) for v in range(n) if u != v and not G.has(u, v)]
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        N = n * (n - 1)
        codes = np.empty(0, dtype=np.int64)
        while True:
            extra = rng.integers(0, N, size=int(1.2 * max_dyads) + 16)
            r = extra // (n - 1); c = extra % (n - 1); c = c + (c >= r)
            ok = np.array([not G.has(int(a), int(b)) for a, b in zip(r, c)])
            codes = np.unique(np.concatenate([codes, r[ok] * n + c[ok]]))
            if codes.size >= max_dyads:
                break
        codes = rng.choice(codes, size=max_dyads, replace=False)
        non = [(int(x // n), int(x % n)) for x in codes]
    X0 = [compute_delta_g_sparse(G, u, v, spec) for u, v in non]
    X = np.array(X1 + X0, dtype=float)
    y = np.r_[np.ones(len(X1)), np.zeros(len(X0))]
    w = np.r_[np.ones(len(X1)), np.full(len(X0), n_non / max(len(X0), 1))]
    return X, y, w


def fit_logistic(X, y, w):
    """Weighted logistic regression without intercept and without penalty (scipy)."""
    from scipy.optimize import minimize

    def nll(b):
        z = X @ b
        return float(np.sum(w * (np.logaddexp(0.0, z) - y * z)))

    def grad(b):
        p = 0.5 * (1.0 + np.tanh(0.5 * (X @ b)))       # stable sigmoid
        return X.T @ (w * (p - y))

    res = minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS",
                   options={"gtol": 1e-8, "maxiter": 10_000})
    return res.x


def compute_mple_sparse(A, spec, max_dyads=1_000_000, rng=None):
    """MPLE starting point on a sparse network (see mple_design_sparse)."""
    return fit_logistic(*mple_design_sparse(A, spec, max_dyads=max_dyads, rng=rng))


def bernoulli_theta(A, spec):
    """Starting point: the Bernoulli graph at the observed density (theta_edges = logit p,
    0 for the other terms). It replaces the MPLE, which needs all n(n-1) pairs."""
    A = _csr(A)
    n = A.shape[0]
    p = A.nnz / (n * (n - 1))
    theta = np.zeros(len(spec))
    theta[list(spec.labels).index("edges")] = np.log(p / (1 - p))
    return theta


# ---------------------------------------------------------------------------
#   MCMC on the PAC distribution, sparse state. Same algorithm and same random
#   draws as sampler.run_mcmc_inner, so that on the same network the two chains coincide.
# ---------------------------------------------------------------------------
def run_mcmc_inner_sparse(
    adj_init, theta, nu, g_obs, epsilon, norm, n_samples,
    spec, rng, burnin=0, proposal="tnt", p_nbhd=0.5, scale=None,
    interval=1, penalty="hard", tau=None,
):
    if interval < 1:
        raise ValueError("interval must be >= 1")
    G = SparseDigraph(adj_init)                                # a fresh copy for this chain
    n = G.n
    D = n * (n - 1)                                            # ORDERED pairs

    use_logistic = (penalty == "logistic")
    if use_logistic and (tau is None or tau <= 0):
        raise ValueError("penalty='logistic' needs tau > 0")

    g_current       = compute_stats_sparse(G, spec)
    d_current       = compute_norm(g_current - g_obs, norm, scale)
    in_band_current = int(d_current <= epsilon)
    s_current       = logistic_membership(d_current, epsilon, tau) if use_logistic else 0.0

    k = len(spec)
    samples_g       = np.zeros((n_samples, k))
    samples_in_band = np.zeros(n_samples, dtype=np.int8)

    if proposal in ("tnt", "neighborhood"):
        edges = G.edges()
        pos   = {e: t for t, e in enumerate(edges)}

    def rand_ordered_nonedge():
        while True:
            i = int(rng.integers(0, n)); j = int(rng.integers(0, n))
            if i != j and not G.has(i, j):
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
            log_h = _tnt_log_h(n_e, D, adding=not G.has(i, j))

        elif proposal == "neighborhood":                       # directed transitive move i->k->j
            n_e = len(edges)
            if rng.uniform() < p_nbhd:
                i = int(rng.integers(0, n))
                outs_i = G.out[i]
                if not outs_i:
                    null_move = True
                else:
                    kk_ = outs_i[int(rng.integers(0, len(outs_i)))]
                    outs_k = [x for x in G.out[kk_] if x != i]  # j != i
                    if not outs_k:
                        null_move = True
                    else:
                        j = outs_k[int(rng.integers(0, len(outs_k)))]
                if not null_move:
                    d_before = G.outdeg(i)
                    d_after  = d_before + (-1 if G.has(i, j) else 1)
                    if d_before <= 0 or d_after <= 0:
                        raise ValueError("neighborhood proposal with invalid out-degree")
                    log_h = float(np.log(d_before) - np.log(d_after))
            else:
                tie = (rng.uniform() < 0.5 and n_e > 0) or (n_e == D)
                if tie:
                    i, j = edges[int(rng.integers(0, n_e))]
                else:
                    i, j = rand_ordered_nonedge()
                log_h = _tnt_log_h(n_e, D, adding=not G.has(i, j))
        else:                                                  # uniform over an ordered pair
            i = int(rng.integers(0, n)); j = int(rng.integers(0, n))
            if i == j:
                null_move = True
            log_h = 0.0

        if null_move:
            record(step); continue

        delta_g      = compute_delta_g_sparse(G, i, j, spec)
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
            adding = not G.has(i, j)
            G.toggle(i, j)
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

    return samples_g, samples_in_band, G.tocsr()
