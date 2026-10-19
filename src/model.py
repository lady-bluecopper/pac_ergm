"""
model.py — the PAC-MaxEnt model: network loading, statistics registry, spec.

The model is declared in the config as an ergm-like string:

    statistic_terms: "edges + nodematch(GroupC) + nodefactor(GroupC, base=3) + gwesp(0.25)"

Each statistic is a self-contained class registered in REGISTRY (see @register):
it knows how to configure itself, its column labels, its value on a graph and its
change-statistic. Adding a statistic = adding one class; nothing else changes.

Available terms:
    edges, mutual, gwesp, gwodeg, gwideg,
    nodematch(attr), nodefactor(attr, base=k), edgecov(name)

Directed networks for now; `directed=False` is wired through TermContext and each
term declares what it supports.
Validated against ergm/R and by finite differences.
"""
from dataclasses import dataclass, field
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#   STATISTICS REGISTRY
#   Each statistic is one self-contained class that knows how to configure
#   itself, its column labels, its value on a graph, and its change-statistic.
#   Adding a new statistic = adding one class. The engine never changes.
# ---------------------------------------------------------------------------
REGISTRY = {}


def register(name):
    """Decorator: enrol a statistic class under the name used in the config."""
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


@dataclass
class TermContext:
    """Everything a term might need to configure itself."""
    n:         int                                  # number of nodes
    directed:  bool = True                          # network type
    nodes_df:  object = None                        # nodal attributes
    edgecov:   dict = field(default_factory=dict)   # {name: n x n matrix}
    cfg_model: object = None                        # model block of the config


class StatTerm:
    """Base class for one statistic (one or more columns of g)."""
    name     = None                            # set by @register
    supports = {"directed", "undirected"}      # network types this term allows
    dyad_independent = False                   # True if delta(u,v) does not depend on the rest of the graph

    def __init__(self, args=None, kwargs=None, ctx=None):
        self.args   = list(args or [])          # positional, e.g. ['GroupC'] or ['0.5']
        self.kwargs = dict(kwargs or {})        # named, e.g. {'base': '3'}
        self.arg    = self.args[0] if self.args else None   # shorthand: first positional
        self.ctx    = ctx
        self.setup()

    def _param(self, name, pos=None, default=None, cast=str):
        """Read a parameter: named form first, then positional #pos, then default."""
        if name in self.kwargs:
            return cast(self.kwargs[name])
        if pos is not None and len(self.args) > pos:
            return cast(self.args[pos])
        return default

    def setup(self):
        """Prepare data (attribute codes, matrices, decay...). Override if needed."""
        pass

    @property
    def labels(self):
        """Column names produced by this term. A term may produce several."""
        return [self.name]

    @property
    def n_cols(self):
        return len(self.labels)

    def value(self, A):
        """List of floats, length == n_cols."""
        raise NotImplementedError

    def delta(self, A, u, v, sign):
        """Change-statistic for toggling the ordered pair (u,v). sign=+1 add, -1 remove."""
        raise NotImplementedError

# ---------------------------------------------------------------------------
#   STATISTICS — structural terms with no parameters
# ---------------------------------------------------------------------------
@register("edges")
class Edges(StatTerm):
    """Number of edges."""
    supports = {"directed", "undirected"}
    dyad_independent = True

    def value(self, A):
        s = float(A.sum())
        return [s if self.ctx.directed else s / 2.0]

    def delta(self, A, u, v, sign):
        return [float(sign)]


@register("mutual")
class Mutual(StatTerm):
    """Number of reciprocal dyads i<->j. Directed networks only."""
    supports = {"directed"}

    def value(self, A):
        return [0.5 * float((A * A.T).sum())]

    def delta(self, A, u, v, sign):
        # changes only if the opposite edge v->u already exists
        return [float(sign) * float(A[v, u])]

@register("gwesp")
class Gwesp(StatTerm):
    """Geometrically weighted edgewise shared partners (transitivity). Directed only"""
    supports = {"directed"}

    def setup(self):
        self.decay = self._param("decay", pos=0, cast=float)
        if self.decay is None:
            raise ValueError("gwesp needs a decay: write e.g. gwesp(0.5) in statistic_terms")
        self.esp_type = self._param("esp_type", pos=1, default="OTP", cast=str)

    def value(self, A):
        S = _esp_S(A, self.esp_type)
        r, c = np.nonzero(A)
        if r.size == 0:
            return [0.0]
        return [float(np.sum(_h(S[r, c].astype(float), self.decay)))]

    def delta(self, A, u, v, sign):
        assert self.esp_type == "OTP", "the gwesp change-stat is derived for OTP"
        return [_gwesp_delta_otp(A, u, v, self.decay, sign)]

class _GWDegree(StatTerm):
    """Shared machinery for the geometrically weighted degree terms.
    Not registered: only its two subclasses below are."""
    supports = {"directed"}
    axis     = None            # 1 = out-degree (sum over rows), 0 = in-degree (over columns)

    def setup(self):
        self.decay = self._param("decay", pos=0, cast=float)
        if self.decay is None:
            raise ValueError(f"{self.name} needs a decay: write e.g. {self.name}(0.5) "
                             f"in statistic_terms")

    def _toggled_degree(self, A, u, v):
        """Degree of the ONE node whose degree the toggle changes."""
        raise NotImplementedError

    def value(self, A):
        deg = A.sum(axis=self.axis).astype(float)
        return [float(np.sum(_h(deg, self.decay)))]

    def delta(self, A, u, v, sign):
        d = self._toggled_degree(A, u, v)
        return [_h(d + sign, self.decay) - _h(d, self.decay)]


@register("gwodeg")
class GWOutDegree(_GWDegree):
    """Geometrically weighted OUT-degree (how much some nodes send far more than others)."""
    axis = 1

    def _toggled_degree(self, A, u, v):
        return float(A[u].sum())          # toggling u->v changes the out-degree of u


@register("gwideg")
class GWInDegree(_GWDegree):
    """Geometrically weighted IN-degree (popularity: how much some nodes are nominated more)."""
    axis = 0

    def _toggled_degree(self, A, u, v):
        return float(A[:, v].sum())       # toggling u->v changes the in-degree of v

def _level_codes(values):
    """Categorical values -> 0-based level indices, ordered like ergm (sorted uniques)."""
    vals   = np.asarray(values)
    levels = sorted(np.unique(vals).tolist())
    index  = {lab: i for i, lab in enumerate(levels)}
    codes  = np.array([index[x] for x in vals], dtype=int)
    return codes, levels


@register("nodematch")
class Nodematch(StatTerm):
    """Edges whose two endpoints share the same value of a categorical attribute."""
    supports = {"directed", "undirected"}
    dyad_independent = True

    def setup(self):
        if not self.arg:
            raise ValueError("nodematch needs an attribute, e.g. 'nodematch.team'")
        if self.arg not in self.ctx.nodes_df.columns:
            raise KeyError(f"attribute '{self.arg}' not in nodes "
                           f"(available: {list(self.ctx.nodes_df.columns)})")
        self.codes, self.levels = _level_codes(self.ctx.nodes_df[self.arg].values)

    @property
    def labels(self):
        return [f"nodematch.{self.arg}"]

    def value(self, A):
        same = (self.codes[:, None] == self.codes[None, :])
        s = float((A * same).sum())
        return [s if self.ctx.directed else s / 2.0]

    def delta(self, A, u, v, sign):
        return [float(sign) if self.codes[u] == self.codes[v] else 0.0]


@register("edgecov")
class EdgeCov(StatTerm):
    """Sum of an n x n covariate matrix over the edges present in the graph."""
    supports = {"directed", "undirected"}
    dyad_independent = True

    def setup(self):
        if not self.arg:
            raise ValueError("edgecov needs a matrix name, e.g. 'edgecov.metric'")
        if self.arg not in self.ctx.edgecov:
            raise KeyError(f"edgecov '{self.arg}' not found "
                           f"(available: {list(self.ctx.edgecov)})")
        M = np.asarray(self.ctx.edgecov[self.arg], dtype=float)
        n = self.ctx.n
        if M.shape != (n, n):
            raise ValueError(f"edgecov '{self.arg}' shape {M.shape}, expected ({n},{n})")
        self.M = M

    @property
    def labels(self):
        return [f"edgecov.{self.arg}"]

    def value(self, A):
        s = float((A * self.M).sum())
        return [s if self.ctx.directed else s / 2.0]

    def delta(self, A, u, v, sign):
        return [float(sign) * float(self.M[u, v])]


@register("nodefactor")
class Nodefactor(StatTerm):
    """Total degree summed by level of a categorical attribute.
    MULTI-COLUMN: produces K-1 columns, one per level, dropping the base level."""
    supports = {"directed"}
    dyad_independent = True

    def setup(self):
        if not self.arg:
            raise ValueError("nodefactor needs an attribute, e.g. 'nodefactor(GroupC, base=3)'")
        self.codes, self.levels = _level_codes(self.ctx.nodes_df[self.arg].values)
        K = len(self.levels)
        base = self._param("base", default=1, cast=int)          # 1-based level to DROP
        if not (1 <= base <= K):
            raise ValueError(f"nodefactor base={base} out of range for {K} levels")
        self.kept = [l for l in range(K) if l != base - 1]

    @property
    def labels(self):
        return [f"nodefactor.{self.arg}.{self.levels[l]}" for l in self.kept]

    def value(self, A):
        tot = A.sum(axis=1) + A.sum(axis=0)                     # total degree (in + out)
        K = len(self.levels)
        per_level = np.array([tot[self.codes == l].sum() for l in range(K)], dtype=float)
        return [float(x) for x in per_level[self.kept]]

    def delta(self, A, u, v, sign):
        full = np.zeros(len(self.levels))
        full[self.codes[u]] += sign                             # u's total degree changes
        full[self.codes[v]] += sign                             # v's total degree changes
        return [float(x) for x in full[self.kept]]

# ---------------------------------------------------------------------------
#   SPEC + BUILDER — turn the config's statistic_terms into ordered term objects
# ---------------------------------------------------------------------------
@dataclass
class StatSpec:
    n:        int
    directed: bool
    terms:    list                                  # StatTerm instances, in g order
    labels:   list = field(default_factory=list)

    def __post_init__(self):
        if not self.labels:                         # flatten: a term may give many columns
            self.labels = [lab for t in self.terms for lab in t.labels]
    def term(self, name):
        """First term with this name, or None.  E.g. spec.term("gwesp").esp_type"""
        return next((t for t in self.terms if t.name == name), None)
    def __len__(self):  return len(self.labels)     # number of COLUMNS of g, not of terms
    def __iter__(self): return iter(self.labels)
    def __repr__(self): return f"StatSpec({self.labels})"


_TERM_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s*$")


def _parse_term(text):
    """'nodefactor(GroupC, base=3)' -> ('nodefactor', ['GroupC'], {'base': '3'})
       'gwesp(0.25)'                -> ('gwesp',      ['0.25'],   {})
       'edges'                      -> ('edges',      [],         {})
    """
    m = _TERM_RE.match(str(text))
    if not m:
        raise ValueError(f"cannot parse statistic term '{str(text).strip()}' — "
                         f"expected 'name' or 'name(arg, key=value)'")
    kind, inside = m.group(1), (m.group(2) or "")
    args, kwargs = [], {}
    for piece in inside.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" in piece:
            k, _, val = piece.partition("=")
            kwargs[k.strip()] = val.strip().strip("'\"")
        else:
            args.append(piece.strip("'\""))
    return kind, args, kwargs


def build_spec(nodes_df, edgecov, cfg_model, directed=True):
    """Build the StatSpec from cfg_model.statistic_terms, a STRING in ergm-like form:

        "edges + nodematch(GroupC) + nodefactor(GroupC, base=3) + gwesp(0.25)"
    """
    n   = len(nodes_df)
    ctx = TermContext(n=n, directed=directed, nodes_df=nodes_df,
                      edgecov=dict(edgecov or {}), cfg_model=cfg_model)

    raw = cfg_model.get("statistic_terms", None)
    if raw is None:
        raise ValueError("config: model.statistic_terms is missing")
    if not isinstance(raw, str):
        raise TypeError("model.statistic_terms must be a STRING, e.g. "
                        f"'edges + mutual + gwesp(0.5)' (got {type(raw).__name__})")

    net_type = "directed" if directed else "undirected"
    terms = []
    for token in raw.split("+"):
        if not token.strip():
            continue
        kind, args, kwargs = _parse_term(token)
        if kind not in REGISTRY:
            raise ValueError(f"unknown statistic '{kind}' in statistic_terms "
                             f"(available: {sorted(REGISTRY)})")
        cls = REGISTRY[kind]
        if net_type not in cls.supports:
            raise ValueError(f"statistic '{kind}' is not available on {net_type} "
                             f"networks (it supports: {sorted(cls.supports)})")
        terms.append(cls(args, kwargs, ctx))

    if not terms:
        raise ValueError("model.statistic_terms is empty")
    return StatSpec(n=n, directed=directed, terms=terms)


# ---------------------------------------------------------------------------
#   ENGINE — generic over the spec
# ---------------------------------------------------------------------------
def compute_stats(adj, spec):
    """g(G): every term's value, concatenated in spec order."""
    A = np.asarray(adj)
    out = []
    for term in spec.terms:
        out.extend(term.value(A))
    return np.asarray(out, dtype=float)


def compute_delta_g(adj, u, v, spec):
    """g(G') - g(G) for toggling the ORDERED pair (u,v)."""
    A    = adj
    sign = 1 if A[u, v] == 0 else -1
    out  = []
    for term in spec.terms:
        out.extend(term.delta(A, u, v, sign))
    return np.asarray(out, dtype=float)



# ---------------------------------------------------------------------------
#    Data loading: directed edge list + nodal attributes + edge-covariate matrices
# ---------------------------------------------------------------------------
def load_data(cfg_data):
    """
    Returns (adj, nodes_df, edgecov_dict).
      edges_path      : CSV with columns 'from','to' (0-based, directed)
      nodes_path      : CSV with 'id' + attribute columns (e.g. team, floor)
      cfg_data.edgecov: dict {name: path} of n x n matrices (order = rows of nodes)
    """
    nodes = pd.read_csv(cfg_data.nodes_path)
    edges = pd.read_csv(cfg_data.edges_path)
    n     = len(nodes)
    id2idx = {int(i): k for k, i in enumerate(nodes["id"].values)}

    adj = np.zeros((n, n), dtype=np.int32)
    for f, t in edges[["from", "to"]].values:
        adj[id2idx[int(f)], id2idx[int(t)]] = 1          # DIRECTED: only [f,t]

    edgecov_dict = {}
    for name, path in dict(cfg_data.get("edgecov", {})).items():
        M = pd.read_csv(path, header=None).values.astype(float)
        if M.shape != (n, n):
            raise ValueError(f"matrix '{name}' shape {M.shape}, expected ({n},{n})")
        edgecov_dict[name] = M

    print(f"Network loaded — nodes: {n}, edges: {int(adj.sum())}, "
          f"edgecov: {list(edgecov_dict)}")
    return adj, nodes, edgecov_dict


# ---------------------------------------------------------------------------
#   GWESP: per-edge weight and shared-partner matrix (by directed type)
# ---------------------------------------------------------------------------
def _h(s, alpha):
    """GWESP weight of an edge with s shared partners: h(s)=e^a(1-(1-e^-a)^s), h(0)=0."""
    ea = np.exp(alpha)
    b  = 1.0 - np.exp(-alpha)
    return ea * (1.0 - b ** s)

def _esp_S(adj, esp_type):
    """S[i,j] = #shared partners of the ordered pair (i,j). OTP = i->k->j (default)."""
    if esp_type == "OTP": return adj @ adj
    if esp_type == "OSP": return adj @ adj.T
    if esp_type == "ISP": return adj.T @ adj
    if esp_type == "ITP": return (adj @ adj).T
    raise ValueError(esp_type)



# ---------------------------------------------------------------------------
#    GWESP (OTP) change-stat for toggling edge (u->v). sign=+1 if adding.
# ---------------------------------------------------------------------------
def _gwesp_delta_otp(adj, u, v, alpha, sign):
    ea = np.exp(alpha); b = 1.0 - np.exp(-alpha)
    def h(s): return ea * (1.0 - b ** s)

    # Part 1 — the edge (u->v): s = (A^2)[u,v], invariant to toggling (u,v)
    s_uv  = float(adj[u] @ adj[:, v])
    part1 = sign * h(s_uv)

    # Part 2 — base edges (u->j) that gain/lose v as an OTP partner (u->v->j)
    Js = np.nonzero((adj[u] == 1) & (adj[v] == 1))[0]
    part2 = 0.0
    if Js.size:
        s_uj  = (adj[u] @ adj[:, Js]).astype(float)            # S[u, Js]
        part2 = float(np.sum(h(s_uj + sign) - h(s_uj)))

    # Part 3 — base edges (i->v) that gain/lose u as an OTP partner (i->u->v)
    Is = np.nonzero((adj[:, v] == 1) & (adj[:, u] == 1))[0]
    part3 = 0.0
    if Is.size:
        s_iv  = (adj[Is] @ adj[:, v]).astype(float)            # S[Is, v]
        part3 = float(np.sum(h(s_iv + sign) - h(s_iv)))

    return part1 + part2 + part3