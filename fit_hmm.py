"""Example HMM fitting script.

Each 10 ms bin is either "no spike" or spikes from m units. Each of those units gets weight
1/m, and the bin's log likelihood under state k is sum_s W[t, s] log E[k, s]. This is the
expectation of the original HMM (adapted from Recanatesi et al., 2022), which instead
draws one of the active units at random.

The forward-backward pass, the transition updates and the HMM container come from `ssm`
(github.com/lindermanlab/ssm @ eb6c8aa). This file adds the emission class, the chain
entry from state 1 and the stopping rule.

The chain entry from state 1 is a legacy decision kept from MATLAB's `hmmtrain`, which the
original model was fitted with. The chain is assumed to sit in state 1 before the first
bin, so pi_0 = A[0, :] and the first bin counts as a transition out of state 1. It has no
separate initial-state parameter. Over a whole session it barely changes A, and it is kept
so the fits reproduce the original ones.

Requires numpy, scipy and `ssm` (h5py only for `read_session_h5`):

    pip install --no-build-isolation \\
        "ssm @ git+https://github.com/lindermanlab/ssm@eb6c8aa33e5311d3564075807dec340759dd8081"

Example:

    spikes, duration_s = read_session_h5("mock_session.h5")   # or your own spike times
    W = bin_weights(spikes, duration_s)
    best, fits = fit_best(W, K=4, seeds=range(60))
    post = posterior(best, W)                                     # (T, K)
    rates = state_rates(post, bin_counts(spikes, duration_s))     # (K, N), Hz

Command line (`python fit_hmm.py --help`):

| Option | Default | Meaning |
|---|---|---|
| `--h5 PATH` | none | fit a session file; without it, fit a simulated session |
| `-K` | `3` | number of states (at least 2) |
| `--seeds` | `5` | number of random starts (seeds 0 .. n-1) |
"""

from __future__ import annotations

import numpy as np
import ssm
from scipy import sparse
from ssm.init_state_distns import InitialStateDistribution
from ssm.observations import Observations
from ssm.transitions import StationaryTransitions

FLOOR = 1e-300
BIN_S = 0.01


# ----------------------------------------------------------------------------- input


def read_session_h5(path: str) -> tuple[list[np.ndarray], float]:
    """Read one session file (`SSM_impl/data/sessions/<session>.h5`).

    The file stores every unit's spike times end to end in `spikes/times` (float, s).
    Unit i's spikes are `times[index[i]:index[i + 1]]`, with `spikes/index` (int, length
    N + 1). The session length is the root attribute `duration_s`.

    Parameters
    ----------
    path : str
        Path to the `.h5` file.

    Returns
    -------
    spikes : list of np.ndarray (float64)
        One array of spike times (s) per unit, length N.
    duration_s : float
        Session length (s).
    """
    import h5py

    with h5py.File(path, "r") as h:
        t, idx = h["spikes/times"][:], h["spikes/index"][:]
        duration_s = float(h.attrs["duration_s"])
    return [t[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)], duration_s


def _bin_units(spikes, n_bins, bin_s):
    """Distinct (bin, unit) pairs and their spike counts."""
    n_units = len(spikes)
    parts = []
    for u, sp in enumerate(spikes):
        sp = np.asarray(sp, float)
        sp = sp[(sp > 0) & (sp <= n_bins * bin_s)]
        if len(sp):
            parts.append(np.clip(np.ceil(sp / bin_s).astype(np.int64) - 1, 0, n_bins - 1)
                         * n_units + u)
    key = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    pair, cnt = np.unique(key, return_counts=True)
    return pair // n_units, pair % n_units, cnt, n_units


def n_bins_of(duration_s: float, bin_s: float = BIN_S) -> int:
    """Number of whole bins in the session (a partial last bin is dropped)."""
    return int(duration_s / bin_s)


def bin_weights(spikes, duration_s: float, bin_s: float = BIN_S) -> sparse.csr_matrix:
    """Bin spike times into the model's input W.

    A spike at time t goes to bin ceil(t / bin_s) (1-based); spikes at t <= 0 or past the
    last whole bin are dropped. If m units fire in a bin, each gets 1/m; a unit that fires
    twice in the bin still counts once. An empty bin puts 1 in the last column ("no
    spike"). Every row sums to 1.

    Parameters
    ----------
    spikes : list of array-like (float)
        One array of spike times (s) per unit, length N.
    duration_s : float
        Session length (s).
    bin_s : float, default 0.01
        Bin width (s).

    Returns
    -------
    W : scipy.sparse.csr_matrix (float64), shape (T, N + 1)
        Weights per bin; column N is "no spike".
    """
    n_bins = n_bins_of(duration_s, bin_s)
    b, u, _, n = _bin_units(spikes, n_bins, bin_s)
    m = np.bincount(b, minlength=n_bins)
    empty = np.ones(n_bins, bool)
    empty[b] = False
    rows = np.r_[b, np.flatnonzero(empty)]
    cols = np.r_[u, np.full(empty.sum(), n)]
    vals = np.r_[1.0 / m[b], np.ones(empty.sum())]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_bins, n + 1))


def bin_counts(spikes, duration_s: float, bin_s: float = BIN_S) -> sparse.csr_matrix:
    """Sparse `(T, N)` spike counts on the same grid as `bin_weights` (for `state_rates`)."""
    n_bins = n_bins_of(duration_s, bin_s)
    b, u, c, n = _bin_units(spikes, n_bins, bin_s)
    return sparse.csr_matrix((c.astype(float), (b, u)), shape=(n_bins, n))


# ----------------------------------------------------------------------------- model


class WeightedCategorical(Observations):
    """Categorical emission scored on a weight vector per bin.

    `data` is the sparse W from `bin_weights`. Each state k has an emission row E[k, :]
    over the N units plus "no spike" (shape (K, N + 1); each row sums to 1 after the first
    M-step).

    log p(bin t | k) = sum_s W[t, s] log E[k, s]
    M-step:  E[k, s] ∝ sum_t gamma_t(k) W[t, s]
    """

    def __init__(self, K: int, n_sym: int):
        super().__init__(K, n_sym)
        self.E = np.full((K, n_sym), 1.0 / n_sym)

    @property
    def params(self):
        return self.E

    @params.setter
    def params(self, value):
        self.E = value

    def permute(self, perm):
        self.E = self.E[perm]

    def log_likelihoods(self, data, input, mask, tag):
        """`(T, K)` log likelihood of each bin under each state."""
        return np.asarray(data @ np.log(np.maximum(self.E, FLOOR)).T)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Set E to the posterior-weighted sum of W, normalized per state."""
        e = sum(np.asarray((w.T @ ez).T) for (ez, _, _), w in zip(expectations, datas))
        self.E = e / np.maximum(e.sum(1, keepdims=True), FLOOR)


class EntryFromState1(StationaryTransitions):
    """Stationary transitions, with the chain entering from state 1 (MATLAB `hmmtrain`).

    The chain is taken to sit in state 1 before the first bin. The first bin's state
    therefore counts as one more transition out of state 1: its posterior gamma_0 is added
    to row 0 of the expected transition counts before each row is normalized. There is no
    pseudocount.
    """

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """Set A to the expected transition counts, plus gamma_0 on row 0, normalized per row."""
        xi = sum(np.asarray(ezz).reshape(self.K, self.K) for _, ezz, _ in expectations)
        xi = xi + np.outer(np.eye(self.K)[0], sum(ez[0] for ez, _, _ in expectations))
        p = xi / np.maximum(xi.sum(1, keepdims=True), FLOOR)
        self.log_Ps = np.log(np.maximum(p, FLOOR))


class Row0InitialState(InitialStateDistribution):
    """Initial-state distribution tied to row 0 of A (pi_0 = A[0, :]); it learns nothing itself."""

    def __init__(self, K: int, D: int, transitions: StationaryTransitions):
        super().__init__(K, D)
        self.transitions = transitions

    @property
    def log_initial_state_distn(self):
        lp = self.transitions.log_Ps[0]
        return lp - np.logaddexp.reduce(lp)

    @property
    def initial_state_distn(self):
        return np.exp(self.log_initial_state_distn)

    def permute(self, perm):
        pass

    def m_step(self, *args, **kwargs):
        pass


def build_hmm(K: int, n_sym: int) -> ssm.HMM:
    """An `ssm.HMM` with the unit-average emission and the state-1 chain entry."""
    obs = WeightedCategorical(K, n_sym)
    tr = EntryFromState1(K, n_sym)
    hmm = ssm.HMM(K, n_sym, observations=obs, transitions=tr)
    hmm.init_state_distn = Row0InitialState(K, n_sym, tr)
    return hmm


def init_params(K: int, n_sym: int, seed: int, sticky_low: float = 0.98
                ) -> tuple[np.ndarray, np.ndarray]:
    """Draw a sticky random start.

    E ~ U(0, 1), left unnormalized (the first M-step normalizes it). Each self-transition
    A_kk ~ U(sticky_low, 1), and the remaining 1 - A_kk is split evenly over the other
    states.

    Parameters
    ----------
    K : int
        Number of states, at least 2.
    n_sym : int
        Number of symbols, N + 1.
    seed : int
        Seed for `np.random.default_rng`.
    sticky_low : float, default 0.98
        Lower bound of the self-transition probabilities.

    Returns
    -------
    E : np.ndarray (float64), shape (K, n_sym)
    A : np.ndarray (float64), shape (K, K)
    """
    if K < 2:
        raise ValueError(f"K must be at least 2, got {K}")
    rng = np.random.default_rng(seed)
    e = rng.random((K, n_sym))
    d = rng.uniform(sticky_low, 1.00, K)
    a = np.repeat(((1 - d) / (K - 1))[:, None], K, 1)
    a[np.arange(K), np.arange(K)] = d
    return e, a


def set_params(hmm, E, A) -> None:
    """Write E `(K, N + 1)` and A `(K, K)` into an HMM from `build_hmm`."""
    hmm.observations.E = np.array(E, float)
    hmm.transitions.log_Ps = np.log(np.maximum(np.asarray(A, float), FLOOR))


def get_params(hmm) -> tuple[np.ndarray, np.ndarray]:
    """Read `(E, A)` from an HMM from `build_hmm`."""
    return np.array(hmm.observations.E), np.array(hmm.transitions.transition_matrix)


def _mask(W):
    """An all-true `(T, 1)` mask (ssm would otherwise build a dense one the size of W)."""
    return np.ones((W.shape[0], 1), bool)


# ----------------------------------------------------------------------------- fitting


def fit(W, K: int = 4, seed: int = 0, max_iter: int = 500, tol: float = 1e-10,
        sticky_low: float = 0.98) -> dict:
    """Fit one HMM by EM from the sticky random start for `seed`.

    EM stops when three changes are all below `tol`: the relative change in log
    likelihood, |dL| / (1 + |L|); the largest per-row L1 change in A divided by K; and the
    largest per-row L1 change in E divided by N + 1.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (T, N + 1)
        Model input from `bin_weights`.
    K : int, default 4
        Number of states, at least 2.
    seed : int, default 0
        Seed of the random start.
    max_iter : int, default 500
        Maximum number of EM iterations.
    tol : float, default 1e-10
        Stopping tolerance.
    sticky_low : float, default 0.98
        Lower bound of the starting self-transitions.

    Returns
    -------
    dict
        E : np.ndarray (float64), shape (K, N + 1) — emission rows.
        A : np.ndarray (float64), shape (K, K) — transition matrix.
        loglik : float — log likelihood of the session under the final parameters.
        n_iter : int — EM iterations run.
        converged : bool — whether the stopping rule fired before `max_iter`.
        seed : int — the seed used.
    """
    if K < 2:
        raise ValueError(f"K must be at least 2, got {K}")
    hmm = build_hmm(K, W.shape[1])
    set_params(hmm, *init_params(K, W.shape[1], seed, sticky_low))
    datas, inputs, masks, tags = [W], [np.zeros((W.shape[0], 0))], [_mask(W)], [None]
    old, converged = -np.inf, False
    for it in range(1, max_iter + 1):
        ex = [hmm.expected_states(d, i, m, t) for d, i, m, t in zip(datas, inputs, masks, tags)]
        ll = float(sum(e[2] for e in ex))
        e_old, a_old = get_params(hmm)
        hmm.init_state_distn.m_step(ex, datas, inputs, masks, tags)
        hmm.transitions.m_step(ex, datas, inputs, masks, tags)
        hmm.observations.m_step(ex, datas, inputs, masks, tags)
        e_new, a_new = get_params(hmm)
        d_ll = abs(ll - old) / (1 + abs(old)) if np.isfinite(old) else np.inf
        d_a = np.abs(a_new - a_old).sum(1).max() / K
        d_e = np.abs(e_new - e_old).sum(1).max() / e_new.shape[1]
        old = ll
        if d_ll < tol and d_a < tol and d_e < tol:
            converged = True
            break
    E, A = get_params(hmm)
    return dict(E=E, A=A, loglik=float(hmm.log_likelihood([W], masks=[_mask(W)])),
                n_iter=it, converged=converged, seed=int(seed))


def fit_best(W, K: int = 4, seeds=range(10), **kw) -> tuple[dict, list[dict]]:
    """Fit once per seed (`**kw` goes to `fit`); return the highest-log-likelihood fit and all fits."""
    fits = [fit(W, K, s, **kw) for s in seeds]
    return max(fits, key=lambda f: f["loglik"]), fits


def as_hmm(params: dict) -> ssm.HMM:
    """Rebuild an `ssm.HMM` from a `fit` result."""
    hmm = build_hmm(params["A"].shape[0], params["E"].shape[1])
    set_params(hmm, params["E"], params["A"])
    return hmm


def posterior(params: dict, W) -> np.ndarray:
    """Posterior P(state at t | whole session), shape `(T, K)`, float64."""
    return as_hmm(params).expected_states(W, mask=_mask(W))[0]


def decode(params: dict, W) -> np.ndarray:
    """Most probable state per bin (posterior argmax), shape `(T,)`, int64."""
    return posterior(params, W).argmax(1)


def state_rates(post: np.ndarray, counts, bin_s: float = BIN_S) -> np.ndarray:
    """Each unit's firing rate in each state, weighted by the posterior.

    rate[k, u] = sum_t gamma_t(k) n_tu / (bin_s * sum_t gamma_t(k)). E is not a rate:
    E[k, u] is unit u's share of the single symbol each bin carries. Use these rates to
    compare states with observed firing.

    Parameters
    ----------
    post : np.ndarray (float64), shape (T, K)
        Posterior from `posterior`.
    counts : scipy.sparse matrix, shape (T, N)
        Spike counts from `bin_counts`.
    bin_s : float, default 0.01
        Bin width (s).

    Returns
    -------
    np.ndarray (float64), shape (K, N)
        Firing rate (Hz) of each unit in each state.
    """
    num = np.asarray((counts.T @ post).T)
    return num / (bin_s * np.maximum(post.sum(0)[:, None], FLOOR))


# ----------------------------------------------------------------------------- demo


def _simulate(n_units: int = 30, duration_s: float = 600.0, K: int = 3, dwell_s: float = 8.0,
              seed: int = 0) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Simulate a session of Poisson spikes from K rate profiles with exponential dwells.

    Parameters
    ----------
    n_units : int, default 30
    duration_s : float, default 600.0
    K : int, default 3
        Number of true states.
    dwell_s : float, default 8.0
        Mean dwell (s) in each state.
    seed : int, default 0

    Returns
    -------
    spikes : list of np.ndarray (float64)
        Spike times (s) per unit.
    z : np.ndarray (int64), shape (T,)
        True state per bin.
    duration_s : float
    """
    rng = np.random.default_rng(seed)
    n = n_bins_of(duration_s)
    lam = rng.gamma(2.0, 4.0, (K, n_units)) * rng.choice([0.3, 3.0], (K, n_units))
    z, t, k = np.zeros(n, int), 0, 0
    while t < n:
        d = max(1, int(rng.exponential(dwell_s) / BIN_S))
        z[t:t + d] = k
        t, k = t + d, (k + rng.integers(1, K)) % K
    spikes = []
    for u in range(n_units):
        c = rng.poisson(lam[z, u] * BIN_S)
        b = np.repeat(np.arange(n), c)
        spikes.append(np.sort((b + rng.random(len(b))) * BIN_S))
    return spikes, z, duration_s


if __name__ == "__main__":
    import argparse
    from itertools import permutations

    ap = argparse.ArgumentParser(description="Fit the unit-average HMM to a simulated or "
                                             "recorded session.")
    ap.add_argument("--h5", default=None, help="session .h5 to fit (default: simulate one)")
    ap.add_argument("-K", type=int, default=3, help="number of states, at least 2 (default 3)")
    ap.add_argument("--seeds", type=int, default=5, help="number of random starts (default 5)")
    args = ap.parse_args()
    if args.K < 2:
        ap.error(f"-K must be at least 2, got {args.K}")

    z = None
    if args.h5:
        spikes, dur = read_session_h5(args.h5)
        print(f"read {args.h5}")
    else:
        spikes, z, dur = _simulate(K=args.K)
        print(f"simulated {len(spikes)} units, {dur:.0f} s, {args.K} true states")

    W = bin_weights(spikes, dur)
    best, fits = fit_best(W, K=args.K, seeds=range(args.seeds))
    path = decode(best, W)
    print(f"T = {W.shape[0]} bins, N = {W.shape[1] - 1} units; "
          f"log L per seed: {[round(f['loglik'], 1) for f in fits]}")
    print(f"best seed {best['seed']}: {best['n_iter']} iterations, "
          f"converged {best['converged']}; E rows sum to {best['E'].sum(1).round(6)}")
    if z is not None:
        agree = max(np.mean(np.asarray(p)[path] == z) for p in permutations(range(args.K)))
        print(f"agreement with the true states {agree:.3f}")
    rates = state_rates(posterior(best, W), bin_counts(spikes, dur))
    print(f"state rates (Hz), shape {rates.shape}; mean per state {rates.mean(1).round(2)}")
