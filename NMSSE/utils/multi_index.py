from __future__ import annotations

from typing import List, Tuple, Iterator, Optional

import math
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

try:
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover
    njit = None
    prange = range

Index = Tuple[int, ...]


def _numba_available() -> bool:
    return njit is not None


def _int64_max() -> int:
    return int(np.iinfo(np.int64).max)


def _build_binom_table(max_n: int) -> np.ndarray:
    """Build binomial coefficient table C[n, k] for 0<=n<=max_n.

    Values are stored as int64 and assumed to fit (caller should guard).
    """
    C = np.zeros((max_n + 1, max_n + 1), dtype=np.int64)
    for n in range(max_n + 1):
        C[n, 0] = 1
        C[n, n] = 1
        for k in range(1, n):
            C[n, k] = C[n - 1, k - 1] + C[n - 1, k]
    return C


if _numba_available():
    @njit(cache=True)
    def _unrank_combination_lex(rank: int, n: int, k: int, binom: np.ndarray) -> np.ndarray:
        """Unrank combinations in lexicographic order.

        Returns a sorted length-k int64 array with values in [0, n-1].
        """
        out = np.empty(k, dtype=np.int64)
        prev = -1
        r = rank
        for i in range(k):
            # p ranges such that there are enough remaining elements
            # remaining to pick: (k - i - 1)
            max_p = n - (k - i)
            chosen = -1
            for p in range(prev + 1, max_p + 1):
                # number of combinations if we choose p here
                rem_n = n - p - 1
                rem_k = k - i - 1
                c = 1
                if rem_k > 0:
                    c = binom[rem_n, rem_k]
                if r >= c:
                    r -= c
                else:
                    chosen = p
                    break
            out[i] = chosen
            prev = chosen
        return out


    @njit(parallel=True, cache=True)
    def _weak_compositions_layer(k_sum: int, r: int, out_count: int, binom: np.ndarray) -> np.ndarray:
        """Generate all weak compositions of k_sum into r parts.

        Uses stars-and-bars: choose (r-1) bar positions among n_pos = k_sum + r - 1 slots.
        Output is NOT sorted; caller sorts lexicographically.
        """
        comps = np.empty((out_count, r), dtype=np.int64)
        n_pos = k_sum + r - 1
        k_bars = r - 1
        for t in prange(out_count):
            if k_bars == 0:
                comps[t, 0] = k_sum
                continue

            bars = _unrank_combination_lex(t, n_pos, k_bars, binom)
            prev = -1
            for i in range(r - 1):
                comps[t, i] = bars[i] - prev - 1
                prev = bars[i]
            comps[t, r - 1] = n_pos - prev - 1
        return comps

class MultiIndex:
    def __init__(
        self,
        r: int,
        n: int,
        show_progress: bool = True,
        progress_desc: str = "MultiIndex",
    ):
        self.dim = r
        self.max_order = n
        self._layers: List[List[Index]] = []
        self._show_progress = bool(show_progress)
        self._progress_desc = progress_desc
        self._generate()

    def _generate(self) -> None:
        if _numba_available() and self._should_use_numba():
            try:
                self._generate_numba()
                return
            except Exception:
                # Fall back to the original pure-Python generator for robustness.
                self._layers.clear()

        self._generate_python()

    def _generate_python(self) -> None:
        r, n = self.dim, self.max_order

        layer_set = {(0,) * r}
        self._layers.append(sorted(layer_set))

        it = range(n)
        if self._show_progress and tqdm is not None and n > 1:
            it = tqdm(it, desc=f"{self._progress_desc} layers", total=n)

        for _ in it:
            next_set = set()
            for x in layer_set:
                for i in range(r):
                    y = x[:i] + (x[i] + 1,) + x[i + 1 :]
                    next_set.add(y)
            layer_set = next_set
            self._layers.append(sorted(layer_set))

    def _should_use_numba(self) -> bool:
        """Heuristic guardrails to avoid memory blow-ups.

        Multi-index counts grow combinatorially: layer(k) size is C(r+k-1, r-1).
        When this is huge, generation (with or without numba) will be infeasible.
        """
        r, n = self.dim, self.max_order

        # Always safe for small problems.
        if r <= 1 or n <= 1:
            return True

        # Hard caps to keep memory reasonable.
        max_layer = 2_000_000
        max_total = 5_000_000

        total = 0
        for k in range(n + 1):
            cnt = math.comb(r + k - 1, r - 1)
            if cnt > max_layer:
                return False
            total += cnt
            if total > max_total:
                return False
        return True

    def _generate_numba(self) -> None:
        r, n = self.dim, self.max_order
        if r <= 0 or n < 0:
            raise ValueError("r must be positive and n must be non-negative")

        # Guard int64 range for the combination unranking.
        i64max = _int64_max()
        for k in range(n + 1):
            cnt = math.comb(r + k - 1, r - 1)
            if cnt > i64max:
                raise OverflowError("layer size exceeds int64; fallback required")

        max_n_pos = n + r - 1  # maximum (k_sum + r - 1)
        binom = _build_binom_table(max_n_pos)

        self._layers = []

        it = range(n + 1)
        if self._show_progress and tqdm is not None and n > 1:
            it = tqdm(it, desc=f"{self._progress_desc} layers", total=n + 1)

        for k_sum in it:
            out_count = int(math.comb(r + k_sum - 1, r - 1))
            comps = _weak_compositions_layer(k_sum, r, out_count, binom)

            # Sort to match the original deterministic order: sorted(tuple) => lexicographic.
            if comps.shape[0] > 1:
                order = np.lexsort(comps.T[::-1])
                comps = comps[order]

            self._layers.append([tuple(int(x) for x in row) for row in comps])


    def layer(self, s: int) -> List[Index]:
        return self._layers[s]
    
    def iter_eq(self, k: int) -> Iterator[Index]:
        """
        go through all indices with sum(x) == k
        """
        k = min(k, self.max_order)
        for idx in self._layers[k]:
            yield idx

    def iter_leq(self, k: int) -> Iterator[Index]:
        """
        go through all indices with sum(x) <= k
        """
        k = min(k, self.max_order)
        for s in range(k + 1):
            for idx in self._layers[s]:
                yield idx

    def __iter__(self) -> Iterator[Index]:
        return self.iter_leq(self.max_order)

    def __len__(self) -> int:
        return sum(len(layer) for layer in self._layers)

    def count_leq(self, k: int) -> int:
        k = min(k, self.max_order)
        return sum(len(self._layers[s]) for s in range(k + 1))
