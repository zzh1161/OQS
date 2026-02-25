from __future__ import annotations

from typing import Iterator, List, Tuple

import itertools

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

Index = Tuple[int, ...]


def _weak_compositions(k_sum: int, r: int) -> List[Index]:
    """All weak compositions of `k_sum` into `r` parts (non-negative integers).

    Uses stars-and-bars: choose (r-1) bar positions among (k_sum + r - 1) slots.
    The returned list is lexicographically sorted for deterministic iteration.
    """
    if r <= 0:
        raise ValueError("r must be positive")
    if k_sum < 0:
        raise ValueError("k_sum must be non-negative")

    if r == 1:
        return [(int(k_sum),)]

    n_pos = k_sum + r - 1
    out: List[Index] = []

    # `itertools.combinations` yields bar positions in lex order.
    for bars in itertools.combinations(range(n_pos), r - 1):
        prev = -1
        parts = [0] * r
        for i, b in enumerate(bars):
            parts[i] = b - prev - 1
            prev = b
        parts[r - 1] = n_pos - prev - 1
        out.append(tuple(int(x) for x in parts))

    out.sort()
    return out


class MultiIndex:
    """Multi-index layers for hierarchy truncation.

    Parameters
    ----------
    r:
        Dimension of the index (tuple length).
    n:
        Maximum layer number to generate (0..n inclusive).
    Notes
    -----
    This class is **lazy**: `__init__` does not generate any layers.
    Call `generate(order=...)` when you know which method/order you need.
    """

    def __init__(
        self,
        r: int,
        n: int,
        show_progress: bool = True,
        progress_desc: str = "MultiIndex",
    ):
        self.dim = int(r)
        self.N_layers = int(n)
        self.order: int | None = None
        self._layers: List[List[Index]] = []
        self._show_progress = bool(show_progress)
        self._progress_desc = str(progress_desc)

        if self.dim <= 0:
            raise ValueError("r must be positive")
        if self.N_layers < 0:
            raise ValueError("n must be non-negative")

    def generate(self, order: int = 1) -> "MultiIndex":
        """(Re)generate layers.

        After calling, `layer(s)` is defined by `sum(idx) == s * order`.
        """
        order = int(order)
        if order <= 0:
            raise ValueError("order must be a positive integer")
        self.order = order
        self._layers = []

        it = range(self.N_layers + 1)
        if self._show_progress and tqdm is not None and self.N_layers > 1:
            it = tqdm(it, desc=f"{self._progress_desc} layers", total=self.N_layers + 1)

        for level in it:
            if self.order == 1:
                k_sum = level
            elif self.order == 2:
                k_sum = level
            else:
                raise ValueError("order > 2 is not supported")
            self._layers.append(_weak_compositions(k_sum, self.dim))

        return self

    def _require_generated(self) -> None:
        if not self._layers:
            raise RuntimeError(
                "MultiIndex is not generated yet. Call generate(order=...) before using layer()/iter_*()."
            )


    def layer(self, s: int) -> List[Index]:
        self._require_generated()
        return self._layers[s]
    
    def iter_eq(self, k: int) -> Iterator[Index]:
        """Iterate over all indices in layer `k`.

        Layer definition:
        - `order == 1`: sum(idx) == k
        - `order > 1`: sum(idx) == k * order
        """
        self._require_generated()
        k = min(k, self.N_layers)
        for idx in self._layers[k]:
            yield idx

    def iter_leq(self, k: int) -> Iterator[Index]:
        """Iterate over all indices in layers `0..k` (inclusive)."""
        self._require_generated()
        k = min(k, self.N_layers)
        for s in range(k + 1):
            for idx in self._layers[s]:
                yield idx

    def __iter__(self) -> Iterator[Index]:
        self._require_generated()
        return self.iter_leq(self.N_layers)

    def __len__(self) -> int:
        self._require_generated()
        return sum(len(layer) for layer in self._layers)

    def count_leq(self, k: int) -> int:
        self._require_generated()
        k = min(k, self.N_layers)
        return sum(len(self._layers[s]) for s in range(k + 1))
