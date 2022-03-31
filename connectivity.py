import numpy as np
import itertools
from typing import Iterable, Tuple

from distribution import draw_bernoulli


def all2all(
    source: Iterable[int], dest: Iterable[int]
) -> Tuple[np.ndarray, np.ndarray]:

    # dest = np.asarray(dest)
    # source = np.asarray(source)
    # return np.repeat(source, dest.size), np.tile(dest, source.size)

    return tuple(np.array(e) for e in zip(*[*itertools.product(source, dest)]))


def bernoulli(
    source: Iterable[int], dest: Iterable[int], p: float
) -> Tuple[np.ndarray, np.ndarray]:

    dest = np.asarray(dest)
    source = np.asarray(source)

    dest_candidates = np.tile(dest, source.size)
    source_candidates = np.repeat(source, dest.size)

    source_samples = draw_bernoulli(p=p, size=dest_candidates.size)

    return source_candidates[source_samples], dest_candidates[source_samples]
