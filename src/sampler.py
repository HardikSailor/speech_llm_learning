"""Stage 5a - length bucketing.

A naive shuffled batch pairs a 2 s clip with a 19 s clip: the short one gets padded to 19 s
and you pay ~10x its real cost. Sorting by duration and cutting batches out of the sorted
list makes every batch nearly uniform; we then shuffle the *batch order* so the model still
sees a random sequence of examples.

Trade-off worth naming in an interview: perfect bucketing correlates batch content with
utterance length (all short clips together), which slightly biases gradient estimates.
The usual compromise is a shuffled pool: sort within a window of ~50 batches, not globally.
"""
from __future__ import annotations

import random
from typing import Iterator, List, Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[List[int]]):
    def __init__(self, durations: Sequence[float], batch_size: int, shuffle: bool = True,
                 pool_batches: int = 50, seed: int = 0, drop_last: bool = False):
        self.durations = list(durations)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool = pool_batches * batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.durations)))
        if self.shuffle:
            rng.shuffle(idx)
        # sort by duration inside a pool, so bucketing is local rather than global
        out: List[List[int]] = []
        for p in range(0, len(idx), self.pool):
            chunk = sorted(idx[p:p + self.pool], key=lambda i: self.durations[i])
            out += [chunk[i:i + self.batch_size] for i in range(0, len(chunk), self.batch_size)]
        if self.drop_last:
            out = [b for b in out if len(b) == self.batch_size]
        if self.shuffle:
            rng.shuffle(out)
        return out

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


def padding_waste(durations: Sequence[float], batches: Sequence[Sequence[int]]) -> float:
    """Fraction of batched audio that is padding. Print it before and after bucketing."""
    real = padded = 0.0
    for b in batches:
        d = [durations[i] for i in b]
        real += sum(d)
        padded += max(d) * len(d)
    return 1.0 - real / padded
