"""Independent, reproducible random streams (Appendix B.4).

Final runs use "independent recorded streams for training, view selection, and evaluation".
Seeds are derived statelessly from ``(run seed, stream, indices)`` so that every draw is
reproducible from the iteration index alone, which also makes resuming trivial.
"""

from __future__ import annotations

import numpy as np

_STREAMS = {
    "order": 0,
    "rollout": 1,
    "views": 2,
    "weights": 3,
    "evaluation": 4,
    "init": 5,
    # Streams of the supplementary experiments. They are separate from the ones above, so
    # enabling student-side augmentation (E14) or sampling analysis states (E16, E17) leaves
    # prompt order, rollouts and teacher-view selection of a run unchanged.
    "student_views": 6,
    "states": 7,
}


def derive_seed(seed: int, stream: str, *indices: int) -> int:
    """Derive a 32-bit seed for ``stream`` (one of ``order``, ``rollout``, ``views``,
    ``weights``, ``evaluation``, ``init``, ``student_views``, ``states``) and optional
    integer indices."""
    if stream not in _STREAMS:
        raise KeyError(f"unknown random stream {stream!r}")
    entropy = [int(seed), _STREAMS[stream], *[int(i) for i in indices]]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def derive_rng(seed: int, stream: str, *indices: int) -> np.random.Generator:
    """NumPy generator seeded with :func:`derive_seed`."""
    return np.random.default_rng(derive_seed(seed, stream, *indices))
