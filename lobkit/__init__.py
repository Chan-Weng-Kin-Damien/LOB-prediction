"""Shared library for the Wunder Challenge 2 LOB prediction work.

Extracted from five near-duplicate training scripts (``weighted_corr_loss`` and
``mixed_loss`` were byte-identical across all five; ``iter_sequence_chunks`` in
four of five).

Import policy: ``metric``, ``data``, ``windows``, ``splits`` and ``blend`` are
numpy-only so the evaluation harness runs anywhere. ``losses`` and ``fullseq``
import torch lazily and are only needed for training.

The shipped ``submission_safe_3model/solution.py`` deliberately does **not**
depend on this package -- it must stay self-contained inside the submission zip.
"""

from .metric import (  # noqa: F401
    CLIP_LIMIT,
    bootstrap_score,
    paired_bootstrap,
    score_both_targets,
    weighted_pearson_correlation,
)
from .data import (  # noqa: F401
    FEATURE_DIM,
    RAW_DIM,
    RAW_FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    WARMUP_STEPS,
    LobCorpus,
    engineered_features_numpy,
    iter_sequence_chunks,
)
from .windows import (  # noqa: F401
    ShuffleBuffer,
    make_window,
    make_windows_view,
    parse_dilations,
    receptive_field,
    sample_training_windows,
)
from .splits import (  # noqa: F401
    BLEND_FIT,
    HOLDOUT,
    INNER_VAL,
    TRAIN,
    build_splits,
    group_kfold,
    load_splits,
    save_splits,
)

__all__ = [name for name in dir() if not name.startswith("_")]
