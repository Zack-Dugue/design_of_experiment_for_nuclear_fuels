"""
time_schedules.py

Central place for describing the regular/periodic time grids used by the HGR
models.

Why this file exists
--------------------
The original code treated the time-step format as a boolean:
    False -> schedule 0
    True  -> schedule 1
That made it impossible to cleanly add a third schedule because any nonzero
schedule id was collapsed to True inside the positional encoding.

The code now supports both:
    1. named/integer schedule ids, for grids you know ahead of time; and
    2. explicit time-value vectors parsed from CSV headers, for arbitrary
       regular/periodic grids.

To add a third known format, add one entry to TIME_SCHEDULE_GAPS, for example:

    TIME_SCHEDULE_GAPS[2] = [0.001, 0.999, 3, 3, 6, 6]

The model can also consume an explicit tensor of time values, so the loader can
handle CSVs whose time columns do not exactly match one of these registered
patterns.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


# Existing schedules from the original PositionalEncoding._create_time_seqs.
# The keys are stable schedule ids. The values are the repeating gaps.
TIME_SCHEDULE_GAPS: Dict[int, list[float]] = {
    0: [0.001, 0.999, 4, 5, 5, 5, 5],
    1: [0.001, 0.999, 2, 2, 5, 5, 5, 2, 2, 2],
    2: [0.001, 0.999, 2, 2, 5, 5, 5, 2, 2, 1],
}


N_STATIC_COLUMNS = 9

def create_time_sequence_from_gaps(gaps: Sequence[float], max_len: int) -> torch.Tensor:
    """
    Create the absolute time sequence implied by a periodic gap pattern.

    This preserves the behavior of the original implementation:
    - generate cumulative sums from the repeated gaps;
    - drop the leading zero-like cumulative entry;
    - force the first returned time value to exactly 0.

    Args:
        gaps: Repeating list of time gaps.
        max_len: Number of time points to return.

    Returns:
        Tensor with shape [max_len].
    """
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")
    if len(gaps) == 0:
        raise ValueError("gaps must contain at least one gap")

    # Original code internally used MAX_LEN + 1 before dropping the first value.
    needed = max_len + 1
    repeated: list[float] = []
    for _ in range((needed // len(gaps)) + 2):
        repeated.extend(float(g) for g in gaps)
    repeated = repeated[:needed]

    cumulative = [sum(repeated[:i]) for i in range(len(repeated))]
    seq = cumulative[1:]
    seq = seq[:max_len]
    seq[0] = 0.0
    return torch.tensor(seq, dtype=torch.float32)


def create_time_sequence(schedule_id: int, max_len: int) -> torch.Tensor:
    """Create a time sequence for a registered integer schedule id."""
    schedule_id = int(schedule_id)
    if schedule_id not in TIME_SCHEDULE_GAPS:
        raise KeyError(
            f"Unknown time schedule id {schedule_id}. "
            f"Known ids are {sorted(TIME_SCHEDULE_GAPS)}. "
            "Add the new periodic gap pattern to TIME_SCHEDULE_GAPS in time_schedules.py, "
            "or pass explicit time values instead of an integer schedule id."
        )
    return create_time_sequence_from_gaps(TIME_SCHEDULE_GAPS[schedule_id], max_len=max_len)


def create_all_registered_time_sequences(max_len: int) -> torch.Tensor:
    """
    Return all registered time sequences stacked as [num_schedules, max_len].

    Schedule ids are expected to be contiguous 0..N-1 for fast indexing inside
    the model. This catches accidental ids like {0, 1, 7} early.
    """
    ids = sorted(TIME_SCHEDULE_GAPS)
    if ids != list(range(len(ids))):
        raise ValueError(
            f"TIME_SCHEDULE_GAPS keys must be contiguous ids 0..N-1, got {ids}."
        )
    return torch.stack([create_time_sequence(i, max_len=max_len) for i in ids], dim=0)


def parse_time_columns(columns: Iterable[object]) -> np.ndarray:
    """
    Parse CSV time-series column labels into floats.

    The patched preprocessing uses the first N_STATIC_COLUMNS columns as static/configuration
    features and the remaining columns as time points. Pandas may load those
    labels as strings, ints, or floats depending on the file, so this function
    normalizes them.
    """
    out: list[float] = []
    for col in columns:
        try:
            out.append(float(col))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Could not parse time-series column label {col!r} as a float. "
                "Expected all columns after the first N_STATIC_COLUMNS static columns to be time values."
            ) from e
    return np.asarray(out, dtype=np.float32)


def infer_registered_schedule_id(
    time_values: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    schedules: Mapping[int, Sequence[float]] | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> int | None:
    """
    Try to match an explicit time vector to one of the registered schedules.

    Returns None if there is no exact-enough match. The dataset no longer
    requires a match, because the model can use explicit time values directly.
    """
    if schedules is None:
        schedules = TIME_SCHEDULE_GAPS
    values = torch.as_tensor(time_values, dtype=torch.float32).cpu().numpy()
    max_len = int(values.shape[0])
    for schedule_id, gaps in schedules.items():
        expected = create_time_sequence_from_gaps(gaps, max_len=max_len).cpu().numpy()
        if np.allclose(values, expected, atol=atol, rtol=rtol):
            return int(schedule_id)
    return None


def resolve_time_values(t: int | Sequence[float] | np.ndarray | torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Convert either a schedule id or explicit time values to a [max_len] tensor.

    This is useful for synthetic CSV creation and active-learning queries.
    """
    if isinstance(t, (int, np.integer)):
        return create_time_sequence(int(t), max_len=max_len)

    tensor = torch.as_tensor(t, dtype=torch.float32).flatten()
    if tensor.numel() < max_len:
        raise ValueError(
            f"Explicit time vector has length {tensor.numel()}, but max_len={max_len}."
        )
    return tensor[:max_len]
