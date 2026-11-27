"""
sequence_buffer.py — Sliding window buffer for action classification inference.

Maintains a fixed-length deque of normalised keypoint frames.
When full, exposes tensors shaped for LSTM and ST-GCN respectively.
Handles missing-person / low-confidence frames consistently with training
extraction (skeleton_utils.py) per spec.md Section 4 & 5.
"""

from __future__ import annotations

import collections
from typing import Optional

import numpy as np
import torch

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from utils.config import WINDOW_SIZE, NUM_JOINTS, NUM_COORDS


class SequenceBuffer:
    """
    Fixed-length sliding window buffer for normalised keypoint frames.

    Missing-frame policy (synchronised with `skeleton_utils.clip_to_keypoint_sequence`):
      • Brief detector dropouts (`1 <= consecutive_missing <= max_missing_gap`,
        default 3 frames) append a zero keypoint vector `(NUM_JOINTS, NUM_COORDS)`
        with confidence = 0.0 — matching the exact zero-padding policy used during
        training data extraction so there is no train/inference skew.
      • Prolonged absence (`consecutive_missing > max_missing_gap`) flushes the
        buffer so stale frames from a prior action are never mixed with a new
        action when a person re-enters the camera view.

    Parameters
    ----------
    window_size     : int — number of frames to buffer (default: WINDOW_SIZE)
    max_missing_gap : int — max consecutive missing frames tolerated before flushing
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        max_missing_gap: int = 3,
    ) -> None:
        self._window_size = window_size
        self._max_missing_gap = max_missing_gap
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=window_size)
        self._valid_flags: collections.deque[bool] = collections.deque(maxlen=window_size)
        self.consecutive_missing: int = 0

    # ── Mutation ──────────────────────────────────────────────────────────────

    def append(self, keypoint_frame: Optional[np.ndarray]) -> None:
        """
        Push one normalised keypoint frame into the buffer.

        Parameters
        ----------
        keypoint_frame : np.ndarray of shape (NUM_JOINTS, NUM_COORDS), or None.
        """
        if keypoint_frame is None:
            self.consecutive_missing += 1
            if self.consecutive_missing > self._max_missing_gap:
                # Prolonged absence: flush stale frames so old and new actions never mix
                self._buffer.clear()
                self._valid_flags.clear()
                return

            # Brief dropout (<= max_missing_gap): zero-pad (confidence=0.0),
            # identical to skeleton_utils.clip_to_keypoint_sequence()
            frame = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
            self._buffer.append(frame)
            self._valid_flags.append(False)
        else:
            frame = np.asarray(keypoint_frame, dtype=np.float32)
            if frame.shape != (NUM_JOINTS, NUM_COORDS):
                raise ValueError(
                    f"keypoint_frame must have shape ({NUM_JOINTS}, {NUM_COORDS}), "
                    f"got {frame.shape}"
                )
            self.consecutive_missing = 0
            self._buffer.append(frame)
            self._valid_flags.append(True)

    def reset(self) -> None:
        """Clear all frames and state from the buffer."""
        self._buffer.clear()
        self._valid_flags.clear()
        self.consecutive_missing = 0

    # ── State ─────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Return True when the buffer holds exactly WINDOW_SIZE frames."""
        return len(self._buffer) == self._window_size

    def has_valid_subject(self, min_valid_frames: Optional[int] = None) -> bool:
        """
        Return True if the buffer is full and contains at least `min_valid_frames`
        genuine person detections (default: >= 50% of window_size, matching
        `max_missing_ratio = 0.5` in `skeleton_utils.clip_to_keypoint_sequence`).
        """
        if not self.is_ready():
            return False
        if min_valid_frames is None:
            min_valid_frames = max(1, (self._window_size + 1) // 2)
        return sum(self._valid_flags) >= min_valid_frames

    @property
    def valid_frame_count(self) -> int:
        """Number of observed (non-zero-padded) frames currently in the window."""
        return sum(self._valid_flags)

    def __len__(self) -> int:
        return len(self._buffer)

    # ── Tensor conversion ─────────────────────────────────────────────────────

    def as_lstm_tensor(self) -> torch.Tensor:
        """
        Return buffer as a float32 tensor for LSTM inference.

        Shape: (1, WINDOW_SIZE, NUM_JOINTS * NUM_COORDS)
        Raises RuntimeError if the buffer is not ready.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer not ready: has {len(self._buffer)}/{self._window_size} frames."
            )
        arr = np.stack(list(self._buffer), axis=0)                      # (T, J, C)
        arr = arr.reshape(self._window_size, NUM_JOINTS * NUM_COORDS)   # (T, J*C)
        return torch.from_numpy(arr).unsqueeze(0)                       # (1, T, J*C)

    def as_stgcn_tensor(self) -> torch.Tensor:
        """
        Return buffer as a float32 tensor for ST-GCN inference.

        Shape: (1, NUM_COORDS, WINDOW_SIZE, NUM_JOINTS) — (N, C, T, V) convention.
        Raises RuntimeError if the buffer is not ready.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer not ready: has {len(self._buffer)}/{self._window_size} frames."
            )
        arr = np.stack(list(self._buffer), axis=0)   # (T, J, C)
        arr = arr.transpose(2, 0, 1)                 # (C, T, J)
        arr = arr[np.newaxis, ...]                   # (1, C, T, J)
        return torch.from_numpy(arr.copy())
