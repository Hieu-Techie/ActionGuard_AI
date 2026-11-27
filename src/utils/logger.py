"""
logger.py — Event Transition Logger for ActionGuard AI (Milestone 5A)
=====================================================================
Persists alert state transitions and operator actions (e.g., RED latch
acknowledgment) to `logs/events.csv` and provides helpers for reading
recent events and exporting CSV bytes in the Streamlit UI.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_CSV = PROJECT_ROOT / "logs" / "events.csv"

EVENT_COLUMNS: List[str] = [
    "timestamp",
    "source",
    "model",
    "prev_level",
    "new_level",
    "action_label",
    "confidence",
    "frame_idx",
    "video_time_s",
    "note",
]


def _resolve_csv_path(csv_path: Optional[Union[str, Path]] = None) -> Path:
    return Path(csv_path) if csv_path is not None else DEFAULT_EVENTS_CSV


def ensure_events_csv(csv_path: Optional[Union[str, Path]] = None) -> Path:
    """Ensure `logs/events.csv` exists with its standard header row."""
    path = _resolve_csv_path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
            writer.writeheader()
    return path


def log_event(
    *,
    source: str,
    model: str,
    prev_level: str,
    new_level: str,
    action_label: str,
    confidence: float,
    frame_idx: int = 0,
    video_time_s: float = 0.0,
    note: str = "",
    csv_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """
    Append one state-transition or operator event row to `logs/events.csv`.

    Returns the written row dictionary (all string-formatted values).
    """
    path = ensure_events_csv(csv_path)
    row: Dict[str, str] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "source": str(source),
        "model": str(model).upper(),
        "prev_level": str(prev_level),
        "new_level": str(new_level),
        "action_label": str(action_label),
        "confidence": f"{float(confidence):.4f}",
        "frame_idx": str(int(frame_idx)),
        "video_time_s": f"{float(video_time_s):.2f}",
        "note": str(note),
    }
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
        writer.writerow(row)
    return row


def read_recent_events(
    limit: int = 25,
    csv_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, str]]:
    """
    Read up to `limit` most recent event rows from `logs/events.csv`,
    ordered newest-first.
    """
    path = ensure_events_csv(csv_path)
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    if limit > 0:
        rows = rows[-limit:]
    rows.reverse()
    return rows


def clear_events_log(csv_path: Optional[Union[str, Path]] = None) -> None:
    """Reset `logs/events.csv` to a header-only file."""
    path = _resolve_csv_path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
        writer.writeheader()


def export_events_csv_bytes(csv_path: Optional[Union[str, Path]] = None) -> bytes:
    """Return UTF-8 bytes of `logs/events.csv` for `st.download_button`."""
    path = ensure_events_csv(csv_path)
    return path.read_bytes()
