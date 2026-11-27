"""
overlay.py — Burn the predicted action label and alert status directly onto video frames.

Uses cv2.putText (ASCII only), so all on-frame strings are plain English.
"""

from __future__ import annotations

import cv2
import numpy as np

# BGR colours, status text, text colour
_LEVEL_STYLE = {
    "GREEN":  ((70, 170, 60),  "NORMAL",     (255, 255, 255)),
    "YELLOW": ((0, 190, 255),  "WARNING",    (0, 0, 0)),
    "RED":    ((40, 40, 230),  "FALL ALERT", (255, 255, 255)),
}

_LABEL_TEXT = {
    "Unknown": "Analyzing...",
    "No_Person": "No person",
    "Low_Pose_Quality": "Low pose quality",
    "Idle": "Idle",
    "Sit_Down": "Sit Down",
    "Stand_Up": "Stand Up",
}


def _filled_label(img, text, origin, scale, thick, bg, fg, pad):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = origin
    cv2.rectangle(img, (x, y), (x + tw + 2 * pad, y + th + base + 2 * pad), bg, -1)
    cv2.putText(img, text, (x + pad, y + th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, thick, cv2.LINE_AA)
    return tw + 2 * pad, th + base + 2 * pad


def annotate_frame(frame_bgr: np.ndarray, label: str, confidence: float, level_name: str) -> np.ndarray:
    """Return a copy of `frame_bgr` with action label + confidence (top-left),
    alert status (top-right) and a coloured border for WARNING / FALL ALERT."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    bg, status, fg = _LEVEL_STYLE.get(level_name, _LEVEL_STYLE["GREEN"])

    scale = max(0.5, w / 800.0)
    thick = max(1, int(round(scale * 2)))
    pad = max(4, int(round(6 * scale)))

    text = _LABEL_TEXT.get(label, label.replace("_", " "))
    if label in ("Unknown", "No_Person", "Low_Pose_Quality", "") or confidence <= 0.0:
        label_text = text
    else:
        label_text = f"{text} {confidence * 100:.0f}%"
    _filled_label(out, label_text, (pad, pad), scale, thick, bg, fg, pad)

    (sw, _), _b = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, thick)
    _filled_label(out, status, (max(pad, w - sw - 3 * pad), pad), scale * 0.8, thick, bg, fg, pad)

    if level_name in ("YELLOW", "RED"):
        border = max(3, int(round((8 if level_name == "RED" else 4) * scale)))
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), bg, border)
    return out
