from __future__ import annotations

import statistics

from app.ocr_postprocess.utils import bbox_center_y, bbox_height


def _line_threshold(words: list[dict], default: float = 14.0) -> float:
    heights = [bbox_height(w) for w in words if bbox_height(w) > 0]
    if not heights:
        return default
    median_h = statistics.median(heights)
    return max(10.0, min(18.0, median_h * 0.55))


def group_into_lines(words: list[dict], y_threshold: float | None = None) -> list[list[dict]]:
    usable = [
        w for w in words
        if (w.get("text") or "").strip() and w.get("bbox") and len(w.get("bbox")) == 4
    ]
    if not usable:
        return []

    threshold = y_threshold if y_threshold is not None else _line_threshold(usable)
    sorted_words = sorted(usable, key=bbox_center_y)
    lines: list[list[dict]] = []
    line_centers: list[float] = []

    for word in sorted_words:
        cy = bbox_center_y(word)
        best_idx = None
        best_delta = threshold
        for idx, center in enumerate(line_centers):
            delta = abs(cy - center)
            if delta <= best_delta:
                best_idx = idx
                best_delta = delta

        if best_idx is None:
            lines.append([word])
            line_centers.append(cy)
            continue

        lines[best_idx].append(word)
        line_centers[best_idx] = (
            line_centers[best_idx] * (len(lines[best_idx]) - 1) + cy
        ) / len(lines[best_idx])

    return lines
