from __future__ import annotations

from app.ocr_postprocess.utils import bbox_center_x, bbox_center_y, is_arabic


def sort_words_in_line(line: list[dict]) -> list[dict]:
    if line and all(w.get("synthetic_line_tokens") for w in line):
        return sorted(line, key=lambda w: int(w.get("source_token_index", 0)))
    arabic_line = any(is_arabic(w.get("text", "")) for w in line)
    return sorted(line, key=bbox_center_x, reverse=arabic_line)


def sort_lines(lines: list[list[dict]]) -> list[list[dict]]:
    return sorted(
        (sort_words_in_line(line) for line in lines if line),
        key=lambda line: sum(bbox_center_y(w) for w in line) / len(line),
    )
