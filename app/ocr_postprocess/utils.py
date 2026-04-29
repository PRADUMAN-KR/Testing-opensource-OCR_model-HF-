from __future__ import annotations

import re


ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
TOKEN_RE = re.compile(r"\S+")


def is_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


def bbox_center_y(word: dict) -> float:
    x1, y1, x2, y2 = word.get("bbox") or [0, 0, 0, 0]
    return (float(y1) + float(y2)) / 2.0


def bbox_center_x(word: dict) -> float:
    x1, y1, x2, y2 = word.get("bbox") or [0, 0, 0, 0]
    return (float(x1) + float(x2)) / 2.0


def bbox_height(word: dict) -> float:
    x1, y1, x2, y2 = word.get("bbox") or [0, 0, 0, 0]
    return max(0.0, float(y2) - float(y1))


def normalize_token_text(text: str) -> str:
    return (text or "").replace("\u0640", "").strip()


def levenshtein(a: str, b: str, max_distance: int | None = None) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if max_distance is not None and abs(len(a) - len(b)) > max_distance:
        return max_distance + 1

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        row_min = current[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
            row_min = min(row_min, current[-1])
        if max_distance is not None and row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def split_line_tokens(
    line_text: str,
    confidence: float,
    bbox: list[int] | None,
    *,
    synthetic_line_tokens: bool = False,
) -> list[dict]:
    tokens = TOKEN_RE.findall(line_text or "")
    if not tokens:
        return []
    token_bboxes: list[list[int] | None] = [bbox for _ in tokens]
    if bbox and len(bbox) == 4 and len(tokens) > 1:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        width = max(1, x2 - x1)
        weights = [max(1, len(token)) for token in tokens]
        total_weight = max(1, sum(weights))
        rtl = any(is_arabic(token) for token in tokens)
        token_bboxes = []
        if rtl:
            cursor = x2
            for idx, weight in enumerate(weights):
                if idx == len(weights) - 1:
                    nx1 = x1
                else:
                    nx1 = cursor - max(1, round(width * (weight / total_weight)))
                    nx1 = max(x1, nx1)
                token_bboxes.append([nx1, y1, cursor, y2])
                cursor = nx1
        else:
            cursor = x1
            for idx, weight in enumerate(weights):
                if idx == len(weights) - 1:
                    nx2 = x2
                else:
                    nx2 = cursor + max(1, round(width * (weight / total_weight)))
                    nx2 = min(x2, nx2)
                token_bboxes.append([cursor, y1, nx2, y2])
                cursor = nx2
    out = []
    for idx, token in enumerate(tokens):
        out.append({
            "text": token,
            "confidence": confidence,
            "bbox": token_bboxes[idx],
            "source_token_index": idx,
            "synthetic_line_tokens": synthetic_line_tokens,
        })
    return out
