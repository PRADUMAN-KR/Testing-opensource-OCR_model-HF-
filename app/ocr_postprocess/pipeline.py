from __future__ import annotations

import re
import statistics
from typing import Any

from app.ocr_postprocess.correction import (
    apply_final_regex_cleanup,
    correct_low_confidence_words,
    fix_broken_words,
    load_arabic_dictionary,
    paragraphs_to_text,
)
from app.ocr_postprocess.grouping import group_into_lines
from app.ocr_postprocess.merging import build_lines, merge_lines
from app.ocr_postprocess.normalization import remove_noise
from app.ocr_postprocess.sorting import sort_lines
from app.ocr_postprocess.utils import (
    bbox_center_x,
    bbox_height,
    is_arabic,
    normalize_token_text,
    split_line_tokens,
)


EXCLUDED_NOISE_REASONS = {
    "low_confidence_repeated_short_arabic_fragments",
}
TITLE_LINES = {
    "حدود الكويت مع العراق",
    "حقائق تاريخية",
}


def _expand_line_or_word_items(words: list[dict]) -> list[dict]:
    expanded: list[dict] = []
    for item in words:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        bbox = item.get("bbox")
        confidence = float(item.get("confidence", 1.0))
        tokens = split_line_tokens(
            text,
            confidence,
            bbox,
            synthetic_line_tokens=bool(item.get("line_level", False)),
        )
        if len(tokens) <= 1:
            expanded.append({
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
            })
        else:
            expanded.extend(tokens)
    return expanded


def _bbox_top(line: dict) -> float:
    bbox = line.get("bbox") or [0, 0, 0, 0]
    return float(bbox[1])


def _bbox_bottom(line: dict) -> float:
    bbox = line.get("bbox") or [0, 0, 0, 0]
    return float(bbox[3])


def _bbox_center_y(line: dict) -> float:
    bbox = line.get("bbox") or [0, 0, 0, 0]
    return (float(bbox[1]) + float(bbox[3])) / 2.0


def _bbox_union(items: list[dict]) -> list[int]:
    boxes = [item.get("bbox") for item in items if item.get("bbox") and len(item.get("bbox")) == 4]
    if not boxes:
        return [0, 0, 0, 0]
    return [
        int(min(box[0] for box in boxes)),
        int(min(box[1] for box in boxes)),
        int(max(box[2] for box in boxes)),
        int(max(box[3] for box in boxes)),
    ]


def _avg_confidence(items: list[dict]) -> float:
    if not items:
        return 0.0
    return sum(float(item.get("confidence", 0.0)) for item in items) / len(items)


def _x_overlap_ratio(a: list[int], b: list[int]) -> float:
    inter = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    denom = max(1.0, min(float(a[2] - a[0]), float(b[2] - b[0])))
    return inter / denom


def _is_digit_text(text: str) -> bool:
    normalized = (text or "").strip().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    normalized = re.sub(r"[,\.\s]", "", normalized)
    return bool(normalized) and normalized.isdigit()


def _numeric_text(text: str) -> str:
    translated = (text or "").strip().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return re.sub(r"[^\d.]", "", translated)


def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _median_line_height_from_debug(lines: list[dict]) -> float:
    heights = [
        float(line["bbox"][3] - line["bbox"][1])
        for line in lines
        if line.get("bbox") and len(line["bbox"]) == 4 and line["bbox"][3] > line["bbox"][1]
    ]
    return float(sorted(heights)[len(heights) // 2]) if heights else 18.0


def _cluster_by_axis(items: list[dict], center_fn, threshold: float) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    centers: list[float] = []
    for item in sorted(items, key=center_fn):
        center = float(center_fn(item))
        best_idx = None
        best_delta = threshold
        for idx, cluster_center in enumerate(centers):
            delta = abs(center - cluster_center)
            if delta <= best_delta:
                best_idx = idx
                best_delta = delta
        if best_idx is None:
            clusters.append([item])
            centers.append(center)
            continue
        clusters[best_idx].append(item)
        centers[best_idx] = (
            centers[best_idx] * (len(clusters[best_idx]) - 1) + center
        ) / len(clusters[best_idx])
    return clusters


def _detect_table_grid_from_image(image_rgb: Any | None) -> dict:
    if image_rgb is None:
        return {
            "table_mode": False,
            "source": "image_unavailable",
            "horizontal_line_count": 0,
            "vertical_line_count": 0,
        }
    try:
        import cv2
        import numpy as np

        img = np.asarray(image_rgb)
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("empty image")

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12,
        )
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 18), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 18)))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

        def count_lines(mask, orientation: str) -> int:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if orientation == "horizontal":
                    if bw >= w * 0.18 and bw >= bh * 5:
                        count += 1
                else:
                    if bh >= h * 0.12 and bh >= bw * 5:
                        count += 1
            return count

        horizontal_count = count_lines(horizontal, "horizontal")
        vertical_count = count_lines(vertical, "vertical")
        return {
            "table_mode": horizontal_count >= 4 and vertical_count >= 3,
            "source": "image_grid",
            "horizontal_line_count": horizontal_count,
            "vertical_line_count": vertical_count,
        }
    except Exception as exc:
        return {
            "table_mode": False,
            "source": "image_grid_error",
            "horizontal_line_count": 0,
            "vertical_line_count": 0,
            "error": str(exc),
        }


def _detect_table_grid_from_bboxes(lines: list[dict]) -> dict:
    usable = [line for line in lines if line.get("bbox") and len(line.get("bbox")) == 4]
    if len(usable) < 9:
        return {
            "table_mode": False,
            "source": "bbox_grid",
            "row_count": 0,
            "column_count": 0,
        }
    heights = [float(line["bbox"][3] - line["bbox"][1]) for line in usable if line["bbox"][3] > line["bbox"][1]]
    widths = [float(line["bbox"][2] - line["bbox"][0]) for line in usable if line["bbox"][2] > line["bbox"][0]]
    median_h = statistics.median(heights) if heights else 18.0
    median_w = statistics.median(widths) if widths else 45.0
    rows = _cluster_by_axis(usable, _bbox_center_y, max(10.0, median_h * 0.80))
    cols = _cluster_by_axis(usable, bbox_center_x, max(18.0, median_w * 0.95))
    populated_rows = sum(1 for row in rows if len(row) >= 2)
    return {
        "table_mode": len(rows) >= 4 and len(cols) >= 3 and populated_rows >= 3,
        "source": "bbox_grid",
        "row_count": len(rows),
        "column_count": len(cols),
        "populated_row_count": populated_rows,
    }


def detect_table_layout(lines: list[dict], image_rgb: Any | None = None) -> dict:
    image_detection = _detect_table_grid_from_image(image_rgb)
    if image_detection.get("table_mode"):
        return image_detection
    bbox_detection = _detect_table_grid_from_bboxes(lines)
    if bbox_detection.get("table_mode"):
        bbox_detection["image_detection"] = image_detection
        return bbox_detection
    image_detection["bbox_detection"] = bbox_detection
    return image_detection


def _merge_layout_lines(layout_lines: list[dict]) -> list[list[dict]]:
    if not layout_lines:
        return []
    median_h = _median_line_height_from_debug(layout_lines)
    paragraphs: list[list[dict]] = []
    current: list[dict] = []
    current_bbox: list[int] | None = None

    for line in sorted(layout_lines, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        line_words = line.get("words") or []
        line_bbox = line.get("bbox") or _bbox_union(line_words)
        line_text = " ".join(word.get("text", "") for word in line_words).strip()
        if not current:
            current = list(line_words)
            current_bbox = line_bbox
            continue

        assert current_bbox is not None
        current_text = " ".join(word.get("text", "") for word in current).strip()
        vertical_gap = float(line_bbox[1] - current_bbox[3])
        center_delta = abs(
            ((line_bbox[1] + line_bbox[3]) / 2.0)
            - ((current_bbox[1] + current_bbox[3]) / 2.0)
        )
        should_merge = (
            vertical_gap >= 0
            and vertical_gap < 1.6 * median_h
            and _x_overlap_ratio(current_bbox, line_bbox) > 0.45
            and center_delta < 2.6 * median_h
            and current_text != "TODAY"
            and line_text != "TODAY"
        )
        if should_merge:
            current.extend(line_words)
            current_bbox = _bbox_union(current)
        else:
            paragraphs.append(current)
            current = list(line_words)
            current_bbox = line_bbox

    if current:
        paragraphs.append(current)
    return paragraphs


def _arabic_word_tokens(words: list[dict]) -> list[str]:
    out = []
    for word in words:
        text = normalize_token_text(word.get("text", ""))
        if is_arabic(text) and sum(1 for ch in text if "\u0600" <= ch <= "\u06FF") >= 2:
            out.append(text)
    return out


def _arabic_base_len(text: str) -> int:
    return sum(
        1
        for ch in normalize_token_text(text)
        if "\u0600" <= ch <= "\u06FF" and not ("\u064B" <= ch <= "\u0652")
    )


def line_noise_score(text: str) -> dict:
    """
    Score whether a line is detector noise. Confidence is intentionally not part
    of the score; deletion requires low confidence plus this independent signal.
    """
    raw = text or ""
    tokens = re.findall(r"\S+", raw)
    arabic_lengths = [_arabic_base_len(token) for token in tokens]
    arabic_lengths = [length for length in arabic_lengths if length > 0]
    arabic_token_count = len(arabic_lengths)

    repeated_fragment_ratio = (
        sum(1 for length in arabic_lengths if length <= 2) / arabic_token_count
        if arabic_token_count else 0.0
    )

    non_space_chars = [ch for ch in raw if not ch.isspace()]
    latin_chars = sum(1 for ch in non_space_chars if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    latin_garbage_ratio = latin_chars / len(non_space_chars) if non_space_chars else 0.0

    avg_word_length = (
        sum(arabic_lengths) / arabic_token_count
        if arabic_token_count else 0.0
    )
    short_average_word_length = avg_word_length < 2.5

    dictionary = load_arabic_dictionary()
    normalized_tokens = [normalize_token_text(token) for token in tokens]
    arabic_tokens = [
        token for token in normalized_tokens
        if _arabic_base_len(token) > 0
    ]
    known = sum(1 for token in arabic_tokens if token in dictionary)
    dictionary_coverage = known / len(arabic_tokens) if arabic_tokens else 0.0
    low_dictionary_coverage = dictionary_coverage < 0.25
    has_meaningful_arabic_word = any(length >= 4 for length in arabic_lengths)

    score = (
        repeated_fragment_ratio * 0.35
        + min(1.0, latin_garbage_ratio * 3.0) * 0.20
        + (0.15 if short_average_word_length else 0.0)
        + (0.15 if low_dictionary_coverage else 0.0)
        + (0.15 if not has_meaningful_arabic_word else 0.0)
    )

    return {
        "score": round(min(1.0, score), 4),
        "repeated_fragment_ratio": round(repeated_fragment_ratio, 4),
        "latin_garbage_ratio": round(latin_garbage_ratio, 4),
        "avg_word_length": round(avg_word_length, 4),
        "short_average_word_length": short_average_word_length,
        "dictionary_coverage": round(dictionary_coverage, 4),
        "low_dictionary_coverage": low_dictionary_coverage,
        "has_meaningful_arabic_word": has_meaningful_arabic_word,
    }


def _detect_columns(words: list[dict]) -> list[list[dict]]:
    usable = [w for w in words if w.get("bbox") and len(w.get("bbox")) == 4]
    if len(usable) < 6:
        return [usable]

    centers = sorted(bbox_center_x(word) for word in usable)
    gaps = [(centers[idx + 1] - centers[idx], idx) for idx in range(len(centers) - 1)]
    max_gap, split_idx = max(gaps, key=lambda item: item[0])
    page_width = max((float(w["bbox"][2]) for w in usable), default=0.0)
    if page_width <= 0 or max_gap / page_width < 0.12:
        return [usable]

    split_x = (centers[split_idx] + centers[split_idx + 1]) / 2.0
    left = [word for word in usable if bbox_center_x(word) < split_x]
    right = [word for word in usable if bbox_center_x(word) >= split_x]
    columns = [col for col in (right, left) if col]
    return columns or [usable]


def reconstruct_arabic_layout(words: list[dict]) -> dict:
    """
    Build deterministic Arabic reading order from word-level boxes.

    Layout order is: columns right-to-left, blocks top-to-bottom inside each
    column, lines by y-center inside each block, words right-to-left in each
    Arabic line.
    """
    source_words = [
        dict(word)
        for word in words
        if (word.get("text") or "").strip() and word.get("bbox") and len(word.get("bbox")) == 4
    ]
    line_level_source = [word for word in source_words if word.get("line_level")]
    column_source = line_level_source if line_level_source else _expand_line_or_word_items(source_words)
    columns = _detect_columns(column_source)
    debug_lines: list[dict] = []
    debug_blocks: list[dict] = []

    for column_index, column_words in enumerate(columns):
        if not column_words:
            continue
        column_words = (
            _expand_line_or_word_items(column_words)
            if line_level_source
            else column_words
        )
        lines = sort_lines(group_into_lines(column_words))
        if not lines:
            continue

        line_heights = [max(bbox_height(word) for word in line) for line in lines if line]
        median_height = sorted(line_heights)[len(line_heights) // 2] if line_heights else 18.0
        block_gap = median_height * 1.8
        current_block: list[list[dict]] = []
        previous_bottom: float | None = None

        def flush_block() -> None:
            if not current_block:
                return
            block_line_indices: list[int] = []
            block_words: list[dict] = []
            for line in current_block:
                line_words = sorted(line, key=bbox_center_x, reverse=True)
                block_words.extend(line_words)
                line_index = len(debug_lines)
                block_line_indices.append(line_index)
                text = " ".join(word["text"] for word in line_words if word.get("text")).strip()
                debug_lines.append({
                    "line_index": line_index,
                    "column_index": column_index,
                    "block_index": len(debug_blocks),
                    "text": text,
                    "confidence": round(_avg_confidence(line_words), 4),
                    "bbox": _bbox_union(line_words),
                    "words": [
                        {
                            "text": word.get("text", ""),
                            "confidence": round(float(word.get("confidence", 0.0)), 4),
                            "bbox": word.get("bbox"),
                        }
                        for word in line_words
                    ],
                })
            debug_blocks.append({
                "block_index": len(debug_blocks),
                "column_index": column_index,
                "bbox": _bbox_union(block_words),
                "line_indices": block_line_indices,
                "text": "\n".join(debug_lines[idx]["text"] for idx in block_line_indices),
            })

        for line in lines:
            line_bbox = _bbox_union(line)
            if (
                previous_bottom is not None
                and line_bbox[1] - previous_bottom > block_gap
            ):
                flush_block()
                current_block = []
            current_block.append(line)
            previous_bottom = float(line_bbox[3])
        flush_block()

    return {
        "words": _expand_line_or_word_items(source_words),
        "lines": debug_lines,
        "blocks": debug_blocks,
    }


def score_ocr_words(words: list[dict]) -> dict:
    expanded_words = _expand_line_or_word_items(words)
    confidences = [float(word.get("confidence", 0.0)) for word in expanded_words]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    arabic_tokens = _arabic_word_tokens(expanded_words)
    arabic_word_count = len(arabic_tokens)
    arabic_word_count_score = min(arabic_word_count / 80.0, 1.0) * 0.25

    dictionary = load_arabic_dictionary()
    known = sum(1 for token in arabic_tokens if token in dictionary)
    dictionary_coverage = known / arabic_word_count if arabic_word_count else 0.0
    dictionary_coverage_score = dictionary_coverage * 0.25

    low_confidence_count = sum(1 for conf in confidences if conf < 0.70)
    low_confidence_penalty = (
        low_confidence_count / len(confidences) * 0.25
        if confidences else 0.0
    )

    broken_count = 0
    for token in arabic_tokens:
        if len(token) <= 2 and token not in dictionary:
            broken_count += 1
    broken_word_penalty = (
        broken_count / arabic_word_count * 0.20
        if arabic_word_count else 0.0
    )

    score = (
        avg_confidence
        + arabic_word_count_score
        + dictionary_coverage_score
        - low_confidence_penalty
        - broken_word_penalty
    )

    return {
        "score": round(score, 4),
        "avg_confidence": round(avg_confidence, 4),
        "arabic_word_count": arabic_word_count,
        "arabic_word_count_score": round(arabic_word_count_score, 4),
        "dictionary_coverage": round(dictionary_coverage, 4),
        "dictionary_coverage_score": round(dictionary_coverage_score, 4),
        "low_confidence_penalty": round(low_confidence_penalty, 4),
        "broken_word_penalty": round(broken_word_penalty, 4),
    }


def _is_digit_only_line(line: dict, max_bottom: float) -> bool:
    text = (line.get("text") or "").strip()
    if not text:
        return False
    normalized_digits = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    if not normalized_digits.isdigit():
        return False
    return _bbox_top(line) >= max_bottom - 80


def _split_lines_for_final_text(lines: list[dict]) -> tuple[list[dict], list[dict]]:
    if not lines:
        return [], []
    min_top = min(_bbox_top(line) for line in lines)
    title_lines: list[dict] = []
    body_lines: list[dict] = []
    for line in lines:
        text = (line.get("text") or "").strip()
        is_title = (
            text in TITLE_LINES
            and float(line.get("confidence", 0.0)) >= 0.85
            and _bbox_top(line) <= min_top + 90
        )
        if is_title:
            title_lines.append(line)
        else:
            body_lines.append(line)
    return title_lines, body_lines


def _group_table_rows(lines: list[dict]) -> list[list[dict]]:
    usable = [line for line in lines if line.get("bbox") and len(line.get("bbox")) == 4]
    if not usable:
        return []
    heights = [float(line["bbox"][3] - line["bbox"][1]) for line in usable if line["bbox"][3] > line["bbox"][1]]
    median_h = statistics.median(heights) if heights else 18.0
    rows = _cluster_by_axis(usable, _bbox_center_y, max(10.0, median_h * 0.85))
    return sorted(rows, key=lambda row: min(_bbox_top(item) for item in row))


def _table_column_centers(rows: list[list[dict]]) -> list[float]:
    items = [item for row in rows for item in row if item.get("bbox") and len(item.get("bbox")) == 4]
    if not items:
        return []
    widths = [float(item["bbox"][2] - item["bbox"][0]) for item in items if item["bbox"][2] > item["bbox"][0]]
    median_w = statistics.median(widths) if widths else 45.0
    clusters = _cluster_by_axis(items, bbox_center_x, max(18.0, median_w * 0.95))
    centers = [
        sum(bbox_center_x(item) for item in cluster) / len(cluster)
        for cluster in clusters
        if cluster
    ]
    return sorted(centers, reverse=True)


def _table_cell_text(items: list[dict]) -> str:
    if not items:
        return ""
    arabic_cell = any(_has_arabic(item.get("text", "")) for item in items)
    ordered = sorted(items, key=bbox_center_x, reverse=arabic_cell)
    return " ".join((item.get("text") or "").strip() for item in ordered if (item.get("text") or "").strip())


def _assign_row_to_columns(row: list[dict], centers: list[float]) -> list[str]:
    if not centers:
        return [_table_cell_text(row)]
    cells: list[list[dict]] = [[] for _ in centers]
    for item in row:
        center = bbox_center_x(item)
        idx = min(range(len(centers)), key=lambda col_idx: abs(center - centers[col_idx]))
        cells[idx].append(item)
    return [_table_cell_text(cell) for cell in cells]


def _looks_like_header(cells: list[str]) -> bool:
    non_empty = [cell for cell in cells if cell.strip()]
    if len(non_empty) < 2:
        return False
    arabic_count = sum(1 for cell in non_empty if _has_arabic(cell))
    numeric_count = sum(1 for cell in non_empty if _is_digit_text(cell))
    return arabic_count >= 1 and numeric_count <= max(1, len(non_empty) // 2)


def _extract_table_totals(rows: list[list[str]]) -> dict:
    totals: dict[str, str] = {}
    total_keywords = (
        "ضريبة",
        "الانفاق",
        "الاستهلاكي",
        "الإدارة",
        "الادارة",
        "إعادة",
        "اعادة",
        "الإعمار",
        "الاعمار",
        "المجموع",
        "النهائي",
        "total",
    )
    for cells in rows:
        non_empty = [cell.strip() for cell in cells if cell and cell.strip()]
        if len(non_empty) < 2:
            continue
        numeric_cells = [cell for cell in non_empty if _numeric_text(cell)]
        label_cells = [cell for cell in non_empty if _has_arabic(cell) and not _is_digit_text(cell)]
        if not numeric_cells or not label_cells:
            continue
        label = " ".join(label_cells).strip()
        if not any(keyword in label for keyword in total_keywords):
            continue
        totals[label] = _numeric_text(numeric_cells[-1]) or numeric_cells[-1]
    return totals


def reconstruct_table_layout(lines: list[dict]) -> dict:
    rows = _group_table_rows(lines)
    centers = _table_column_centers(rows)
    table_rows = [_assign_row_to_columns(row, centers) for row in rows]
    table_rows = [
        [cell.strip() for cell in row]
        for row in table_rows
        if any(cell.strip() for cell in row)
    ]

    headers: list[str] = []
    data_rows = table_rows
    for idx, row in enumerate(table_rows[:3]):
        if _looks_like_header(row):
            headers = row
            data_rows = table_rows[idx + 1:]
            break

    totals = _extract_table_totals(data_rows)
    readable_rows = []
    for row in table_rows:
        readable_rows.append(" | ".join(cell for cell in row if cell.strip()))
    return {
        "text": "\n".join(readable_rows),
        "tables": [
            {
                "headers": headers,
                "rows": data_rows,
            }
        ] if table_rows else [],
        "totals": totals,
        "debug": {
            "column_centers_rtl": [round(center, 2) for center in centers],
            "row_count": len(table_rows),
        },
    }


def _run_text_pipeline(words: list[dict]) -> str:
    layout = reconstruct_arabic_layout(words)
    block_texts: list[str] = []
    for block in layout["blocks"]:
        block_lines = [
            layout["lines"][line_idx]
            for line_idx in block.get("line_indices", [])
            if 0 <= line_idx < len(layout["lines"]) and layout["lines"][line_idx].get("words")
        ]
        paragraphs = _merge_layout_lines(block_lines)
        fixed_lines = fix_broken_words(paragraphs)
        corrected_lines = correct_low_confidence_words(fixed_lines)
        block_text = paragraphs_to_text(corrected_lines)
        if block_text:
            block_texts.append(block_text)
    normalized = "\n\n".join(block_texts)
    regex_cleaned = apply_final_regex_cleanup(normalized)
    cleaned = remove_noise(regex_cleaned)
    return cleaned


def postprocess_ocr(words: list[dict]) -> str:
    return postprocess_ocr_result(words)["text"]


def postprocess_ocr_result(
    words: list[dict],
    raw_text: str = "",
    selected_variant: str | None = None,
    variant_scores: list[dict] | None = None,
    image_rgb: Any | None = None,
) -> dict:
    sorted_lines = sorted(
        [dict(line) for line in words if (line.get("text") or "").strip()],
        key=lambda line: (_bbox_top(line), line.get("bbox", [0, 0, 0, 0])[0]),
    )
    max_bottom = max((_bbox_bottom(line) for line in sorted_lines), default=0.0)
    table_detection = detect_table_layout(sorted_lines, image_rgb=image_rgb)
    table_mode = bool(table_detection.get("table_mode"))

    flagged_lines = [line for line in sorted_lines if line.get("filter_reason")]
    accepted_lines: list[dict] = []
    review_lines: list[dict] = []
    excluded_noise_lines: list[dict] = []
    included_lines: list[dict] = []
    per_line_confidence: list[dict] = []
    per_line_noise_score: list[dict] = []

    for line in sorted_lines:
        line = dict(line)
        confidence = float(line.get("confidence", 0.0))
        noise = line_noise_score(line.get("text", ""))
        noise_score = float(noise["score"])
        line["noise_score"] = noise_score
        line["noise_details"] = noise

        per_line_confidence.append({
            "line_index": int(line.get("line_index", len(per_line_confidence))),
            "text": line.get("text", ""),
            "confidence": round(confidence, 4),
        })
        per_line_noise_score.append({
            "line_index": int(line.get("line_index", len(per_line_noise_score))),
            "text": line.get("text", ""),
            "noise_score": noise_score,
            "noise_details": noise,
        })

        reason = line.get("filter_reason")
        if reason in EXCLUDED_NOISE_REASONS:
            excluded = dict(line)
            excluded["status"] = "excluded_noise_line"
            excluded["exclude_reason"] = reason
            excluded_noise_lines.append(excluded)
            continue

        if not table_mode and _is_digit_only_line(line, max_bottom):
            excluded = dict(line)
            excluded["status"] = "excluded_noise_line"
            excluded["exclude_reason"] = "bottom_page_number"
            excluded_noise_lines.append(excluded)
            continue

        if not _is_digit_text(line.get("text", "")) and confidence < 0.60 and noise_score >= 0.65:
            excluded = dict(line)
            excluded["status"] = "excluded_noise_line"
            excluded["exclude_reason"] = "low_confidence_high_noise_score"
            excluded_noise_lines.append(excluded)
            continue

        if line.get("bbox_valid") is False:
            line["status"] = "review_line"
            line["review_reason"] = line.get("review_reason") or "invalid_bbox"
            review_lines.append(line)
            continue

        if table_mode and _is_digit_text(line.get("text", "")):
            line["status"] = "accepted"
            accepted_lines.append(line)
        elif table_mode and confidence >= 0.45:
            line["status"] = "accepted"
            accepted_lines.append(line)
        elif confidence < 0.70:
            line["status"] = "review_line"
            review_lines.append(line)
        else:
            line["status"] = "accepted"
            accepted_lines.append(line)
        included_lines.append(line)

    tables: list[dict] = []
    totals: dict[str, str] = {}
    table_debug: dict = {}
    if table_mode:
        table_result = reconstruct_table_layout(accepted_lines)
        final_text = table_result["text"]
        tables = table_result["tables"]
        totals = table_result["totals"]
        table_debug = table_result["debug"]
        layout = {"lines": [], "blocks": []}
    else:
        title_lines, body_lines = _split_lines_for_final_text(accepted_lines)
        title_text = "\n".join(line["text"].strip() for line in title_lines)
        layout = reconstruct_arabic_layout([
            {**line, "line_level": True}
            for line in body_lines
        ])
        body_text = _run_text_pipeline([
            {**line, "line_level": True}
            for line in body_lines
        ])
        final_parts = [part for part in (title_text, body_text) if part]
        final_text = "\n".join(final_parts)
        final_text = remove_noise(apply_final_regex_cleanup(final_text))

    accepted_conf = [
        float(line.get("confidence", 0.0))
        for line in accepted_lines
    ]
    avg_confidence = (
        sum(accepted_conf) / len(accepted_conf)
        if accepted_conf else 0.0
    )
    score_details = score_ocr_words([
        {**line, "line_level": True}
        for line in accepted_lines
    ])
    warnings: list[str] = []
    if flagged_lines:
        warnings.append("arabic_noise_lines_flagged")
    if review_lines:
        warnings.append("low_confidence_review_lines_present")
    if excluded_noise_lines:
        warnings.append("excluded_high_noise_lines")

    return {
        "final_text": final_text,
        "text": final_text,
        "raw_text": raw_text,
        "selected_variant": selected_variant,
        "confidence_score": score_details["score"],
        "score_details": score_details,
        "variant_scores": variant_scores or [],
        "layout_mode": "table" if table_mode else "text",
        "tables": tables,
        "totals": totals,
        "debug": {
            "table_detection": table_detection,
            "table_layout": table_debug,
        },
        "debug_lines": layout["lines"],
        "debug_blocks": layout["blocks"],
        "accepted_lines": accepted_lines,
        "review_lines": review_lines,
        "flagged_lines": flagged_lines,
        "excluded_noise_lines": excluded_noise_lines,
        "excluded_lines": excluded_noise_lines,
        "per_line_confidence": per_line_confidence,
        "per_line_noise_score": per_line_noise_score,
        "avg_confidence": round(avg_confidence, 4),
        "quality": {
            "status": "review" if warnings else "ok",
            "warnings": warnings,
            "flagged_line_count": len(flagged_lines),
            "review_line_count": len(review_lines),
            "excluded_noise_line_count": len(excluded_noise_lines),
            "excluded_line_count": len(excluded_noise_lines),
        },
    }
