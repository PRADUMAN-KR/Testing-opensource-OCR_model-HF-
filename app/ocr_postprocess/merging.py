from __future__ import annotations


PUNCTUATION_END = tuple(".،؛؟!:")


def _line_text(line: list[dict]) -> str:
    return " ".join(token["text"] for token in line if token.get("text")).strip()


def build_lines(lines: list[list[dict]]) -> list[list[dict]]:
    text_lines: list[list[dict]] = []
    for line in lines:
        tokens = []
        for word in line:
            text = (word.get("text") or "").strip()
            if not text:
                continue
            tokens.append({
                "text": text,
                "confidence": float(word.get("confidence", 1.0)),
                "bbox": word.get("bbox"),
            })
        if tokens:
            text_lines.append(tokens)
    return text_lines


def merge_lines(text_lines: list[list[dict]]) -> list[list[dict]]:
    paragraphs: list[list[dict]] = []
    current: list[dict] = []

    for line in text_lines:
        line_text = _line_text(line)
        if line_text == "TODAY":
            if current:
                paragraphs.append(current)
                current = []
            paragraphs.append(list(line))
            continue

        if not current:
            current = list(line)
            continue

        previous_text = _line_text(current)
        if previous_text == "TODAY":
            paragraphs.append(current)
            current = list(line)
            continue

        should_merge = (
            not previous_text.endswith(PUNCTUATION_END)
            or len(line) < 3
        )
        if should_merge:
            current.extend(line)
        else:
            paragraphs.append(current)
            current = list(line)

    if current:
        paragraphs.append(current)
    return paragraphs
