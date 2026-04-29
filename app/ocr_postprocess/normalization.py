from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u0640", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([،.؟؛!:])", r"\1", text)
    text = re.sub(r"([،.؟؛!:])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_noise(text: str) -> str:
    # Keep this intentionally conservative: do not delete OCR tokens that may
    # carry meaning. Noise handling is exposed as flags in the model metadata.
    lines = (text or "").splitlines()
    cleaned_lines = [normalize_text(line) for line in lines]
    out: list[str] = []
    previous_blank = False
    for line in cleaned_lines:
        if not line:
            if out and not previous_blank:
                out.append("")
            previous_blank = True
            continue
        out.append(line)
        previous_blank = False
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)
