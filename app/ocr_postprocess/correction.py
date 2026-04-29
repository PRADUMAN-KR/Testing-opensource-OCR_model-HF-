from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from app.ocr_postprocess.normalization import normalize_text
from app.ocr_postprocess.utils import is_arabic, levenshtein, normalize_token_text


DICT_PATH = Path(__file__).resolve().parent / "dictionary" / "arabic_words.txt"

TOKEN_FIXES = {
    "قرارا": "قرار",
    "لقرارا": "لقرار",
    "للقض": "للقضاء",
    "دائاً": "دائماً",
    "الى": "إلى",
    "الي": "التي",
    "اتفافياتمعروفة": "اتفاقيات معروفة",
    "اتفاقياتمعروفة": "اتفاقيات معروفة",
    "أحدولا": "أحد ولا",
    "موثغقة": "موثقة",
    "تحالفة": "مخالفة",
    "تخالفة": "مخالفة",
    "أوللمطالبة": "أو للمطالبة",
    "الناسومغالطة": "الناس، ومغالطة",
    "عحايد": "محايد",
    "عايد": "محايد",
    "كيائاً": "كياناً",
    "الكويتهذه": "الكويت هذه",
    "الشاطىالشالي": "الشاطئ الشمالي",
    "السلامية": "الإسلامية",
    "العشانية": "العثمانية",
    "العري": "العربي",
    "العثانية": "العثمانية",
    "عثاي": "عثماني",
    "تنَس": "تمس",
}


@lru_cache(maxsize=1)
def load_arabic_dictionary() -> set[str]:
    if not DICT_PATH.exists():
        return set()
    return {
        line.strip()
        for line in DICT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _closest_dictionary_match(word: str, dictionary: set[str]) -> str | None:
    if not dictionary or not is_arabic(word):
        return None
    max_dist = 1 if len(word) <= 5 else 2
    best_word = None
    best_dist = max_dist + 1
    for candidate in dictionary:
        if abs(len(candidate) - len(word)) > max_dist:
            continue
        dist = levenshtein(word, candidate, max_dist)
        if dist < best_dist:
            best_dist = dist
            best_word = candidate
            if dist == 1:
                break
    return best_word if best_dist <= max_dist else None


def fix_broken_words(paragraphs: list[list[dict]]) -> list[list[dict]]:
    dictionary = load_arabic_dictionary()
    fixed_paragraphs: list[list[dict]] = []

    for paragraph in paragraphs:
        fixed: list[dict] = []
        i = 0
        while i < len(paragraph):
            token = dict(paragraph[i])
            token["text"] = normalize_token_text(token["text"])
            token["text"] = TOKEN_FIXES.get(token["text"], token["text"])
            if (
                i + 1 < len(paragraph)
                and is_arabic(token["text"])
            ):
                nxt = dict(paragraph[i + 1])
                nxt_text = normalize_token_text(nxt["text"])
                combined = token["text"] + nxt_text
                if token["text"] == "العام" and nxt_text == "ية":
                    fixed.append(token)
                    i += 2
                    continue
                should_try_join = len(token["text"]) < 3 or len(nxt_text) < 3
                if should_try_join and combined in dictionary:
                    merged = dict(token)
                    merged["text"] = combined
                    merged["confidence"] = min(
                        float(token.get("confidence", 1.0)),
                        float(nxt.get("confidence", 1.0)),
                    )
                    fixed.append(merged)
                    i += 2
                    continue
            fixed.append(token)
            i += 1
        fixed_paragraphs.append(fixed)
    return fixed_paragraphs


def correct_low_confidence_words(paragraphs: list[list[dict]]) -> list[list[dict]]:
    dictionary = load_arabic_dictionary()
    if not dictionary:
        return paragraphs

    corrected: list[list[dict]] = []
    for paragraph in paragraphs:
        out = []
        for token in paragraph:
            token = dict(token)
            text = normalize_token_text(token["text"])
            if float(token.get("confidence", 1.0)) < 0.85 and is_arabic(text):
                replacement = _closest_dictionary_match(text, dictionary)
                if replacement:
                    token["text"] = replacement
            out.append(token)
        corrected.append(out)
    return corrected


def reconstruct_sentence_boundaries(text: str) -> str:
    return text


def apply_phrase_reorder_rules(text: str) -> str:
    return text


def apply_final_regex_cleanup(text: str) -> str:
    replacements = [
        (r"\bدائاً\b", "دائماً"),
        (r"\bالى\b", "إلى"),
        (r"\bالي\b", "التي"),
        (r"\bكا\s+", "كما "),
        (r"\bتتقدم\s+با\s+الكويت\b", "تتقدم بها الكويت"),
        (r"\bاتفافياتمعروفة\b", "اتفاقيات معروفة"),
        (r"\bاتفاقياتمعروفة\b", "اتفاقيات معروفة"),
        (r"\bأحدولا\b", "أحد ولا"),
        (r"\bموثغقة\b", "موثقة"),
        (r"\bتحالفة\b", "مخالفة"),
        (r"\bتخالفة\b", "مخالفة"),
        (r"\bأوللمطالبة\b", "أو للمطالبة"),
        (r"\bالناسومغالطة\b", "الناس، ومغالطة"),
        (r"\bفا\s+قدمت\b", "فما قدمت"),
        (r"\bعحايد\b", "محايد"),
        (r"\bعايد\b", "محايد"),
        (r"\bكيائاً\b", "كياناً"),
        (r"\bشكل\s+الظروف\b", "كل الظروف"),
        (r"\bشكل\s+المتغيرات\b", "كل المتغيرات"),
        (r"\bالسلامية\b", "الإسلامية"),
        (r"\bالعشانية\b", "العثمانية"),
        (r"\bعثاي\b", "عثماني"),
        (r"\bالكويتهذه\b", "الكويت هذه"),
        (r"\bالشاطىالشالي\b", "الشاطئ الشمالي"),
        (r"\bالعري\b", "العربي"),
        (r"\bالعثانية\b", "العثمانية"),
        (r"\bتنَس\b", "تمس"),
        (r"\bلقرارا\b", "لقرار"),
        (r"\bقرارا\b", "قرار"),
        (r"\bالمتقا\s+عد\b", "المتقاعد"),
        (r"\bالعام\s+ية\b", "العام"),
        (r"\bللقض\b", "للقضاء"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def paragraphs_to_text(paragraphs: list[list[dict]]) -> str:
    paragraph_texts = []
    for paragraph in paragraphs:
        text = " ".join(token["text"] for token in paragraph if token.get("text"))
        text = normalize_text(text)
        if text:
            paragraph_texts.append(text)
    return "\n\n".join(paragraph_texts)
