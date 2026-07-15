"""
Lyric fabrication validator for Kritiq feedback.

PURPOSE
    The feedback model occasionally writes lyric fragments it did not hear —
    it pulls a plausible line from training-data memory and drops it at a
    timestamp with no evidence (e.g. "the line 'You're getting it'" on a
    Coldplay cover where Whisper never captured those words).

    The prompt already forbids this. Prompts are soft and this one failed.
    This module is the HARD enforcement: it runs on the model's output AFTER
    generation and mechanically strips any lyric reference that is not backed
    by the Whisper transcript — regardless of what the model did.

SCOPE (be honest about this)
    CATCHES:  quoted lyric spans  ("...", '...')
              heuristic lyric references ("the line/lyric/phrase/words <X>")
              capitalized multi-word runs that read like sung lyric fragments
    DOES NOT CATCH:
              un-quoted invented PERFORMANCE EVENTS ("the energy drops, reads
              as fatigue"). Those aren't quotable strings checkable against a
              transcript — different problem, out of reach for a text check.

    So this kills "fabricated lyrics", not "all hallucination". Frame it that
    way to Andy.

USAGE (wire into feedback.generate_feedback, right before `return feedback`)
    from lyric_validator import scrub_fabricated_lyrics
    feedback = scrub_fabricated_lyrics(feedback, lyrics_transcript)
"""

import re
import logging

logger = logging.getLogger(__name__)

# Generic referents the model can fall back to once a fabricated lyric is
# removed. Chosen so the surrounding coaching still reads naturally.
_FALLBACK_REFERENT = "the phrase there"

# Words that commonly start a lyric reference in the model's prose.
_LYRIC_LEAD_PATTERN = re.compile(
    r"\b(?:the\s+)?(?:line|lyric|lyrics|words?|phrase)\b\s*"
    r"(?:like|such as|where (?:they|you|the\s+\w+)\s+sings?\b[^A-Za-z0-9]*)?",
    re.IGNORECASE,
)

# A run of 3+ capitalized/Title-case words — the shape of a sung lyric
# fragment dropped mid-sentence without quotes ("...around Getting It Right
# near the end..."). Tuned to avoid normal Title Case proper nouns by
# requiring length >= 3 words.
_CAP_RUN_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z']+(?:\s+|$)){3,}"
)

# Quote characters by family. We find an opening quote, then the matching
# closing quote of the same family, allowing apostrophes inside straight-single
# spans ("You're getting it"). Regex alone mishandles the apostrophe ambiguity,
# so quoted-span extraction is done procedurally in _find_quoted_spans.
_OPEN_TO_CLOSE = {
    '"': '"',
    '\u201c': '\u201d',  # smart double
    '\u2018': '\u2019',  # smart single
}


def _find_quoted_spans(text: str):
    """
    Yield (start, end, inner) for each quoted span. Handles:
      - double quotes and smart doubles
      - smart singles
      - straight single quotes used as lyric delimiters, allowing apostrophes
        inside (e.g. 'You're getting it') by treating a "'" as a delimiter only
        when it is at a word boundary (preceded/followed by non-letter), and
        the closing "'" likewise. Apostrophes mid-word (You're) are not
        boundaries, so they're skipped.
    """
    spans = []
    n = len(text)

    # Paired quote families (double, smart-double, smart-single).
    i = 0
    while i < n:
        ch = text[i]
        if ch in _OPEN_TO_CLOSE:
            close = _OPEN_TO_CLOSE[ch]
            j = text.find(close, i + 1)
            if j != -1:
                spans.append((i, j + 1, text[i + 1:j]))
                i = j + 1
                continue
        i += 1

    # Straight single quotes as lyric delimiters.
    # An opening ' is one where the previous char is not a letter (start of a
    # quoted lyric). The matching closing ' is one where the next char is not a
    # letter. This lets "You're" pass through (apostrophe sits between letters).
    i = 0
    while i < n:
        if text[i] == "'":
            prev_is_letter = i > 0 and text[i - 1].isalpha()
            if not prev_is_letter:
                # opening delimiter — find a closing ' at a word boundary
                k = i + 1
                while k < n:
                    if text[k] == "'":
                        next_is_letter = k + 1 < n and text[k + 1].isalpha()
                        if not next_is_letter:
                            spans.append((i, k + 1, text[i + 1:k]))
                            i = k
                            break
                    k += 1
        i += 1

    spans.sort()
    return spans


def _strip_quoted(text: str, norm_transcript: str, has_transcript: bool):
    """Replace unverified multi-word quoted lyric spans with a generic referent."""
    spans = _find_quoted_spans(text)
    if not spans:
        return text, 0

    removed = 0
    out = []
    idx = 0
    for start, end, inner in spans:
        if start < idx:
            continue  # overlapping; already consumed
        if len(_normalize(inner).split()) >= 2 and (
            not has_transcript or not _in_transcript(inner, norm_transcript)
        ):
            out.append(text[idx:start])
            out.append(_FALLBACK_REFERENT)
            idx = end
            removed += 1
    out.append(text[idx:])
    return "".join(out), removed


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation for substring tests."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _in_transcript(span: str, norm_transcript: str) -> bool:
    """True if the span (normalized) is a substring of the transcript."""
    norm_span = _normalize(span)
    if not norm_span:
        return True  # empty / punctuation-only — nothing to fabricate
    # Single short word (<=2 words) is too generic to be a "fabricated lyric";
    # only police multi-word spans. Avoids stripping "your" or "grace".
    if len(norm_span.split()) < 2:
        return True
    return norm_span in norm_transcript


def _scrub_text(text: str, norm_transcript: str, has_transcript: bool) -> tuple:
    """
    Remove fabricated lyric references from one string.
    Returns (cleaned_text, num_removed).
    """
    if not text:
        return text, 0

    removed = 0

    # ── 1. Quoted spans (procedural, apostrophe-safe) ────────────────
    text, n_q = _strip_quoted(text, norm_transcript, has_transcript)
    removed += n_q

    # ── 2. "the line/lyric/phrase <X>" lead-ins followed by a cap run ─
    # Catch un-quoted references the model introduces with a lyric lead word.
    # Rebuild around lead-in matches (manual, since we consume following text).
    out = []
    idx = 0
    for m in _LYRIC_LEAD_PATTERN.finditer(text):
        if m.start() < idx:
            continue
        tail = text[m.end():m.end() + 80]
        stripped = tail.lstrip()
        offset = len(tail) - len(stripped)
        cap = _CAP_RUN_PATTERN.match(stripped)
        if cap:
            fragment = cap.group(0).strip()
            if not has_transcript or not _in_transcript(fragment, norm_transcript):
                out.append(text[idx:m.start()])
                out.append(_FALLBACK_REFERENT)
                idx = m.end() + offset + len(cap.group(0))
                removed += 1
    out.append(text[idx:])
    text = "".join(out)

    # ── 3. Bare capitalized runs (no lead-in, no quotes) ─────────────
    def _cap_repl(m):
        nonlocal removed
        fragment = m.group(0).strip()
        if not has_transcript or not _in_transcript(fragment, norm_transcript):
            removed += 1
            return ""  # excise; surrounding sentence usually survives
        return m.group(0)

    text = _CAP_RUN_PATTERN.sub(_cap_repl, text)

    # Tidy double spaces / orphaned punctuation left by excisions.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Ensure a fused referent gets a space ("theredrops" -> "there drops").
    text = text.replace(_FALLBACK_REFERENT + "drops", _FALLBACK_REFERENT + " drops")
    text = re.sub(rf"({re.escape(_FALLBACK_REFERENT)})(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\(\s*\)", "", text)

    return text, removed


def scrub_fabricated_lyrics(feedback: dict, lyrics_transcript: str = None) -> dict:
    """
    Post-process model feedback to remove lyric references not backed by the
    Whisper transcript. Mutates and returns the feedback dict.

    Safe to call unconditionally — if there is no transcript, it strips any
    multi-word quoted/capitalized lyric-looking span (nothing is verifiable).
    """
    if not isinstance(feedback, dict):
        return feedback

    has_transcript = bool(lyrics_transcript and lyrics_transcript.strip())
    norm_transcript = _normalize(lyrics_transcript) if has_transcript else ""

    total_removed = 0

    for section in ("what_worked", "needs_improvement"):
        items = feedback.get(section) or []
        for item in items:
            if not isinstance(item, dict):
                continue
            for field in ("detail", "point"):
                original = item.get(field)
                if isinstance(original, str):
                    cleaned, n = _scrub_text(original, norm_transcript, has_transcript)
                    if n:
                        item[field] = cleaned
                        total_removed += n

    summary = feedback.get("summary")
    if isinstance(summary, str):
        cleaned, n = _scrub_text(summary, norm_transcript, has_transcript)
        if n:
            feedback["summary"] = cleaned
            total_removed += n

    if total_removed:
        feedback["_lyric_scrub_count"] = total_removed

    logger.info(
        f"lyric_validator: scrub ran (transcript_present={has_transcript}, "
        f"stripped={total_removed})"
    )

    return feedback


# ─── Self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    transcript = "where do we go nobody knows gave you style and gave you grace"

    fb = {
        "what_worked": [
            {"point": "Natural phrasing",
             "detail": "The delivery of 'Where do we go, nobody knows' sits well rhythmically.",
             "timestamp": "0:02"},
        ],
        "needs_improvement": [
            {"point": "Final section loses focus",
             "detail": "From 0:53 onward — through the line 'You're getting it' — the "
                       "vocal delivery softens significantly and reads as fatigue.",
             "timestamp": "0:53"},
            {"point": "Breath control",
             "detail": "The phrase 'gave you grace' trails off as breath runs out.",
             "timestamp": "0:17"},
        ],
        "summary": "Competent cover; the closing line Getting It Right Near The End drops energy.",
    }

    out = scrub_fabricated_lyrics(fb, transcript)
    import json
    print(json.dumps(out, indent=2))
