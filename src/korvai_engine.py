"""
Korvai rhythmic pattern engine.

Parses Carnatic solkattu syllables into timed MIDI events and generates
chord voicings placed at korvai attack points instead of metronomic stabs.

A korvai is structured as A B A B A  (not A-silence-A-silence-A).
    A = main phrase (repeated 3×, building intensity)
    B = connector phrase (repeated 2×, a distinct motif between A phrases)

The A phrases build in voicing density and velocity:
    Phrase 1 (A): outer voices only, soft
    Phrase 2 (A): shell + inner voice, medium
    Phrase 3 (A): full voicing, full velocity — resolves to sam (mukthāyi)

The B connector phrase plays at a quieter level with minimal voicing.
Only explicit rest notation (`,` or `;`) within any phrase produces silence.

Future forms (not yet implemented):
    A B A' B' A''       — incremental variation on each repetition
    Srotaswaha yati     — each A phrase longer than the last (ascending)
    Gopuccha yati       — each A phrase shorter than the last (descending)
"""

from __future__ import annotations
import random as _random
from src.harmony import note_name_to_midi, midi_to_note_name
from src.voice_leading import voice_lead


# ---------------------------------------------------------------------------
# Syllable weight tables (matra counts per syllable)
# ---------------------------------------------------------------------------

# Long syllables (dirgham) = 2 matras. Listed before short ones so the
# greedy scanner won't swallow a short prefix and leave a dangling tail.
_LONG: list[tuple[str, float]] = [
    ("thaam", 2.0), ("taam",  2.0), ("dheem", 2.0), ("dhim",  2.0),
    ("naam",  2.0), ("thoom", 2.0), ("toom",  2.0), ("daam",  2.0),
    ("deem",  2.0), ("jhem",  2.0), ("jem",   2.0),
]

# Short syllables (hrasva) = 1 matra
_SHORT: list[tuple[str, float]] = [
    ("jenu", 1.0), ("thom", 1.0), ("num",  1.0), ("tom",  1.0),
    ("tha",  1.0), ("dhi",  1.0), ("thu",  1.0),
    ("tin",  1.0), ("din",  1.0), ("gin",  1.0),
    ("ta",   1.0), ("ka",   1.0), ("di",   1.0), ("mi",   1.0),
    ("ki",   1.0), ("na",   1.0), ("ga",   1.0), ("ri",   1.0),
    ("nu",   1.0), ("cu",   1.0), ("gi",   1.0), ("jo",   1.0),
    ("gu",   1.0), ("ti",   1.0), ("ni",   1.0), ("la",   1.0),
    ("lan",  1.0),
]

# Sorted longest-first so the scanner never takes a short prefix
# when a longer match is available (e.g. "dhim" before "dhi").
_ALL_SYLLABLES: list[tuple[str, float]] = sorted(
    _LONG + _SHORT, key=lambda x: -len(x[0])
)

# ---------------------------------------------------------------------------
# Numeric korvai notation
# ---------------------------------------------------------------------------

_SYLLABLE_CYCLE = ["ta", "ka", "dhi", "mi", "ta", "ki", "ta", "dhi", "mi", "ki"]


def _expand_numeric(text: str) -> str:
    """
    Expand a numeric korvai pattern to solkattu syllables.

    Each integer N expands to N syllables drawn from the canonical cycle.
    Commas between integers become 1-matra rests, giving rhythmic spacing.

        "5"       → "ta ka dhi mi ta"
        "5, 4, 3" → "ta ka dhi mi ta , ta ka dhi mi , ta ka dhi"  (gopuccha yati)
        "1, 1, 1" → "ta , ta , ta"

    Strings containing any non-digit/comma/space pass through unchanged so
    plain solkattu phrases ("ta ka dhi mi") are never mangled.
    """
    import re as _re
    tokens = [t.strip() for t in _re.split(r"[\s,]+", text.strip()) if t.strip()]
    if not tokens or not all(t.isdigit() for t in tokens):
        return text  # not numeric — return unchanged
    parts = []
    for tok in tokens:
        n = int(tok)
        syls = [_SYLLABLE_CYCLE[i % len(_SYLLABLE_CYCLE)] for i in range(n)]
        parts.append(" ".join(syls))
    return " , ".join(parts)

# Gati (nadai) — controls the subdivision feel of each beat.
# Chatusram = 4-syllable feel (standard); Tisram = 3-syllable triplet feel.
# Note: "Mel kalam" in gati is a legacy label kept for compat; the Kalam
# control below is the canonical way to set overall speed.
GATI_OPTIONS: dict[str, float] = {
    "Chatusram (4-feel)": 1.0,
    "Tisram (3-feel)":    1.5,
    "Khandam (5-feel)":   1.25,
}

# Kalam (speed) — how many quarter-note beats each syllable occupies.
# Default is Normal kalam: one syllable = one sixteenth note (1/4 of the BPM click).
# This is the natural pace at which korvai patterns are audible as rhythmic patterns.
# Keezh kalam slows to eighth notes; Madyama to quarter notes (very slow / teaching tempo).
MATRA_OPTIONS: dict[str, float] = {
    "Normal kalam  (♬ sixteenth note)":  0.25,
    "Mel kalam     (faster, 1/32)":      0.125,
    "Keezh kalam   (♪ eighth note)":     0.5,
    "Vilambita     (♩ quarter note)":    1.0,
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _scan_word(word: str) -> list[tuple[str, float]]:
    """Greedily scan one solkattu word into (syllable, matra_weight) pairs."""
    events: list[tuple[str, float]] = []
    i = 0
    while i < len(word):
        matched = False
        for syl, w in _ALL_SYLLABLES:
            if word[i: i + len(syl)] == syl:
                events.append((syl, w))
                i += len(syl)
                matched = True
                break
        if not matched:
            i += 1
    return events or [(word, 1.0)]


def tokenize_solkattu(text: str) -> list[tuple[str, float]]:
    """
    Parse a solkattu string into ordered (syllable, matra_weight) tuples.

    Space-separated syllables give the most accurate per-attack resolution
    (e.g. "ta ka dhi mi"). Run-together forms ("takadhimi") are handled
    by greedy character scanning.

    Rest notation:
        ,   (comma)     — 1-matra rest (one silent quarter note)
        ;   (semicolon) — 2-matra rest (one silent half note)

    Both produce audible silence in the MIDI output so the listener's ear
    can detect the gap as a structural element of the korvai.
    """
    text = _expand_numeric(text)  # convert numeric patterns before tokenizing
    # Normalise punctuation before splitting so they become isolated tokens.
    normalised = (
        text.strip().lower()
        .replace("-", " ")
        .replace(";", " ; ")
        .replace(",", " , ")
    )
    events: list[tuple[str, float]] = []
    for tok in normalised.split():
        if tok == ",":
            events.append(("rest", 1.0))   # 1-matra silence
        elif tok == ";":
            events.append(("rest", 2.0))   # 2-matra silence
        else:
            events.extend(_scan_word(tok))
    return events


def phrase_matras(syl_str: str, gati_ratio: float = 1.0) -> float:
    """Total matra count for a syllable string at the given gati."""
    return sum(w for _, w in tokenize_solkattu(syl_str)) / gati_ratio


def korvai_info(
    phrase:    str,
    connector: str,
    gati_ratio: float = 1.0,
    target: int = 32,
) -> dict:
    """
    Return a summary dict for the live UI indicator.
    Keys: phrase_matras, connector_matras, total, remainder, fits.
    """
    p = phrase_matras(phrase,    gati_ratio)
    c = phrase_matras(connector, gati_ratio)
    total = 3 * p + 2 * c
    return {
        "phrase_matras":    p,
        "connector_matras": c,
        "total":            total,
        "remainder":        round(target - total, 4),
        "fits":             abs(target - total) < 0.01,
    }


# ---------------------------------------------------------------------------
# Frame event builder
# ---------------------------------------------------------------------------

def _build_frame(
    phrase_syl:      str,
    connector_syl:   str,
    gati_ratio:      float,
    beats_per_matra: float,
) -> tuple[list[dict], float]:
    """
    Build one complete korvai frame: A B A B A.

    Returns (events, frame_duration_beats).
    Each event dict:
        beat        – attack position from frame start (beats)
        duration    – note duration in beats
        is_rest     – True = explicit rest (comma/semicolon notation); no note placed
        is_long     – True = dirgham syllable (sustained articulation)
        phrase_num  – 0 = connector (B phrase)
                      1 / 2 / 3 = main A phrase (1=sparse, 2=medium, 3=mukthāyi)
    """
    p_toks = tokenize_solkattu(phrase_syl)
    c_toks = tokenize_solkattu(connector_syl)

    events: list[dict] = []
    cursor = 0.0

    def _emit(toks: list[tuple[str, float]], phrase_num: int) -> None:
        nonlocal cursor
        for syl, raw_w in toks:
            dur_beats = (raw_w / gati_ratio) * beats_per_matra
            is_rest   = (syl == "rest")
            is_long   = (raw_w >= 2.0)
            note_dur  = (dur_beats * 0.9) if is_long else min(0.5, dur_beats * 0.8)
            events.append({
                "beat":       cursor,
                "duration":   max(0.05, note_dur),
                "is_rest":    is_rest,
                "is_long":    is_long,
                "phrase_num": phrase_num,
            })
            cursor += dur_beats

    _emit(p_toks, 1)   # A phrase 1 — sparse
    _emit(c_toks, 0)   # B connector 1 — quiet
    _emit(p_toks, 2)   # A phrase 2 — medium
    _emit(c_toks, 0)   # B connector 2 — quiet
    _emit(p_toks, 3)   # A phrase 3 — full (mukthāyi)

    return events, cursor


# ---------------------------------------------------------------------------
# Active-chord lookup
# ---------------------------------------------------------------------------

def _active_chord(chord_timeline: list[tuple[float, str]], beat: float) -> str:
    result = chord_timeline[0][1]
    for cb, cs in chord_timeline:
        if beat >= cb:
            result = cs
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Main MIDI generator
# ---------------------------------------------------------------------------

# Velocity fraction per phrase number.
# Base vel 100; connector (B) is softer than even P1 to keep it subordinate.
_VEL_SCALE = {0: 0.55, 1: 0.70, 2: 0.83, 3: 1.00}

# Voice count per phrase — builds intensity toward the mukthāyi resolution.
# Connector and P1: single note, so the RHYTHM is heard cleanly (not buried in chords).
# P2: 2-voice shell (outer voices only) — harmonic colour emerges.
# P3 (mukthāyi): 3 voices — full chord, the culminating stab.
_VOICE_COUNT = {0: 1, 1: 1, 2: 2, 3: 3}


def korvai_frame_beats(
    phrase_syl: str,
    connector_syl: str,
    gati_ratio: float = 1.0,
    beats_per_matra: float = 0.25,
) -> float:
    """Total beat duration of one korvai frame (3×phrase + 2×gap)."""
    _, dur = _build_frame(phrase_syl, connector_syl, gati_ratio, beats_per_matra)
    return dur


def make_korvai_chord_part(
    chord_timeline:   list[tuple[float, str]],
    total_beats:      float,
    lo:               int,
    hi:               int,
    key:              str,
    phrase_syl:       str   = "ta ka dhi mi ta ka dhi mi",
    connector_syl:    str   = "ta ka",
    gati_ratio:       float = 1.0,
    beats_per_matra:  float = 0.25,
    start_beat:       float = 0.0,
) -> list[dict]:
    """
    Generate MIDI chord notes placed at korvai syllable attack points.

    The korvai frame tiles to fill total_beats starting from start_beat.
    The active chord at each attack is read from chord_timeline using
    absolute beat positions, so korvai can sit anywhere in a song timeline.
    """
    if not chord_timeline:
        return []

    frame_events, frame_duration = _build_frame(
        phrase_syl, connector_syl, gati_ratio, beats_per_matra
    )
    if not frame_events or frame_duration <= 0:
        return []

    notes: list[dict] = []
    center = (lo + hi) // 2
    prev_midis: list[int] = []
    end_beat = start_beat + total_beats
    beat_offset = start_beat

    while beat_offset < end_beat:
        for ev in frame_events:
            abs_beat   = beat_offset + ev["beat"]
            phrase_num = ev["phrase_num"]

            if abs_beat >= end_beat:
                break
            if ev["is_rest"]:
                continue  # explicit rest notation (`,` or `;`) → silence

            chord  = _active_chord(chord_timeline, abs_beat)
            midis  = voice_lead(chord, prev_midis, lo, hi, center)
            prev_midis = midis[:]

            n_voices = _VOICE_COUNT[phrase_num]
            if len(midis) > n_voices:
                if n_voices == 1:
                    midis = [midis[len(midis) // 2]]   # middle voice ≈ root position
                elif n_voices == 2:
                    midis = [midis[0], midis[-1]]
                else:
                    midis = [midis[0], midis[len(midis) // 2], midis[-1]]

            vel_base  = 100
            vel_scale = _VEL_SCALE[phrase_num]
            note_dur  = min(ev["duration"], end_beat - abs_beat)

            for j, midi in enumerate(midis):
                vel = max(45, min(120, int(vel_base * vel_scale) - j * 4))
                notes.append({
                    "note":           midi_to_note_name(midi),
                    "start_beat":     round(abs_beat * 8) / 8,
                    "duration_beats": note_dur,
                    "velocity":       vel,
                })

        beat_offset += frame_duration

    return notes


def make_korvai_bass_part(
    chord_timeline:   list[tuple[float, str]],
    total_beats:      float,
    lo:               int,
    hi:               int,
    phrase_syl:       str   = "ta ka dhi mi ta ka dhi mi",
    connector_syl:    str   = "ta ka",
    gati_ratio:       float = 1.0,
    beats_per_matra:  float = 0.25,
    start_beat:       float = 0.0,
) -> list[dict]:
    """
    Bass line that follows the korvai structure: root attacks on phrase events,
    silence during gaps. This makes the structural gaps audible to the ear.

    Phrase 1 and 2 play the root only; Phrase 3 adds the fifth on alternating
    syllables to build intensity into the mukthāyi.
    """
    if not chord_timeline:
        return []

    frame_events, frame_duration = _build_frame(
        phrase_syl, connector_syl, gati_ratio, beats_per_matra
    )
    if not frame_events or frame_duration <= 0:
        return []

    from src.harmony import chord_tones_from_symbol, _NOTE_PC  # type: ignore

    def _nearest(target: int, pc: int, rng_lo: int, rng_hi: int) -> int:
        for d in range(0, 13):
            for delta in ([d, -d] if d > 0 else [0]):
                c = target + delta
                if rng_lo <= c <= rng_hi and c % 12 == pc:
                    return c
        for m in range(rng_lo, rng_hi + 1):
            if m % 12 == pc:
                return m
        return target

    notes: list[dict] = []
    end_beat = start_beat + total_beats
    beat_offset = start_beat
    center = (lo + hi) // 2

    while beat_offset < end_beat:
        phrase3_note_idx = 0
        for ev in frame_events:
            abs_beat   = beat_offset + ev["beat"]
            phrase_num = ev["phrase_num"]

            if abs_beat >= end_beat:
                break
            if ev["is_rest"]:
                continue  # explicit rest notation → silence

            chord = _active_chord(chord_timeline, abs_beat)
            root, intervals = chord_tones_from_symbol(chord)
            root_pc = _NOTE_PC.get(root, 0)
            root_midi = _nearest(center, root_pc, lo, hi)

            if phrase_num == 3 and phrase3_note_idx % 2 == 1 and len(intervals) > 1:
                fifth_pc = (root_pc + intervals[1]) % 12
                midi = _nearest(root_midi, fifth_pc, lo, hi)
            else:
                midi = root_midi

            if phrase_num == 3:
                phrase3_note_idx += 1

            vel_scale = _VEL_SCALE[phrase_num]
            note_dur  = min(ev["duration"], end_beat - abs_beat)
            vel = max(45, min(110, int(85 * vel_scale)))
            notes.append({
                "note":           midi_to_note_name(midi),
                "start_beat":     round(abs_beat * 8) / 8,
                "duration_beats": note_dur,
                "velocity":       vel,
            })

        beat_offset += frame_duration

    return notes


# ---------------------------------------------------------------------------
# Korvai randomizer
# ---------------------------------------------------------------------------

# Pre-verified syllable building blocks (every syllable exists in _ALL_SYLLABLES).
# Organised by raw matra count so the filler can pick groups that sum exactly.
_BLOCKS: dict[int, list[str]] = {
    4: [
        "ta ka dhi mi",
        "ta ka ta ka",
        "ta ki ta ka",
        "dhi mi ta ka",
        "ta ka ta ki",
        "ta din gi na",
        "ta ki ta dhi",
        "gin na thom na",
        "ti ta ka ta",
        "ta ka na ka",
    ],
    3: [
        "ta ki ta",
        "ta di na",
        "dhi mi ta",
        "ta ka ta",
        "ki ta ka",
        "ta ka dhi",
        "ta gi na",
    ],
    2: [
        "ta ka",
        "dhi mi",
        "ta ki",
        "ki ta",
        "ta na",
        "taam",
        "dheem",
        "naam",
    ],
    1: ["ta", "ka", "dhi", "mi", "ki", "ti", "na", "din", "gin", "tin"],
}

def _fill_matras(n: int, rest_rate: float = 0.0) -> str:
    """
    Generate a syllable string of exactly n raw matras.

    rest_rate controls how aggressively rests (`,` and `;`) are inserted:
      0.0  — no rests (pure syllable phrases)
      0.2  — occasional rests in A phrases (adds rhythmic interest)
      0.5  — frequent rests, common in connector phrases

    Note: long syllables (taam, dheem…) are *sustained notes*, not note+rest.
    Only `,` (1 matra) and `;` (2 matras) produce actual silence.

    Prefers groupings of 4, 3, 2 (in that order) for musicality.
    """
    parts: list[str] = []
    remaining = n

    while remaining > 0:
        options: list[tuple[int, str]] = []

        if remaining >= 4:
            options += [(4, b) for b in _BLOCKS[4]] * 3
        if remaining >= 3:
            options += [(3, b) for b in _BLOCKS[3]] * 2
        if remaining >= 2:
            options += [(2, b) for b in _BLOCKS[2]] * 2
        options += [(1, s) for s in _BLOCKS[1]]

        # Rest options — weighted by rest_rate relative to the syllable pool.
        # Matra counts verified against tokenize_solkattu:
        #   ","  → rest(1)           = 1 matra
        #   ";"  → rest(2)           = 2 matras
        #   "ta ,"  → ta(1)+rest(1)  = 2 matras
        #   ", ta"  → rest(1)+ta(1)  = 2 matras
        #   "; ta"  → rest(2)+ta(1)  = 3 matras
        #   "ta ;"  → ta(1)+rest(2)  = 3 matras
        if rest_rate > 0:
            rest_weight = max(1, int(len(options) * rest_rate))
            if remaining >= 3:
                options += [(3, "; ta"), (3, "ta ;")] * rest_weight
            if remaining >= 2:
                options += [(2, ";"), (2, "ta ,"), (2, ", ta")] * rest_weight
            if remaining >= 1:
                options += [(1, ",")] * (rest_weight * 2)  # commas most common

        size, chosen = _random.choice(options)
        parts.append(chosen)
        remaining -= size

    return " ".join(parts)


def _valid_pairs(target_matras: int, gati_ratio: float = 1.0) -> list[tuple[int, int]]:
    """
    Return ALL (p, c) int pairs where 3p + 2c = target_matras × gati_ratio.

    Only hard constraints:
      p ≥ 1   (phrase must have at least one syllable)
      c ≥ 1   (connector must have at least one syllable)

    No ratio or ordering constraint — connector may be longer than phrase,
    enabling future forms like A A' A'' B (progression) or short-phrase
    high-speed patterns. The caller weights pairs by musical preference.
    """
    T = target_matras * gati_ratio
    T_int = round(T)
    if abs(T - T_int) > 0.05:
        return []   # non-integer — no exact solution
    T_int = int(T_int)

    pairs: list[tuple[int, int]] = []
    for p in range(1, T_int // 3 + 1):
        rem = T_int - 3 * p
        if rem >= 2 and rem % 2 == 0:   # c = rem/2 ≥ 1
            pairs.append((p, rem // 2))
    return pairs


def random_korvai(
    target_matras: int = 32,
    gati_ratio: float = 1.0,
    seed: int | None = None,
) -> tuple[str, str]:
    """
    Generate a random valid korvai: (phrase_syl, connector_syl).

    The only constraint is that the A-B-A-B-A frame totals exactly
    target_matras at the given gati_ratio:
        3 × phrase_matras + 2 × connector_matras = target_matras

    All valid (p, c) splits are in play. Weighting nudges toward pairs
    where both phrases are long enough to be musically interesting (p ≥ 3,
    c ≥ 2) but does not exclude short or asymmetric structures — short
    punchy phrases with long connectors, or the inverse, are both valid
    korvai shapes, especially as precursors to A/A'/A'' variation forms.
    """
    if seed is not None:
        _random.seed(seed)

    pairs = _valid_pairs(target_matras, gati_ratio)

    if not pairs:
        return "ta ka dhi mi ta ka dhi mi", "ta ka"  # fallback

    def _weight(p: int, c: int) -> float:
        # Prefer both phrase and connector to be at least 3/2 matras (audible motifs)
        # but allow anything — just reduce weight for very short segments
        pw = 1.0 if p >= 3 else (0.5 if p == 2 else 0.15)
        cw = 1.0 if c >= 2 else 0.4
        return pw * cw

    weights = [_weight(p, c) for p, c in pairs]
    (p, c) = _random.choices(pairs, weights=weights, k=1)[0]

    # A phrase: moderate rests — rhythmic interest without losing the motif
    # B connector: heavier rests — breathing space between A statements
    phrase    = _fill_matras(p, rest_rate=0.20)
    connector = _fill_matras(c, rest_rate=0.50)

    return phrase, connector


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def gati_from_label(label: str) -> float:
    return GATI_OPTIONS.get(label, 1.0)


def beats_per_matra_from_label(label: str) -> float:
    return MATRA_OPTIONS.get(label, 0.25)
