"""
Common chord progressions and song-form builder.

Progressions are defined in scale-degree notation so they transpose to any key.
The form builder concatenates per-section chord timelines into one flat list.
"""

from __future__ import annotations
import re

_NOTE_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "F": 5, "E#": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}
_PC_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

# ── Roman numeral chord parser ─────────────────────────────────────────────
# Matches: optional accidental (b/#), Roman numeral (I–VII, case-sensitive),
# optional quality suffix (7, maj7, m7, dim, sus4, …)
_ROMAN_RE = re.compile(
    r'^(b|#)?'
    r'(VII|III|IV|VI|II|I|V|vii|iii|iv|vi|ii|i|v)'
    r'(maj13|maj11|maj9|maj7|m7b5|m7|min7|min|m9|m|9|11|13|sus4|sus2|add9|dim7|dim|aug|7|6)?$'
)
_ROMAN_TO_DEG: dict[str, int] = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
}


def is_roman_progression(text: str) -> bool:
    """Return True if *any* space/dash/comma-separated token looks like a Roman-numeral chord."""
    return any(
        _ROMAN_RE.match(t)
        for t in re.split(r'[\s\-,|]+', (text or "").strip())
        if t
    )


def parse_roman_progression(text: str, key: str) -> list[str]:
    """
    Resolve Roman-numeral chord notation to absolute chord names in *key*.

    Uppercase numeral → major quality by default; lowercase → minor.
    Flat/sharp prefix (b/# before the numeral) shifts the scale degree by a semitone.

    Examples (key = "C major"):
        "I7-iii-V"      → ["C7", "Em", "G"]
        "i-iv-V7-i"     → ["Cm", "Fm", "G7", "Cm"]
        "I-bVII-IV-I"   → ["C", "Bb", "F", "C"]
    """
    key_tokens = (key or "C major").strip().split()
    key_root   = key_tokens[0] if key_tokens else "C"
    mode_str   = " ".join(key_tokens[1:]).lower() if len(key_tokens) > 1 else "major"
    scale      = _MINOR_SCALE if ("minor" in mode_str or "aeolian" in mode_str) else _MAJOR_SCALE
    key_pc     = _NOTE_PC.get(key_root, 0)

    result: list[str] = []
    for tok in re.split(r'[\s\-,|]+', text.strip()):
        if not tok:
            continue
        m = _ROMAN_RE.match(tok)
        if not m:
            result.append(tok)   # pass through unrecognised tokens as-is
            continue
        accidental, roman, quality = m.group(1) or "", m.group(2), m.group(3) or ""
        is_lower = roman[0].islower()
        deg      = _ROMAN_TO_DEG[roman.upper()]
        interval = scale[(deg - 1) % 7]
        if accidental == "b":
            interval -= 1
        elif accidental == "#":
            interval += 1
        pc   = (key_pc + interval) % 12
        note = _PC_TO_NOTE[pc]
        if not quality:
            quality = "m" if is_lower else ""
        result.append(f"{note}{quality}")

    return result

# Each progression: list of (scale_degree_1_indexed, quality_suffix)
COMMON_PROGRESSIONS: dict[str, list[tuple[int, str]]] = {
    # ── Jazz ──────────────────────────────────────────────────────────────────
    "ii-V-I":                           [(2, "m7"), (5, "7"), (1, "maj7")],
    "ii-V-I-VI (turnaround)":           [(2, "m7"), (5, "7"), (1, "maj7"), (6, "7")],
    "I-vi-ii-V (rhythm changes A)":     [(1, "maj7"), (6, "m7"), (2, "m7"), (5, "7")],
    "I-IV-iii-VI-ii-V-I":               [(1, "maj7"), (4, "maj7"), (3, "m7"),
                                          (6, "7"), (2, "m7"), (5, "7"), (1, "maj7")],
    "iii-VI-ii-V (jazz cycle)":         [(3, "m7"), (6, "7"), (2, "m7"), (5, "7")],
    # ── Classical / pop ───────────────────────────────────────────────────────
    "I-IV-V-I":                         [(1, ""), (4, ""), (5, ""), (1, "")],
    "I-V-vi-IV (axis / pop)":           [(1, ""), (5, ""), (6, "m"), (4, "")],
    "I-vi-IV-V (50s)":                  [(1, ""), (6, "m"), (4, ""), (5, "")],
    "IV-I-V-vi":                        [(4, ""), (1, ""), (5, ""), (6, "m")],
    "I-iii-IV-V":                       [(1, ""), (3, "m"), (4, ""), (5, "")],
    "I-IV-I-V":                         [(1, ""), (4, ""), (1, ""), (5, "")],
    # ── Minor ─────────────────────────────────────────────────────────────────
    "i-VII-VI-VII (Andalusian)":        [(1, "m"), (7, ""), (6, ""), (7, "")],
    "i-iv-v-i (natural minor)":         [(1, "m"), (4, "m"), (5, "m"), (1, "m")],
    "i-iv-V-i (harmonic minor)":        [(1, "m7"), (4, "m7"), (5, "7"), (1, "m7")],
    "i-VI-III-VII":                     [(1, "m"), (6, ""), (3, ""), (7, "")],
    "i-v-iv-i":                         [(1, "m"), (5, "m"), (4, "m"), (1, "m")],
    # ── Blues ─────────────────────────────────────────────────────────────────
    "12-bar blues":                     [(1, "7"), (4, "7"), (1, "7"), (1, "7"),
                                          (4, "7"), (4, "7"), (1, "7"), (1, "7"),
                                          (5, "7"), (4, "7"), (1, "7"), (5, "7")],
    "8-bar blues":                      [(1, "7"), (1, "7"), (4, "7"), (4, "7"),
                                          (1, "7"), (5, "7"), (4, "7"), (1, "7")],
    # ── Modal / world ─────────────────────────────────────────────────────────
    "I-bVII-IV-I (Mixolydian)":         [(1, ""), (7, ""), (4, ""), (1, "")],
    "i-bVI-bVII-i (Aeolian)":           [(1, "m"), (6, ""), (7, ""), (1, "m")],
    "i-II-i-II (Phrygian)":             [(1, "m"), (2, ""), (1, "m"), (2, "")],
    "i-bII-V-i (double harmonic)":      [(1, "m"), (2, ""), (5, "7"), (1, "m")],
    # ── Carnatic-flavoured ────────────────────────────────────────────────────
    "I-II-V-I (Bhairav / Purvi)":       [(1, ""), (2, ""), (5, "7"), (1, "")],
    "i-bII-bVII-i (Hijaz)":             [(1, "m"), (2, ""), (7, ""), (1, "m")],
    "i-IV-bII-V (Raga Yaman flavour)":  [(1, "m"), (4, ""), (2, ""), (5, "7")],
}

PROGRESSION_NAMES: list[str] = list(COMMON_PROGRESSIONS.keys())


def progression_to_chords(progression_name: str, key: str) -> list[str]:
    """Convert a named progression to chord symbols in the given key."""
    degrees = COMMON_PROGRESSIONS.get(progression_name)
    if not degrees:
        return []
    tokens = key.strip().split()
    key_root = tokens[0] if tokens else "C"
    mode_str = " ".join(tokens[1:]).lower() if len(tokens) > 1 else "major"
    scale = _MINOR_SCALE if ("minor" in mode_str or "aeolian" in mode_str) else _MAJOR_SCALE
    key_pc = _NOTE_PC.get(key_root, 0)
    result = []
    for degree, quality in degrees:
        interval = scale[(degree - 1) % 7]
        pc = (key_pc + interval) % 12
        result.append(f"{_PC_TO_NOTE[pc]}{quality}")
    return result


def parse_form(form_str: str) -> list[str]:
    """
    Parse 'A B A A', 'AABA', 'A B K1 A', 'K1' into a list of section labels.

    Uses a token regex that recognises:
      K<digits>  → korvai bridge labels (K1, K2, K12, …)
      [A-Z]      → single-letter section labels (A, B, C, …)

    This means 'AABA' → ['A','A','B','A'] and 'A B K1 A' → ['A','B','K1','A']
    and 'K1' → ['K1'] all work correctly.
    """
    s = (form_str or "").strip().upper()
    if not s:
        return []
    return re.findall(r'K\d+|[A-Z]', s)


def build_song_timeline(
    form: list[str],
    section_chords: dict[str, list[str]],
    beats_per_chord: float,
    beats_per_bar: float,
    bars_per_section: dict[str, int],
    korvai_definitions: dict[str, dict] | None = None,
    fallback_chord: str = "Cmaj7",
) -> tuple[list[tuple[float, str]], float, list[dict]]:
    """
    Build a flat chord timeline from a song form and section chord definitions.

    Supports korvai bridge sections (K1, K2, …) when korvai_definitions is
    provided. Each K section occupies exactly one korvai frame (3×phrase +
    2×gap) and sustains the last active chord (or fallback_chord if first).

    Returns:
        (timeline, total_beats, section_map)
        timeline:    [(beat_offset, chord_symbol), …]
        total_beats: sum of all section durations
        section_map: [{"label", "start", "end", "korvai_params"}, …]
                     korvai_params is None for A/B/C sections.
    """
    from src.korvai_engine import korvai_frame_beats  # local import avoids circularity

    if korvai_definitions is None:
        korvai_definitions = {}

    _K_RE = re.compile(r'^K\d+$')

    timeline: list[tuple[float, str]] = []
    section_map: list[dict] = []
    current_beat = 0.0

    for label in form:
        if _K_RE.match(label) and label in korvai_definitions:
            kp = korvai_definitions[label]
            duration = korvai_frame_beats(
                kp["phrase_syl"], kp["connector_syl"],
                kp["gati_ratio"], kp["beats_per_matra"],
            )
            if duration <= 0:
                continue
            last_chord = timeline[-1][1] if timeline else fallback_chord
            if not timeline or timeline[-1][0] < current_beat:
                timeline.append((current_beat, last_chord))
            section_map.append({
                "label":        label,
                "start":        current_beat,
                "end":          current_beat + duration,
                "korvai_params": kp,
            })
            current_beat += duration

        else:
            chords = section_chords.get(label, [])
            if not chords:
                continue
            bars = bars_per_section.get(label, 8)
            section_beats = bars * beats_per_bar

            t = 0.0
            idx = 0
            while t < section_beats - 0.001:
                chord = chords[idx % len(chords)]
                timeline.append((current_beat + t, chord))
                t += beats_per_chord
                idx += 1

            section_map.append({
                "label":        label,
                "start":        current_beat,
                "end":          current_beat + section_beats,
                "korvai_params": None,
            })
            current_beat += section_beats

    return timeline, current_beat, section_map
