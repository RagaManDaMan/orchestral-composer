"""
Voice leading, functional harmony analysis, and chord substitution.

The goal is to turn a flat list of chord symbols into a musically coherent
arrangement with smooth voice movement, proper tension-resolution shaping,
and optional jazz/symphonic substitutions.

All functions operate on pitch-class sets and MIDI note numbers — no LLM needed.
"""

from __future__ import annotations
from src.harmony import chord_tones_from_symbol, _NOTE_PC, midi_to_note_name, note_name_to_midi

# ---------------------------------------------------------------------------
# Functional harmony
# ---------------------------------------------------------------------------

# Semitone intervals from root that define chord quality
_DOMINANT_QUALITIES  = {"7", "dom7", "aug7", "7sus4", "7#11", "9", "13"}
_TONIC_QUALITIES     = {"", "maj", "maj7", "maj9", "add9"}
_SUBDOMINANT_QUAL    = {"m", "min", "m7", "min7", "m9", "m7b5", "sus4", "sus2"}
_DIMINISHED_QUAL     = {"dim", "dim7", "m7b5"}

# Intervals that define the scale degree of a chord root relative to a key root
# major key scale degrees (semitones from tonic): I=0, II=2, III=4, IV=5, V=7, VI=9, VII=11
_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # natural minor


def _parse_quality(symbol: str) -> tuple[str, str]:
    """Return (root, quality_suffix) from a chord symbol."""
    s = symbol.strip()
    if len(s) >= 2 and s[1] in ("#", "b"):
        return s[:2], s[2:]
    return s[:1], s[1:]


def chord_function(symbol: str, key_root: str, scale: list[int] = None) -> str:
    """
    Classify a chord as 'T' (tonic), 'SD' (subdominant), 'D' (dominant),
    or 'X' (chromatic/borrowed).
    """
    if scale is None:
        scale = _MAJOR_SCALE
    root, quality = _parse_quality(symbol)
    # Strip slash bass note
    quality = quality.split("/")[0]

    root_pc  = _NOTE_PC.get(root, 0)
    key_pc   = _NOTE_PC.get(key_root, 0)
    interval = (root_pc - key_pc) % 12

    if quality in _DOMINANT_QUALITIES:
        return "D"
    if interval in (0, 4, 9) and quality in _TONIC_QUALITIES:
        return "T"   # I, iii, vi — tonic function
    if interval in (5, 2) and quality in _SUBDOMINANT_QUAL:
        return "SD"  # IV, ii — subdominant function
    if interval == 7 and quality in _DOMINANT_QUALITIES | {"", "maj"}:
        return "D"   # V major / V7 — dominant function
    if quality in _DIMINISHED_QUAL:
        return "D"   # dim chords have dominant tension
    return "X"


def analyze_progression(
    chord_timeline: list[tuple[float, str]],
    key: str,
) -> list[tuple[float, str, str]]:
    """
    Return [(beat, symbol, function), ...] with T/SD/D/X labels.
    key is e.g. 'D major', 'G minor', 'D Purvi'.
    """
    tokens = key.strip().split()
    key_root  = tokens[0] if tokens else "C"
    mode_str  = " ".join(tokens[1:]).lower() if len(tokens) > 1 else "major"
    scale     = _MINOR_SCALE if "minor" in mode_str or "aeolian" in mode_str else _MAJOR_SCALE

    return [
        (beat, sym, chord_function(sym, key_root, scale))
        for beat, sym in chord_timeline
    ]


def chord_complexity(symbol: str) -> float:
    """
    Return a complexity score [0.0, 1.0] from a chord symbol.
    Used to shape velocities in non-functional (raga / modal) contexts where
    T/SD/D labels don't apply.
    """
    _, quality = _parse_quality(symbol)
    quality = quality.split("/")[0]
    _scores = {
        "": 0.15, "maj": 0.15, "m": 0.15, "min": 0.15,
        "sus2": 0.20, "sus4": 0.22, "add9": 0.25,
        "aug": 0.40, "dim": 0.45,
        "maj7": 0.35, "m7": 0.40, "7": 0.55, "dom7": 0.55,
        "m7b5": 0.65, "dim7": 0.70,
        "aug7": 0.60, "7sus4": 0.50, "MajMaj7": 0.45,
        "9": 0.45, "maj9": 0.40, "m9": 0.48,
        "7#11": 0.72, "13": 0.62,
    }
    return _scores.get(quality, 0.30)


def tension_velocity(function: str, base: int = 72, symbol: str = "") -> tuple[int, int]:
    """
    Return (chord_vel, bass_vel) shaped by harmonic function.
    When function is 'X' (non-diatonic / modal context), fall back to
    chord_complexity so raga arrangements still get meaningful dynamics.
    """
    if function == "D":
        return base + 8, base + 10
    if function == "SD":
        return base + 3, base + 5
    if function == "T":
        return base - 4, base - 2
    # 'X': use chord complexity to vary velocity
    if symbol:
        offset = int((chord_complexity(symbol) - 0.3) * 20)  # range ≈ -6 to +8
        return base + offset, base + offset + 2
    return base, base


# ---------------------------------------------------------------------------
# Chord substitutions
# ---------------------------------------------------------------------------

_TRITONE_SUB: dict[str, str] = {
    # Maps a dominant chord root to its tritone-substitution root
    # tritone = 6 semitones away
}

def _transpose_root(root: str, semitones: int) -> str:
    pc  = (_NOTE_PC.get(root, 0) + semitones) % 12
    # Use sharps for up, flats for down
    _S  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return _S[pc]


def tritone_substitute(symbol: str) -> str | None:
    """Return the tritone substitution of a dominant chord, or None if not dominant."""
    root, quality = _parse_quality(symbol)
    quality = quality.split("/")[0]
    if quality not in _DOMINANT_QUALITIES:
        return None
    new_root = _transpose_root(root, 6)
    return f"{new_root}7"


def secondary_dominant(target_symbol: str) -> str:
    """
    Return the secondary dominant (V7/target) for any chord.
    e.g. target=Am7 → E7, target=Dm7 → A7
    """
    root, _ = _parse_quality(target_symbol)
    dom_root = _transpose_root(root, 7)   # perfect 5th above = the V that resolves here
    return f"{dom_root}7"


def apply_tritone_substitutions(
    chord_timeline: list[tuple[float, str]],
) -> list[tuple[float, str]]:
    """
    Replace every dominant chord with its tritone substitution (root +6 semitones).
    Non-dominant chords are left unchanged.
    """
    result = []
    for beat, sym in chord_timeline:
        sub = tritone_substitute(sym)
        result.append((beat, sub if sub is not None else sym))
    return result


def insert_secondary_dominants(
    chord_timeline: list[tuple[float, str]],
    key: str,
    min_chord_beats: float = 2.0,
) -> list[tuple[float, str]]:
    """
    Before each non-tonic chord that lasts ≥ 2*min_chord_beats, insert a
    secondary dominant for half the duration. Only applied where the
    secondary dominant is not already in the key.

    Returns a new (possibly longer) chord timeline.
    """
    tokens  = key.strip().split()
    key_root = tokens[0] if tokens else "C"
    scale    = _MINOR_SCALE if len(tokens) > 1 and "minor" in " ".join(tokens[1:]).lower() else _MAJOR_SCALE
    key_pc   = _NOTE_PC.get(key_root, 0)
    scale_pcs = frozenset((key_pc + i) % 12 for i in scale)

    result: list[tuple[float, str]] = []
    for i, (beat, sym) in enumerate(chord_timeline):
        beat_end = chord_timeline[i + 1][0] if i + 1 < len(chord_timeline) else None
        if beat_end is None:
            result.append((beat, sym))
            continue
        duration = beat_end - beat
        fn = chord_function(sym, key_root, scale)
        sec_dom = secondary_dominant(sym)
        sec_root, _ = _parse_quality(sec_dom)
        sec_pc = _NOTE_PC.get(sec_root, 0)

        # A secondary dominant is useful when its major 3rd is chromatic (outside the key scale).
        # Dominant chords have a major 3rd (4 semitones above root) — that's the chromatic note
        # that creates tension. Root alone can be in the scale and it's still a chromatic dom.
        sec_major_third_pc = (sec_pc + 4) % 12
        is_chromatic_dom = sec_major_third_pc not in scale_pcs

        # Only insert if: chord lasts long enough, function is non-tonic, sec_dom is chromatic
        if (fn in ("SD", "X") and duration >= 2 * min_chord_beats and is_chromatic_dom):
            half = duration / 2
            result.append((beat, sec_dom))
            result.append((beat + half, sym))
        else:
            result.append((beat, sym))

    return result


# ---------------------------------------------------------------------------
# Voice leading
# ---------------------------------------------------------------------------

def _find_pc_near(pc: int, reference: int, lo: int, hi: int, exclude: set[int]) -> int | None:
    """Find nearest MIDI note with pitch class pc to reference, within [lo,hi], not in exclude."""
    for delta in range(0, 13):
        for d in ([delta, -delta] if delta > 0 else [0]):
            m = reference + d
            if lo <= m <= hi and m % 12 == pc and m not in exclude:
                return m
    return None


def voice_lead(
    new_symbol: str,
    prev_midis: list[int],
    lo: int,
    hi: int,
    center: int | None = None,
) -> list[int]:
    """
    Voice the new chord by smooth voice leading from prev_midis.

    Rules (in priority order):
    1. Common tones (same pitch class in both chords) stay at the same MIDI note.
    2. Other voices move to the nearest available chord tone.
    3. No two voices share the same MIDI note.
    4. All voices stay within [lo, hi].
    """
    root, intervals = chord_tones_from_symbol(new_symbol)
    root_pc   = _NOTE_PC.get(root, 0)
    new_pcs   = [(root_pc + iv) % 12 for iv in intervals]

    if not prev_midis:
        # No previous voicing — use close-position voicing near center
        if center is None:
            center = (lo + hi) // 2
        result = []
        ref = center
        used: set[int] = set()
        for pc in new_pcs:
            m = _find_pc_near(pc, ref, lo, hi, used)
            if m is not None:
                result.append(m)
                used.add(m)
                ref = m
        return sorted(result)

    used: set[int] = set()
    result: list[int] = []

    # Pass 1: keep common tones
    prev_by_pc: dict[int, list[int]] = {}
    for m in prev_midis:
        prev_by_pc.setdefault(m % 12, []).append(m)

    kept_pcs: set[int] = set()
    for pc in new_pcs:
        if pc in prev_by_pc:
            # Pick the lowest common-tone midi (most stable)
            m = prev_by_pc[pc][0]
            if m not in used and lo <= m <= hi:
                result.append(m)
                used.add(m)
                kept_pcs.add(pc)

    # Pass 2: resolve remaining voices by minimal movement
    remaining_pcs = [pc for pc in new_pcs if pc not in kept_pcs]
    # Sort previous midis so each remaining pc maps to the closest previous note
    for pc in remaining_pcs:
        best = None
        best_dist = 999
        for prev in prev_midis:
            m = _find_pc_near(pc, prev, lo, hi, used)
            if m is not None and abs(m - prev) < best_dist:
                best      = m
                best_dist = abs(m - prev)
        if best is not None:
            result.append(best)
            used.add(best)

    return sorted(result) if result else prev_midis


# ---------------------------------------------------------------------------
# Additional chord transformations
# ---------------------------------------------------------------------------

_TRIAD_TO_SEVENTH: dict[str, str] = {
    "":    "maj7",   # major triad → major 7th
    "m":   "m7",     # minor triad → minor 7th
    "dim": "m7b5",   # diminished triad → half-diminished
    "aug": "aug7",   # augmented triad → augmented 7th
}


def extend_triads_to_sevenths(
    chord_timeline: list[tuple[float, str]],
    key: str = "",  # reserved for future key-aware dominant detection
) -> list[tuple[float, str]]:
    """
    Upgrade plain triads to their natural 7th form:
      major → maj7, minor → m7, dim → m7b5, aug → aug7.
    Chords already carrying a 7th extension are left unchanged.
    """
    result = []
    for beat, sym in chord_timeline:
        slash = ""
        base = sym
        if "/" in sym:
            base, rest = sym.split("/", 1)
            slash = "/" + rest
        root, quality = _parse_quality(base)
        new_q = _TRIAD_TO_SEVENTH.get(quality)
        result.append((beat, f"{root}{new_q}{slash}" if new_q is not None else sym))
    return result


def insert_back_cycling(
    chord_timeline: list[tuple[float, str]],
    key: str = "",
    min_chord_beats: float = 2.0,
) -> list[tuple[float, str]]:
    """
    Before each dominant-7th chord lasting ≥ 2 × min_chord_beats, insert its
    related ii7 chord for the first half of the duration.
    e.g.  G7 (4 beats)  →  Dm7 (2 beats)  G7 (2 beats)
    """
    result: list[tuple[float, str]] = []
    for i, (beat, sym) in enumerate(chord_timeline):
        beat_end = chord_timeline[i + 1][0] if i + 1 < len(chord_timeline) else None
        if beat_end is None:
            result.append((beat, sym))
            continue

        duration = beat_end - beat
        root, quality = _parse_quality(sym)
        quality_base = quality.split("/")[0]

        if quality_base in _DOMINANT_QUALITIES and duration >= 2 * min_chord_beats:
            # ii root is a perfect 4th below the dominant root (5 semitones down)
            # e.g. V7 = G7 → ii7 = Dm7  (D is 5 semitones below G)
            ii_root = _transpose_root(root, -5)
            half = duration / 2
            result.append((beat, f"{ii_root}m7"))
            result.append((beat + half, sym))
        else:
            result.append((beat, sym))

    return result


def insert_passing_chords(
    chord_timeline: list[tuple[float, str]],
    min_chord_beats: float = 2.0,
) -> list[tuple[float, str]]:
    """
    Insert a chromatic passing diminished chord between consecutive chords
    whose roots are a whole step apart (ascending by 2 semitones).
    The preceding chord must last ≥ 2 × min_chord_beats for the split to be worthwhile.
    e.g.  C (4 beats) Dm  →  C (2 beats) C#dim (2 beats) Dm
    """
    result: list[tuple[float, str]] = []
    for i, (beat, sym) in enumerate(chord_timeline):
        if i + 1 >= len(chord_timeline):
            result.append((beat, sym))
            continue

        beat_end = chord_timeline[i + 1][0]
        duration = beat_end - beat
        next_sym = chord_timeline[i + 1][1]

        root, _ = _parse_quality(sym)
        next_root, _ = _parse_quality(next_sym)
        interval = (_NOTE_PC.get(next_root, 0) - _NOTE_PC.get(root, 0)) % 12

        if interval == 2 and duration >= 2 * min_chord_beats:
            passing_root = _transpose_root(next_root, -1)  # semitone below target
            half = duration / 2
            result.append((beat, sym))
            result.append((beat + half, f"{passing_root}dim"))
        else:
            result.append((beat, sym))

    return result


# ---------------------------------------------------------------------------
# Dynamic shaping
# ---------------------------------------------------------------------------

def shape_velocities(
    notes: list[dict],
    function: str,
    base_vel: int = 72,
    style: str = "Block chords",
) -> list[dict]:
    """
    Apply velocity shaping based on harmonic function and style.
    Modifies notes in-place and returns them.
    """
    chord_vel, _ = tension_velocity(function, base_vel)
    for i, note in enumerate(notes):
        # Layer notes in a voicing: bottom note slightly louder
        rank = i  # assumes notes are sorted by pitch (lowest first)
        vel  = max(40, min(120, chord_vel - rank * 3))
        note["velocity"] = vel
    return notes
