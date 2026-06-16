"""
Diatonic chord palette generator.

Given a root note and a scale (by mode/raga name or raw interval input),
derives every natural chord available in that scale: triads, 7ths, 9ths,
aug, dim, sus combinations, and half-diminished. Surfaces enharmonic
equivalents where they exist.

Designed to feed chord suggestions into the orchestral-composer LLM prompt
as a harmonic palette for improvisation — not rigid rules.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Chromatic note tables
# ---------------------------------------------------------------------------

_SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLATS  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_FLAT_ROOTS = {"F", "Bb", "Eb", "Ab", "Db", "Gb"}

def _chromatic(root: str) -> list[str]:
    table = _FLATS if (root in _FLAT_ROOTS or "b" in root) else _SHARPS
    try:
        start = table.index(root)
    except ValueError:
        start = _SHARPS.index(root)
    return [table[(start + i) % 12] for i in range(12)]


# ---------------------------------------------------------------------------
# Mode / scale library  (semitone intervals from root, 0-based)
# ---------------------------------------------------------------------------

MODES: dict[str, list[int]] = {
    # Western diatonic
    "Major (Ionian)":            [0, 2, 4, 5, 7, 9, 11],
    "Dorian":                    [0, 2, 3, 5, 7, 9, 10],
    "Phrygian":                  [0, 1, 3, 5, 7, 8, 10],
    "Lydian":                    [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian":                [0, 2, 4, 5, 7, 9, 10],
    "Minor (Aeolian)":           [0, 2, 3, 5, 7, 8, 10],
    "Locrian":                   [0, 1, 3, 5, 6, 8, 10],

    # Melodic / harmonic variants
    "Harmonic Minor":            [0, 2, 3, 5, 7, 8, 11],
    "Melodic Minor":             [0, 2, 3, 5, 7, 9, 11],
    "Phrygian Dominant":         [0, 1, 4, 5, 7, 8, 10],
    "Hungarian Minor":           [0, 2, 3, 6, 7, 8, 11],
    "Double Harmonic":           [0, 1, 4, 5, 7, 8, 11],

    # Pentatonic
    "Major Pentatonic":          [0, 2, 4, 7, 9],
    "Minor Pentatonic":          [0, 3, 5, 7, 10],
    "Blues":                     [0, 3, 5, 6, 7, 10],

    # Carnatic Melakarta (selected)
    "Mayamalavagowla":           [0, 1, 4, 5, 7, 8, 11],   # = Double Harmonic / Bhairav
    "Shankarabharanam":          [0, 2, 4, 5, 7, 9, 11],   # = Major
    "Kalyani":                   [0, 2, 4, 6, 7, 9, 11],   # = Lydian
    "Kharaharapriya":            [0, 2, 3, 5, 7, 9, 10],   # = Dorian
    "Harikambhoji":              [0, 2, 4, 5, 7, 9, 10],   # = Mixolydian
    "Shanmukha Priya":           [0, 2, 3, 6, 7, 8, 10],   # S R2 G1 M2 P D1 N2 (Melakarta 56)
    "Simhendramadhyamam":        [0, 2, 3, 6, 7, 8, 11],   # S R2 G1 M2 P D1 N3 (Melakarta 57)
    "Hemavathi":                 [0, 2, 3, 6, 7, 9, 10],   # S R2 G1 M2 P D2 N2 (Melakarta 58)
    "Dharmavathi":               [0, 2, 3, 6, 7, 9, 11],   # S R2 G1 M2 P D2 N3 (Melakarta 59)
    "Neetimathi":                [0, 2, 3, 6, 7, 10, 11],  # S R2 G1 M2 P D3 N3 (Melakarta 60)
    "Kantamani":                 [0, 2, 4, 6, 7, 8, 10],   # S R2 G2 M2 P D1 N2 (Melakarta 61)
    "Rishabhapriya":             [0, 2, 4, 6, 7, 8, 11],   # S R2 G2 M2 P D1 N3 (Melakarta 62)
    "Latangi":                   [0, 2, 4, 6, 7, 9, 10],   # S R2 G2 M2 P D2 N2 (Melakarta 63)
    "Suvarnangi":                [0, 1, 3, 6, 7, 9, 10],   # S R1 G1 M2 P D2 N2 (Melakarta 47)
    "Divyamani":                 [0, 1, 3, 6, 7, 9, 11],   # S R1 G1 M2 P D2 N3 (Melakarta 48)
    "Dhavalambari":              [0, 1, 3, 6, 7, 10, 11],  # S R1 G1 M2 P D3 N3 (Melakarta 49)
    "Namanarayani":              [0, 1, 4, 6, 7, 8, 11],   # S R1 G2 M2 P D1 N3 (Melakarta 50)
    "Ramapriya":                 [0, 1, 4, 6, 7, 9, 10],   # S R1 G2 M2 P D2 N2 (Melakarta 52)
    "Gamanashrama":              [0, 1, 4, 6, 7, 9, 11],   # S R1 G2 M2 P D2 N3 (Melakarta 53)
    "Vishwambhari":              [0, 1, 4, 6, 7, 10, 11],  # S R1 G2 M2 P D3 N3 (Melakarta 54)
    "Syamalangi":                [0, 2, 3, 5, 7, 8, 11],   # = Harmonic Minor variant (Melakarta 55)
    "Natabhairavi":              [0, 2, 3, 5, 7, 8, 10],   # = Natural Minor
    "Kiravani":                  [0, 2, 3, 5, 7, 8, 11],   # = Harmonic Minor
    "Charukesi":                 [0, 2, 4, 5, 7, 8, 10],
    "Todi":                      [0, 1, 3, 6, 7, 8, 11],
    "Bhairavi":                  [0, 1, 3, 5, 7, 8, 10],   # = Phrygian
    "Mohanam":                   [0, 2, 4, 7, 9],           # = Major Pentatonic
    "Hamsadhwani":               [0, 2, 4, 7, 11],
    "Hindolam":                  [0, 3, 5, 8, 10],
    "Abhogi":                    [0, 2, 3, 5, 9],
    "Malkauns":                  [0, 3, 5, 8, 10],          # = Hindolam

    # Hindustani ragas
    "Yaman":                     [0, 2, 4, 6, 7, 9, 11],   # = Lydian
    "Bhairav":                   [0, 1, 4, 5, 7, 8, 11],   # = Mayamalavagowla
    "Kafi":                      [0, 2, 3, 5, 7, 9, 10],   # = Dorian
    "Bhimpalasi":                [0, 2, 3, 5, 7, 8, 10],
    "Marwa":                     [0, 1, 4, 6, 7, 9, 11],
    "Purvi":                     [0, 1, 4, 6, 7, 8, 11],
    "Desh":                      [0, 2, 4, 5, 7, 9, 10, 11],  # both komal (♭VII) and shuddha (VII) nishad
    "Bageshri":                  [0, 2, 3, 5, 7, 8, 10],
    "Bihag":                     [0, 2, 4, 5, 6, 9, 11],
    "Yaman Kalyan":              [0, 2, 4, 5, 6, 9, 11],
    "Darbari Kanada":            [0, 2, 3, 5, 7, 8, 10],
    "Miya Malhar":               [0, 2, 3, 5, 7, 8, 10],   # komal Ga, komal Ni variant
    "Kedar":                     [0, 2, 4, 5, 6, 7, 10],
    "Jhinjhoti":                 [0, 2, 4, 5, 7, 9, 10],   # Khamaj variant
    "Khamaj":                    [0, 2, 4, 5, 7, 9, 10],   # = Mixolydian
    "Lalit":                     [0, 1, 4, 5, 6, 8, 11],
    "Sindh Bhairavi":            [0, 1, 3, 5, 7, 8, 10],   # = Bhairavi
    "Shree":                     [0, 1, 4, 5, 7, 8, 11],
    "Multani":                   [0, 1, 3, 6, 7, 8, 11],   # = Todi variant

    # Middle Eastern / Turkish
    "Hijaz":                     [0, 1, 4, 5, 7, 8, 10],   # Arabic minor with aug 2nd
    "Hijaz Kar":                 [0, 1, 4, 5, 7, 8, 11],   # Hijaz with major 7th
    "Saba":                      [0, 1, 3, 5, 6, 8, 10],
    "Rast":                      [0, 2, 3, 5, 7, 9, 10],   # like Dorian but with neutral 3rd (approx)
    "Nahawand":                  [0, 2, 3, 5, 7, 8, 11],   # = Harmonic Minor
    "Kurd":                      [0, 1, 3, 5, 7, 8, 10],   # = Phrygian
    "Nikriz":                    [0, 2, 3, 6, 7, 9, 10],

    # Western exotic / synthetic
    "Whole Tone":                [0, 2, 4, 6, 8, 10],
    "Diminished (HW)":          [0, 1, 3, 4, 6, 7, 9, 10],  # half-whole diminished (octatonic)
    "Diminished (WH)":          [0, 2, 3, 5, 6, 8, 9, 11],  # whole-half diminished
    "Chromatic":                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Enigmatic":                 [0, 1, 4, 6, 8, 10, 11],
    "Neapolitan Major":          [0, 1, 3, 5, 7, 9, 11],
    "Neapolitan Minor":          [0, 1, 3, 5, 7, 8, 11],
    "Persian":                   [0, 1, 4, 5, 6, 8, 11],
    "Arabian":                   [0, 2, 4, 5, 6, 8, 10],
    "Byzantine":                 [0, 1, 4, 5, 7, 8, 11],    # = Double Harmonic
    "Eight-tone Spanish":        [0, 1, 3, 4, 5, 6, 8, 10],
    "Prometheus":                [0, 2, 4, 6, 9, 10],
    "Tritone":                   [0, 1, 4, 6, 7, 10],
    "Augmented":                 [0, 3, 4, 7, 8, 11],

    # Balinese / Javanese
    "Pelog":                     [0, 1, 3, 7, 8],
    "Slendro":                   [0, 2, 5, 7, 10],

    # Flamenco / Spanish
    "Flamenco":                  [0, 1, 4, 5, 7, 8, 11],    # = Double Harmonic
    "Spanish Gypsy":             [0, 1, 4, 5, 7, 8, 10],
    "Andalusian":                [0, 2, 3, 5, 6, 8, 10],
}

# Aliases from raga-harmony-studio numerical format (1-based swarasthanas)
# Users may type these directly; parse_mode_input handles conversion.


# ---------------------------------------------------------------------------
# Chord data
# ---------------------------------------------------------------------------

@dataclass
class ChordInfo:
    symbol:      str          # e.g. "Am7"
    root:        str          # e.g. "A"
    quality:     str          # e.g. "min7"  (internal label)
    degree:      int          # 0-indexed scale degree
    intervals:   list[int]    # semitones above chord root present in scale
    enharmonics: list[str] = field(default_factory=list)

    # Display category
    @property
    def category(self) -> str:
        if any(x in self.quality for x in ("9", "11", "13", "add")):
            return "Extended"
        if any(x in self.quality for x in ("7", "maj7", "dim7", "m7b5")):
            return "7th"
        if "sus" in self.quality:
            return "Sus"
        if self.quality in ("aug",):
            return "Aug"
        return "Triad"


# Canonical chord tones for each quality (used for inversion / slash generation)
_QUALITY_TONES: dict[str, list[int]] = {
    "maj":     [0, 4, 7],
    "min":     [0, 3, 7],
    "dim":     [0, 3, 6],
    "aug":     [0, 4, 8],
    "sus2":    [0, 2, 7],
    "sus4":    [0, 5, 7],
    "maj7":    [0, 4, 7, 11],
    "dom7":    [0, 4, 7, 10],
    "min7":    [0, 3, 7, 10],
    "minMaj7": [0, 3, 7, 11],
    "m7b5":    [0, 3, 6, 10],
    "dim7":    [0, 3, 6, 9],
    "aug7":    [0, 4, 8, 10],
    "7sus4":   [0, 5, 7, 10],
}


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_scale_chords(root: str, intervals: list[int]) -> list[ChordInfo]:
    """
    Derive all natural chords available in a scale.

    For every scale degree, check which harmonic intervals exist within
    the scale and yield every valid chord type (triad through extended).
    """
    notes = _chromatic(root)
    chords: list[ChordInfo] = []

    for degree, semitone in enumerate(intervals):
        degree_root = notes[semitone % 12]
        # Intervals from this degree that exist in the scale
        avail = {(s - semitone) % 12 for s in intervals}

        has = avail.__contains__   # shorthand

        # Quality detectors
        m3, M3 = has(3),  has(4)
        p4      = has(5)
        d5, p5, A5 = has(6), has(7), has(8)
        M2      = has(2)   # also = major 9th mod 12
        m2      = has(1)   # also = minor 9th mod 12
        d7      = has(9)   # diminished 7th (= major 6th)
        m7, M7  = has(10), has(11)
        A4      = has(6)   # aug 11th = tritone

        def chord(quality: str, suffix: str) -> ChordInfo:
            return ChordInfo(
                symbol=f"{degree_root}{suffix}",
                root=degree_root,
                quality=quality,
                degree=degree,
                intervals=sorted(avail),
            )

        # --- Triads ---
        if M3 and p5:
            chords.append(chord("maj",  ""))
        if m3 and p5:
            chords.append(chord("min",  "m"))
        if m3 and d5 and not m7 and not d7:
            chords.append(chord("dim",  "dim"))
        if M3 and A5:
            chords.append(chord("aug",  "aug"))
        if M2 and p5 and not m3 and not M3:
            chords.append(chord("sus2", "sus2"))
        if p4 and p5 and not m3 and not M3:
            chords.append(chord("sus4", "sus4"))

        # --- 7th chords ---
        if M3 and p5 and M7:
            chords.append(chord("maj7",    "maj7"))
        if M3 and p5 and m7:
            chords.append(chord("dom7",    "7"))
        if m3 and p5 and m7:
            chords.append(chord("min7",    "m7"))
        if m3 and p5 and M7:
            chords.append(chord("minMaj7", "mMaj7"))
        if m3 and d5 and m7:
            chords.append(chord("m7b5",    "m7b5"))   # half-diminished ø
        if m3 and d5 and d7:
            chords.append(chord("dim7",    "dim7"))
        if M3 and A5 and m7:
            chords.append(chord("aug7",    "aug7"))
        if p4 and p5 and m7 and not m3 and not M3:
            chords.append(chord("7sus4",   "7sus4"))

        # --- Extended (9th, add9, #11) ---
        if M3 and p5 and M2 and not m7 and not M7:
            chords.append(chord("add9",  "add9"))
        if M3 and p5 and m7 and M2:
            chords.append(chord("9",     "9"))
        if M3 and p5 and M7 and M2:
            chords.append(chord("maj9",  "maj9"))
        if m3 and p5 and m7 and M2:
            chords.append(chord("min9",  "m9"))
        if M3 and p5 and m7 and A4:
            chords.append(chord("7#11",  "7#11"))   # Lydian dominant

    _mark_enharmonics(chords)
    return chords


# ---------------------------------------------------------------------------
# Chord-tone analysis, MIDI helpers, and compliance enforcement
# ---------------------------------------------------------------------------

_NOTE_PC: dict[str, int] = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4, 'F': 5, 'E#': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11, 'B#': 0,
}

_CHORD_QUALITY_INTERVALS: dict[str, list[int]] = {
    "":       [0, 4, 7],      # major
    "m":      [0, 3, 7],      # minor
    "min":    [0, 3, 7],
    "dim":    [0, 3, 6],
    "aug":    [0, 4, 8],
    "sus2":   [0, 2, 7],
    "sus4":   [0, 5, 7],
    "maj7":   [0, 4, 7, 11],
    "M7":     [0, 4, 7, 11],
    "7":      [0, 4, 7, 10],
    "m7":     [0, 3, 7, 10],
    "min7":   [0, 3, 7, 10],
    "mMaj7":  [0, 3, 7, 11],
    "mM7":    [0, 3, 7, 11],
    "m7b5":   [0, 3, 6, 10],
    "dim7":   [0, 3, 6, 9],
    "aug7":   [0, 4, 8, 10],
    "7sus4":  [0, 5, 7, 10],
    "add9":   [0, 2, 4, 7],
    "9":      [0, 2, 4, 7, 10],
    "maj9":   [0, 2, 4, 7, 11],
    "m9":     [0, 2, 3, 7, 10],
    "7#11":   [0, 4, 6, 7, 10],
}


def chord_tones_from_symbol(symbol: str) -> tuple[str, list[int]]:
    """Parse chord symbol → (root_note, [semitone_intervals]). Strips slash bass note."""
    if "/" in symbol:
        symbol = symbol.split("/")[0]
    s = symbol.strip()
    root, quality_str = (s[:2], s[2:]) if len(s) >= 2 and s[1] in ("#", "b") else (s[:1], s[1:])
    return root, _CHORD_QUALITY_INTERVALS.get(quality_str, [0, 4, 7])


def chord_tone_names(symbol: str) -> list[str]:
    """'Am7' → ['A', 'C', 'E', 'G']"""
    root, intervals = chord_tones_from_symbol(symbol)
    chromatic = _chromatic(root)
    return [chromatic[i % 12] for i in intervals]


def _chord_pc_set(symbol: str) -> frozenset[int]:
    root, intervals = chord_tones_from_symbol(symbol)
    root_pc = _NOTE_PC.get(root, 0)
    return frozenset((root_pc + i) % 12 for i in intervals)


def note_name_to_midi(note: str) -> int:
    """'D4' → 62, 'C#4' → 61, 'Bb3' → 58"""
    note = note.strip()
    if len(note) >= 3 and note[1] in ('#', 'b'):
        pitch, octave_str = note[:2], note[2:]
    else:
        pitch, octave_str = note[0], note[1:]
    pc = _NOTE_PC.get(pitch)
    if pc is None:
        raise ValueError(f"Unknown pitch: {pitch!r} in {note!r}")
    return (int(octave_str) + 1) * 12 + pc


def midi_to_note_name(midi: int) -> str:
    """62 → 'D4', 61 → 'C#4' (always uses sharps)"""
    return f"{_SHARPS[midi % 12]}{midi // 12 - 1}"


def build_chord_timeline(
    chords: list[str], beats_per_chord: float, total_beats: float
) -> list[tuple[float, str]]:
    """Expand repeating chord list → [(beat, symbol), ...] covering total_beats."""
    timeline, beat, idx = [], 0.0, 0
    while beat < total_beats:
        timeline.append((beat, chords[idx % len(chords)]))
        beat += beats_per_chord
        idx += 1
    return timeline


def _active_chord(timeline: list[tuple[float, str]], beat: float) -> str:
    result = timeline[0][1]
    for cb, cs in timeline:
        if beat >= cb:
            result = cs
        else:
            break
    return result


def _snap_to_chord_tone(midi: int, chord: str, lo: int, hi: int) -> int:
    """Return nearest in-range MIDI note that is a chord tone of chord."""
    root, intervals = chord_tones_from_symbol(chord)
    root_pc = _NOTE_PC.get(root, 0)
    valid_pcs = frozenset((root_pc + i) % 12 for i in intervals)
    if midi % 12 in valid_pcs:
        return midi
    candidates = [m for m in range(max(lo, midi - 12), min(hi + 1, midi + 13))
                  if m % 12 in valid_pcs]
    if not candidates:
        candidates = [m for m in range(lo, hi + 1) if m % 12 in valid_pcs]
    return min(candidates, key=lambda m: abs(m - midi)) if candidates else midi


def enforce_chord_compliance(
    orchestration: dict,
    timeline: list[tuple[float, str]],
    melody_instrument: str,
    instrument_ranges: dict[str, tuple[int, int]],
    forbidden_symbols: list[str] | None = None,
) -> dict:
    """
    Post-process: snap every non-chord-tone note to the nearest chord tone,
    then check for forbidden chord formations and re-voice them.
    Melody track is always skipped — only harmony/bass parts are touched.
    """
    if not timeline:
        return orchestration

    forbidden_pc_sets = [_chord_pc_set(s) for s in (forbidden_symbols or [])]

    # --- Pass 1: snap every note to its active chord's tones ---
    for inst, notes in orchestration.get("parts", {}).items():
        if inst == melody_instrument:
            continue
        lo, hi = instrument_ranges.get(inst, (36, 96))
        for note in notes:
            if not isinstance(note, dict) or "note" not in note:
                continue
            try:
                midi = note_name_to_midi(note["note"])
            except (ValueError, KeyError):
                continue
            chord = _active_chord(timeline, note.get("start_beat", 0.0))
            snapped = _snap_to_chord_tone(midi, chord, lo, hi)
            note["note"] = midi_to_note_name(snapped)

    # --- Pass 2: forbidden chord detection and re-voicing ---
    if not forbidden_pc_sets:
        return orchestration

    # Collect notes per 16th-note grid slot across all non-melody parts
    from collections import defaultdict
    slot_notes: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for inst, notes in orchestration.get("parts", {}).items():
        if inst == melody_instrument:
            continue
        for note in notes:
            if not isinstance(note, dict):
                continue
            slot = round(note.get("start_beat", 0.0) * 4)
            slot_notes[slot].append((inst, note))

    for slot, inst_notes in slot_notes.items():
        beat = slot / 4.0
        chord = _active_chord(timeline, beat)
        root, intervals = chord_tones_from_symbol(chord)
        root_pc = _NOTE_PC.get(root, 0)
        valid_pcs = frozenset((root_pc + iv) % 12 for iv in intervals)

        try:
            pcs = frozenset(note_name_to_midi(n["note"]) % 12 for _, n in inst_notes)
        except (ValueError, KeyError):
            continue

        for forbidden_pcs in forbidden_pc_sets:
            if not forbidden_pcs.issubset(pcs):
                continue
            # Re-voice: find the offending note and move it to a different chord tone
            for inst, note in inst_notes:
                try:
                    midi = note_name_to_midi(note["note"])
                except ValueError:
                    continue
                if midi % 12 not in forbidden_pcs:
                    continue
                lo, hi = instrument_ranges.get(inst, (36, 96))
                # Try adjacent chord tones that break the forbidden pattern
                for delta in [7, -7, 4, -4, 2, -2, 12, -12]:
                    candidate = midi + delta
                    if lo <= candidate <= hi and candidate % 12 in valid_pcs:
                        new_pcs = (pcs - {midi % 12}) | {candidate % 12}
                        if not forbidden_pcs.issubset(new_pcs):
                            note["note"] = midi_to_note_name(candidate)
                            pcs = new_pcs
                            break
                break  # fix one note per slot per forbidden chord

    return orchestration


def fill_chord_voicings(
    orchestration: dict,
    chord_timeline: list[tuple[float, str]],
    chordal_instruments: frozenset[str],
    instrument_ranges: dict[str, tuple[int, int]],
) -> dict:
    """
    Guarantee that every chordal instrument has a full chord voicing for every chord slot.

    For each chord slot, if the LLM placed fewer note objects than there are chord tones
    (or placed none at all), the missing tones are added above the existing notes in close
    position. This converts sparse single-note LLM output into actual chord voicings.
    """
    if not chord_timeline:
        return orchestration

    for inst in chordal_instruments:
        part = orchestration.get("parts", {}).get(inst)
        if not part:
            continue

        lo, hi = instrument_ranges.get(inst, (36, 96))
        new_notes: list[dict] = list(part)

        for i, (beat_start, symbol) in enumerate(chord_timeline):
            beat_end = chord_timeline[i + 1][0] if i + 1 < len(chord_timeline) else float("inf")
            slot_dur = (chord_timeline[i + 1][0] - beat_start) if i + 1 < len(chord_timeline) else 4.0

            try:
                root, intervals = chord_tones_from_symbol(symbol)
                root_pc = _NOTE_PC.get(root, 0)
                chord_pcs = [(root_pc + iv) % 12 for iv in intervals]
            except Exception:
                continue

            slot_notes = [
                n for n in part
                if isinstance(n, dict) and beat_start <= n.get("start_beat", -1) < beat_end
            ]

            if not slot_notes:
                # Build a root-position voicing from scratch
                center = (lo + hi) // 2
                root_midi = root_pc + 12 * round((center - root_pc) / 12)
                if root_midi < lo:
                    root_midi += 12
                if root_midi > hi:
                    root_midi -= 12
                if not (lo <= root_midi <= hi):
                    continue
                for j, iv in enumerate(intervals):
                    midi = root_midi + iv
                    if midi > hi:
                        midi -= 12
                    if lo <= midi <= hi:
                        new_notes.append({
                            "note": midi_to_note_name(midi),
                            "start_beat": beat_start,
                            "duration_beats": slot_dur,
                            "velocity": 72 if j == 0 else 67,
                        })
                continue

            # Find present pitch classes and highest midi in this slot
            present_pcs: set[int] = set()
            midis_in_slot: list[int] = []
            template = slot_notes[0]
            for n in slot_notes:
                try:
                    midi = note_name_to_midi(n["note"])
                    present_pcs.add(midi % 12)
                    midis_in_slot.append(midi)
                except Exception:
                    pass

            if not midis_in_slot:
                continue

            missing_pcs = [pc for pc in chord_pcs if pc not in present_pcs]
            if not missing_pcs:
                continue

            t_beat = template.get("start_beat", beat_start)
            t_dur  = template.get("duration_beats", slot_dur)
            t_vel  = max(40, template.get("velocity", 72) - 8)

            # Stack missing tones in close position above the highest existing note
            top = max(midis_in_slot)
            for pc in missing_pcs:
                midi = top + 1
                while midi % 12 != pc:
                    midi += 1
                if midi > hi:
                    midi -= 12
                if lo <= midi <= hi:
                    new_notes.append({
                        "note": midi_to_note_name(midi),
                        "start_beat": t_beat,
                        "duration_beats": t_dur,
                        "velocity": t_vel,
                    })
                    top = midi

        orchestration["parts"][inst] = new_notes

    return orchestration


def scale_note_names(root: str, intervals: list[int]) -> list[str]:
    """Return the pitch-class names in a scale, e.g. D Double Harmonic → ['D','Eb','F#','G','A','Bb','C#']."""
    chromatic = _chromatic(root)
    return [chromatic[i % 12] for i in intervals]


_KEY_ALIASES: dict[str, str] = {
    "major":                 "Major (Ionian)",
    "ionian":                "Major (Ionian)",
    "minor":                 "Minor (Aeolian)",
    "aeolian":               "Minor (Aeolian)",
    "natural minor":         "Minor (Aeolian)",
    "harmonic minor":        "Harmonic Minor",
    "melodic minor":         "Melodic Minor",
    "dorian":                "Dorian",
    "phrygian":              "Phrygian",
    "lydian":                "Lydian",
    "mixolydian":            "Mixolydian",
    "locrian":               "Locrian",
    "phrygian dominant":     "Phrygian Dominant",
    "hungarian minor":       "Hungarian Minor",
    "double harmonic":       "Double Harmonic",
    "double harmonic major": "Double Harmonic",
    "bhairav":               "Bhairav",
    "mayamalavagowla":       "Mayamalavagowla",
    "kalyani":               "Kalyani",
    "todi":                  "Todi",
    "yaman":                 "Yaman",
    "kafi":                  "Kafi",
    "bhairavi":              "Bhairavi",
    "mohanam":               "Mohanam",
    "shankarabharanam":      "Shankarabharanam",
    "kharaharapriya":        "Kharaharapriya",
    "kiravani":              "Kiravani",
    "charukesi":             "Charukesi",
    "harikambhoji":          "Harikambhoji",
    "natabhairavi":          "Natabhairavi",
    "blues":                 "Blues",
    "major pentatonic":      "Major Pentatonic",
    "minor pentatonic":      "Minor Pentatonic",
    # Middle Eastern
    "hijaz":                 "Hijaz",
    "hijaz kar":             "Hijaz Kar",
    "nahawand":              "Nahawand",
    "rast":                  "Rast",
    "saba":                  "Saba",
    "nikriz":                "Nikriz",
    # Western exotic
    "whole tone":            "Whole Tone",
    "diminished":            "Diminished (HW)",
    "octatonic":             "Diminished (HW)",
    "enigmatic":             "Enigmatic",
    "neapolitan major":      "Neapolitan Major",
    "neapolitan minor":      "Neapolitan Minor",
    "persian":               "Persian",
    "byzantine":             "Byzantine",
    "prometheus":            "Prometheus",
    "augmented":             "Augmented",
    # Hindustani extras
    "lalit":                 "Lalit",
    "kedar":                 "Kedar",
    "shree":                 "Shree",
    "multani":               "Multani",
    "khamaj":                "Khamaj",
    # Flamenco / Spanish
    "flamenco":              "Flamenco",
    "spanish gypsy":         "Spanish Gypsy",
    "andalusian":            "Andalusian",
    # World
    "pelog":                 "Pelog",
    "slendro":               "Slendro",
}


def parse_key_string(key_str: str) -> tuple[str, list[int]] | None:
    """
    Parse a free-form key string like 'D Double Harmonic' or 'G Todi' into
    (root, scale_intervals). Returns None if the root or mode is unrecognisable.
    """
    tokens = key_str.strip().split()
    if not tokens:
        return None
    root = tokens[0]
    if root not in _NOTE_PC:
        return None
    if len(tokens) == 1:
        return root, MODES["Major (Ionian)"]
    mode_str = " ".join(tokens[1:])
    # Encoded custom intervals: "D custom:0,2,3,6,7,8,10"
    if mode_str.startswith("custom:"):
        try:
            intervals = sorted(set(int(x) for x in mode_str[7:].split(",")))
            return root, intervals
        except (ValueError, IndexError):
            return None
    # Direct MODES key
    if mode_str in MODES:
        return root, MODES[mode_str]
    # Case-insensitive alias
    key_lower = mode_str.lower()
    if key_lower in _KEY_ALIASES:
        return root, MODES[_KEY_ALIASES[key_lower]]
    # Partial match (first word)
    first = tokens[1].lower()
    if first in _KEY_ALIASES:
        return root, MODES[_KEY_ALIASES[first]]
    return None


def check_chord_scale_compatibility(
    chord_timeline: list[tuple[float, str]],
    key: str,
) -> list[str]:
    """
    Return a list of warning strings for chords that contain notes outside the scale.
    Each unique chord is checked once. Empty list = all chords are scale-compatible.
    """
    parsed = parse_key_string(key)
    if not parsed:
        return []
    root, intervals = parsed
    root_pc = _NOTE_PC.get(root, 0)
    scale_pcs = frozenset((root_pc + i) % 12 for i in intervals)

    warnings: list[str] = []
    seen: set[str] = set()
    for _, chord in chord_timeline:
        base = chord.split("/")[0]
        if base in seen:
            continue
        seen.add(base)
        try:
            tones = chord_tone_names(base)
        except Exception:
            continue
        clashes = [t for t in tones if note_name_to_midi(f"{t}4") % 12 not in scale_pcs]
        if clashes:
            warnings.append(f"{base} has out-of-scale tones: {', '.join(clashes)}")
    return warnings


def snap_all_parts_to_scale(
    orchestration: dict,
    scale_pcs: frozenset[int],
    instrument_ranges: dict[str, tuple[int, int]],
) -> dict:
    """
    Snap every note in every part to the nearest pitch class in scale_pcs.
    Applied in compose mode so the melody obeys the mode/raga, not just the harmony.
    """
    for inst, notes in orchestration.get("parts", {}).items():
        lo, hi = instrument_ranges.get(inst, (36, 96))
        for note in notes:
            if not isinstance(note, dict) or "note" not in note:
                continue
            try:
                midi = note_name_to_midi(note["note"])
            except (ValueError, KeyError):
                continue
            if midi % 12 in scale_pcs:
                continue
            candidates = [m for m in range(max(lo, midi - 6), min(hi + 1, midi + 7))
                          if m % 12 in scale_pcs]
            if not candidates:
                candidates = [m for m in range(lo, hi + 1) if m % 12 in scale_pcs]
            if candidates:
                note["note"] = midi_to_note_name(min(candidates, key=lambda m: abs(m - midi)))
    return orchestration


# ---------------------------------------------------------------------------
# Enharmonic equivalence
# ---------------------------------------------------------------------------

def _mark_enharmonics(chords: list[ChordInfo]) -> None:
    """
    Mark chords that are enharmonically equivalent to each other.

    Two chords are enharmonic if they share the same interval set from
    different roots. The classic example: Bdim7 = Ddim7 = Fdim7 = Abdim7.
    """
    from collections import defaultdict
    # Map frozenset(intervals) + quality → list of symbols
    groups: dict[tuple, list[ChordInfo]] = defaultdict(list)
    for c in chords:
        key = (c.quality, frozenset(c.intervals))
        groups[key].append(c)
    for group in groups.values():
        if len(group) > 1:
            symbols = [c.symbol for c in group]
            for c in group:
                c.enharmonics = [s for s in symbols if s != c.symbol]


# ---------------------------------------------------------------------------
# Slash / inversion variants
# ---------------------------------------------------------------------------

def generate_slash_variants(
    chords: list[ChordInfo],
    root: str,
    scale_intervals: list[int],
) -> list[str]:
    """
    For every triad and 7th chord in the palette, generate slash-chord variants
    where a non-root chord tone (that is also in the scale) sits in the bass.

    This covers every scale note as a reachable bass note, including tones that
    are not chord roots in the scale (common in pentatonic and raga scales).

    Returns deduplicated list of "Chord/Bass" strings, e.g. ["C/E", "C/G", "Am7/C"].
    """
    chromatic      = _chromatic(root)
    scale_set      = set(scale_intervals)
    st_to_note     = {i: chromatic[i % 12] for i in range(12)}

    variants: list[str] = []
    seen:     set[str]  = set()

    for c in chords:
        if c.category not in ("Triad", "7th"):
            continue
        tones = _QUALITY_TONES.get(c.quality, [])
        chord_root_st = scale_intervals[c.degree]

        for interval in tones[1:]:          # skip root (interval 0)
            bass_st = (chord_root_st + interval) % 12
            if bass_st in scale_set:
                symbol = f"{c.symbol}/{st_to_note[bass_st]}"
                if symbol not in seen:
                    seen.add(symbol)
                    variants.append(symbol)

    return variants


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_mode_input(mode_name: str, custom_text: str = "") -> list[int]:
    """
    Return semitone intervals (0-based) for a given mode or raw interval string.

    Custom text always takes priority over mode_name when non-empty.

    Accepts:
      - A key in MODES
      - Swarasthanas (1-based, 1-12):  "1 2 5 6 8 9 12"
      - Semitones (0-based, 0-11):     "0 1 4 5 7 8 11"
    """
    # Custom intervals take priority over the named mode
    text = custom_text.strip()
    if text:
        nums = [int(x) for x in text.replace(",", " ").split() if x.strip().lstrip("-").isdigit()]
        if nums:
            # Distinguish 1-based (swarasthanas, min≥1, max≤12)
            # from 0-based semitones (min==0)
            if min(nums) >= 1:
                return sorted(set((n - 1) % 12 for n in nums))
            else:
                return sorted(set(n % 12 for n in nums))

    if mode_name in MODES:
        return MODES[mode_name]

    return MODES["Major (Ionian)"]


# ---------------------------------------------------------------------------
# Palette display
# ---------------------------------------------------------------------------

_CATEGORY_ORDER = ["Triad", "7th", "Extended", "Sus", "Aug"]
_CATEGORY_COLOURS = {
    "Triad":    "#dbeafe",   # blue-100
    "7th":      "#dcfce7",   # green-100
    "Extended": "#fef9c3",   # yellow-100
    "Sus":      "#fce7f3",   # pink-100
    "Aug":      "#ede9fe",   # purple-100
}

def palette_html(
    chords: list[ChordInfo],
    root: str,
    mode_name: str,
    slash_variants: list[str] | None = None,
) -> str:
    """Return an HTML chord palette grouped by category, with optional slash row."""
    from collections import defaultdict
    by_cat: dict[str, list[ChordInfo]] = defaultdict(list)
    for c in chords:
        by_cat[c.category].append(c)

    def _badge(label: str, colour: str, enharmonics: list[str] | None = None) -> str:
        enh = ""
        if enharmonics:
            enh = f' <span style="color:#666;font-size:10px;">= {", ".join(enharmonics)}</span>'
        return (
            f'<span style="background:{colour};color:#1a1a1a;padding:3px 8px;border-radius:12px;'
            f'margin:2px;display:inline-block;font-size:12px;font-weight:600;">'
            f'{label}{enh}</span>'
        )

    def _label(text: str) -> str:
        return (
            f'<span style="font-size:10px;color:#1a1a1a;background:#e5e7eb;padding:2px 6px;'
            f'border-radius:4px;font-weight:700;text-transform:uppercase;margin-right:6px;">{text}</span>'
        )

    rows = []
    for cat in _CATEGORY_ORDER:
        if cat not in by_cat:
            continue
        colour = _CATEGORY_COLOURS[cat]
        badges = []
        seen: set[str] = set()
        for c in by_cat[cat]:
            if c.symbol in seen:
                continue
            seen.add(c.symbol)
            badges.append(_badge(c.symbol, colour, c.enharmonics or None))
        rows.append(
            f'<div style="margin-bottom:6px;">{_label(cat)}{"".join(badges)}</div>'
        )

    if slash_variants:
        badges = [_badge(s, "#f1f5f9") for s in slash_variants]
        rows.append(
            f'<div style="margin-bottom:6px;">{_label("Slash / Inv")}{"".join(badges)}</div>'
        )

    header = (
        f'<div style="font-size:12px;color:#1a1a1a;background:#f3f4f6;padding:6px 10px;'
        f'border-radius:6px;margin-bottom:8px;font-weight:600;">'
        f'{root} {mode_name} — chords available in this scale<br>'
        f'<span style="font-weight:400;font-size:11px;color:#555;">'
        f'Check chords below then click <em>Use selected</em></span></div>'
    )
    return header + "\n".join(rows) if rows else "<em>No chords generated.</em>"


# ---------------------------------------------------------------------------
# Progression suggester
# ---------------------------------------------------------------------------

# Standard progressions by mode (scale degree indices, 0-based, prefer richer chords)
_MODE_PROGRESSIONS: dict[str, list[int]] = {
    "Major (Ionian)":   [1, 4, 0],       # ii – V – I
    "Dorian":           [0, 3, 0, 4],     # i – IV – i – V
    "Phrygian":         [0, 6, 7, 0],     # i – bVII – bVIII – i  (approx)
    "Lydian":           [0, 1, 4, 0],     # I – II – V – I
    "Mixolydian":       [0, 6, 3, 0],     # I – bVII – IV – I
    "Minor (Aeolian)":  [0, 3, 4, 0],     # i – iv – v – i
    "Locrian":          [0, 1, 3, 0],
    "Harmonic Minor":   [0, 3, 4, 0],     # i – iv – V – i  (V is major in harm. minor)
    "Melodic Minor":    [0, 3, 4, 0],
    "Phrygian Dominant":[0, 6, 3, 0],
}

def suggest_progression(
    chords: list[ChordInfo],
    mode_name: str,
    beats_per_chord: float = 4.0,
) -> str:
    """
    Return one best chord per scale degree as space-separated symbols.

    Mode-specific characteristic degrees come first (if defined); remaining
    degrees follow in ascending order. No degree repeats.
    """
    _quality_rank = {
        "maj9": 10, "min9": 10, "9": 10, "maj7": 9, "min7": 9, "dom7": 9,
        "m7b5": 8,  "dim7": 8,  "7sus4": 7, "maj": 6, "min": 6, "dim": 5,
        "aug": 4,   "sus4": 3,  "sus2": 3,
    }
    by_degree: dict[int, ChordInfo] = {}
    for c in chords:
        prev = by_degree.get(c.degree)
        if prev is None or _quality_rank.get(c.quality, 0) > _quality_rank.get(prev.quality, 0):
            by_degree[c.degree] = c

    all_degrees = sorted(by_degree.keys())
    characteristic = _MODE_PROGRESSIONS.get(mode_name, [])
    # Deduplicate while preserving order: characteristic first, then remaining
    seen: set[int] = set()
    ordered: list[int] = []
    for d in characteristic:
        d = d % len(all_degrees) if all_degrees else d
        if d in by_degree and d not in seen:
            ordered.append(d)
            seen.add(d)
    for d in all_degrees:
        if d not in seen:
            ordered.append(d)
            seen.add(d)

    return "  ".join(by_degree[d].symbol for d in ordered)