"""
Phrase boundary detection from a quantized melody.

Identifies where musical phrases end by looking for silence gaps between
consecutive notes. Each detected phrase gets its best-fitting diatonic chord
from suggest_chords_from_melody, producing a chord timeline that follows the
actual shape of the recording rather than a rigid bar-count grid.
"""
from __future__ import annotations
from src.chord_detect import suggest_chords_from_melody


def detect_phrases(
    notes: list[dict],
    beats_per_bar: float = 4.0,
    min_rest_beats: float | None = None,
    min_phrase_beats: float | None = None,
) -> list[tuple[float, float]]:
    """
    Detect phrase boundaries from quantized notes.

    A phrase boundary is declared whenever there is a rest gap ≥ min_rest_beats
    AND the preceding phrase is at least min_phrase_beats long.  The boundary
    beat is snapped to the nearest bar line when it falls within a quarter-bar.

    Args:
        notes:            Quantized notes with 'start_beat' and 'duration_beats'.
        beats_per_bar:    Beats per measure (e.g. 4.0 for 4/4, 3.0 for 3/4).
        min_rest_beats:   Minimum silence gap to consider a phrase break.
                          Defaults to one quarter-note (1.0 beat).
        min_phrase_beats: Minimum phrase length in beats.
                          Defaults to two bars.

    Returns:
        Sorted list of (phrase_start_beat, phrase_end_beat) tuples.
        If no boundaries are found, returns the whole melody as one phrase.
    """
    if not notes:
        return []

    if min_rest_beats is None:
        min_rest_beats = 1.0  # one quarter note — language-agnostic default
    if min_phrase_beats is None:
        min_phrase_beats = beats_per_bar * 2

    sorted_notes = sorted(notes, key=lambda n: n.get("start_beat", 0))
    first_start = sorted_notes[0].get("start_beat", 0)
    last_end = max(n.get("start_beat", 0) + n.get("duration_beats", 1) for n in sorted_notes)

    boundaries: list[float] = [first_start]

    for i in range(len(sorted_notes) - 1):
        n = sorted_notes[i]
        nxt = sorted_notes[i + 1]
        n_end = n.get("start_beat", 0) + n.get("duration_beats", 1)
        gap = nxt.get("start_beat", 0) - n_end

        if gap < min_rest_beats:
            continue

        phrase_so_far = nxt.get("start_beat", 0) - boundaries[-1]
        if phrase_so_far < min_phrase_beats:
            continue

        candidate = nxt.get("start_beat", 0)
        # Snap to nearest bar line if within ¼ bar — gives cleaner chord changes
        bar = round(candidate / beats_per_bar) * beats_per_bar
        if abs(bar - candidate) <= beats_per_bar * 0.25:
            candidate = bar

        if candidate > boundaries[-1] + 0.01:
            boundaries.append(candidate)

    phrases: list[tuple[float, float]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else last_end
        if end > start:
            phrases.append((start, end))

    return phrases if phrases else [(first_start, last_end)]


def auto_chord_timeline_from_melody(
    notes: list[dict],
    key: str,
    total_beats: float,
    beats_per_bar: float = 4.0,
    beats_per_chord: float = 4.0,
) -> list[tuple[float, str]]:
    """
    Build a chord timeline by detecting phrase boundaries in the melody and
    choosing the best diatonic chord for each phrase.

    Within each phrase, chords change every beats_per_chord beats (or every
    phrase if the phrase is shorter than beats_per_chord).

    Args:
        notes:          Quantized melody notes.
        key:            Key string e.g. 'D major', 'G Kedar'.
        total_beats:    Total recording duration in beats.
        beats_per_bar:  Beats per measure.
        beats_per_chord: How often chords change within a long phrase.

    Returns:
        Sorted list of (beat_offset, chord_symbol) tuples.
    """
    phrases = detect_phrases(notes, beats_per_bar)

    if not phrases:
        # Fallback: treat whole melody as one phrase
        chords = suggest_chords_from_melody(notes, key, beats_per_chord, total_beats)
        return [(i * beats_per_chord, c) for i, c in enumerate(chords)]

    timeline: list[tuple[float, str]] = []

    for phrase_start, phrase_end in phrases:
        phrase_notes = [
            n for n in notes
            if n.get("start_beat", 0) < phrase_end
            and n.get("start_beat", 0) + n.get("duration_beats", 1) > phrase_start
        ]

        if not phrase_notes:
            if timeline:
                timeline.append((phrase_start, timeline[-1][1]))
            continue

        phrase_beats = phrase_end - phrase_start
        # One chord per phrase when the phrase is shorter than beats_per_chord
        effective_bpc = beats_per_chord if phrase_beats >= beats_per_chord else phrase_beats

        phrase_chords = suggest_chords_from_melody(
            phrase_notes, key, effective_bpc, phrase_beats
        )

        for i, chord in enumerate(phrase_chords):
            beat = phrase_start + i * effective_bpc
            if beat < phrase_end - 0.01:
                timeline.append((beat, chord))

    if not timeline:
        return []

    # If the audio is longer than the last detected phrase (pyin often misses trailing
    # audio), vamp on the closing chords rather than leaving one long sustained note.
    melody_end = phrases[-1][1] if phrases else 0.0
    if total_beats > melody_end + beats_per_chord:
        # Collect the chords from the final phrase as the vamp material
        final_phrase_chords = [c for b, c in timeline if b >= phrases[-1][0]]
        if not final_phrase_chords:
            final_phrase_chords = [timeline[-1][1]]
        vamp_idx = 0
        t = melody_end
        while t < total_beats - 0.01:
            timeline.append((t, final_phrase_chords[vamp_idx % len(final_phrase_chords)]))
            t += beats_per_chord
            vamp_idx += 1

    return sorted(timeline, key=lambda t: t[0])
