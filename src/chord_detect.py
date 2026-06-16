"""
Algorithmic chord detection from a transcribed melody.

Scores diatonic chord candidates by weighted pitch-class overlap with active
melody notes in each time slot. No LLM required.
"""
from __future__ import annotations
import re
from src.harmony import (
    parse_key_string, generate_scale_chords,
    _NOTE_PC, chord_tones_from_symbol,
)


def suggest_chords_from_melody(
    notes: list[dict],
    key: str,
    beats_per_chord: float = 4.0,
    total_beats: float = 0.0,
) -> list[str]:
    """
    Return one chord symbol per beats_per_chord slot, chosen from diatonic candidates.

    Each note's contribution to a slot is weighted by its overlap duration.
    Scoring: sum of weighted coverage for chord tones minus a penalty for
    melody notes that fall outside the chord (passing-tone tolerance = 0.4).

    Args:
        notes: quantised notes with 'start_beat', 'duration_beats', and
               'note_name' or 'note' keys.
        key: key string, e.g. 'D major', 'G Todi'.
        beats_per_chord: how many beats each chord slot spans.
        total_beats: total piece duration; inferred from notes if 0.

    Returns:
        List of chord symbols, one per slot (may repeat consecutive chords).
    """
    if not notes:
        return []

    if total_beats <= 0:
        total_beats = max(
            n.get("start_beat", 0) + n.get("duration_beats", 1) for n in notes
        )

    parsed = parse_key_string(key)
    if not parsed:
        return []

    root, intervals = parsed
    scale_chords = generate_scale_chords(root, intervals)

    # Quality preference rank (prefer richer chords when scores are tied)
    _quality_rank: dict[str, int] = {
        "maj9": 10, "min9": 10, "9": 10,
        "maj7": 9, "min7": 9, "dom7": 9,
        "m7b5": 8, "dim7": 8, "7sus4": 7,
        "add9": 6, "maj": 5, "min": 5,
        "dim": 4, "aug": 3, "sus4": 2, "sus2": 2,
    }

    # De-duplicate by symbol, keeping the highest-ranked quality
    seen: dict[str, int] = {}
    candidates: list[tuple[str, frozenset[int]]] = []
    for chord in scale_chords:
        sym = chord.symbol
        rank = _quality_rank.get(chord.quality, 0)
        if seen.get(sym, -1) >= rank:
            continue
        seen[sym] = rank
        c_root, c_ivs = chord_tones_from_symbol(sym)
        c_root_pc = _NOTE_PC.get(c_root, 0)
        pcs = frozenset((c_root_pc + i) % 12 for i in c_ivs)
        candidates = [(s, p) for s, p in candidates if s != sym]
        candidates.append((sym, pcs))

    if not candidates:
        return []

    num_slots = max(1, round(total_beats / beats_per_chord))
    result: list[str] = []

    for slot in range(num_slots):
        slot_start = slot * beats_per_chord
        slot_end = slot_start + beats_per_chord

        # Pitch-class histogram weighted by overlap duration in this slot
        pc_weights: dict[int, float] = {}
        for n in notes:
            n_start = n.get("start_beat", 0)
            n_end = n_start + n.get("duration_beats", 1)
            name = n.get("note_name", n.get("note", ""))
            overlap = min(n_end, slot_end) - max(n_start, slot_start)
            if overlap <= 0:
                continue
            pc = _note_to_pc(name)
            if pc is not None:
                pc_weights[pc] = pc_weights.get(pc, 0.0) + overlap

        if not pc_weights:
            result.append(result[-1] if result else candidates[0][0])
            continue

        total_weight = sum(pc_weights.values())
        best_sym = candidates[0][0]
        best_score = -float("inf")
        prev_sym = result[-1] if result else None

        for sym, pcs in candidates:
            coverage = sum(pc_weights.get(pc, 0.0) for pc in pcs)
            non_chord = total_weight - coverage
            # 0.4 tolerance: up to 40% non-chord weight is acceptable (passing tones)
            score = coverage - 0.4 * non_chord
            # Penalise exact repeat of the previous chord to encourage progression.
            if sym == prev_sym:
                score -= total_weight * 0.25
            if score > best_score:
                best_score = score
                best_sym = sym

        result.append(best_sym)

    return result


def _note_to_pc(note_name: str) -> int | None:
    """'D4' → 2, 'C#3' → 1, 'Bb2' → 10, etc."""
    if not note_name:
        return None
    m = re.match(r"([A-Ga-g][#b]?)", note_name)
    if not m:
        return None
    name = m.group(1)
    return _NOTE_PC.get(name[0].upper() + name[1:])
