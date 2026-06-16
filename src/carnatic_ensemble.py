"""
Carnatic ensemble logic — raga-aware harmony only.

All layers must use notes from the raga scale. Embellishments and violin
harmony are scheduled only in gaps between veena phrases so nothing clashes.
"""

from __future__ import annotations

import librosa
import numpy as np

from src.raga_engine import RagaDefinition, raga_scale_pcs, snap_note_to_raga

_NOTE_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def split_phrases(notes: list[dict], gap_beats: float = 0.75) -> list[list[dict]]:
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: n["start_beat"])
    phrases: list[list[dict]] = [[sorted_notes[0]]]
    for note in sorted_notes[1:]:
        prev = phrases[-1][-1]
        gap = note["start_beat"] - (prev["start_beat"] + prev["duration_beats"])
        if gap >= gap_beats:
            phrases.append([note])
        else:
            phrases[-1].append(note)
    return phrases


def find_gap_windows(notes: list[dict], min_gap: float = 1.0) -> list[tuple[float, float]]:
    """Silent gaps between phrases — safe windows for harmony/embellishment."""
    main = sorted(
        [n for n in notes if n.get("role") in (None, "main", "melody")],
        key=lambda n: n["start_beat"],
    )
    if len(main) < 2:
        return []

    windows: list[tuple[float, float]] = []
    for i in range(len(main) - 1):
        gap_start = main[i]["start_beat"] + main[i]["duration_beats"]
        gap_end = main[i + 1]["start_beat"]
        if gap_end - gap_start >= min_gap:
            windows.append((gap_start + 0.15, gap_end - 0.1))
    return windows


def tag_phrase_boundaries(notes: list[dict], gap_beats: float = 0.75) -> list[dict]:
    if not notes:
        return notes
    sorted_notes = sorted(notes, key=lambda n: n["start_beat"])
    phrases = split_phrases(sorted_notes, gap_beats)
    phrase_end_keys = {(p[-1]["start_beat"], p[-1].get("note")) for p in phrases if p}

    out: list[dict] = []
    for note in sorted_notes:
        n = dict(note)
        dur = n.get("duration_beats", 0.5)
        if dur <= 0.2 and n.get("velocity", 80) < 62:
            n["role"] = "grace"
        elif n.get("role") not in ("grace", "harmony", "embellishment"):
            n.setdefault("role", "main")
        if (n["start_beat"], n.get("note")) in phrase_end_keys:
            n["phrase_end"] = True
        out.append(n)
    return out


def _midi_to_note_safe(midi: int) -> str:
    return librosa.midi_to_note(int(midi), unicode=False)


def _nearest_midi_in_scale(midi: int, scale: frozenset[int], lo: int = 48, hi: int = 84) -> int:
    if midi % 12 in scale:
        return midi
    candidates = [m for m in range(max(lo, midi - 6), min(hi + 1, midi + 7)) if m % 12 in scale]
    if not candidates:
        candidates = [m for m in range(lo, hi + 1) if m % 12 in scale]
    return min(candidates, key=lambda m: abs(m - midi))


def make_violin_harmony_pops(
    notes: list[dict],
    key: str,
    raga: RagaDefinition,
) -> list[dict]:
    """
    Raga-safe harmony: Sa and Pa only, placed in silent gaps between phrases.
    """
    if not notes:
        return []

    scale = raga_scale_pcs(key, raga)
    root_pc = _NOTE_TO_PC.get(key.strip().split()[0], 0)
    pa_pc = (root_pc + 7) % 12
    gaps = find_gap_windows(notes, min_gap=1.2)
    if not gaps:
        return []

    harmony: list[dict] = []
    for i, (gap_start, gap_end) in enumerate(gaps):
        if i % 2 != 0:
            continue
        dur = min(1.5, gap_end - gap_start - 0.1)
        if dur < 0.4:
            continue

        # Sa (tonic) in a comfortable octave
        sa_midi = _nearest_midi_in_scale(60 + root_pc, scale)
        if pa_pc in scale:
            pa_midi = _nearest_midi_in_scale(sa_midi + 7, scale)
            while pa_midi % 12 != pa_pc:
                pa_midi = _nearest_midi_in_scale(pa_midi + 1, scale)
        else:
            # Fall back to perfect fourth (Ma) or just sa_midi if Pa doesn't exist in scale (like Marwa)
            ma_pc = (root_pc + 5) % 12
            if ma_pc in scale:
                pa_midi = _nearest_midi_in_scale(sa_midi + 5, scale)
                while pa_midi % 12 != ma_pc:
                    pa_midi = _nearest_midi_in_scale(pa_midi + 1, scale)
            else:
                pa_midi = sa_midi

        harmony.append({
            "note": snap_note_to_raga(_midi_to_note_safe(sa_midi), key, raga),
            "start_beat": round(gap_start, 4),
            "duration_beats": round(dur * 0.55, 4),
            "velocity": 48,
            "role": "harmony",
        })
        if dur > 0.8:
            harmony.append({
                "note": snap_note_to_raga(_midi_to_note_safe(pa_midi), key, raga),
                "start_beat": round(gap_start + dur * 0.45, 4),
                "duration_beats": round(dur * 0.45, 4),
                "velocity": 42,
                "role": "harmony",
            })

    return harmony


def _arohana_run_from(midi: int, key: str, raga: RagaDefinition, steps: int = 4) -> list[int]:
    """Next notes along the raga arohana from a starting pitch."""
    root_pc = _NOTE_TO_PC.get(key.strip().split()[0], 0)
    scale = raga_scale_pcs(key, raga)
    aro_pcs = [(root_pc + pc) % 12 for pc in raga.arohana]
    if not aro_pcs:
        return [midi]

    start_pc = midi % 12
    try:
        idx = aro_pcs.index(start_pc)
    except ValueError:
        idx = 0
        midi = _nearest_midi_in_scale(midi, scale)

    run = [midi]
    cur_midi = midi
    for _ in range(steps - 1):
        idx = (idx + 1) % len(aro_pcs)
        next_pc = aro_pcs[idx]
        cur_midi += 1
        while cur_midi % 12 != next_pc:
            cur_midi += 1
        if cur_midi > 84:
            break
        run.append(cur_midi)
    return run


def make_embellishments(
    notes: list[dict],
    instrument: str,
    key: str,
    raga: RagaDefinition,
    max_flourishes: int = 3,
) -> list[dict]:
    """
  Bansuri/sitar — short raga arohana runs only in phrase gaps, never over the veena.
    """
    gaps = find_gap_windows(notes, min_gap=1.0)
    if not gaps:
        return []

    flourishes: list[dict] = []
    step = max(1, len(gaps) // max_flourishes)

    for gi in range(0, len(gaps), step):
        gap_start, gap_end = gaps[gi]
        window = gap_end - gap_start
        if window < 0.6:
            continue

        # Anchor on Sa or phrase-ending note pitch class
        anchor_midi = 60 + _NOTE_TO_PC.get(key.strip().split()[0], 0)
        run = _arohana_run_from(anchor_midi, key, raga, steps=min(4, int(window / 0.15)))
        t = gap_start
        for j, midi in enumerate(run):
            if t + 0.12 > gap_end:
                break
            flourishes.append({
                "note": snap_note_to_raga(_midi_to_note_safe(midi), key, raga),
                "start_beat": round(t, 4),
                "duration_beats": 0.14,
                "velocity": 62 + j * 2,
                "role": "embellishment",
                "instrument": instrument,
            })
            t += 0.14

    return flourishes


# Continuous adi tala with fills
ADI_TALA_FULL: dict[int, list[tuple[str, float, float]]] = {
    0: [("thom", 0.00, 0.88), ("ta", 0.50, 0.62)],
    1: [("ta",   0.00, 0.55)],
    2: [("ta",   0.00, 0.52), ("din", 0.75, 0.45)],
    3: [("ta",   0.00, 0.50)],
    4: [("thom", 0.00, 0.82), ("ta", 0.50, 0.58)],
    5: [("din",  0.00, 0.48), ("ta", 0.50, 0.45)],
    6: [("ta",   0.00, 0.52), ("din", 0.50, 0.45)],
    7: [("ta",   0.00, 0.48), ("din", 0.75, 0.42)],
}


def mridangam_pattern_for_beat(_section: str = "full") -> dict[int, list[tuple[str, float, float]]]:
    return ADI_TALA_FULL


def melody_activity_envelope(
    notes: list[dict],
    total_samples: int,
    tempo_bpm: float,
    sr: int,
) -> np.ndarray:
    beat_dur = 60.0 / tempo_bpm
    env = np.zeros(total_samples, dtype=np.float32)
    for note in notes:
        if note.get("role") in ("grace", "harmony", "embellishment"):
            continue
        start = int(note["start_beat"] * beat_dur * sr)
        end = int((note["start_beat"] + note["duration_beats"]) * beat_dur * sr)
        end = min(end, total_samples)
        if start < total_samples and end > start:
            env[start:end] = np.maximum(env[start:end], note.get("velocity", 75) / 90.0)

    kernel = int(sr * 0.08)
    if kernel > 1:
        kernel_w = np.ones(kernel, dtype=np.float32) / kernel
        env = np.convolve(env, kernel_w, mode="same")
    return np.clip(env, 0, 1)
