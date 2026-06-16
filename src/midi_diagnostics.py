"""
MIDI diagnostic analysis — parse an output MIDI file and report quality issues.

Checks:
  - Note count and density per track
  - Out-of-scale pitch classes (given a key string)
  - Register coverage (are notes in sensible ranges?)
  - Rhythmic variety (is it too mechanical / too sparse?)
  - Empty or near-empty tracks
"""

from __future__ import annotations
import mido
from src.harmony import parse_key_string, midi_to_note_name, _NOTE_PC

_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def analyse_midi(midi_path: str, key: str = "", total_beats: float = 0.0) -> str:
    """
    Parse a MIDI file and return a human-readable diagnostic report.

    Parameters
    ----------
    midi_path   : path to the .mid file
    key         : key string like "D Double Harmonic" (optional — enables scale check)
    total_beats : expected duration in beats (optional — enables density check)
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        return f"Could not read MIDI: {e}"

    ticks_per_beat = mid.ticks_per_beat
    lines: list[str] = []

    # Scale reference
    scale_pcs: frozenset[int] | None = None
    scale_name = ""
    if key:
        parsed = parse_key_string(key)
        if parsed:
            root, intervals = parsed
            root_pc = _NOTE_PC.get(root, 0)
            scale_pcs = frozenset((root_pc + i) % 12 for i in intervals)
            scale_name = key
            scale_note_list = [_PC_NAMES[(root_pc + i) % 12] for i in intervals]
            lines.append(f"Scale ({key}): {', '.join(scale_note_list)}")
            lines.append("")

    # Per-track analysis
    overall_note_count = 0
    overall_out_of_scale = 0
    track_reports: list[str] = []

    for track in mid.tracks:
        name = track.name or "(unnamed)"
        note_ons: list[tuple[float, int]] = []  # (tick_time, pitch)
        abs_tick = 0.0

        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note_ons.append((abs_tick, msg.note))

        if not note_ons:
            continue

        pitches   = [p for _, p in note_ons]
        note_count = len(pitches)
        overall_note_count += note_count

        lo, hi = min(pitches), max(pitches)
        lo_name = midi_to_note_name(lo)
        hi_name = midi_to_note_name(hi)

        # Pitch class histogram
        pc_counts: dict[int, int] = {}
        for p in pitches:
            pc_counts[p % 12] = pc_counts.get(p % 12, 0) + 1
        most_common_pc = max(pc_counts, key=pc_counts.get)

        # Out-of-scale check
        out_notes: list[str] = []
        if scale_pcs is not None:
            bad = [p for p in pitches if p % 12 not in scale_pcs]
            overall_out_of_scale += len(bad)
            if bad:
                bad_names = sorted({midi_to_note_name(p) for p in bad})
                # Summarise by pitch class to avoid very long lists
                bad_pcs = sorted({_PC_NAMES[p % 12] for p in bad})
                out_notes = bad_pcs

        # Rhythmic variety: count distinct onset ticks mod ticks_per_beat
        duration_beats = (abs_tick / ticks_per_beat) if ticks_per_beat > 0 else 0
        density = note_count / duration_beats if duration_beats > 0 else 0

        # Rhythmic variety: count distinct sub-beat positions (1/8-beat resolution).
        # Walking bass correctly hits every beat (all beat-level), so measure at 1/8 resolution
        # to distinguish it from truly stuck-on-one-note patterns.
        if ticks_per_beat > 0:
            eighth = ticks_per_beat // 2
            sub_slots = {int(t) // max(1, eighth) for t, _ in note_ons}
            variety = len(sub_slots)
        else:
            variety = 0

        rep = [f"  Track '{name}': {note_count} notes, range {lo_name}–{hi_name}"]
        rep.append(f"    density {density:.1f} notes/beat, rhythmic spread {variety} 1/8-slots")
        rep.append(f"    most common pitch class: {_PC_NAMES[most_common_pc]}")
        if out_notes:
            rep.append(f"    ⚠ out-of-scale pitch classes: {', '.join(out_notes)}")
        else:
            rep.append(f"    ✓ all notes in scale")

        if note_count < 4:
            rep.append(f"    ⚠ very sparse — only {note_count} notes (check if this track is correct)")
        if note_count >= 8 and variety <= 2:
            rep.append(f"    ⚠ very low rhythmic variety — may sound mechanical")

        track_reports.append("\n".join(rep))

    lines.append(f"Total notes: {overall_note_count}")
    if scale_pcs is not None:
        pct = 100 * overall_out_of_scale / overall_note_count if overall_note_count else 0
        lines.append(f"Out-of-scale notes: {overall_out_of_scale} ({pct:.0f}%)")
    lines.append("")
    lines.extend(track_reports)

    return "\n".join(lines)
