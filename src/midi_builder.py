"""
Build a multi-track Type-1 MIDI file from an orchestration dict.

Each instrument part becomes its own MIDI track. Track 0 carries the
tempo map and time signature. General MIDI program numbers are assigned
based on the part name so the file plays correctly in any MIDI player
or DAW without manual patch assignment.
"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.prompts import INSTRUMENT_PROGRAMS

TICKS_PER_BEAT = 480   # standard resolution; fine enough for 16th-note accuracy
QUANTIZE_GRID  = 0.25  # 16th-note grid used for Synfire-friendly export

# Enharmonic spellings the LLM might use, normalised to sharp form
_ENHARMONIC = {
    "Db": "C#", "Eb": "D#", "Fb": "E",  "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B",  "E#": "F", "B#": "C",
}
_PITCH_CLASS = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4,  "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
}


def note_name_to_midi(name: str) -> int:
    """
    Convert a note name string to a MIDI note number.

    Handles: 'D4', 'F#3', 'Bb5', 'Db2', 'E#4', etc.

    Raises:
        ValueError: If the note name cannot be parsed.
    """
    name = name.strip()

    # Split into pitch class and octave
    if len(name) >= 3 and name[1] in "#b":
        pitch_str = name[:2]
        octave_str = name[2:]
    else:
        pitch_str = name[0]
        octave_str = name[1:]

    pitch_str = _ENHARMONIC.get(pitch_str, pitch_str)

    if pitch_str not in _PITCH_CLASS:
        raise ValueError(f"Unrecognised pitch class: '{pitch_str}' in '{name}'")

    try:
        octave = int(octave_str)
    except ValueError:
        raise ValueError(f"Unrecognised octave '{octave_str}' in '{name}'")

    return (octave + 1) * 12 + _PITCH_CLASS[pitch_str]


def build_midi(orchestration: dict, output_path: str,
               time_sig_num: int = 4, time_sig_den: int = 4,
               exclude_parts: set[str] | None = None,
               chord_timeline: list[tuple[float, str]] | None = None,
               key: str | None = None,
               section_map: list[dict] | None = None,
               quantize: bool = False,
               raga=None) -> str:
    """
    Write a multi-track MIDI file from an orchestration dict.

    Args:
        raga:  RagaDefinition (from raga_engine) — when supplied, pitch bend
               gamaka ornaments are inserted on melodic Indian instrument tracks.
               Pass None for Western presets (no pitch bend).
    """
    tempo_bpm        = float(orchestration.get("tempo", 120))
    tempo_us         = mido.bpm2tempo(tempo_bpm)
    parts: dict      = orchestration["parts"]
    skip             = exclude_parts or set()

    # Instruments that carry the melody in Indian presets — get gamaka treatment
    _INDIAN_MELODIC = {"veena", "bansuri", "sitar", "sarangi", "carnatic_violin",
                       "nadaswaram", "harmonium", "strings_melody", "flute_melody"}
    gamaka_map = (raga.gamakas if raga and hasattr(raga, "gamakas") else {})

    mid = MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)

    # --- Track 0: tempo map, key signature, section markers ---
    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(MetaMessage("track_name", name="Master", time=0))
    tempo_track.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))
    tempo_track.append(MetaMessage("time_signature", numerator=time_sig_num, denominator=time_sig_den,
                                   clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    key_sig = _parse_key_sig(key) if key else None
    if key_sig:
        tempo_track.append(MetaMessage("key_signature", key=key_sig, time=0))

    # Section markers sorted by start beat → absolute-tick meta-messages
    if section_map:
        marker_events = sorted(
            (int(round(sec["start"] * TICKS_PER_BEAT)), sec["label"])
            for sec in section_map
        )
        current_tick = 0
        for abs_tick, label in marker_events:
            tempo_track.append(MetaMessage("marker", text=label, time=abs_tick - current_tick))
            current_tick = abs_tick

    tempo_track.append(MetaMessage("end_of_track", time=0))

    # --- Track 1: chord progression (text events) — Synfire reads these directly ---
    if chord_timeline:
        chord_track = MidiTrack()
        mid.tracks.append(chord_track)
        chord_track.append(MetaMessage("track_name", name="Chords", time=0))
        current_tick = 0
        for beat, chord_name in sorted(chord_timeline, key=lambda x: x[0]):
            abs_tick = int(round(beat * TICKS_PER_BEAT))
            chord_track.append(MetaMessage("text", text=chord_name, time=abs_tick - current_tick))
            current_tick = abs_tick
        chord_track.append(MetaMessage("end_of_track", time=0))

    # --- One track per instrument part ---
    channel_idx = 0
    for part_name, note_list in parts.items():
        if part_name in skip:
            continue
        # MIDI has 16 channels; channel 9 is reserved for drums — skip it
        channel = channel_idx if channel_idx < 9 else channel_idx + 1
        if channel > 15:
            break   # hard MIDI limit
        channel_idx += 1

        track = MidiTrack()
        mid.tracks.append(track)

        track.append(MetaMessage("track_name", name=part_name, time=0))

        program = INSTRUMENT_PROGRAMS.get(part_name, 40)   # default: Violin
        track.append(Message("program_change", channel=channel, program=program, time=0))

        notes = _quantize_note_list(note_list) if quantize else note_list

        # Use gamaka pitch bend for Indian melodic instruments when raga is provided
        if gamaka_map and part_name in _INDIAN_MELODIC:
            events = _build_events_with_gamaka(notes, channel, gamaka_map)
        else:
            events = _build_events(notes, channel)

        _append_delta_messages(track, events, channel)

        track.append(MetaMessage("end_of_track", time=0))

    mid.save(output_path)
    return output_path


def build_midi_sections(
    orchestration: dict,
    section_map: list[dict],
    stem_path: str,
    time_sig_num: int = 4,
    time_sig_den: int = 4,
    exclude_parts: set[str] | None = None,
    chord_timeline: list[tuple[float, str]] | None = None,
    key: str | None = None,
    quantize: bool = False,
) -> list[tuple[str, str]]:
    """
    Write one MIDI file per unique section label, sliced from the full orchestration.

    Each file contains only the notes within that section's beat range, re-baselined
    to start at beat 0 so the file can be dragged directly into Logic (or any DAW)
    as a self-contained loopable region.

    Returns a list of (label, path) pairs for the files that were written.
    """
    if not section_map:
        return []

    tempo_bpm = float(orchestration.get("tempo", 120))
    tempo_us  = mido.bpm2tempo(tempo_bpm)
    parts     = orchestration["parts"]
    skip      = exclude_parts or set()

    written: list[tuple[str, str]] = []
    seen_labels: set[str] = set()

    for sec in section_map:
        label = sec["label"]
        if label in seen_labels:
            continue  # write only the first occurrence of each label
        seen_labels.add(label)

        sec_start = sec["start"]
        sec_end   = sec["end"]

        mid = MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)

        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(MetaMessage("track_name", name="Master", time=0))
        tempo_track.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))
        tempo_track.append(MetaMessage("time_signature",
                                       numerator=time_sig_num, denominator=time_sig_den,
                                       clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
        key_sig = _parse_key_sig(key) if key else None
        if key_sig:
            tempo_track.append(MetaMessage("key_signature", key=key_sig, time=0))
        tempo_track.append(MetaMessage("end_of_track", time=0))

        # Chord sub-track: slice chords to this section, re-baseline to beat 0
        if chord_timeline:
            chord_track = MidiTrack()
            mid.tracks.append(chord_track)
            chord_track.append(MetaMessage("track_name", name="Chords", time=0))
            current_tick = 0
            for beat, chord_name in sorted(chord_timeline, key=lambda x: x[0]):
                if sec_start <= beat < sec_end:
                    abs_tick = int(round((beat - sec_start) * TICKS_PER_BEAT))
                    chord_track.append(MetaMessage("text", text=chord_name,
                                                   time=abs_tick - current_tick))
                    current_tick = abs_tick
            chord_track.append(MetaMessage("end_of_track", time=0))

        channel_idx = 0
        for part_name, note_list in parts.items():
            if part_name in skip:
                continue
            channel = channel_idx if channel_idx < 9 else channel_idx + 1
            if channel > 15:
                break
            channel_idx += 1

            # Slice notes to this section, shift start_beat to 0
            sliced = []
            for n in note_list:
                sb = n["start_beat"]
                if sec_start <= sb < sec_end:
                    shifted = dict(n)
                    shifted["start_beat"] = sb - sec_start
                    max_dur = (sec_end - sec_start) - shifted["start_beat"]
                    shifted["duration_beats"] = min(n["duration_beats"], max_dur)
                    sliced.append(shifted)

            if quantize:
                sliced = _quantize_note_list(sliced)

            track = MidiTrack()
            mid.tracks.append(track)
            track.append(MetaMessage("track_name", name=part_name, time=0))
            program = INSTRUMENT_PROGRAMS.get(part_name, 40)
            track.append(Message("program_change", channel=channel, program=program, time=0))
            events = _build_events(sliced, channel)
            _append_delta_messages(track, events, channel)
            track.append(MetaMessage("end_of_track", time=0))

        out_path = f"{stem_path}_{label}.mid"
        mid.save(out_path)
        written.append((label, out_path))

    return written


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_events(note_list: list[dict], channel: int) -> list[tuple]:
    """
    Convert a list of note dicts to (absolute_tick, type, note, velocity) tuples.

    Skips any note that cannot be parsed rather than crashing.
    """
    events = []
    for note in note_list:
        try:
            midi_note = note_name_to_midi(note["note"])
            start_tick = _beats_to_ticks(note["start_beat"])
            end_tick   = _beats_to_ticks(note["start_beat"] + note["duration_beats"])
            velocity   = int(max(1, min(127, note.get("velocity", 80))))

            events.append((start_tick, "note_on",  midi_note, velocity))
            events.append((end_tick,   "note_off", midi_note, 0))
        except (ValueError, KeyError, TypeError):
            continue   # malformed note from LLM — skip silently

    events.sort(key=lambda e: e[0])
    return events


# ---------------------------------------------------------------------------
# Gamaka pitch bend
# ---------------------------------------------------------------------------

# MIDI pitch bend range: ±8192 units = ±2 semitones (standard DAW setting)
# 1 semitone = 4096 units
_SEMITONE_BEND = 4096
_BEND_CENTER   = 8192   # no bend

def _gamaka_bend_events(
    note: dict,
    gamaka_type: str,
    channel: int,
) -> list[tuple]:
    """
    Generate pitch bend events for a single note based on its gamaka type.

    Returns list of (abs_tick, "pitchwheel", channel, pitch) tuples.
    These are interleaved with note_on/note_off events in the track.

    Gamaka types:
      meend     — smooth glide up from 1 semitone below to pitch, over first 30% of note
      andolan   — slow oscillation ±0.5 semitone, 2 cycles over note duration
      kampita   — fast oscillation ±0.25 semitone, 4+ cycles (classical kampita)
      sphurita  — quick touch of note below then immediate return to pitch
      kan       — very brief grace touch from semitone below at note start
    """
    import math

    start_beat = note["start_beat"]
    dur_beats  = note["duration_beats"]
    if dur_beats < 0.1:
        return []

    start_tick = _beats_to_ticks(start_beat)
    end_tick   = _beats_to_ticks(start_beat + dur_beats)
    dur_ticks  = end_tick - start_tick
    events: list[tuple] = []

    def _add(tick: int, bend: int) -> None:
        bend = max(0, min(16383, bend))
        events.append((tick, "pitchwheel", channel, bend - _BEND_CENTER))

    if gamaka_type == "meend":
        # Glide up from -1 semitone to 0 over first 35% of note, then hold
        glide_end = start_tick + int(dur_ticks * 0.35)
        steps = max(8, (glide_end - start_tick) // 30)
        for i in range(steps + 1):
            t    = start_tick + int((glide_end - start_tick) * i / steps)
            bend = _BEND_CENTER + int(-_SEMITONE_BEND + _SEMITONE_BEND * i / steps)
            _add(t, bend)
        # Reset at note end
        _add(end_tick - 1, _BEND_CENTER)

    elif gamaka_type == "andolan":
        # Slow oscillation ±0.5 semitone, ~2 full cycles
        amplitude = _SEMITONE_BEND // 2
        cycles    = 2.0
        steps     = max(16, dur_ticks // 20)
        for i in range(steps + 1):
            t    = start_tick + int(dur_ticks * i / steps)
            phase = 2 * math.pi * cycles * i / steps
            bend = _BEND_CENTER + int(amplitude * math.sin(phase))
            _add(t, bend)
        _add(end_tick - 1, _BEND_CENTER)

    elif gamaka_type == "kampita":
        # Fast tight oscillation ±0.25 semitone, 4+ cycles
        amplitude = _SEMITONE_BEND // 4
        # Start after first 15% (establish the note first)
        osc_start = start_tick + int(dur_ticks * 0.15)
        osc_ticks = end_tick - osc_start
        cycles    = max(4.0, dur_beats * 5)
        steps     = max(24, osc_ticks // 15)
        _add(start_tick, _BEND_CENTER)
        for i in range(steps + 1):
            t     = osc_start + int(osc_ticks * i / steps)
            phase = 2 * math.pi * cycles * i / steps
            bend  = _BEND_CENTER + int(amplitude * math.sin(phase))
            _add(t, bend)
        _add(end_tick - 1, _BEND_CENTER)

    elif gamaka_type == "sphurita":
        # Quick dip to semitone below, back up — all in first 20%
        dip_end = start_tick + int(dur_ticks * 0.20)
        mid_pt  = start_tick + int(dur_ticks * 0.10)
        _add(start_tick, _BEND_CENTER - _SEMITONE_BEND)
        _add(mid_pt,     _BEND_CENTER - _SEMITONE_BEND)
        _add(dip_end,    _BEND_CENTER)
        _add(end_tick - 1, _BEND_CENTER)

    elif gamaka_type == "kan":
        # Grace touch: very brief (first 8%) from semitone below
        grace_end = start_tick + int(dur_ticks * 0.08)
        _add(start_tick, _BEND_CENTER - _SEMITONE_BEND)
        _add(grace_end,  _BEND_CENTER)
        _add(end_tick - 1, _BEND_CENTER)

    return events


def _build_events_with_gamaka(
    note_list: list[dict],
    channel: int,
    gamakas: dict[int, str],
) -> list[tuple]:
    """
    Build note events with interleaved pitch bend events for gamaka ornaments.

    gamakas: pitch_class → gamaka_type mapping from RagaDefinition.
    """
    events = []

    # Reset pitch bend at track start
    events.append((0, "pitchwheel", channel, 0))

    for note in note_list:
        try:
            midi_note  = note_name_to_midi(note["note"])
            start_tick = _beats_to_ticks(note["start_beat"])
            end_tick   = _beats_to_ticks(note["start_beat"] + note["duration_beats"])
            velocity   = int(max(1, min(127, note.get("velocity", 80))))
        except (ValueError, KeyError, TypeError):
            continue

        pc = midi_note % 12
        gamaka_type = gamakas.get(pc)

        if gamaka_type and gamaka_type != "none" and note["duration_beats"] >= 0.25:
            bend_events = _gamaka_bend_events(note, gamaka_type, channel)
            events.extend(bend_events)

        events.append((start_tick, "note_on",  midi_note, velocity))
        events.append((end_tick,   "note_off", midi_note, 0))

    events.sort(key=lambda e: e[0])
    return events


def _append_delta_messages(track: MidiTrack, events: list[tuple], channel: int) -> None:
    """Convert absolute-tick events to delta-time MIDI messages and append to track."""
    current_tick = 0
    for event in events:
        abs_tick = event[0]
        msg_type = event[1]
        delta    = abs_tick - current_tick
        current_tick = abs_tick

        if msg_type == "pitchwheel":
            track.append(Message("pitchwheel", channel=event[2], pitch=event[3], time=delta))
        else:
            note, velocity = event[2], event[3]
            track.append(Message(msg_type, note=note, velocity=velocity,
                                 time=delta, channel=channel))


def _beats_to_ticks(beats: float) -> int:
    return max(0, int(round(beats * TICKS_PER_BEAT)))


def _quantize_note_list(notes: list[dict]) -> list[dict]:
    """Snap note start_beats to the nearest 16th-note grid (QUANTIZE_GRID beats)."""
    result = []
    for n in notes:
        q = dict(n)
        q["start_beat"] = round(round(n["start_beat"] / QUANTIZE_GRID) * QUANTIZE_GRID, 6)
        result.append(q)
    return result


def _parse_key_sig(key_str: str) -> str | None:
    """Convert 'C major' / 'G minor' to a mido key_signature string ('C' / 'Gm')."""
    parts = key_str.strip().lower().split()
    if len(parts) < 2:
        return None
    root = parts[0].capitalize()
    # Capitalise accidental: 'f#' → 'F#', 'bb' → 'Bb'
    if len(root) == 2 and root[1] in "#b":
        root = root[0].upper() + root[1]
    mode = parts[1]
    if mode == "major":
        return root
    if mode == "minor":
        return root + "m"
    return None