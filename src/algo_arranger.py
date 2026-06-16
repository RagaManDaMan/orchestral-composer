"""
Algorithmic generation of harmony (chord voicings) and bass lines.

These functions bypass the LLM entirely. Given a chord timeline they produce
musically correct, rhythmically coherent parts every time — no model needed.

Used as the default for chordal and bass instruments when a chord chart is
provided. The LLM is still used for melodic voices where free creativity is
actually wanted.
"""

from __future__ import annotations
import math
from src.harmony import (
    chord_tones_from_symbol, _NOTE_PC, note_name_to_midi, midi_to_note_name,
    parse_key_string, snap_all_parts_to_scale,
)
from src.voice_leading import voice_lead, analyze_progression, tension_velocity, shape_velocities


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _voice_chord(
    symbol: str,
    center_midi: int,
    lo: int,
    hi: int,
) -> list[int]:
    """Return MIDI note numbers for a close-position chord voicing near center_midi."""
    root, intervals = chord_tones_from_symbol(symbol)
    root_pc = _NOTE_PC.get(root, 0)
    root_midi = root_pc + 12 * round((center_midi - root_pc) / 12)
    if root_midi < lo:
        root_midi += 12
    if root_midi > hi:
        root_midi -= 12
    midis = []
    for iv in intervals:
        midi = root_midi + iv
        if midi > hi:
            midi -= 12
        if midi < lo:
            midi += 12
        if lo <= midi <= hi:
            midis.append(midi)
    return sorted(set(midis))


def _chord_beat_end(chord_timeline: list[tuple[float, str]], idx: int, total_beats: float) -> float:
    return chord_timeline[idx + 1][0] if idx + 1 < len(chord_timeline) else total_beats


# ---------------------------------------------------------------------------
# Voicing styles
# ---------------------------------------------------------------------------

_BLOCK_REARTICULATE_BEATS = 4.0  # re-attack sustained chords every N beats


def _block_chords(chord_timeline, total_beats, lo, hi, center, key="C major") -> list[dict]:
    notes = []
    analysis = analyze_progression(chord_timeline, key)
    n_chords = max(1, len(analysis))
    prev_midis: list[int] = []
    inversion_offset = 0   # tracks how many times we've rotated the voicing up

    for i, (beat_start, symbol, fn) in enumerate(analysis):
        beat_end = _chord_beat_end(chord_timeline, i, total_beats)
        dur = beat_end - beat_start
        is_final_chord = (i == len(analysis) - 1)
        midis = voice_lead(symbol, prev_midis, lo, hi, center)

        # Inversion cycling: every 4 chords rotate the lowest voice up an octave.
        # This adds register variety without disrupting smooth voice leading.
        if i > 0 and i % 4 == 0 and len(midis) >= 2:
            lowest = midis[0]
            if lowest + 12 <= hi:
                midis = sorted(midis[1:] + [lowest + 12])
                inversion_offset += 1
            elif inversion_offset > 0:
                # Come back down when we've gone too high
                highest = midis[-1]
                if highest - 12 >= lo:
                    midis = sorted([highest - 12] + midis[:-1])
                    inversion_offset -= 1

        prev_midis = midis

        # Long-form dynamic arc: velocity rises toward 2/3 of the piece then tapers.
        # Works even when harmonic function is unknown (e.g. raga / modal scales).
        arc_pos = i / n_chords                                  # 0→1 over whole piece
        arc_scale = 0.75 + 0.25 * math.sin(math.pi * min(arc_pos * 1.5, 1.0))
        chord_vel, _ = tension_velocity(fn, symbol=symbol)
        chord_vel = max(40, min(110, int(chord_vel * arc_scale)))

        # Re-attack long chords at regular intervals so the harmony stays audible
        # and rhythmically engaged rather than one indefinitely-held block.
        # Use 90% duration so note-off lands slightly before the next note-on,
        # preventing the synthesizer click caused by same-tick note-off/note-on collisions.
        if dur > _BLOCK_REARTICULATE_BEATS:
            t = beat_start
            attack_idx = 0
            while t < beat_end - 0.25:
                remaining = beat_end - t
                is_last = remaining <= _BLOCK_REARTICULATE_BEATS
                # Final chord of the piece holds to full duration; everything else breathes (5% gap).
                if is_last and is_final_chord:
                    seg_dur = remaining
                elif is_last:
                    seg_dur = remaining * 0.95
                else:
                    seg_dur = _BLOCK_REARTICULATE_BEATS * 0.9
                vel_scale = 1.0 if attack_idx == 0 else 0.75
                for j, midi in enumerate(midis):
                    vel = max(30, min(120, int((chord_vel - j * 3) * vel_scale)))
                    notes.append({
                        "note": midi_to_note_name(midi),
                        "start_beat": round(t * 4) / 4,
                        "duration_beats": seg_dur,
                        "velocity": vel,
                    })
                t += _BLOCK_REARTICULATE_BEATS
                attack_idx += 1
        else:
            # Short chord: hold for full duration; add breathing gap unless it's the final chord.
            note_dur = dur if is_final_chord else dur * 0.95
            for j, midi in enumerate(midis):
                vel = max(40, min(120, chord_vel - j * 3))
                notes.append({
                    "note": midi_to_note_name(midi),
                    "start_beat": beat_start,
                    "duration_beats": note_dur,
                    "velocity": vel,
                })
    return notes


def _arpeggio(chord_timeline, total_beats, lo, hi, center, direction="up", step=0.5, key="C major") -> list[dict]:
    notes = []
    analysis  = analyze_progression(chord_timeline, key)
    prev_midis: list[int] = []
    for i, (beat_start, symbol, fn) in enumerate(analysis):
        beat_end = _chord_beat_end(chord_timeline, i, total_beats)
        midis = voice_lead(symbol, prev_midis, lo, hi, center)
        prev_midis = midis[:]
        chord_vel, _ = tension_velocity(fn)
        seq = list(midis)
        if direction == "down":
            seq = list(reversed(seq))
        elif direction == "alt" and len(seq) >= 3:
            seq = [seq[0], seq[-1], seq[len(seq) // 2], seq[-1]]
        t, idx = beat_start, 0
        while t < beat_end - 0.001:
            midi = seq[idx % len(seq)]
            dur  = min(step, beat_end - t)
            vel  = max(40, min(120, chord_vel if idx % len(seq) == 0 else chord_vel - 6))
            notes.append({
                "note": midi_to_note_name(midi),
                "start_beat": round(t * 4) / 4,
                "duration_beats": dur,
                "velocity": vel,
            })
            t   += step
            idx += 1
    return notes


def _alberti(chord_timeline, total_beats, lo, hi, center, key="C major") -> list[dict]:
    """Alberti bass pattern: low–high–mid–high repeating at 0.5-beat steps."""
    notes    = []
    analysis = analyze_progression(chord_timeline, key)
    prev_midis: list[int] = []
    for i, (beat_start, symbol, fn) in enumerate(analysis):
        beat_end = _chord_beat_end(chord_timeline, i, total_beats)
        midis = voice_lead(symbol, prev_midis, lo, hi, center)
        prev_midis = midis[:]
        chord_vel, _ = tension_velocity(fn)
        if len(midis) < 3:
            midis = (midis * 2)[:3]
        pattern = [midis[0], midis[-1], midis[len(midis) // 2], midis[-1]]
        t, idx = beat_start, 0
        while t < beat_end - 0.001:
            dur = min(0.5, beat_end - t)
            notes.append({
                "note": midi_to_note_name(pattern[idx % len(pattern)]),
                "start_beat": round(t * 4) / 4,
                "duration_beats": dur,
                "velocity": max(40, chord_vel if idx % 4 == 0 else chord_vel - 10),
            })
            t   += 0.5
            idx += 1
    return notes


def _broken_chords(chord_timeline, total_beats, lo, hi, center, key="C major") -> list[dict]:
    """Pairs of notes (root+5th, 3rd+7th) alternating every beat."""
    notes    = []
    analysis = analyze_progression(chord_timeline, key)
    prev_midis: list[int] = []
    for i, (beat_start, symbol, fn) in enumerate(analysis):
        beat_end = _chord_beat_end(chord_timeline, i, total_beats)
        midis = voice_lead(symbol, prev_midis, lo, hi, center)
        prev_midis = midis[:]
        chord_vel, _ = tension_velocity(fn)
        if len(midis) < 2:
            midis = (midis * 2)
        lower = midis[:max(1, len(midis) // 2)]
        upper = midis[max(1, len(midis) // 2):]
        t, idx = beat_start, 0
        while t < beat_end - 0.001:
            pair = lower if idx % 2 == 0 else upper
            dur  = min(0.5, beat_end - t)
            for midi in pair:
                notes.append({
                    "note": midi_to_note_name(midi),
                    "start_beat": round(t * 4) / 4,
                    "duration_beats": dur,
                    "velocity": max(40, chord_vel - 4),
                })
            t   += 1.0
            idx += 1
    return notes


def _jazz_comping(chord_timeline, total_beats, lo, hi, center, beats_per_bar=4.0, key="C major") -> list[dict]:
    """Jazz comping: voice-led shell-voicing stabs on syncopated off-beats."""
    notes    = []
    analysis = analyze_progression(chord_timeline, key)
    prev_midis: list[int] = []

    for i, (beat_start, symbol, fn) in enumerate(analysis):
        beat_end  = _chord_beat_end(chord_timeline, i, total_beats)
        dur       = beat_end - beat_start
        midis     = voice_lead(symbol, prev_midis, lo, hi, center)
        prev_midis = midis[:]
        chord_vel, _ = tension_velocity(fn)
        shell = [midis[0], midis[1], midis[-1]] if len(midis) >= 3 else midis
        stab_dur = min(0.5, dur * 0.25)

        # Offsets as fractions of chord duration — gives syncopated feel regardless of chord length.
        # For 4-beat chords: 0.5*4=2.0, 0.875*4=3.5 (classic jazz comping positions).
        # For 2-beat chords: 0.5*2=1.0, 0.875*2=1.75 (same relative feel).
        offsets = [dur * 0.5, dur * 0.875]
        for off in offsets:
            t = beat_start + off
            if t + stab_dur <= beat_end:
                for j, midi in enumerate(shell):
                    notes.append({
                        "note": midi_to_note_name(midi),
                        "start_beat": round(t * 4) / 4,
                        "duration_beats": stab_dur,
                        "velocity": max(40, chord_vel if j == 0 else chord_vel - 8),
                    })
    return notes


# ---------------------------------------------------------------------------
# Bass line generation
# ---------------------------------------------------------------------------

def _nearest_pc(current: int, pc: int, lo: int, hi: int) -> int:
    """Find the MIDI note with pitch class pc that is closest to current, within [lo, hi]."""
    for delta in range(0, 13):
        for d in ([delta, -delta] if delta > 0 else [0]):
            candidate = current + d
            if lo <= candidate <= hi and candidate % 12 == pc:
                return candidate
    # Fallback: any in-range instance
    for m in range(lo, hi + 1):
        if m % 12 == pc:
            return m
    return current


def _walking_bass(chord_timeline, total_beats, lo, hi, beats_per_bar=4.0, step=2.0) -> list[dict]:
    """
    Stride bass: root on beat 1, 5th on beat 3 (step=2.0 default), notes held for step*0.85.
    Set step=1.0 for a fully walking bass (one note per beat).
    """
    notes   = []
    current = None

    for i, (beat_start, symbol) in enumerate(chord_timeline):
        beat_end  = _chord_beat_end(chord_timeline, i, total_beats)
        root, intervals = chord_tones_from_symbol(symbol)
        root_pc   = _NOTE_PC.get(root, 0)
        chord_pcs = [(root_pc + iv) % 12 for iv in intervals]

        root_target = (lo + hi) // 2
        root_midi   = _nearest_pc(root_target, root_pc, lo, hi)
        current     = root_midi

        t        = beat_start
        tone_idx = 0
        while t < beat_end - 0.001:
            if t == beat_start:
                midi = root_midi
                vel  = 82
            else:
                tone_idx += 1
                pc   = chord_pcs[tone_idx % len(chord_pcs)]
                midi = _nearest_pc(current, pc, lo, hi)
                current = midi
                vel  = 68
            # Hold note for most of the step, leaving a rest gap before the next attack
            note_dur = min(step * 0.85, beat_end - t)
            notes.append({
                "note": midi_to_note_name(midi),
                "start_beat": t,
                "duration_beats": note_dur,
                "velocity": vel,
            })
            t += step
    return notes


def _pedal_bass(chord_timeline, total_beats, lo, hi) -> list[dict]:
    """Pedal bass: root only on beat 1 of each chord change, held for full duration."""
    notes = []
    for i, (beat_start, symbol) in enumerate(chord_timeline):
        beat_end  = _chord_beat_end(chord_timeline, i, total_beats)
        root, _   = chord_tones_from_symbol(symbol)
        root_pc   = _NOTE_PC.get(root, 0)
        root_midi = root_pc + 12 * 3
        while root_midi < lo: root_midi += 12
        while root_midi > hi: root_midi -= 12
        notes.append({
            "note": midi_to_note_name(root_midi),
            "start_beat": beat_start,
            "duration_beats": beat_end - beat_start,
            "velocity": 80,
        })
    return notes


# ---------------------------------------------------------------------------
# Per-instrument textural roles
# ---------------------------------------------------------------------------

# Every chordal instrument has a ROLE that determines its voicing density and
# rhythmic placement, independent of the user-selected harmony style.
# This is the primary mechanism that makes multi-instrument arrangements sound
# distinct rather than a cluster of identical chord stabs.
#
#   comp  — full harmony, rhythmically active (piano is the canonical comp voice)
#   shell — 3-note shell voicings, short staccato stabs every beat (Freddie Green guitar)
#   pad   — sustained: one attack per chord change, held for full duration (strings, organ)
#   fill  — sparse ornament: top 1-2 notes at the start of each chord (harmonica, winds)
#
_PART_ROLE: dict[str, str] = {
    "piano_harmony":   "comp",
    "piano_melody":    "comp",
    "electric_guitar": "shell",
    "vibraphone":      "shell",
    "strings_harmony": "pad",
    "woodwind_harmony":"fill",
    "harmonica":       "fill",
    "alto_sax":        "fill",
    "tenor_sax":       "fill",
    "sitar":           "fill",
}

def _part_role(instrument: str) -> str:
    return _PART_ROLE.get(instrument, "comp")


def _guitar_shell_comp(
    chord_timeline, total_beats, lo, hi, center, beats_per_bar=4.0, key="C major"
) -> list[dict]:
    """
    Freddie Green style: one short stab per beat, 3-note shell voicing.

    The guitar's role in a multi-part arrangement is rhythmic clarity and
    harmonic colour — not to double the piano's full voicings. Three notes
    (bottom, middle, top) played staccato every beat sit inside the texture
    without overwhelming it.
    """
    notes: list[dict] = []
    analysis = analyze_progression(chord_timeline, key)
    prev_midis: list[int] = []
    for i, (beat_start, symbol, fn) in enumerate(analysis):
        beat_end = _chord_beat_end(chord_timeline, i, total_beats)
        midis = voice_lead(symbol, prev_midis, lo, hi, center)
        prev_midis = midis[:]
        # Shell: bottom + middle + top (≤ 3 notes)
        if len(midis) >= 3:
            shell = [midis[0], midis[len(midis) // 2], midis[-1]]
        else:
            shell = midis
        chord_vel, _ = tension_velocity(fn)
        chord_vel = max(36, int(chord_vel * 0.76))
        t = beat_start
        while t < beat_end - 0.001:
            for midi in shell:
                notes.append({
                    "note":           midi_to_note_name(midi),
                    "start_beat":     round(t * 4) / 4,
                    "duration_beats": 0.2,   # staccato
                    "velocity":       chord_vel,
                })
            t += 1.0   # one stab per beat
    return notes


def _sustained_pad(
    chord_timeline, total_beats, lo, hi, center, key="C major"
) -> list[dict]:
    """
    Sustained harmonic wash: one attack per chord change, held for the full duration.

    The strings / organ pad role — smooth, quiet, continuous support beneath
    more rhythmically active instruments. Minimal movement between chords.
    """
    notes: list[dict] = []
    analysis = analyze_progression(chord_timeline, key)
    prev_midis: list[int] = []
    for i, (beat_start, symbol, fn) in enumerate(analysis):
        beat_end = _chord_beat_end(chord_timeline, i, total_beats)
        dur = beat_end - beat_start
        midis = voice_lead(symbol, prev_midis, lo, hi, center)
        prev_midis = midis[:]
        chord_vel, _ = tension_velocity(fn)
        base_vel = max(28, int(chord_vel * 0.68))
        is_final = (i == len(analysis) - 1)
        note_dur = dur if is_final else dur * 0.97   # near-legato, barely breathes
        for j, midi in enumerate(midis):
            notes.append({
                "note":           midi_to_note_name(midi),
                "start_beat":     beat_start,
                "duration_beats": note_dur,
                "velocity":       max(24, base_vel - j * 4),
            })
    return notes


def _sparse_fill(
    chord_timeline, total_beats, lo, hi, center, beats_per_bar=4.0, key="C major"
) -> list[dict]:
    """
    Sparse fills: top 1-2 notes of the chord, short duration, once per chord change.

    The harmonica / woodwind fill role — colour and air in the arrangement,
    not continuous comping. Sounds like a player responding to the harmony
    rather than driving it.
    """
    notes: list[dict] = []
    analysis = analyze_progression(chord_timeline, key)
    prev_midis: list[int] = []
    for i, (beat_start, symbol, fn) in enumerate(analysis):
        beat_end = _chord_beat_end(chord_timeline, i, total_beats)
        dur = beat_end - beat_start
        midis = voice_lead(symbol, prev_midis, lo, hi, center)
        prev_midis = midis[:]
        if not midis:
            continue
        chord_vel, _ = tension_velocity(fn)
        # Top 1 note for thin chords, top 2 for rich voicings
        fill_notes = midis[-2:] if len(midis) >= 4 else [midis[-1]]
        fill_dur = min(beats_per_bar * 0.5, dur * 0.45, 2.0)
        fill_vel = max(32, int(chord_vel * 0.60))
        for midi in fill_notes:
            notes.append({
                "note":           midi_to_note_name(midi),
                "start_beat":     beat_start,
                "duration_beats": fill_dur,
                "velocity":       fill_vel,
            })
    return notes


# ---------------------------------------------------------------------------
# Per-instrument dynamic shaping
# ---------------------------------------------------------------------------

# Velocity scaling applied on top of role-based generation.
_INSTRUMENT_VELOCITY_SCALE: dict[str, float] = {
    "strings_harmony":   0.90,   # pad is already quiet; keep it audible
    "woodwind_harmony":  0.85,
    "piano_harmony":     0.90,
    "electric_guitar":   0.82,
    "harmonica":         0.80,
}


def _scale_velocities(notes: list[dict], instrument: str) -> list[dict]:
    scale = _INSTRUMENT_VELOCITY_SCALE.get(instrument, 1.0)
    if scale == 1.0:
        return notes
    for n in notes:
        n["velocity"] = max(25, min(110, int(n["velocity"] * scale)))
    return notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _freely(
    chord_timeline, total_beats, lo, hi, center,
    beats_per_bar=4.0, key="C major",
) -> list[dict]:
    """
    Per-chord pattern mixing: each chord gets its own style chosen by its
    position and duration, mimicking an improvising accompanist.

    Rotation order (deterministic by chord index so the same session sounds
    consistent): block → jazz comping → arpeggio up → broken → arpeggio alt
    → block → …  Short chords (< 2 beats) always get a block chord because
    there is not enough room for an arpeggio or comping stab.
    """
    _PATTERNS = ["block", "jazz", "arp_up", "broken", "arp_alt"]
    notes: list[dict] = []

    analysis = analyze_progression(chord_timeline, key)
    prev_midis: list[int] = []
    n_chords = max(1, len(analysis))

    for i, (beat_start, symbol, fn) in enumerate(analysis):
        beat_end = _chord_beat_end(chord_timeline, i, total_beats)
        dur = beat_end - beat_start

        pattern = _PATTERNS[i % len(_PATTERNS)] if dur >= 2.0 else "block"

        # Build a single-chord timeline for delegation to helpers
        single = [(beat_start, symbol)]
        if i + 1 < len(analysis):
            single_end = beat_end
        else:
            single_end = total_beats

        midis = voice_lead(symbol, prev_midis, lo, hi, center)
        prev_midis = midis

        chord_vel, _ = tension_velocity(fn, symbol=symbol)
        arc_pos = i / n_chords
        arc_scale = 0.75 + 0.25 * math.sin(math.pi * min(arc_pos * 1.5, 1.0))
        chord_vel = max(40, min(110, int(chord_vel * arc_scale)))

        is_final = (i == len(analysis) - 1)

        if pattern == "block":
            note_dur = dur if is_final else dur * 0.95
            for j, midi in enumerate(midis):
                notes.append({
                    "note": midi_to_note_name(midi),
                    "start_beat": beat_start,
                    "duration_beats": note_dur,
                    "velocity": max(35, chord_vel - j * 3),
                })

        elif pattern == "jazz":
            shell = [midis[0], midis[1], midis[-1]] if len(midis) >= 3 else midis
            stab_dur = min(0.5, dur * 0.25)
            offsets = [dur * 0.5, dur * 0.875]
            for off in offsets:
                t = beat_start + off
                if t + stab_dur <= beat_end:
                    for j, midi in enumerate(shell):
                        notes.append({
                            "note": midi_to_note_name(midi),
                            "start_beat": round(t * 4) / 4,
                            "duration_beats": stab_dur,
                            "velocity": max(35, chord_vel if j == 0 else chord_vel - 8),
                        })

        elif pattern in ("arp_up", "arp_alt"):
            seq = list(midis)
            if pattern == "arp_alt" and len(seq) >= 3:
                seq = [seq[0], seq[-1], seq[len(seq) // 2], seq[-1]]
            t, idx = beat_start, 0
            while t < beat_end - 0.001:
                m = seq[idx % len(seq)]
                d = min(0.5, beat_end - t)
                notes.append({
                    "note": midi_to_note_name(m),
                    "start_beat": round(t * 4) / 4,
                    "duration_beats": d,
                    "velocity": max(35, chord_vel if idx % len(seq) == 0 else chord_vel - 6),
                })
                t += 0.5
                idx += 1

        elif pattern == "broken":
            if len(midis) < 2:
                midis = midis * 2
            lower = midis[:max(1, len(midis) // 2)]
            upper = midis[max(1, len(midis) // 2):]
            t, idx = beat_start, 0
            while t < beat_end - 0.001:
                pair = lower if idx % 2 == 0 else upper
                d = min(0.5, beat_end - t)
                for midi in pair:
                    notes.append({
                        "note": midi_to_note_name(midi),
                        "start_beat": round(t * 4) / 4,
                        "duration_beats": d,
                        "velocity": max(35, chord_vel - 4),
                    })
                t += 1.0
                idx += 1

    return notes


def make_chord_part(
    chord_timeline: list[tuple[float, str]],
    total_beats: float,
    instrument: str,
    instrument_ranges: dict[str, tuple[int, int]],
    harmony_style: str = "Block chords",
    beats_per_bar: float = 4.0,
    key: str = "C major",
    korvai_params: dict | None = None,
) -> list[dict]:
    """
    Return an algorithmically generated chord part.

    Dispatch order:
      1. Korvai pattern (overrides everything)
      2. Instrument role (shell / pad / fill) — ensures textural variety in
         multi-instrument arrangements regardless of global harmony_style
      3. User-selected harmony_style for the comp (piano/keyboard) role
    """
    lo, hi = instrument_ranges.get(instrument, (36, 96))
    center  = (lo + hi) // 2
    role    = _part_role(instrument)

    if harmony_style == "Korvai pattern" and korvai_params:
        from src.korvai_engine import make_korvai_chord_part
        notes = make_korvai_chord_part(
            chord_timeline, total_beats, lo, hi, key,
            phrase_syl=korvai_params.get("phrase_syl", "ta ka dhi mi ta ka dhi mi"),
            connector_syl=korvai_params.get("connector_syl", "ta ka"),
            gati_ratio=korvai_params.get("gati_ratio", 1.0),
            beats_per_matra=korvai_params.get("beats_per_matra", 1.0),
        )
        return notes   # korvai stabs are the primary event — skip velocity scaling

    # Role dispatch: shell / pad / fill instruments ignore harmony_style and use
    # their characteristic texture instead. Only the comp role (piano, keyboards)
    # honours the user's global style selector.
    if role == "shell":
        notes = _guitar_shell_comp(chord_timeline, total_beats, lo, hi, center, beats_per_bar, key)
    elif role == "pad":
        notes = _sustained_pad(chord_timeline, total_beats, lo, hi, center, key)
    elif role == "fill":
        notes = _sparse_fill(chord_timeline, total_beats, lo, hi, center, beats_per_bar, key)
    elif harmony_style == "Free":
        notes = _freely(chord_timeline, total_beats, lo, hi, center, beats_per_bar, key)
    elif harmony_style in ("Block chords", "Pad (sustained)"):
        notes = _block_chords(chord_timeline, total_beats, lo, hi, center, key)
    elif harmony_style == "Arpeggio (up)":
        notes = _arpeggio(chord_timeline, total_beats, lo, hi, center, "up", 0.5, key)
    elif harmony_style == "Arpeggio (down)":
        notes = _arpeggio(chord_timeline, total_beats, lo, hi, center, "down", 0.5, key)
    elif harmony_style == "Arpeggio (alt)":
        notes = _arpeggio(chord_timeline, total_beats, lo, hi, center, "alt", 0.5, key)
    elif harmony_style == "Alberti bass":
        notes = _alberti(chord_timeline, total_beats, lo, hi, center, key)
    elif harmony_style == "Broken chords":
        notes = _broken_chords(chord_timeline, total_beats, lo, hi, center, key)
    elif harmony_style == "Jazz comping":
        notes = _jazz_comping(chord_timeline, total_beats, lo, hi, center, beats_per_bar, key)
    else:
        notes = _block_chords(chord_timeline, total_beats, lo, hi, center, key)

    return _scale_velocities(notes, instrument)


def make_bass_part(
    chord_timeline: list[tuple[float, str]],
    total_beats: float,
    instrument: str,
    instrument_ranges: dict[str, tuple[int, int]],
    beats_per_bar: float = 4.0,
    style: str = "walking",
) -> list[dict]:
    """Return an algorithmically generated bass line with smooth voice leading."""
    lo, hi = instrument_ranges.get(instrument, (28, 55))
    if style == "pedal":
        return _pedal_bass(chord_timeline, total_beats, lo, hi)
    return _walking_bass(chord_timeline, total_beats, lo, hi, beats_per_bar)


def _active_chord_at(chord_timeline: list[tuple[float, str]], beat: float) -> str:
    """Return the chord symbol active at the given beat."""
    result = chord_timeline[0][1] if chord_timeline else "Cmaj7"
    for b, c in chord_timeline:
        if beat >= b:
            result = c
        else:
            break
    return result


def _slice_chord_timeline(
    chord_timeline: list[tuple[float, str]],
    start: float,
    end: float,
) -> list[tuple[float, str]]:
    """
    Return chord_timeline entries in [start, end), rebased so start → 0.0.
    Prepends the active chord at `start` if no entry begins exactly there.
    """
    sliced = [(b - start, c) for b, c in chord_timeline if start <= b < end]
    if not sliced or sliced[0][0] > 1e-9:
        sliced.insert(0, (0.0, _active_chord_at(chord_timeline, start)))
    return sliced


def inject_algo_parts(
    orchestration: dict,
    chord_timeline: list[tuple[float, str]],
    total_beats: float,
    chordal_instruments: frozenset[str],
    bass_instruments: frozenset[str],
    instrument_ranges: dict[str, tuple[int, int]],
    harmony_style: str = "Block chords",
    beats_per_bar: float = 4.0,
    key: str = "C major",
    fill_melody_parts: bool = False,
    korvai_params: dict | None = None,   # legacy: single-korvai flat mode
    section_map: list[dict] | None = None,  # preferred: section-aware mode
) -> dict:
    """
    Replace harmony/bass parts with algorithmically generated versions.

    When section_map is supplied, each section is processed independently:
      - Regular sections (A/B/C): use harmony_style for chord voicing, walking bass.
      - Korvai sections (K1/K2/…): use korvai engine for both harmony and bass.

    Without section_map (legacy mode): applies harmony_style across the whole
    piece. korvai_params enables the old flat-korvai path for backward compat.

    Melody parts are kept untouched unless fill_melody_parts=True.
    No scale snapping — chord voicings preserve intentional chromatic tones.
    """
    if section_map:
        return _inject_sectioned(
            orchestration, chord_timeline, total_beats,
            chordal_instruments, bass_instruments, instrument_ranges,
            harmony_style, beats_per_bar, key, fill_melody_parts, section_map,
        )

    # ── Legacy flat mode ──────────────────────────────────────────────────────
    korvai_active = korvai_params is not None

    for inst in list(orchestration.get("parts", {}).keys()):
        if inst in chordal_instruments:
            orchestration["parts"][inst] = make_chord_part(
                chord_timeline, total_beats, inst, instrument_ranges,
                harmony_style, beats_per_bar, key, korvai_params,
            )
        elif inst in bass_instruments:
            if korvai_active:
                from src.korvai_engine import make_korvai_bass_part
                lo, hi = instrument_ranges.get(inst, (28, 55))
                orchestration["parts"][inst] = make_korvai_bass_part(
                    chord_timeline, total_beats, lo, hi,
                    phrase_syl=korvai_params.get("phrase_syl", "ta ka dhi mi ta ka dhi mi"),
                    connector_syl=korvai_params.get("connector_syl", "ta ka"),
                    gati_ratio=korvai_params.get("gati_ratio", 1.0),
                    beats_per_matra=korvai_params.get("beats_per_matra", 0.25),
                )
            else:
                orchestration["parts"][inst] = make_bass_part(
                    chord_timeline, total_beats, inst, instrument_ranges,
                    beats_per_bar,
                )
        elif fill_melody_parts and not orchestration["parts"].get(inst):
            orchestration["parts"][inst] = make_chord_part(
                chord_timeline, total_beats, inst, instrument_ranges,
                harmony_style, beats_per_bar, key, korvai_params,
            )

    return orchestration


def _inject_sectioned(
    orchestration: dict,
    chord_timeline: list[tuple[float, str]],
    total_beats: float,
    chordal_instruments: frozenset[str],
    bass_instruments: frozenset[str],
    instrument_ranges: dict[str, tuple[int, int]],
    harmony_style: str,
    beats_per_bar: float,
    key: str,
    fill_melody_parts: bool,
    section_map: list[dict],
) -> dict:
    """Process each section independently, routing K sections to korvai engine."""
    from src.korvai_engine import make_korvai_chord_part, make_korvai_bass_part

    # Accumulate notes per instrument
    part_notes: dict[str, list[dict]] = {
        inst: [] for inst in orchestration.get("parts", {})
    }

    for sec in section_map:
        start    = sec["start"]
        end      = sec["end"]
        sec_dur  = end - start
        kparams  = sec.get("korvai_params")

        for inst in part_notes:
            # Melody parts are not touched (unless fill mode)
            if inst not in chordal_instruments and inst not in bass_instruments:
                if fill_melody_parts and not orchestration["parts"].get(inst):
                    # Fill empty melody part with chord content for this section
                    stl = _slice_chord_timeline(chord_timeline, start, end)
                    notes = make_chord_part(
                        stl, sec_dur, inst, instrument_ranges,
                        harmony_style, beats_per_bar, key,
                    )
                    for n in notes:
                        n["start_beat"] += start
                    part_notes[inst].extend(notes)
                continue

            if kparams:
                # ── Korvai section ────────────────────────────────────────
                lo, hi = instrument_ranges.get(inst, (36, 96))
                if inst in chordal_instruments:
                    notes = make_korvai_chord_part(
                        chord_timeline, sec_dur, lo, hi, key,
                        phrase_syl=kparams["phrase_syl"],
                        connector_syl=kparams["connector_syl"],
                        gati_ratio=kparams["gati_ratio"],
                        beats_per_matra=kparams["beats_per_matra"],
                        start_beat=start,
                    )
                else:  # bass
                    lo_b, hi_b = instrument_ranges.get(inst, (28, 55))
                    notes = make_korvai_bass_part(
                        chord_timeline, sec_dur, lo_b, hi_b,
                        phrase_syl=kparams["phrase_syl"],
                        connector_syl=kparams["connector_syl"],
                        gati_ratio=kparams["gati_ratio"],
                        beats_per_matra=kparams["beats_per_matra"],
                        start_beat=start,
                    )
            else:
                # ── Regular section ───────────────────────────────────────
                stl = _slice_chord_timeline(chord_timeline, start, end)
                if inst in chordal_instruments:
                    notes = make_chord_part(
                        stl, sec_dur, inst, instrument_ranges,
                        harmony_style, beats_per_bar, key,
                    )
                    for n in notes:
                        n["start_beat"] += start
                else:  # bass
                    notes = make_bass_part(
                        stl, sec_dur, inst, instrument_ranges,
                        beats_per_bar,
                    )
                    for n in notes:
                        n["start_beat"] += start

            part_notes[inst].extend(notes)

    # Write accumulated notes back; preserve existing melody content
    for inst in part_notes:
        if inst in chordal_instruments or inst in bass_instruments or fill_melody_parts:
            orchestration["parts"][inst] = part_notes[inst]

    return orchestration