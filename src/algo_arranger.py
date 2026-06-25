"""
Algorithmic generation of harmony (chord voicings), bass lines, and Synfire-style
contour/motif melodies.

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


# ---------------------------------------------------------------------------
# Synfire-style melody generators (contour + motif)
# ---------------------------------------------------------------------------

def make_contour_melody(
    chord_timeline: list[tuple[float, str]],
    total_beats: float,
    lo: int,
    hi: int,
    contour: str = "Arch",
    notes_per_beat: float = 2.0,
) -> list[dict]:
    """
    Generate a melody whose pitch trajectory follows a named contour shape.

    Contour shapes (Synfire-inspired: shape defined independently of harmony,
    then fitted to the chord tones at each moment):
      Rise    — pitch climbs steadily from lo to hi over the piece
      Fall    — pitch descends from hi to lo
      Arch    — rises to the midpoint then falls (classic melodic arc)
      Valley  — falls to the midpoint then rises
      Wave    — two full sine-wave cycles (up-down-up-down)
      Step    — alternates between low and high registers every bar

    For each note slot, the target pitch height (0–1) is mapped to the MIDI
    range, then the nearest available chord tone is chosen.  This separates
    *shape* from *harmony* exactly as Synfire does.
    """
    step = 1.0 / max(0.5, notes_per_beat)
    notes: list[dict] = []
    n_steps = max(1, int(total_beats / step))

    for i in range(n_steps):
        t = i * step
        if t >= total_beats:
            break

        # Position in piece as 0→1
        pos = t / total_beats

        # Contour → target height 0→1
        if contour == "Rise":
            height = pos
        elif contour == "Fall":
            height = 1.0 - pos
        elif contour == "Arch":
            height = math.sin(math.pi * pos)
        elif contour == "Valley":
            height = 1.0 - math.sin(math.pi * pos)
        elif contour == "Wave":
            height = 0.5 + 0.5 * math.sin(2 * math.pi * 2 * pos)
        else:  # Step
            height = 0.8 if int(pos * 8) % 2 == 0 else 0.3

        target_midi = lo + int(height * (hi - lo))

        # Get chord tones at this beat
        chord_sym = chord_timeline[0][1]
        for b, sym in chord_timeline:
            if t >= b:
                chord_sym = sym
        try:
            root, intervals = chord_tones_from_symbol(chord_sym)
        except Exception:
            intervals = [0, 4, 7]
            root = "C"
        root_pc = _NOTE_PC.get(root, 0)
        candidates = []
        for octave in range(2, 8):
            for iv in intervals:
                m = (root_pc + iv) % 12 + octave * 12
                if lo <= m <= hi:
                    candidates.append(m)

        if not candidates:
            continue

        # Pick the candidate closest to the target height
        midi = min(candidates, key=lambda m: abs(m - target_midi))

        # Velocity: louder on strong beats, softer off-beats
        is_downbeat = abs((t % 1.0)) < 0.05
        vel = 78 if is_downbeat else 62

        dur = min(step * 0.88, total_beats - t)
        notes.append({
            "note": midi_to_note_name(midi),
            "start_beat": round(t * 4) / 4,
            "duration_beats": dur,
            "velocity": vel,
        })

    return notes


def make_motif_melody(
    chord_timeline: list[tuple[float, str]],
    total_beats: float,
    lo: int,
    hi: int,
    motif_str: str = "C4 E4 G4 E4",
    notes_per_beat: float = 2.0,
) -> list[dict]:
    """
    Stamp a short motif across the full piece, transposing it to fit each chord.

    The motif is parsed as space-separated note names (e.g. "C4 E4 G4 E4").
    Its *intervals* (semitone differences between consecutive notes) are extracted.
    At each chord change, the motif is re-rooted to the nearest chord tone to
    the previous note, preserving the interval shape but fitting the harmony.

    This is the Synfire "figure" concept: a melodic cell that reharmonises
    automatically as the chords change beneath it.
    """
    # Parse motif into MIDI note numbers
    motif_midis: list[int] = []
    for token in motif_str.strip().split():
        try:
            motif_midis.append(note_name_to_midi(token))
        except Exception:
            pass
    if not motif_midis:
        return []

    # Extract relative intervals from the motif
    intervals_semitones = [0] + [motif_midis[i] - motif_midis[i - 1] for i in range(1, len(motif_midis))]
    step = 1.0 / max(0.5, notes_per_beat)
    motif_len = len(motif_midis)

    notes: list[dict] = []
    current_root = (lo + hi) // 2  # starting anchor
    t = 0.0
    motif_idx = 0

    while t < total_beats:
        # Find active chord
        chord_sym = chord_timeline[0][1]
        for b, sym in chord_timeline:
            if t >= b:
                chord_sym = sym

        # On chord boundaries, snap the motif root to nearest chord tone
        is_chord_start = any(abs(t - b) < 0.05 for b, _ in chord_timeline)
        if is_chord_start or t < 0.01:
            try:
                root, chord_ivs = chord_tones_from_symbol(chord_sym)
            except Exception:
                chord_ivs = [0, 4, 7]
                root = "C"
            root_pc = _NOTE_PC.get(root, 0)
            # Find chord tone closest to current_root
            best = current_root
            best_dist = 999
            for octave in range(2, 8):
                for iv in chord_ivs:
                    m = (root_pc + iv) % 12 + octave * 12
                    if lo <= m <= hi and abs(m - current_root) < best_dist:
                        best_dist = abs(m - current_root)
                        best = m
            current_root = best
            motif_idx = 0  # restart motif on chord change

        # Build note from motif interval
        midi = current_root + intervals_semitones[motif_idx % motif_len]
        midi = max(lo, min(hi, midi))

        is_downbeat = abs((t % 1.0)) < 0.05
        vel = 80 if is_downbeat else 64
        dur = min(step * 0.88, total_beats - t)

        notes.append({
            "note": midi_to_note_name(midi),
            "start_beat": round(t * 4) / 4,
            "duration_beats": dur,
            "velocity": vel,
        })

        current_root = midi
        motif_idx += 1
        t += step

    return notes


def vary_orchestration(orchestration: dict, technique: str = "invert") -> dict:
    """
    Return a copy of the orchestration with the melody part transformed.

    Synfire-style section variation: take a generated melody and derive a B-section
    variant from it without touching the harmony/bass.

    Techniques:
      invert     — flip all intervals upside-down (melodic inversion around the
                   midpoint of the melody's range)
      retrograde — reverse the time order of all notes
      augment    — double all note durations, halving note density
      diminish   — halve all note durations, doubling note density
    """
    import copy
    result = copy.deepcopy(orchestration)

    # Find melody parts: anything not bass/harmony
    _MELODY_INST = frozenset({
        "piano_melody", "violin_melody", "flute_melody", "cello_melody",
        "strings_melody", "alto_sax", "tenor_sax", "harmonica",
        "sitar", "nadaswaram", "sarod",
    })
    mel_parts = [p for p in result.get("parts", {}) if p in _MELODY_INST]
    if not mel_parts:
        # Fall back to first non-empty part
        mel_parts = [p for p, ns in result["parts"].items() if ns][:1]

    for part in mel_parts:
        ns = result["parts"][part]
        if not ns:
            continue

        if technique == "invert":
            midis = [note_name_to_midi(n["note"]) for n in ns]
            if not midis:
                continue
            axis = (min(midis) + max(midis)) // 2
            for i, n in enumerate(ns):
                m = note_name_to_midi(n["note"])
                n["note"] = midi_to_note_name(axis * 2 - m)

        elif technique == "retrograde":
            beats = [n["start_beat"] for n in ns]
            durs  = [n["duration_beats"] for n in ns]
            rev_notes = list(reversed([n["note"] for n in ns]))
            rev_vels  = list(reversed([n["velocity"] for n in ns]))
            for i, n in enumerate(ns):
                n["note"]     = rev_notes[i]
                n["velocity"] = rev_vels[i]
                n["duration_beats"] = durs[i]

        elif technique == "augment":
            total = max((n["start_beat"] + n["duration_beats"]) for n in ns)
            for n in ns:
                n["start_beat"]      = n["start_beat"] * 2
                n["duration_beats"]  = n["duration_beats"] * 2
            # Filter notes that now exceed total_beats*2 (caller decides)

        elif technique == "diminish":
            for n in ns:
                n["start_beat"]     = n["start_beat"] / 2
                n["duration_beats"] = max(0.125, n["duration_beats"] / 2)

    return result


# GM channel-9 drum note names (MIDI note → note name mapping)
_DRUM_KICK   = "C2"   # MIDI 36
_DRUM_SNARE  = "D2"   # MIDI 38
_DRUM_HIHAT  = "F#2"  # MIDI 42 closed hi-hat
_DRUM_HIHAT_O= "A#2"  # MIDI 46 open hi-hat
_DRUM_RIDE   = "D#3"  # MIDI 51 ride cymbal
_DRUM_CRASH  = "C#3"  # MIDI 49 crash
_DRUM_HI_TOM = "D3"   # MIDI 50
_DRUM_MID_TOM= "B2"   # MIDI 47
_DRUM_FLOOR  = "A2"   # MIDI 45 floor tom

def _drum_note(note_name: str, beat: float, vel: int, dur: float = 0.2) -> dict:
    return {"note": note_name, "start_beat": round(beat, 4),
            "duration_beats": dur, "velocity": vel}

def make_drum_part(
    total_beats: float,
    beats_per_bar: float = 4.0,
    style: str = "pop",
) -> list[dict]:
    """
    Generate a genre-appropriate drum track on GM channel 9.

    Notes use pitch names that map to GM drum numbers:
      C2=kick(36)  D2=snare(38)  F#2=closed-hat(42)
      A#2=open-hat(46)  D#3=ride(51)  C#3=crash(49)

    Styles: pop, rock, jazz, funk, bossa, hiphop, brush, waltz, latin
    """
    notes: list[dict] = []
    bar = 0.0
    bpb = beats_per_bar

    while bar < total_beats:
        remaining = total_beats - bar

        if style in ("pop", "rock"):
            # Kick: beat 1 & 3, Snare: beat 2 & 4, 8th-note hats
            for b in range(int(bpb)):
                if b in (0, 2):
                    notes.append(_drum_note(_DRUM_KICK,  bar + b, 95))
                if b in (1, 3):
                    notes.append(_drum_note(_DRUM_SNARE, bar + b, 85))
                notes.append(_drum_note(_DRUM_HIHAT, bar + b,       60))
                notes.append(_drum_note(_DRUM_HIHAT, bar + b + 0.5, 45))

        elif style == "funk":
            # 16th-note kick syncopation, snare on 2.5 & 4, tight hats
            kick_offsets = [0, 0.75, 2.25, 2.5, 3.75]
            snare_offsets = [1, 2.5, 3]
            for k in kick_offsets:
                if bar + k < total_beats:
                    notes.append(_drum_note(_DRUM_KICK,  bar + k, 90 if k == 0 else 75))
            for s in snare_offsets:
                if bar + s < total_beats:
                    notes.append(_drum_note(_DRUM_SNARE, bar + s, 80))
            for h in [i * 0.25 for i in range(int(bpb * 4))]:
                if bar + h < total_beats:
                    vel = 55 if h % 0.5 == 0 else 38
                    notes.append(_drum_note(_DRUM_HIHAT, bar + h, vel))

        elif style in ("jazz", "bebop"):
            # Ride cymbal on beat + offbeat (swing), sparse kick & snare
            ride_offsets = [0, 0.67, 1, 1.67, 2, 2.67, 3, 3.67]
            for r in ride_offsets:
                if bar + r < total_beats:
                    notes.append(_drum_note(_DRUM_RIDE, bar + r, 58 if r % 1 == 0 else 44))
            if remaining >= 1:
                notes.append(_drum_note(_DRUM_KICK,  bar, 60))           # light kick on 1
            if remaining >= 3:
                notes.append(_drum_note(_DRUM_SNARE, bar + 2, 50))      # brush-like snare on 3

        elif style == "brush":
            # Very sparse: ride + occasional snare
            for b in range(int(bpb)):
                if bar + b < total_beats:
                    notes.append(_drum_note(_DRUM_RIDE, bar + b, 48))
                    if b % 2 == 1:
                        notes.append(_drum_note(_DRUM_SNARE, bar + b, 38))

        elif style == "bossa":
            # Bossa nova: rim-shot pattern (use snare for rim), light kick
            rim_offsets = [0, 0.75, 1.5, 2.5, 3.25]
            for r in rim_offsets:
                if bar + r < total_beats:
                    notes.append(_drum_note(_DRUM_SNARE, bar + r, 52))
            if remaining >= 0.5:
                notes.append(_drum_note(_DRUM_KICK, bar, 70))
            if remaining >= 2.5:
                notes.append(_drum_note(_DRUM_KICK, bar + 2, 65))
            for h in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]:
                if bar + h < total_beats:
                    notes.append(_drum_note(_DRUM_HIHAT, bar + h, 35))

        elif style == "hiphop":
            # Boom-bap: kick 1 + 2.5, snare on 3, sparse hats
            boom_bap = [(0, _DRUM_KICK, 95), (2, _DRUM_KICK, 80),
                        (2.5, _DRUM_KICK, 70), (1, _DRUM_SNARE, 88), (3, _DRUM_SNARE, 85)]
            for off, note, vel in boom_bap:
                if bar + off < total_beats:
                    notes.append(_drum_note(note, bar + off, vel))
            for h in [0, 1, 2, 3]:
                if bar + h < total_beats:
                    notes.append(_drum_note(_DRUM_HIHAT, bar + h, 42))

        elif style == "waltz":
            # 3/4 or 6/8: kick on 1, hi-hat on 2 & 3
            notes.append(_drum_note(_DRUM_KICK,  bar, 88))
            for b in range(1, int(bpb)):
                if bar + b < total_beats:
                    notes.append(_drum_note(_DRUM_HIHAT, bar + b, 50))

        elif style == "latin":
            # Clave-like: hi-hat 8ths, kick + snare syncopated
            clave = [0, 0.75, 1.5, 2.5, 3.25]
            for c in clave:
                if bar + c < total_beats:
                    notes.append(_drum_note(_DRUM_SNARE, bar + c, 60))
            for h in [i * 0.5 for i in range(int(bpb * 2))]:
                if bar + h < total_beats:
                    notes.append(_drum_note(_DRUM_HIHAT, bar + h, 42))
            if remaining >= 0.5:
                notes.append(_drum_note(_DRUM_KICK, bar, 80))

        else:  # default: basic 4/4
            notes.append(_drum_note(_DRUM_KICK,  bar,     88))
            notes.append(_drum_note(_DRUM_SNARE, bar + 2, 80))
            for h in range(int(bpb)):
                if bar + h < total_beats:
                    notes.append(_drum_note(_DRUM_HIHAT, bar + h, 50))

        bar += bpb

    return notes


# Drum style per preset name
_DRUM_STYLE_MAP: dict[str, str] = {
    "Jazz Quartet":     "jazz",
    "Jazz Big Band":    "jazz",
    "Jazz Trio (Brush)":"brush",
    "Bebop Quintet":    "bebop",
    "Blues Band":       "rock",
    "Blues Trio":       "rock",
    "Funk Band":        "funk",
    "Pop Band":         "pop",
    "Retro Pop (80s)":  "pop",
    "City Pop":         "funk",
    "Hip-Hop Beat":     "hiphop",
    "Rock Band":        "rock",
    "Bossa Nova":       "bossa",
    "Bollywood Modern": "latin",
    "Dholak Party":     "latin",
    "Koothu / Folk":    "latin",
}


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

    # Generate drum part if present in orchestration
    preset_name = orchestration.get("preset", "")
    drum_style = _DRUM_STYLE_MAP.get(preset_name, "pop")
    for inst in list(orchestration.get("parts", {}).keys()):
        if inst == "drums":
            orchestration["parts"][inst] = make_drum_part(total_beats, beats_per_bar, drum_style)
            continue
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
            # Drums: generated once for the full piece, handled after loop
            if inst == "drums":
                continue
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

    # Generate drums for full piece (not per-section) and write back
    preset_name = orchestration.get("preset", "")
    drum_style = _DRUM_STYLE_MAP.get(preset_name, "pop")
    for inst in part_notes:
        if inst == "drums":
            orchestration["parts"][inst] = make_drum_part(total_beats, beats_per_bar, drum_style)
        elif inst in chordal_instruments or inst in bass_instruments or fill_melody_parts:
            orchestration["parts"][inst] = part_notes[inst]

    return orchestration