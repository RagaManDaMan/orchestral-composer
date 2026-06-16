"""
src/drum_engine.py
==================
Genre-specific drum pattern generator.

All patterns output MIDI note events on channel 9 (0-indexed), which is
the standard GM percussion channel (channel 10 in 1-indexed notation).

GM Drum Map (key note numbers used here):
  35 = Acoustic Bass Drum    36 = Bass Drum 1
  38 = Acoustic Snare        40 = Electric Snare
  42 = Closed Hi-Hat         44 = Pedal Hi-Hat
  46 = Open Hi-Hat           49 = Crash Cymbal 1
  51 = Ride Cymbal 1         53 = Ride Bell
  56 = Cowbell               57 = Crash Cymbal 2
  60 = Hi Bongo              61 = Low Bongo
  62 = Mute Hi Conga         63 = Open Hi Conga   64 = Low Conga
  66 = Low Timbale           67 = High Agogo       68 = Low Agogo
  69 = Cabasa                70 = Maracas
  73 = Short Guiro           74 = Long Guiro       75 = Claves
  76 = Hi Wood Block         77 = Low Wood Block

Indian percussion (mapped to closest GM equivalents — use SGM soundfont for authentic sounds):
  41 = Low Floor Tom   → Dholak bass stroke
  43 = High Floor Tom  → Tabla bayan (bass)
  45 = Low Tom         → Tabla dayan (treble)
  47 = Low-Mid Tom     → Mridangam left
  48 = Hi-Mid Tom      → Mridangam right
  50 = High Tom        → Kanjira / frame drum
  39 = Hand Clap       → Qawwali hand clap
  54 = Tambourine      → Kanjira rattle
"""

from __future__ import annotations
import math

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
NoteEvent = dict  # {"note": int, "start_beat": float, "duration_beats": float, "velocity": int, "channel": int}

DRUM_CHANNEL = 9  # 0-indexed GM percussion channel

# ---------------------------------------------------------------------------
# GM drum note constants
# ---------------------------------------------------------------------------
KICK         = 36
KICK2        = 35
SNARE        = 38
SNARE_ELEC   = 40
HIHAT_CLOSED = 42
HIHAT_PEDAL  = 44
HIHAT_OPEN   = 46
RIDE         = 51
RIDE_BELL    = 53
CRASH        = 49
CRASH2       = 57
COWBELL      = 56

# Bongos / congas
HI_BONGO     = 60
LO_BONGO     = 61
CONGA_MUTE   = 62
CONGA_HI     = 63
CONGA_LO     = 64

# Indian GM approximations
TABLA_BAYAN  = 43   # Low floor tom → tabla bass
TABLA_DAYAN  = 45   # Low tom → tabla treble
MRID_LEFT    = 47   # Low-mid tom → mridangam left
MRID_RIGHT   = 48   # Hi-mid tom → mridangam right
DHOLAK_BASS  = 41   # Low floor tom → dholak bass
DHOLAK_TREBLE= 50   # High tom → dholak treble
KANJIRA      = 54   # Tambourine → kanjira
HANDCLAP     = 39   # Hand clap → qawwali clap
FRAME_DRUM   = 50   # High tom → parai/frame drum
THAVIL_HI    = 48
THAVIL_LO    = 47


# ---------------------------------------------------------------------------
# Helper: build note event
# ---------------------------------------------------------------------------

def _hit(note: int, beat: float, dur: float = 0.1, vel: int = 80) -> NoteEvent:
    return {
        "note":           note,
        "start_beat":     round(beat * 4) / 4,
        "duration_beats": dur,
        "velocity":       max(20, min(127, vel)),
        "channel":        DRUM_CHANNEL,
    }


def _tile(events: list[NoteEvent], bar_beats: float, total_beats: float) -> list[NoteEvent]:
    """Tile a single-bar pattern across total_beats."""
    result = []
    bar = 0.0
    while bar < total_beats:
        for e in events:
            new_beat = e["start_beat"] + bar
            if new_beat < total_beats:
                result.append({**e, "start_beat": round(new_beat * 4) / 4})
        bar += bar_beats
    return result


# ---------------------------------------------------------------------------
# Genre patterns — each returns a list of NoteEvents for ONE bar
# Call _tile() to fill total_beats
# ---------------------------------------------------------------------------

def _jazz_brush(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Jazz brush pattern — ride cymbal on every beat, snare brushes on 2 and 4,
    kick on 1, subtle hi-hat pedal on 2 and 4. Swing feel implied.
    """
    events = []
    # Ride on every beat + upbeats (swing 8ths at X.0 and X.67)
    for beat in range(int(bpb)):
        events.append(_hit(RIDE, beat + 0.0, 0.1, 65))
        events.append(_hit(RIDE, beat + 0.67, 0.1, 48))
    # Kick on beat 1
    events.append(_hit(KICK, 0.0, 0.2, 75))
    # Snare brush (low velocity) on 2 and 4
    events.append(_hit(SNARE, 1.0, 0.15, 52))
    events.append(_hit(SNARE, 3.0, 0.15, 55))
    # Hi-hat pedal on 2 and 4
    events.append(_hit(HIHAT_PEDAL, 1.0, 0.1, 42))
    events.append(_hit(HIHAT_PEDAL, 3.0, 0.1, 42))
    return events


def _jazz_ride(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Jazz ride pattern — classic bebop comping feel.
    Ride: beat 1, beat 2.67 (and of 2), beat 3, beat 4.67
    Kick: feathered on 1 and 3 (very soft)
    Snare: cross-stick on 2 and 4
    """
    events = []
    ride_hits = [0.0, 1.0, 1.67, 2.0, 3.0, 3.67]
    for b in ride_hits:
        if b < bpb:
            vel = 72 if b in (0.0, 2.0) else 58
            events.append(_hit(RIDE, b, 0.1, vel))
    # Ride bell on 1
    events.append(_hit(RIDE_BELL, 0.0, 0.1, 55))
    # Feathered kick
    events.append(_hit(KICK, 0.0, 0.15, 45))
    events.append(_hit(KICK, 2.0, 0.15, 38))
    # Cross-stick snare on 2 and 4
    events.append(_hit(SNARE, 1.0, 0.1, 60))
    events.append(_hit(SNARE, 3.0, 0.1, 63))
    # Hi-hat pedal on 2 and 4
    events.append(_hit(HIHAT_PEDAL, 1.0, 0.1, 45))
    events.append(_hit(HIHAT_PEDAL, 3.0, 0.1, 45))
    return events


def _blues_shuffle(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Blues shuffle — triplet feel hi-hat, kick on 1 and 3, snare on 2 and 4.
    Hi-hat: X.0 and X.67 per beat (2:1 triplet).
    """
    events = []
    for beat in range(int(bpb)):
        events.append(_hit(HIHAT_CLOSED, beat + 0.0,  0.1, 72))
        events.append(_hit(HIHAT_CLOSED, beat + 0.67, 0.1, 55))
    events.append(_hit(KICK,  0.0, 0.2, 88))
    events.append(_hit(KICK,  2.0, 0.2, 80))
    events.append(_hit(SNARE, 1.0, 0.15, 82))
    events.append(_hit(SNARE, 3.0, 0.15, 85))
    # Open hi-hat on beat 4-and for energy
    events.append(_hit(HIHAT_OPEN, 3.67, 0.2, 65))
    return events


def _rock_straight(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Straight rock — 8th note hi-hats, kick on 1 and 3, snare on 2 and 4.
    """
    events = []
    for i in range(int(bpb * 2)):
        beat = i * 0.5
        vel  = 75 if beat == int(beat) else 62
        events.append(_hit(HIHAT_CLOSED, beat, 0.1, vel))
    events.append(_hit(KICK,  0.0, 0.2, 95))
    events.append(_hit(KICK,  2.0, 0.2, 90))
    events.append(_hit(SNARE, 1.0, 0.15, 90))
    events.append(_hit(SNARE, 3.0, 0.15, 92))
    events.append(_hit(CRASH, 0.0, 0.3, 70))
    return events


def _pop_straight(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Pop/Max Martin style — four-on-the-floor kick, closed hi-hats on 8ths,
    open hi-hat on beat 4-and, snare on 2 and 4.
    """
    events = []
    # Closed hi-hat on every 8th
    for i in range(int(bpb * 2)):
        beat = i * 0.5
        vel  = 78 if beat == int(beat) else 60
        events.append(_hit(HIHAT_CLOSED, beat, 0.1, vel))
    # Open hi-hat on 4-and
    events.append(_hit(HIHAT_OPEN, 3.5, 0.2, 68))
    # Four-on-the-floor kick
    for b in range(int(bpb)):
        events.append(_hit(KICK, float(b), 0.2, 92))
    # Snare on 2 and 4
    events.append(_hit(SNARE, 1.0, 0.15, 88))
    events.append(_hit(SNARE, 3.0, 0.15, 90))
    return events


def _hiphop_808(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Hip-hop / trap feel — 808 kick pattern, snare on 2 and 4,
    hi-hat 16th rolls with velocity variation.
    """
    events = []
    # 16th note hi-hats with velocity humanization
    vels = [80, 45, 60, 45, 75, 45, 55, 45, 78, 45, 58, 45, 72, 45, 52, 45]
    for i in range(int(bpb * 4)):
        beat = i * 0.25
        if beat < bpb:
            events.append(_hit(HIHAT_CLOSED, beat, 0.08, vels[i % len(vels)]))
    # 808 kick — syncopated
    for b in [0.0, 0.75, 2.0, 2.5, 3.75]:
        if b < bpb:
            events.append(_hit(KICK, b, 0.3, 95))
    # Snare on 2 and 4
    events.append(_hit(SNARE, 1.0, 0.15, 85))
    events.append(_hit(SNARE, 3.0, 0.15, 88))
    # Open hi-hat on offbeats
    for b in [0.5, 2.5]:
        events.append(_hit(HIHAT_OPEN, b, 0.15, 60))
    return events


def _bossa_nova(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Bossa nova — rimshot on 2 and 4, kick on 1 and 3,
    hi-hat on dotted rhythm, maracas feel.
    """
    events = []
    # Bossa clave pattern on hi-hat: 1, 1.5, 2.5, 3, 4.5 (mod bpb)
    clave = [0.0, 0.5, 1.5, 2.0, 3.5]
    for b in clave:
        if b < bpb:
            events.append(_hit(HIHAT_CLOSED, b, 0.1, 62))
    # Kick: bossa pattern
    for b in [0.0, 1.5, 3.0]:
        if b < bpb:
            events.append(_hit(KICK, b, 0.2, 75))
    # Rim shot / cross-stick on 2 and 4
    events.append(_hit(SNARE, 1.0, 0.1, 65))
    events.append(_hit(SNARE, 3.0, 0.1, 65))
    # Cabasa feel (maracas)
    for i in range(int(bpb * 2)):
        events.append(_hit(70, i * 0.5, 0.08, 42))  # 70 = maracas
    return events


def _funk_16th(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Funk 16th — tight 16th grid, syncopated kick, snare on 2 and 4,
    ghost notes on snare, open hi-hat on offbeats.
    """
    events = []
    # 16th note hi-hats
    for i in range(int(bpb * 4)):
        beat = i * 0.25
        is_downbeat = (i % 4 == 0)
        is_upbeat   = (i % 2 == 1)
        vel = 82 if is_downbeat else (38 if is_upbeat else 55)
        events.append(_hit(HIHAT_CLOSED, beat, 0.08, vel))
    # Open hi-hat on the "and" of 2 and 4
    events.append(_hit(HIHAT_OPEN, 1.5, 0.15, 65))
    events.append(_hit(HIHAT_OPEN, 3.5, 0.15, 65))
    # Syncopated kick
    for b in [0.0, 0.75, 2.0, 2.75]:
        if b < bpb:
            events.append(_hit(KICK, b, 0.2, 92))
    # Snare on 2 and 4
    events.append(_hit(SNARE, 1.0, 0.15, 90))
    events.append(_hit(SNARE, 3.0, 0.15, 92))
    # Ghost notes (very soft snare)
    for b in [0.5, 1.5, 2.5, 3.5]:
        events.append(_hit(SNARE, b, 0.08, 28))
    return events


def _waltz(bpb: float = 3.0) -> list[NoteEvent]:
    """3/4 waltz — kick on 1, hi-hat on 2 and 3."""
    events = []
    events.append(_hit(KICK,         0.0, 0.2,  82))
    events.append(_hit(HIHAT_CLOSED, 1.0, 0.1,  65))
    events.append(_hit(HIHAT_CLOSED, 2.0, 0.1,  60))
    events.append(_hit(SNARE,        1.0, 0.12, 58))
    events.append(_hit(SNARE,        2.0, 0.12, 55))
    return events


# ---------------------------------------------------------------------------
# Indian percussion patterns
# ---------------------------------------------------------------------------

def _tabla_teentaal(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Tabla teentaal approximation — 16 matras in 4 beats.
    Simplified pattern: na tin na dha | dha dhi na na | na tin na dha | dha dhi na na
    Bayan (bass) on dha strokes, dayan (treble) on na/tin/dhi.
    """
    events = []
    # One cycle = 4 beats = 16 16th notes
    # Pattern syllables mapped to positions:
    # na=dayan, tin=dayan hard, dha=both, dhi=both, rest=silence
    pattern = [
        # beat, bayan_vel, dayan_vel
        (0.00,  0,  78),   # na
        (0.25,  0,  62),   # —
        (0.50,  0,  72),   # tin
        (0.75,  0,  58),   # —
        (1.00, 85,  78),   # dha (both)
        (1.25,  0,  62),   # —
        (1.50, 75,  70),   # dhi (both)
        (1.75,  0,  0),    # rest
        (2.00,  0,  65),   # na
        (2.25,  0,  0),    # rest
        (2.50,  0,  72),   # tin
        (2.75,  0,  58),   # —
        (3.00, 88,  80),   # dha (both) — sam approach
        (3.25,  0,  65),   # —
        (3.50, 78,  72),   # dhi
        (3.75,  0,  0),    # rest
    ]
    for beat, bayan_vel, dayan_vel in pattern:
        if beat < bpb:
            if bayan_vel > 0:
                events.append(_hit(TABLA_BAYAN, beat, 0.12, bayan_vel))
            if dayan_vel > 0:
                events.append(_hit(TABLA_DAYAN, beat, 0.10, dayan_vel))
    return events


def _mridangam_adi(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Mridangam adi talam (8-beat cycle, shown here as 4 beats = half cycle).
    Basic thom-ta pattern.
    """
    events = []
    # thom = left (MRID_LEFT), ta/ki/ta = right (MRID_RIGHT)
    pattern = [
        (0.00, 'L', 85),   # thom
        (0.25, 'R', 65),   # ta
        (0.50, 'R', 58),   # ki
        (0.75, 'R', 62),   # ta
        (1.00, 'L', 78),   # thom
        (1.25, 'R', 60),   # —
        (1.50, 'R', 68),   # ta
        (1.75, 'R', 55),   # —
        (2.00, 'L', 88),   # thom (strong)
        (2.25, 'R', 70),   # ta
        (2.50, 'R', 62),   # ki
        (2.75, 'R', 65),   # ta
        (3.00, 'L', 80),   # thom
        (3.25, 'R', 58),   # —
        (3.50, 'R', 72),   # dhi
        (3.75, 'R', 60),   # —
    ]
    for beat, side, vel in pattern:
        if beat < bpb:
            note = MRID_LEFT if side == 'L' else MRID_RIGHT
            events.append(_hit(note, beat, 0.10, vel))
    return events


def _dholak_bhangra(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Dholak bhangra pattern — energetic, driving.
    Bass (dhaa) on 1 and 3, treble (na) fills in between.
    """
    events = []
    pattern = [
        (0.00, DHOLAK_BASS,   92),   # dhaa
        (0.25, DHOLAK_TREBLE, 65),
        (0.50, DHOLAK_TREBLE, 72),
        (0.75, DHOLAK_BASS,   78),   # dhaa
        (1.00, DHOLAK_TREBLE, 68),
        (1.25, DHOLAK_TREBLE, 60),
        (1.50, DHOLAK_BASS,   88),   # dhaa
        (1.75, DHOLAK_TREBLE, 65),
        (2.00, DHOLAK_BASS,   95),   # dhaa (strong)
        (2.25, DHOLAK_TREBLE, 70),
        (2.50, DHOLAK_TREBLE, 75),
        (2.75, DHOLAK_BASS,   80),
        (3.00, DHOLAK_TREBLE, 65),
        (3.25, DHOLAK_TREBLE, 62),
        (3.50, DHOLAK_BASS,   85),
        (3.75, DHOLAK_TREBLE, 68),
    ]
    for beat, note, vel in pattern:
        if beat < bpb:
            events.append(_hit(note, beat, 0.12, vel))
    return events


def _qawwali_clap(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Qawwali hand clap pattern + harmonium drone hits.
    Heavy cupped claps on 2 and 4, lighter fills on offbeats.
    """
    events = []
    # Heavy claps on 2 and 4
    events.append(_hit(HANDCLAP, 1.0, 0.15, 90))
    events.append(_hit(HANDCLAP, 3.0, 0.15, 92))
    # Lighter claps on offbeats
    for b in [0.5, 1.5, 2.5, 3.5]:
        events.append(_hit(HANDCLAP, b, 0.10, 55))
    # Tabla support
    events.append(_hit(TABLA_BAYAN, 0.0, 0.12, 80))
    events.append(_hit(TABLA_DAYAN, 0.5, 0.10, 65))
    events.append(_hit(TABLA_BAYAN, 2.0, 0.12, 78))
    events.append(_hit(TABLA_DAYAN, 2.5, 0.10, 62))
    return events


def _koothu_parai(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Dappan Koothu / Parai pattern — aggressive, high-energy folk percussion.
    Frame drum (parai) + thavil combination.
    """
    events = []
    # Parai (frame drum) — aggressive hits
    parai_pattern = [0.0, 0.25, 0.75, 1.0, 1.5, 2.0, 2.25, 2.75, 3.0, 3.5]
    for b in parai_pattern:
        if b < bpb:
            vel = 95 if b in (0.0, 1.0, 2.0, 3.0) else 72
            events.append(_hit(FRAME_DRUM, b, 0.10, vel))
    # Thavil hi hits
    thavil_hi = [0.5, 1.25, 2.5, 3.25]
    for b in thavil_hi:
        if b < bpb:
            events.append(_hit(THAVIL_HI, b, 0.10, 80))
    # Thavil low
    thavil_lo = [0.0, 2.0]
    for b in thavil_lo:
        events.append(_hit(THAVIL_LO, b, 0.15, 88))
    return events


def _retro_bollywood(bpb: float = 4.0) -> list[NoteEvent]:
    """
    Retro Bollywood (RD Burman era) — congas + dholak hybrid.
    Syncopated conga pattern with dholak accents.
    """
    events = []
    # Conga pattern
    conga_pattern = [
        (0.00, CONGA_HI,  82),
        (0.50, CONGA_LO,  65),
        (0.75, CONGA_HI,  70),
        (1.00, CONGA_LO,  78),
        (1.50, CONGA_HI,  72),
        (2.00, CONGA_LO,  85),
        (2.25, CONGA_HI,  68),
        (2.50, CONGA_LO,  62),
        (3.00, CONGA_HI,  80),
        (3.50, CONGA_LO,  70),
        (3.75, CONGA_HI,  65),
    ]
    for beat, note, vel in conga_pattern:
        if beat < bpb:
            events.append(_hit(note, beat, 0.12, vel))
    # Dholak bass accents
    events.append(_hit(DHOLAK_BASS, 0.0, 0.15, 88))
    events.append(_hit(DHOLAK_BASS, 2.0, 0.15, 85))
    return events


# ---------------------------------------------------------------------------
# Genre → pattern mapping
# ---------------------------------------------------------------------------

GENRE_DRUM_PATTERNS: dict[str, callable] = {
    "jazz_brush":         _jazz_brush,
    "jazz_ride":          _jazz_ride,
    "blues_shuffle":      _blues_shuffle,
    "rock":               _rock_straight,
    "pop":                _pop_straight,
    "hiphop":             _hiphop_808,
    "bossa_nova":         _bossa_nova,
    "funk":               _funk_16th,
    "waltz":              _waltz,
    "tabla_teentaal":     _tabla_teentaal,
    "mridangam_adi":      _mridangam_adi,
    "dholak_bhangra":     _dholak_bhangra,
    "qawwali":            _qawwali_clap,
    "koothu":             _koothu_parai,
    "retro_bollywood":    _retro_bollywood,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_drum_part(
    genre: str,
    total_beats: float,
    beats_per_bar: float = 4.0,
    intensity: float = 1.0,   # 0.5 = quieter, 1.0 = full, 1.2 = louder
    fill_every_n_bars: int = 4,  # add a fill every N bars (0 = no fills)
) -> list[NoteEvent]:
    """
    Generate a complete drum part for the given genre.

    Parameters
    ----------
    genre           : One of the keys in GENRE_DRUM_PATTERNS
    total_beats     : Total length to fill
    beats_per_bar   : Bar length (4.0 for 4/4, 3.0 for 3/4 etc.)
    intensity       : Velocity multiplier
    fill_every_n_bars: Insert a crash+fill every N bars

    Returns
    -------
    List of NoteEvent dicts on DRUM_CHANNEL
    """
    pattern_fn = GENRE_DRUM_PATTERNS.get(genre, _jazz_brush)
    bar_pattern = pattern_fn(beats_per_bar)

    # Tile across total duration
    events = _tile(bar_pattern, beats_per_bar, total_beats)

    # Apply intensity scaling
    if intensity != 1.0:
        for e in events:
            e["velocity"] = max(20, min(127, int(e["velocity"] * intensity)))

    # Add crash + fill at section boundaries
    if fill_every_n_bars > 0:
        fill_beats = [
            b * beats_per_bar
            for b in range(0, int(total_beats / beats_per_bar), fill_every_n_bars)
            if b > 0
        ]
        for fb in fill_beats:
            if fb < total_beats:
                events.append(_hit(CRASH, fb, 0.3, int(85 * intensity)))
                # Snare fill in the bar before
                fill_start = fb - beats_per_bar
                if fill_start >= 0:
                    for i in range(4):
                        b = fill_start + beats_per_bar - 1.0 + i * 0.25
                        if 0 <= b < total_beats:
                            events.append(_hit(SNARE, b, 0.08, int((70 + i * 5) * intensity)))

    # Sort by start_beat
    events.sort(key=lambda e: e["start_beat"])
    return events


def drum_genre_for_preset(preset_name: str) -> str | None:
    """
    Return the drum genre string for a given instrument preset name.
    Returns None if the preset doesn't use drums.

    Indian classical presets (Carnatic, Hindustani, Sufi, Sitar) return None —
    their percussion is handled by the raga engine as melodic tracks, not
    GM channel 9 drums.
    """
    mapping = {
        # Western
        "Jazz Quartet":           "jazz_ride",
        "Jazz Big Band":          "jazz_ride",
        "Jazz Trio (Brush)":      "jazz_brush",
        "Blues Band":             "blues_shuffle",
        "Blues Trio":             "blues_shuffle",
        "Funk Band":              "funk",
        "Pop Band":               "pop",
        "Retro Pop (80s)":        "pop",
        "City Pop":               "bossa_nova",
        "Hip-Hop Beat":           "hiphop",
        "Rock Band":              "rock",
        "Bossa Nova":             "bossa_nova",
        # Indian folk only — dholak/parai are folk instruments, not classical
        "Bollywood Golden Era":   "retro_bollywood",
        "Bollywood Modern":       "pop",
        "Koothu / Folk":          "koothu",
        "Dholak Party":           "dholak_bhangra",
        # Excluded — no GM drums for these:
        # Carnatic Ensemble, Hindustani Classical, Sufi / Qawwali,
        # Sitar & Strings — percussion handled by raga engine
    }
    return mapping.get(preset_name)
