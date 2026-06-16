"""
src/raga_engine.py
==================
Indian classical music engine for Orchestral Composer.

Provides:
  1. Raga definitions — arohana, avarohana, vadi, samvadi, pakad phrases,
     characteristic gamakas, and time-of-day classification
  2. Drone arrangement — replaces Western chord timeline with Sa-Pa-Sa drone
  3. Raga-aware melody generator — respects arohana/avarohana rules,
     vadi/samvadi note weighting, pakad phrases, and gamaka ornaments
  4. Preset detector — tells the pipeline whether to use Indian or Western mode

All Indian presets bypass the Western chord voicing system entirely.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Note helpers
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_TO_PC: dict[str, int] = {
    "C": 0,  "C#": 1, "Db": 1,
    "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,  "F":  5, "F#": 6, "Gb": 6,
    "G": 7,  "G#": 8, "Ab": 8,
    "A": 9,  "A#":10, "Bb":10,
    "B": 11,
}


def _midi_to_note_name(midi: int) -> str:
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _pc_to_midi(pc: int, center: int = 65) -> int:
    """Find the MIDI note with pitch class pc closest to center."""
    base = (center // 12) * 12 + pc
    if base < center - 6:
        base += 12
    if base > center + 6:
        base -= 12
    return base


# ---------------------------------------------------------------------------
# Raga definition
# ---------------------------------------------------------------------------

@dataclass
class RagaDefinition:
    name:        str
    arohana:     list[int]   # ascending scale — pitch classes 0-11
    avarohana:   list[int]   # descending scale — pitch classes 0-11
    vadi:        int          # most important note (pitch class)
    samvadi:     int          # second most important note (pitch class)
    pakad:       list[int]    # characteristic phrase (pitch classes)
    time:        str = "any"  # "morning", "evening", "night", "any"
    family:      str = "Carnatic"  # "Carnatic", "Hindustani"
    description: str = ""
    # Gamaka specifications: maps pitch class → gamaka type
    # Types: "meend" (glide from below), "andolan" (oscillation),
    #        "kampita" (fast oscillation), "sphurita" (touch note below then land)
    #        "kan" (grace note approach), "none" (hard attack, no ornament)
    gamakas:     dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Raga library
# ---------------------------------------------------------------------------

RAGAS: dict[str, RagaDefinition] = {

    # ── Carnatic ──────────────────────────────────────────────────────────────

    "Shankarabharanam": RagaDefinition(
        name="Shankarabharanam",
        arohana   = [0, 2, 4, 5, 7, 9, 11],   # S R2 G3 M1 P D2 N3
        avarohana = [0, 2, 4, 5, 7, 9, 11],   # same (melakarta — symmetric)
        vadi=7, samvadi=0,
        pakad=[0, 2, 4, 5, 7, 5, 4, 2, 0],
        time="any", family="Carnatic",
        description="The Carnatic equivalent of major. Bright, auspicious.",
        gamakas={4: "kampita", 11: "meend", 9: "andolan"},
    ),

    "Kalyani": RagaDefinition(
        name="Kalyani",
        arohana   = [0, 2, 4, 6, 7, 9, 11],   # S R2 G3 M2 P D2 N3
        avarohana = [0, 2, 4, 6, 7, 9, 11],
        vadi=4, samvadi=11,
        pakad=[0, 2, 4, 6, 7, 6, 4, 2, 0],
        time="evening", family="Carnatic",
        description="Lydian-equivalent. Uplifting, devotional.",
        gamakas={4: "kampita", 6: "meend", 11: "andolan"},
    ),

    "Bhairavi": RagaDefinition(
        name="Bhairavi",
        arohana   = [0, 1, 3, 5, 7, 8, 10],
        avarohana = [0, 1, 3, 5, 7, 8, 10],
        vadi=5, samvadi=0,
        pakad=[0, 1, 3, 5, 3, 1, 0],
        time="morning", family="Carnatic",
        description="Melancholic, introspective. Used to conclude concerts.",
        gamakas={1: "andolan", 3: "kampita", 8: "meend", 10: "sphurita"},
    ),

    "Mohanam": RagaDefinition(
        name="Mohanam",
        arohana   = [0, 2, 4, 7, 9],
        avarohana = [0, 2, 4, 7, 9],
        vadi=4, samvadi=9,
        pakad=[0, 2, 4, 7, 9, 7, 4, 2, 0],
        time="evening", family="Carnatic",
        description="Major pentatonic. Warm, folk-like, accessible.",
        gamakas={4: "kampita", 9: "andolan"},
    ),

    "Hamsadhwani": RagaDefinition(
        name="Hamsadhwani",
        arohana   = [0, 2, 4, 7, 11],
        avarohana = [0, 2, 4, 7, 11],
        vadi=4, samvadi=11,
        pakad=[0, 2, 4, 7, 11, 7, 4, 2, 0],
        time="evening", family="Carnatic",
        description="Pentatonic with major 7th. Auspicious, celebratory.",
        gamakas={4: "kampita", 11: "meend"},
    ),

    "Kharaharapriya": RagaDefinition(
        name="Kharaharapriya",
        arohana   = [0, 2, 3, 5, 7, 9, 10],
        avarohana = [0, 2, 3, 5, 7, 9, 10],
        vadi=2, samvadi=9,
        pakad=[0, 2, 3, 5, 7, 5, 3, 2, 0],
        time="any", family="Carnatic",
        description="Dorian-equivalent. Expressive, versatile.",
        gamakas={3: "kampita", 10: "andolan", 9: "meend"},
    ),

    "Natabhairavi": RagaDefinition(
        name="Natabhairavi",
        arohana   = [0, 2, 3, 5, 7, 8, 10],
        avarohana = [0, 2, 3, 5, 7, 8, 10],
        vadi=5, samvadi=0,
        pakad=[0, 2, 3, 5, 7, 5, 3, 2, 0],
        time="any", family="Carnatic",
        description="Natural minor equivalent. Emotive, sorrowful.",
        gamakas={3: "kampita", 8: "andolan", 10: "meend"},
    ),

    "Kiravani": RagaDefinition(
        name="Kiravani",
        arohana   = [0, 2, 3, 5, 7, 8, 11],
        avarohana = [0, 2, 3, 5, 7, 8, 11],
        vadi=7, samvadi=2,
        pakad=[0, 2, 3, 5, 7, 8, 11, 8, 7, 5, 3, 2, 0],
        time="night", family="Carnatic",
        description="Harmonic minor. Dramatic, intense.",
        gamakas={3: "kampita", 8: "andolan", 11: "meend"},
    ),

    "Mayamalavagowla": RagaDefinition(
        name="Mayamalavagowla",
        arohana   = [0, 1, 4, 5, 7, 8, 11],
        avarohana = [0, 1, 4, 5, 7, 8, 11],
        vadi=5, samvadi=0,
        pakad=[0, 1, 4, 5, 7, 5, 4, 1, 0],
        time="morning", family="Carnatic",
        description="Double harmonic. Used for beginners — very distinctive.",
        gamakas={1: "andolan", 4: "kampita", 8: "andolan", 11: "meend"},
    ),

    "Abhogi": RagaDefinition(
        name="Abhogi",
        arohana   = [0, 2, 3, 5, 9],
        avarohana = [0, 2, 3, 5, 9],
        vadi=3, samvadi=9,
        pakad=[0, 2, 3, 5, 9, 5, 3, 2, 0],
        time="any", family="Carnatic",
        description="Pentatonic. Light, playful.",
        gamakas={3: "kampita", 9: "kan"},
    ),

    "Hindolam": RagaDefinition(
        name="Hindolam",
        arohana   = [0, 3, 5, 8, 10],
        avarohana = [0, 3, 5, 8, 10],
        vadi=3, samvadi=8,
        pakad=[0, 3, 5, 8, 10, 8, 5, 3, 0],
        time="early morning", family="Carnatic",
        description="Minor pentatonic variant. Meditative, haunting.",
        gamakas={3: "andolan", 8: "kampita", 10: "meend"},
    ),

    # ── Hindustani ────────────────────────────────────────────────────────────

    "Yaman": RagaDefinition(
        name="Yaman",
        arohana   = [0, 2, 4, 6, 7, 9, 11],   # Lydian — sharp Ma
        avarohana = [0, 2, 4, 6, 7, 9, 11],
        vadi=4, samvadi=11,
        pakad=[0, 4, 6, 7, 9, 11, 7, 4, 2, 0],
        time="evening", family="Hindustani",
        description="Most popular Hindustani raga. Romantic, expansive.",
        gamakas={4: "meend", 6: "andolan", 11: "kampita"},
    ),

    "Bhairav": RagaDefinition(
        name="Bhairav",
        arohana   = [0, 1, 4, 5, 7, 8, 11],   # Double harmonic
        avarohana = [0, 1, 4, 5, 7, 8, 11],
        vadi=5, samvadi=0,
        pakad=[0, 1, 4, 5, 4, 1, 0],
        time="morning", family="Hindustani",
        description="Morning raga. Devotional, majestic.",
        gamakas={1: "andolan", 4: "meend", 8: "andolan", 11: "kampita"},
    ),

    "Kafi": RagaDefinition(
        name="Kafi",
        arohana   = [0, 2, 3, 5, 7, 9, 10],   # Dorian
        avarohana = [0, 2, 3, 5, 7, 9, 10],
        vadi=2, samvadi=7,
        pakad=[0, 2, 3, 5, 7, 5, 3, 2, 0],
        time="night", family="Hindustani",
        description="Associated with folk, holi, thumri. Earthy, warm.",
        gamakas={3: "kampita", 10: "andolan", 9: "meend"},
    ),

    "Darbari Kanada": RagaDefinition(
        name="Darbari Kanada",
        arohana   = [0, 2, 3, 5, 7, 8, 10],
        avarohana = [0, 2, 3, 5, 7, 8, 10],
        vadi=3, samvadi=7,
        pakad=[0, 2, 3, 5, 7, 8, 7, 5, 3, 2, 0],
        time="night", family="Hindustani",
        description="Regal, profound. Akbar's court raga.",
        # Darbari is famous for its slow, deep andolan on Ga and Dha
        gamakas={3: "andolan", 8: "andolan", 10: "meend"},
    ),

    "Bhimpalasi": RagaDefinition(
        name="Bhimpalasi",
        arohana   = [0, 3, 5, 7, 10],          # pentatonic ascent
        avarohana = [0, 2, 3, 5, 7, 8, 10],   # full descent
        vadi=5, samvadi=0,
        pakad=[0, 3, 5, 7, 5, 3, 2, 0],
        time="afternoon", family="Hindustani",
        description="Afternoon raga. Tender, longing.",
        gamakas={3: "kampita", 8: "andolan", 10: "meend"},
    ),

    "Bageshri": RagaDefinition(
        name="Bageshri",
        arohana   = [0, 3, 5, 7, 10],
        avarohana = [0, 2, 3, 5, 7, 8, 10],
        vadi=3, samvadi=7,
        pakad=[0, 3, 5, 7, 8, 7, 5, 3, 0],
        time="night", family="Hindustani",
        description="Romantic, melancholic. Night raga.",
        gamakas={3: "andolan", 8: "meend", 10: "kampita"},
    ),

    "Marwa": RagaDefinition(
        name="Marwa",
        arohana   = [0, 1, 4, 6, 9, 11],       # no Pa (no perfect fifth!)
        avarohana = [0, 1, 4, 6, 9, 11],
        vadi=4, samvadi=1,
        pakad=[0, 1, 4, 6, 9, 11, 9, 6, 4, 1, 0],
        time="evening", family="Hindustani",
        description="Sunset raga. Restless, tense — no perfect fifth.",
        gamakas={1: "andolan", 4: "meend", 6: "kampita", 9: "andolan"},
    ),

    "Kedar": RagaDefinition(
        name="Kedar",
        arohana   = [0, 4, 5, 6, 7, 10],
        avarohana = [0, 2, 4, 5, 7, 9, 10],
        vadi=7, samvadi=4,
        pakad=[0, 4, 5, 6, 7, 6, 5, 4, 2, 0],
        time="evening", family="Hindustani",
        description="Devotional, serene. Bhajan and thumri staple.",
        gamakas={4: "meend", 6: "sphurita", 10: "andolan"},
    ),
}


# ---------------------------------------------------------------------------
# Indian preset detection
# ---------------------------------------------------------------------------

INDIAN_PRESETS = frozenset({
    "Carnatic Ensemble",
    "Bollywood Golden Era",
    "Bollywood Modern",
    "Sufi / Qawwali",
    "Koothu / Folk",
    "Dholak Party",
    "Hindustani Classical",
    "Sitar & Strings",
})

DRONE_PRESETS = frozenset({
    "Carnatic Ensemble",
    "Hindustani Classical",
    "Sitar & Strings",
    "Sufi / Qawwali",
})

BOLLYWOOD_PRESETS = frozenset({
    "Bollywood Golden Era",
    "Bollywood Modern",
    "Dholak Party",
    "Koothu / Folk",
})


def is_indian_preset(preset_name: str) -> bool:
    return preset_name in INDIAN_PRESETS


def uses_drone_mode(preset_name: str) -> bool:
    """True = replace chord timeline with Sa-Pa drone. False = keep Western harmony."""
    return preset_name in DRONE_PRESETS


def get_raga_for_key(key: str) -> RagaDefinition | None:
    """
    Try to find a matching raga from the key string.
    Accepts "D Yaman", "G Bhairav", "C Kafi" etc.
    Falls back to None if no raga name found.
    """
    tokens = key.strip().split()
    if len(tokens) < 2:
        return None
    raga_name = " ".join(tokens[1:])
    # Direct match
    if raga_name in RAGAS:
        return RAGAS[raga_name]
    # Case-insensitive
    raga_lower = raga_name.lower()
    for name, raga in RAGAS.items():
        if name.lower() == raga_lower:
            return raga
    return None


def resolve_raga(key: str) -> RagaDefinition:
    """Raga for this key, or Shankarabharanam as safe default."""
    return get_raga_for_key(key) or RAGAS["Shankarabharanam"]


def raga_scale_pcs(key: str, raga: RagaDefinition | None = None) -> frozenset[int]:
    """Absolute pitch classes (0–11) allowed in this raga."""
    root_pc = _NOTE_TO_PC.get(key.strip().split()[0], 0)
    raga = raga or resolve_raga(key)
    return frozenset((root_pc + pc) % 12 for pc in set(raga.arohana + raga.avarohana))


def snap_note_to_raga(
    note_name: str,
    key: str,
    raga: RagaDefinition | None = None,
    lo: int = 48,
    hi: int = 84,
) -> str:
    """Snap one note to the nearest degree in the raga."""
    import librosa
    scale = raga_scale_pcs(key, raga)
    try:
        midi = librosa.note_to_midi(note_name)
    except Exception:
        return note_name
    if midi % 12 in scale:
        return note_name
    candidates = [m for m in range(max(lo, midi - 6), min(hi + 1, midi + 7)) if m % 12 in scale]
    if not candidates:
        candidates = [m for m in range(lo, hi + 1) if m % 12 in scale]
    if not candidates:
        return note_name
    best = min(candidates, key=lambda m: abs(m - midi))
    return _midi_to_note_name(best)


def snap_notes_to_raga(
    notes: list[dict],
    key: str,
    raga: RagaDefinition | None = None,
) -> list[dict]:
    """Force every melodic note onto the raga scale before audio render."""
    raga = raga or resolve_raga(key)
    out: list[dict] = []
    for note in notes:
        n = dict(note)
        n["note"] = snap_note_to_raga(n.get("note", "C4"), key, raga)
        out.append(n)
    return out


# ---------------------------------------------------------------------------
# Fix 1 — Drone arrangement
# ---------------------------------------------------------------------------

def make_drone_timeline(
    key: str,
    total_beats: float,
    raga: RagaDefinition | None = None,
) -> list[tuple[float, str]]:
    """
    Build a fake chord timeline that is just a sustained tonic drone.
    Used instead of a Western chord progression for classical Indian presets.
    Returns [(0.0, "Sa")] — a single drone entry.
    """
    root = key.strip().split()[0]
    # Single drone entry — the algo_arranger drone handler converts this
    return [(0.0, f"{root}drone")]


def make_drone_parts(
    key: str,
    total_beats: float,
    instrument_ranges: dict[str, tuple[int, int]],
    preset_name: str,
    raga: RagaDefinition | None = None,
    beats_per_bar: float = 4.0,
) -> dict[str, list[dict]]:
    """
    Generate drone parts for Indian classical presets.

    Returns a dict of {instrument_name: [note_events]} for:
      - tambura: continuous Sa + Pa + upper Sa drone
      - sarangi / strings_harmony: sustained held notes
      - harmonium: long swells on Sa and Pa
    """
    root = key.strip().split()[0]
    root_pc = _NOTE_TO_PC.get(root, 0)

    # Sa, Pa, upper Sa MIDI notes
    sa_midi  = root_pc + 48   # Sa in octave 3
    pa_midi  = root_pc + 55   # Pa (perfect fifth) in octave 3
    sa2_midi = root_pc + 60   # upper Sa in octave 4

    # Adjust to instrument ranges
    def in_range(midi: int, lo: int, hi: int) -> int:
        while midi < lo: midi += 12
        while midi > hi: midi -= 12
        return midi

    parts: dict[str, list[dict]] = {}

    # Tambura: plucked string drone — individual plucks every beat
    if "tambura" in instrument_ranges or preset_name in DRONE_PRESETS:
        lo, hi = instrument_ranges.get("tambura", (36, 60))
        sa  = in_range(sa_midi,  lo, hi)
        pa  = in_range(pa_midi,  lo, hi)
        sa2 = in_range(sa2_midi, lo, hi)
        tambura_notes = []
        pattern = [sa2, pa, sa, sa]   # classic tambura 4-string pattern
        beat = 0.0
        idx  = 0
        while beat < total_beats:
            tambura_notes.append({
                "note":           _midi_to_note_name(pattern[idx % 4]),
                "start_beat":     round(beat, 4),
                "duration_beats": beats_per_bar * 0.24,
                "velocity":       48,
            })
            beat += beats_per_bar * 0.25   # one pluck per quarter bar
            idx += 1
        parts["tambura"] = tambura_notes

    # Strings / sarangi: sustained held chord on Sa and Pa
    for inst in ["strings_harmony", "sarangi"]:
        if inst not in instrument_ranges:
            continue
        lo, hi = instrument_ranges[inst]
        sa  = in_range(sa_midi + 12, lo, hi)
        pa  = in_range(pa_midi + 12, lo, hi)
        held_notes = []
        bar_beats = beats_per_bar * 4   # hold for 4 bars at a time
        beat = 0.0
        while beat < total_beats:
            dur = min(bar_beats, total_beats - beat)
            for midi, vel in [(sa, 42), (pa, 38)]:
                held_notes.append({
                    "note":           _midi_to_note_name(midi),
                    "start_beat":     round(beat, 4),
                    "duration_beats": dur * 0.98,
                    "velocity":       vel,
                })
            beat += bar_beats
        parts[inst] = held_notes

    # Harmonium: swells on Sa every 2 bars
    if "harmonium" in instrument_ranges:
        lo, hi = instrument_ranges["harmonium"]
        sa  = in_range(sa_midi + 12, lo, hi)
        pa  = in_range(pa_midi + 12, lo, hi)
        sa2 = in_range(sa2_midi + 12, lo, hi)
        harm_notes = []
        swell_beats = beats_per_bar * 2
        beat = 0.0
        toggle = 0
        while beat < total_beats:
            dur = min(swell_beats, total_beats - beat)
            pitch = [sa, pa, sa2][toggle % 3]
            # Velocity swell: soft → loud → soft
            for offset, vel in [(0.0, 38), (dur * 0.3, 55), (dur * 0.7, 42)]:
                if beat + offset < total_beats:
                    harm_notes.append({
                        "note":           _midi_to_note_name(pitch),
                        "start_beat":     round(beat + offset, 4),
                        "duration_beats": dur * 0.3,
                        "velocity":       vel,
                    })
            beat += swell_beats
            toggle += 1
        parts["harmonium"] = harm_notes

    return parts


# ---------------------------------------------------------------------------
# Fix 2 — Raga-aware melody generator
# ---------------------------------------------------------------------------

def generate_raga_melody(
    key:         str,
    tempo_bpm:   float,
    total_beats: float,
    raga:        RagaDefinition | None = None,
    temperature: float = 1.0,
    seed:        int   = 0,
) -> list[dict]:
    """
    Generate a melodically authentic raga-based melody.

    Rules respected:
      1. Arohana notes used when ascending, avarohana when descending
      2. Vadi note gets 2.5× weight — appears most often
      3. Samvadi note gets 1.5× weight
      4. Pakad phrases inserted at phrase beginnings
      5. Gamaka ornaments (grace notes) on approach to vadi/samvadi
      6. Phrases end on Sa (tonic) or vadi
      7. No notes outside the combined arohana+avarohana set
    """
    rng = np.random.default_rng(seed if seed > 0 else None)

    root = key.strip().split()[0]
    root_pc = _NOTE_TO_PC.get(root, 0)

    # If no raga found, try to get one from key string, else use Shankarabharanam
    if raga is None:
        raga = get_raga_for_key(key)
    if raga is None:
        raga = RAGAS.get("Shankarabharanam", list(RAGAS.values())[0])

    # Build MIDI note sets for arohana and avarohana
    def _to_midi_set(scale_pcs: list[int], lo: int = 55, hi: int = 83) -> list[int]:
        notes = []
        for pc in scale_pcs:
            abs_pc = (root_pc + pc) % 12
            for m in range(lo, hi + 1):
                if m % 12 == abs_pc:
                    notes.append(m)
        return sorted(set(notes))

    arohana_midi   = _to_midi_set(raga.arohana)
    avarohana_midi = _to_midi_set(raga.avarohana)
    all_midi       = sorted(set(arohana_midi + avarohana_midi))

    if not all_midi:
        # Fallback to major scale
        all_midi = _to_midi_set([0, 2, 4, 5, 7, 9, 11])

    vadi_pc    = (root_pc + raga.vadi)    % 12
    samvadi_pc = (root_pc + raga.samvadi) % 12
    sa_pc      = root_pc % 12

    # Center note — Sa in a comfortable octave
    sa_midi = next((m for m in all_midi if m % 12 == sa_pc and 60 <= m <= 72), all_midi[len(all_midi)//2])
    current = sa_midi

    # Pakad phrase converted to MIDI
    pakad_midi = []
    for pc in raga.pakad:
        abs_pc = (root_pc + pc) % 12
        candidates = [m for m in all_midi if m % 12 == abs_pc]
        if candidates:
            best = min(candidates, key=lambda m: abs(m - current))
            pakad_midi.append(best)
            current = best

    # Rhythm patterns — Indian classical has more freedom, uses longer held notes
    if tempo_bpm < 60:
        # Vilambit (slow) — very long notes, deep exploration
        durations = [(3.0, 0.20), (2.0, 0.30), (1.0, 0.30), (0.5, 0.20)]
    elif tempo_bpm < 120:
        # Madhya (medium) — concert pacing with held vadi notes
        durations = [(2.0, 0.22), (1.0, 0.35), (0.5, 0.25), (0.75, 0.18)]
    else:
        # Drut (fast)
        durations = [(1.0, 0.25), (0.5, 0.35), (0.25, 0.25), (0.75, 0.15)]

    dur_vals  = [d for d, _ in durations]
    dur_probs = np.array([p for _, p in durations], dtype=np.float32)
    dur_probs /= dur_probs.sum()

    notes_out: list[dict] = []
    beat = 0.0
    current = sa_midi
    direction = 1   # 1 = ascending, -1 = descending
    phrase_idx = 0
    phrase_beats = 12.0 if tempo_bpm >= 100 else 20.0

    recent_notes: list[int] = []   # track last 4 notes to prevent spamming

    def weight_for(midi: int, recent: list[int]) -> float:
        pc = midi % 12
        w  = 1.0
        if pc == vadi_pc:    w = 2.5
        elif pc == samvadi_pc: w = 1.5
        elif pc == sa_pc:    w = 1.8

        # Strong repeat penalty — penalize same note played recently
        repeat_count = recent.count(midi)
        if repeat_count >= 3:   w *= 0.05   # almost never repeat 3+ times
        elif repeat_count == 2: w *= 0.15
        elif repeat_count == 1: w *= 0.45

        # Penalize same pitch class even in different octave
        pc_count = sum(1 for n in recent if n % 12 == pc)
        if pc_count >= 2: w *= 0.3

        return max(0.01, w)

    while beat < total_beats:
        remaining = total_beats - beat
        phrase_start = beat

        # Insert pakad at the start of alternating phrases
        if phrase_idx % 3 == 0 and len(pakad_midi) > 0 and remaining > len(pakad_midi) * 0.5:
            for pm in pakad_midi:
                dur = float(rng.choice(dur_vals, p=dur_probs))
                dur = min(dur, remaining - 0.001)
                if dur <= 0:
                    break
                # Gamaka: add a grace note approaching vadi/samvadi
                if pm % 12 in (vadi_pc, samvadi_pc) and beat > 0 and dur >= 0.5:
                    # Prefer raga-adjacent note; fall back to chromatic only if in raga
                    _grace_dir = -1 if pm > current else 1
                    _grace_candidates = [m for m in all_midi if m != pm and abs(m - pm) <= 3 and (m - pm) * _grace_dir >= 0] or \
                                        [m for m in all_midi if m != pm and abs(m - pm) <= 2]
                    grace_midi = min(_grace_candidates, key=lambda m: abs(m - pm)) if _grace_candidates else pm - 1
                    if grace_midi in all_midi:
                        notes_out.append({
                            "note":           _midi_to_note_name(grace_midi),
                            "start_beat":     round(beat, 4),
                            "duration_beats": min(0.125, dur * 0.2),
                            "velocity":       50,
                            "role":           "grace",
                        })
                        beat     += 0.125
                        dur      -= 0.125
                        remaining = total_beats - beat

                notes_out.append({
                    "note":           _midi_to_note_name(pm),
                    "start_beat":     round(beat, 4),
                    "duration_beats": round(dur, 4),
                    "velocity":       int(rng.integers(65, 88)),
                    "role":           "main",
                })
                beat += dur
                current = pm
                recent_notes.append(pm)
                if len(recent_notes) > 4:
                    recent_notes.pop(0)
                remaining = total_beats - beat
                if remaining <= 0:
                    break

        # Generate remaining notes in this phrase
        phrase_end = min(phrase_start + phrase_beats, total_beats)
        while beat < phrase_end - 0.124:
            remaining = phrase_end - beat

            # Choose next note
            active_set = arohana_midi if direction == 1 else avarohana_midi
            if not active_set:
                active_set = all_midi

            # Find index of current (or nearest) in active_set
            try:
                idx = active_set.index(current)
            except ValueError:
                nearest = min(active_set, key=lambda m: abs(m - current))
                idx = active_set.index(nearest)

            offsets = []
            if direction == 1: # Ascending
                offsets = [
                    (1, 1.0),    # step up
                    (2, 0.30),   # skip up
                    (0, 0.18),   # repeat note
                    (-1, 0.20),  # step down (brief zig-zag)
                ]
            else: # Descending
                offsets = [
                    (-1, 1.0),   # step down
                    (-2, 0.30),  # skip down
                    (0, 0.18),   # repeat note
                    (1, 0.20),   # step up (brief zig-zag)
                ]

            candidates = []
            base_weights = []
            for offset, base_w in offsets:
                next_idx = idx + offset
                if 0 <= next_idx < len(active_set):
                    cand_note = active_set[next_idx]
                    # Prevent huge interval jumps if scale degree indexes are too far apart
                    if abs(cand_note - current) <= 5:
                        candidates.append(cand_note)
                        base_weights.append(base_w)

            # Fallback if no valid candidates
            if not candidates:
                candidates = [min(active_set, key=lambda m: abs(m - current))]
                base_weights = [1.0]

            weights = []
            for m, bw in zip(candidates, base_weights):
                rw = weight_for(m, recent_notes)
                weights.append(bw * rw)

            weights = np.array(weights, dtype=np.float32)
            # Apply temperature
            weights = np.power(weights, 1.0 / max(0.1, temperature))
            weights /= weights.sum()

            next_midi = int(rng.choice(candidates, p=weights))

            # Duration
            dur = float(rng.choice(dur_vals, p=dur_probs))
            dur = min(dur, remaining - 0.001)
            if dur <= 0:
                break

            # End phrase on Sa or vadi — hold longer for concert weight
            if remaining <= max(dur_vals) * 1.5:
                sa_candidates = [m for m in active_set if m % 12 in (sa_pc, vadi_pc)]
                if sa_candidates:
                    next_midi = min(sa_candidates, key=lambda m: abs(m - current))
                dur = remaining - 0.001

            # Occasional rest within phrase — space to breathe
            if rng.random() < 0.12 and remaining > dur + 0.5:
                beat += 0.5
                remaining = phrase_end - beat
                continue

            vel = int(rng.integers(58, 82))
            next_pc = next_midi % 12
            if next_pc == vadi_pc:
                dur = min(dur * 1.35, remaining - 0.001)
                vel = min(92, vel + 10)
            elif next_pc == samvadi_pc:
                dur = min(dur * 1.15, remaining - 0.001)

            # Gamaka: add a grace note approaching vadi/samvadi in general melody
            if next_pc in (vadi_pc, samvadi_pc) and beat > 0 and dur >= 0.5:
                _g_dir = -1 if next_midi > current else 1
                _g_candidates = [m for m in all_midi if m != next_midi and abs(m - next_midi) <= 3 and (m - next_midi) * _g_dir >= 0] or \
                                 [m for m in all_midi if m != next_midi and abs(m - next_midi) <= 2]
                grace_midi = min(_g_candidates, key=lambda m: abs(m - next_midi)) if _g_candidates else next_midi - 1
                if grace_midi in all_midi:
                    notes_out.append({
                        "note":           _midi_to_note_name(grace_midi),
                        "start_beat":     round(beat, 4),
                        "duration_beats": 0.125,
                        "velocity":       50,
                        "role":           "grace",
                    })
                    beat += 0.125
                    dur  = max(0.25, dur - 0.125)

            notes_out.append({
                "note":           _midi_to_note_name(next_midi),
                "start_beat":     round(beat, 4),
                "duration_beats": round(max(0.25, dur), 4),
                "velocity":       vel,
                "role":           "main",
            })
            beat    += dur
            current  = next_midi

            # Track recent notes for repeat penalty (keep last 4)
            recent_notes.append(next_midi)
            if len(recent_notes) > 4:
                recent_notes.pop(0)

            # Change direction based on pitch register boundaries and random chance
            if current > 78 and direction == 1:
                direction = -1
            elif current < 54 and direction == -1:
                direction = 1
            elif rng.random() < 0.15:
                direction *= -1

        # Gap between phrases (breath) — 2 to 4 beats of silence
        breath = float(rng.choice([2.0, 2.5, 3.0, 4.0]))
        beat += breath
        phrase_idx += 1

    # Tag phrase boundaries for the audio renderer
    try:
        from src.carnatic_ensemble import tag_phrase_boundaries
        notes_out = tag_phrase_boundaries(notes_out)
    except ImportError:
        from carnatic_ensemble import tag_phrase_boundaries
        notes_out = tag_phrase_boundaries(notes_out)

    return notes_out


# ---------------------------------------------------------------------------
# Fix 4 — Soundfont config for Indian instruments
# ---------------------------------------------------------------------------

# SGM-v2.01 soundfont has proper Indian instrument patches at these program numbers.
# If using GeneralUser GS, falls back to GM approximations in prompts.py.
SGM_INDIAN_PROGRAMS: dict[str, int] = {
    "sitar":        104,  # Sitar (proper in SGM)
    "bansuri":       73,  # Flute (closest in GM; SGM has proper bansuri)
    "tambura":       46,  # Orchestral Harp in GM; drone harp in SGM
    "tabla_bayan":   43,  # Low floor tom in GM
    "tabla_dayan":   45,  # Low tom in GM
    "mridangam":     47,  # Low-mid tom in GM
    "sarangi":       40,  # Violin in GM; proper sarangi in SGM
    "harmonium":     16,  # Drawbar organ in GM; harmonium in SGM
    "veena":        104,  # Sitar in GM; veena in SGM
    "nadaswaram":   111,  # Shanai in GM
    "shehnai":      111,  # Shanai
}