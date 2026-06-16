"""
LLM prompt templates for orchestration and composition.

Keeping all prompts in one module makes them easy to inspect, tune, and
version-control independently from the orchestration logic. Students who
want to change how the AI reasons about music should start here.
"""

from src.harmony import chord_tone_names, note_name_to_midi, scale_note_names, parse_key_string

# Instruments that express harmony as simultaneous chord voicings.
# The LLM must output 3–4 note objects with the SAME start_beat for these.
CHORDAL_INSTRUMENTS: frozenset[str] = frozenset({
    "piano_harmony", "electric_guitar",
    "strings_harmony", "woodwind_harmony",
})

# Instruments that play single-note bass lines.
BASS_INSTRUMENTS: frozenset[str] = frozenset({
    "bass", "acoustic_bass", "cello",
})

# General MIDI program numbers for each named instrument role.
# These map part names (used in prompts and MIDI builder) to GM patch numbers.
INSTRUMENT_PROGRAMS: dict[str, int] = {
    # Strings / orchestral winds
    "strings_melody":   40,   # Violin
    "flute_melody":     73,   # Flute
    "oboe_melody":      68,   # Oboe
    "trumpet_melody":   56,   # Trumpet
    "strings_harmony":  41,   # Viola
    "woodwind_harmony": 71,   # Clarinet
    "bass":             43,   # Contrabass
    "cello":            42,   # Cello
    # Piano
    "piano_melody":      0,   # Acoustic Grand Piano
    "piano_harmony":     0,
    # Jazz / blues
    "alto_sax":         65,   # Alto Sax
    "tenor_sax":        66,   # Tenor Sax
    "electric_guitar":  27,   # Electric Guitar (clean)
    "harmonica":        22,   # Harmonica
    "acoustic_bass":    32,   # Acoustic Bass
    # World / Indian classical
    "sitar":           104,   # Sitar
    "veena":           104,   # Sitar patch (closest GM equivalent; Indian audio engine overrides)
    "bansuri":          73,   # Flute (Indian audio engine uses real samples)
    "harmonium":        20,   # Reed Organ (closest GM equivalent)
    "sarangi":          41,   # Viola (closest GM equivalent)
    "nadaswaram":       68,   # Oboe (closest GM equivalent)
    "tambura":         104,   # Sitar patch (Indian audio engine handles drone)
    "tabla":            -1,   # Percussion — handled by drum_engine / Indian audio engine
    "mridangam":        -1,   # Percussion — handled by drum_engine / Indian audio engine
    "dholak":           -1,   # Percussion — handled by drum_engine
}

# Natural pitch ranges (MIDI note numbers) for basic range validation.
INSTRUMENT_RANGES: dict[str, tuple[int, int]] = {
    "strings_melody":  (55, 88),   # G3–E6
    "flute_melody":    (60, 96),   # C4–C7
    "oboe_melody":     (58, 91),   # Bb3–G6
    "trumpet_melody":  (52, 82),   # E3–Bb5
    "strings_harmony": (48, 79),   # C3–G5
    "woodwind_harmony":(50, 89),   # D3–F6
    "bass":            (28, 55),   # E1–G3
    "cello":           (36, 72),   # C2–C5
    "piano_melody":    (21, 108),  # A0–C8 (full piano)
    "piano_harmony":   (21, 108),
    "alto_sax":        (49, 80),   # Db3–Ab5 (concert pitch)
    "tenor_sax":       (44, 75),   # Ab2–Eb5
    "electric_guitar": (40, 84),   # E2–C6
    "harmonica":       (60, 84),   # C4–C6
    "acoustic_bass":   (28, 55),   # E1–G3
    "sitar":           (50, 86),   # D3–D6
    # Indian instruments
    "veena":           (40, 79),   # E2–G5
    "bansuri":         (60, 88),   # C4–E6
    "harmonium":       (48, 84),   # C3–C6
    "sarangi":         (48, 81),   # C3–A5
    "nadaswaram":      (55, 91),   # G3–G6
    "tambura":         (36, 60),   # C2–C4 (drone only)
    "tabla":           (35, 81),   # B1–A5 (GM percussion range)
    "mridangam":       (35, 81),   # B1–A5
    "dholak":          (35, 81),   # B1–A5
}

TIME_SIGNATURES: dict[str, dict] = {
    "4/4":  {
        "beats_per_bar": 4,    "numerator": 4,  "denominator": 4,
        "prompt": "TIME SIGNATURE: 4/4 — four quarter-note beats per bar. Standard pulse.",
    },
    "3/4":  {
        "beats_per_bar": 3,    "numerator": 3,  "denominator": 4,
        "prompt": "TIME SIGNATURE: 3/4 — three quarter-note beats per bar. Strong accent on beat 1.",
    },
    "2/4":  {
        "beats_per_bar": 2,    "numerator": 2,  "denominator": 4,
        "prompt": (
            "TIME SIGNATURE: 2/4 — two quarter-note beats per bar. March/polka feel. "
            "Strong accent on beat 1, weaker on beat 2. Crisp, driving pulse."
        ),
    },
    "5/4":  {
        "beats_per_bar": 5,    "numerator": 5,  "denominator": 4,
        "prompt": (
            "TIME SIGNATURE: 5/4 — five quarter-note beats per bar. "
            "Group as 3+2 (accents at 0.0 and 3.0) or 2+3 (accents at 0.0 and 2.0). "
            "Place a clear secondary accent at the start of each group."
        ),
    },
    "5/8":  {
        "beats_per_bar": 2.5,  "numerator": 5,  "denominator": 8,
        "prompt": (
            "TIME SIGNATURE: 5/8 — five eighth notes per bar (2.5 quarter beats). "
            "Group as 2+3 or 3+2 eighth notes. Use 0.5-beat subdivisions. "
            "Lively, asymmetric folk-dance feel."
        ),
    },
    "6/8":  {
        "beats_per_bar": 3.0,  "numerator": 6,  "denominator": 8,
        "prompt": (
            "TIME SIGNATURE: 6/8 — compound duple. Two dotted-quarter beats per bar "
            "(each = 1.5 quarter beats). Subdivide each into three 0.5-beat eighth-note pulses. "
            "Accent at 0.0 and 1.5 within each bar."
        ),
    },
    "7/8":  {
        "beats_per_bar": 3.5,  "numerator": 7,  "denominator": 8,
        "prompt": (
            "TIME SIGNATURE: 7/8 — seven eighth notes per bar (3.5 quarter beats). "
            "Group as 3+4 (accents at 0.0 and 1.5) or 4+3 (accents at 0.0 and 2.0). "
            "Use 0.5-beat and 1.0-beat note values to create the characteristic uneven pulse."
        ),
    },
    "7/4":  {
        "beats_per_bar": 7,    "numerator": 7,  "denominator": 4,
        "prompt": (
            "TIME SIGNATURE: 7/4 — seven quarter-note beats per bar. "
            "Group as 4+3 (accents at 0.0 and 4.0) or 3+4 (accents at 0.0 and 3.0)."
        ),
    },
    "9/8":  {
        "beats_per_bar": 4.5,  "numerator": 9,  "denominator": 8,
        "prompt": (
            "TIME SIGNATURE: 9/8 — compound triple. Three dotted-quarter beats per bar "
            "(each = 1.5 quarter beats). Subdivide each into three 0.5-beat pulses. "
            "Accents at 0.0, 1.5, and 3.0 within each bar."
        ),
    },
    "10/8": {
        "beats_per_bar": 5.0,  "numerator": 10, "denominator": 8,
        "prompt": (
            "TIME SIGNATURE: 10/8 — ten eighth notes (5.0 quarter beats). "
            "Group as 3+3+4 or 2+3+2+3. Carnatic/Balkan-influenced. "
            "Accents driven by the sub-grouping pattern."
        ),
    },
    "11/8": {
        "beats_per_bar": 5.5,  "numerator": 11, "denominator": 8,
        "prompt": (
            "TIME SIGNATURE: 11/8 — eleven eighth notes (5.5 quarter beats). "
            "Group as 3+4+4 or 4+3+4. Exotic, limping feel. "
            "Use 0.5-beat and 1.0-beat note values. Accent the start of each sub-group."
        ),
    },
    "12/8": {
        "beats_per_bar": 6.0,  "numerator": 12, "denominator": 8,
        "prompt": (
            "TIME SIGNATURE: 12/8 — compound quadruple. Four dotted-quarter beats per bar "
            "(each = 1.5 quarter beats). Blues/shuffle feel — subdivide in triplets. "
            "Accents at 0.0, 1.5, 3.0, and 4.5 within each bar."
        ),
    },
    "13/8": {
        "beats_per_bar": 6.5,  "numerator": 13, "denominator": 8,
        "prompt": (
            "TIME SIGNATURE: 13/8 — thirteen eighth notes (6.5 quarter beats). "
            "Group as 5+4+4 or 4+4+5. Rare, highly asymmetric. "
            "Establish the grouping in the bass first; other parts follow its accent pattern."
        ),
    },
    "15/8": {
        "beats_per_bar": 7.5,  "numerator": 15, "denominator": 8,
        "prompt": (
            "TIME SIGNATURE: 15/8 — fifteen eighth notes (7.5 quarter beats). "
            "Group as 3+3+3+3+3 or 4+4+4+3. Carnatic Khanda Chapu influence. "
            "Strong accent every 3 eighth notes."
        ),
    },
}

HARMONY_STYLES: dict[str, str] = {
    "Block chords": (
        "PIANO/GUITAR STYLE: Block chords. Voice ALL chord tones simultaneously — multiple notes "
        "with the SAME start_beat and duration equal to the chord duration. "
        "Change voicing with each chord change. Use close or open voicing as fits the register."
    ),
    "Arpeggio (up)": (
        "PIANO/GUITAR STYLE: Ascending arpeggio. Spread chord tones from low to high across the chord "
        "duration — one note per 0.5 beats, bottom note first, top note last. Each note has "
        "duration_beats 0.5 except the last which sustains to the next chord. "
        "Cycle through all chord tones for the full chord duration."
    ),
    "Arpeggio (down)": (
        "PIANO/GUITAR STYLE: Descending arpeggio. Spread chord tones from high to low — top note first, "
        "bottom note last, one note per 0.5 beats. Cycle through all chord tones for the full duration."
    ),
    "Arpeggio (alt)": (
        "PIANO/GUITAR STYLE: Alternating arpeggio. Play chord tones in up-down pattern "
        "(root, 5th, 3rd, 7th, 5th, 3rd…), one note per 0.5 beats, cycling for the full chord duration."
    ),
    "Alberti bass": (
        "PIANO/GUITAR STYLE: Alberti bass pattern in the right hand. Low note, high note, middle note, "
        "high note — repeating every 2 beats. Use chord tones only. Left hand plays chord roots. "
        "Pattern (beat offsets within chord): 0.0=root, 0.5=5th, 1.0=3rd, 1.5=5th, repeat."
    ),
    "Broken chords": (
        "PIANO/GUITAR STYLE: Broken chord pattern. Play chord tones in pairs across the chord duration — "
        "root+5th on beat 1, 3rd+7th on beat 2, root+5th on beat 3, 3rd+7th on beat 4. "
        "Each pair shares the same start_beat (simultaneous). Duration 0.5 beats per pair."
    ),
    "Jazz comping": (
        "PIANO/GUITAR STYLE: Jazz comping — irregular syncopated chord stabs. "
        "Voice the chord as 3–4 simultaneous notes (same start_beat) but place stabs off the beat: "
        "on beat 2, beat 3.5, beat 4-and. Leave space — do NOT comp on every beat. "
        "Use shell voicings: root, 3rd, 7th (omit 5th). Extended chords (9ths, 13ths) preferred."
    ),
    "Pad (sustained)": (
        "PIANO/GUITAR STYLE: Sustained pad. Voice the full chord (ALL tones, same start_beat) and hold "
        "for the entire chord duration. Very long duration_beats = entire chord length. "
        "One chord block per chord change, no inner movement. Low velocity (55–65) for background texture."
    ),
    "Free": (
        "PIANO/GUITAR STYLE: Free accompaniment — vary the pattern chord by chord, "
        "like an experienced pianist improvising. Mix block chords, jazz comping stabs "
        "(on off-beats: beat 2, beat 3.5), arpeggios, and broken pairs. "
        "Short chords (< 2 beats): use block chords. Long chords: prefer arpeggios or comping. "
        "Leave breathing space — not every beat needs a note."
    ),
}

RHYTHM_STYLES: dict[str, str] = {
    "Free":          "",
    "Straight 8ths": (
        "RHYTHM: Straight feel. Place 8th notes evenly at 0.5-beat intervals "
        "(0.0, 0.5, 1.0, 1.5…). No swing. Mechanical, even subdivision."
    ),
    "Swing":         (
        "RHYTHM: Swing feel. 8th notes land at X.0 and X.67 (2:1 triplet ratio), "
        "NOT at X.5. Emphasise beats 2 and 4. Lay back slightly on the upbeats."
    ),
    "Shuffle":       (
        "RHYTHM: Blues shuffle. Triplet-based like swing (upbeats at X.67) but heavier "
        "and more insistent. Strong downbeat, shuffle on every beat."
    ),
    "Double-time":   (
        "RHYTHM: Double-time feel. Write at twice the note density — use 0.25-beat "
        "durations where you'd normally use 0.5, and 0.5 where you'd use 1.0. "
        "Harmonic changes stay at the normal rate."
    ),
    "Half-time":     (
        "RHYTHM: Half-time feel. Sparse and spacious — favour 1.0 and 2.0-beat "
        "durations. Place bass accents on beat 1 only; harmony stabs on beat 3. "
        "Leave room between notes."
    ),
    "Syncopated":    (
        "RHYTHM: Syncopated. Avoid landing on main downbeats (1.0, 2.0, 3.0, 4.0). "
        "Place notes on the 'and' (X.5) and anticipate chord changes by 0.5 beats."
    ),
    "Waltz (3/4)":   (
        "RHYTHM: Waltz, 3/4 feel. Phrase in groups of 3 beats. Strong accent on "
        "beat 1, light on 2 and 3. Bass on beat 1, chord stabs on beats 2 and 3."
    ),
    "Bossa Nova":    (
        "RHYTHM: Bossa nova. Bass plays on beats 1 and 3. Harmony anticipates chord "
        "changes: hit on 1, rest on 1.5, hit on 2.5, hit on 3, rest on 3.5, hit on 4.5. "
        "Subtle, laid-back, syncopated feel throughout."
    ),
    "Funk 16ths":    (
        "RHYTHM: Funk groove. Dense 16th-note grid (0.25-beat steps). Heavy accent "
        "on beat 1. Off-beat 8th-note stabs on the 'and' (X.5). Ghost notes on "
        "X.25 and X.75 subdivisions. Lock tight with the downbeat."
    ),
}

SYSTEM_PROMPT = """\
You are an expert music arranger fluent across orchestral, jazz, blues, world, and contemporary styles.

Your task: given a melody, produce a multi-part arrangement as JSON, following the style and
instrumentation specified in the prompt.

Rules you must follow:
- Output ONLY valid JSON. No explanations, no markdown fences, nothing else.
- Every note must have: "note" (e.g. "D4"), "start_beat" (float), "duration_beats" (float), "velocity" (int 40-100).
- Respect the instrument ranges provided in the prompt.
- The first named part should carry the melody closely; remaining parts provide harmony and bass support.
- Honour the STYLE GUIDE — it determines harmonic language, rhythmic feel, and idiomatic phrasing.
- Total duration of each part should not exceed total_beats.
- Use standard note names: C, C#, D, D#, E, F, F#, G, G#, A, A#, B followed by octave number.

CRITICAL — CHORD VOICING FOR PIANO HARMONY AND GUITAR:
Chordal instruments (piano_harmony, electric_guitar) MUST voice chords as multiple
simultaneous notes. A chord voicing = multiple note objects that share the SAME start_beat and the
SAME duration_beats, one object per chord tone. Writing a SINGLE note for these instruments is WRONG.

Example — Am7 block chord voiced at beat 8.0 lasting 4 beats (write ALL FOUR notes):
  {"note":"A3","start_beat":8.0,"duration_beats":4.0,"velocity":72},
  {"note":"C4","start_beat":8.0,"duration_beats":4.0,"velocity":68},
  {"note":"E4","start_beat":8.0,"duration_beats":4.0,"velocity":72},
  {"note":"G4","start_beat":8.0,"duration_beats":4.0,"velocity":65},

If the HARMONY STYLE is arpeggio, spread the tones across the chord duration as individual notes
at different start_beats (e.g. 8.0, 8.5, 9.0, 9.5) each with short duration_beats (0.5).
Either way, ALL chord tones must appear for every chord change.\
"""


def _instrument_role_section(instruments: list[str]) -> str:
    lines = ["INSTRUMENT ROLES — follow these exactly:"]
    for inst in instruments:
        if inst in CHORDAL_INSTRUMENTS:
            lines.append(
                f"  {inst}: CHORDAL — you MUST write 3–4 simultaneous notes per chord change "
                f"(same start_beat, same duration_beats). A single note here is WRONG."
            )
        elif inst in BASS_INSTRUMENTS:
            lines.append(
                f"  {inst}: BASS — single-note line only. Root on beat 1 of each chord; "
                f"walk through chord tones on inner beats."
            )
        else:
            lines.append(f"  {inst}: MELODIC — single-note line.")
    return "\n".join(lines)


def _chord_voicing_example(beats_per_chord: float = 4.0) -> str:
    """Show a concrete chord voicing example in the prompt."""
    d = min(beats_per_chord, 4.0)
    return (
        f"  Example — Am7 voiced at beat 0.0 for {d:.1f} beats:\n"
        f'    {{"note":"A3","start_beat":0.0,"duration_beats":{d:.1f},"velocity":72}},\n'
        f'    {{"note":"C4","start_beat":0.0,"duration_beats":{d:.1f},"velocity":68}},\n'
        f'    {{"note":"E4","start_beat":0.0,"duration_beats":{d:.1f},"velocity":72}},\n'
        f'    {{"note":"G4","start_beat":0.0,"duration_beats":{d:.1f},"velocity":65}},'
    )


def _part_json_template(instruments: list[str]) -> str:
    """Per-instrument JSON examples — chordal instruments show multi-note voicings."""
    lines = []
    for p in instruments:
        if p in CHORDAL_INSTRUMENTS:
            lines.append(
                f'    "{p}": [\n'
                f'      {{"note":"A3","start_beat":0.0,"duration_beats":4.0,"velocity":72}},\n'
                f'      {{"note":"C4","start_beat":0.0,"duration_beats":4.0,"velocity":68}},\n'
                f'      {{"note":"E4","start_beat":0.0,"duration_beats":4.0,"velocity":72}},\n'
                f'      {{"note":"G4","start_beat":0.0,"duration_beats":4.0,"velocity":65}},\n'
                f'      ...\n'
                f'    ],'
            )
        elif p in BASS_INSTRUMENTS:
            lines.append(
                f'    "{p}": [{{"note":"A2","start_beat":0.0,"duration_beats":1.0,"velocity":80}}, ...],'
            )
        else:
            lines.append(
                f'    "{p}": [{{"note":"D4","start_beat":0.0,"duration_beats":1.0,"velocity":80}}, ...],'
            )
    return "\n".join(lines)


def _chord_root(symbol: str) -> str:
    """Return the root note letter(s) from a chord symbol, e.g. 'Am7' → 'A', 'Bbmaj7' → 'Bb'."""
    s = symbol.strip()
    if len(s) >= 2 and s[1] in ("#", "b"):
        return s[:2]
    return s[:1]


def build_chord_chart_from_timeline(
    chord_timeline: list[tuple[float, str]],
    total_beats: float,
) -> str:
    """
    Build a timed chord chart from a pre-built chord timeline.
    Same format as build_chord_chart() but works with song-structure timelines
    where chords don't repeat at a fixed interval.
    """
    if not chord_timeline:
        return ""
    lines = []
    for beat, chord in chord_timeline:
        root = _chord_root(chord)
        try:
            tones = ", ".join(chord_tone_names(chord))
        except Exception:
            tones = root
        try:
            root_pc   = note_name_to_midi(f"{root}2")
            bass_midi = f"{root}2={root_pc}, {root}3={root_pc+12}"
        except Exception:
            bass_midi = root
        lines.append(
            f"  Beat {beat:6.1f} → {chord:<10} "
            f"tones: {tones:<16} bass root MIDI: {bass_midi}"
        )
    return "\n".join(lines)


def build_chord_chart(
    chords: list[str],
    beats_per_chord: float,
    total_beats: float,
) -> str:
    """
    Expand chord list into a timed chart with chord tones and bass MIDI roots
    explicitly listed so the LLM knows exactly which notes are valid.
    """
    if not chords:
        return ""
    lines = []
    beat, idx = 0.0, 0
    while beat < total_beats:
        chord = chords[idx % len(chords)]
        root  = _chord_root(chord)
        try:
            tones = ", ".join(chord_tone_names(chord))
        except Exception:
            tones = root
        # Bass root MIDI numbers across two octaves (octaves 2 and 3)
        try:
            root_pc = note_name_to_midi(f"{root}2")
            bass_midi = f"{root}2={root_pc}, {root}3={root_pc+12}"
        except Exception:
            bass_midi = root
        lines.append(
            f"  Beat {beat:6.1f} → {chord:<10} "
            f"tones: {tones:<16} bass root MIDI: {bass_midi}"
        )
        beat += beats_per_chord
        idx  += 1
    return "\n".join(lines)


def build_orchestration_prompt(
    notes: list[dict],
    melody_instrument: str,
    key: str,
    tempo_bpm: float,
    total_beats: float,
    harmony_instruments: list[str],
    style_prompt: str = "",
    notes_per_beat: float = 1.0,
    chord_chart: str = "",
    chord_mode: str = "Strict",
    rhythm_style: str = "Free",
    forbidden_chords: list[str] | None = None,
    time_sig: str = "4/4",
    harmony_style: str = "Block chords",
    beats_per_chord: float = 4.0,
) -> str:
    """
    Build the user-turn prompt for an orchestration request.

    The melody is presented as fixed context — the LLM must not reproduce it.
    Only harmony_instruments are requested as output.
    """
    melody_tokens = [
        f"{n['note_name']}:{n['start_beat']:.2f}:{n['duration_beats']:.2f}"
        for n in notes
    ]
    melody_line = " | ".join(melody_tokens)

    range_lines = "\n".join(
        f"  {inst}: MIDI {INSTRUMENT_RANGES.get(inst, (36, 96))[0]}"
        f"–{INSTRUMENT_RANGES.get(inst, (36, 96))[1]}"
        for inst in harmony_instruments
    )

    part_names = ", ".join(f'"{p}"' for p in harmony_instruments)
    style_section = f"\nSTYLE GUIDE:  {style_prompt}" if style_prompt.strip() else ""

    has_chordal = any(i in CHORDAL_INSTRUMENTS for i in harmony_instruments)
    harmony_style_section = (
        f"\n\n{HARMONY_STYLES[harmony_style]}" if has_chordal and HARMONY_STYLES.get(harmony_style) else ""
    )

    if chord_chart.strip():
        voicing_example = _chord_voicing_example(beats_per_chord)
        if chord_mode == "Strict":
            chord_section = (
                f"\n\nCHORD CHART — STRICT COMPLIANCE REQUIRED.\n"
                f"Every note in every harmony and bass part MUST be a chord tone listed below.\n"
                f"Non-chord tones are NOT permitted. Each row shows the only valid notes:\n"
                f"{chord_chart}\n\n"
                f"CHORDAL INSTRUMENT VOICING (piano, guitar):\n"
                f"Voice ALL listed tones simultaneously — one note object per tone, same start_beat.\n"
                f"{voicing_example}\n"
                f"  Bass rule: use the MIDI note numbers shown in 'bass root MIDI' — root on beat 1."
            )
        else:
            chord_section = (
                f"\n\nHARMONIC PALETTE — use as your framework and improvise freely within these chords:\n"
                f"{chord_chart}\n\n"
                f"CHORDAL INSTRUMENT VOICING:\n"
                f"Voice chord tones simultaneously for piano/guitar (same start_beat).\n"
                f"{voicing_example}\n"
                f"  Bass rule: favour chord root on strong beats; walking lines welcome."
            )
    else:
        chord_section = ""

    forbidden_section = ""
    if forbidden_chords:
        lines = []
        for sym in forbidden_chords:
            try:
                tones = ", ".join(chord_tone_names(sym))
            except Exception:
                tones = sym
            lines.append(f"  {sym} ({tones})")
        forbidden_section = (
            f"\n\nFORBIDDEN CHORDS — never voice these note combinations under any circumstances:\n"
            + "\n".join(lines)
            + "\n  If the chord chart would produce one of these, re-voice to avoid it."
        )

    rhythm_section  = f"\n\n{RHYTHM_STYLES[rhythm_style]}" if RHYTHM_STYLES.get(rhythm_style) else ""
    timesig_section = f"\n\n{TIME_SIGNATURES[time_sig]['prompt']}" if time_sig in TIME_SIGNATURES and time_sig != "4/4" else ""
    target_notes    = max(8, int(total_beats * notes_per_beat))

    parsed = parse_key_string(key)
    if parsed:
        _root, _intervals = parsed
        _note_names = ", ".join(scale_note_names(_root, _intervals))
        scale_section = (
            f"\n\nSCALE — all harmony and bass notes MUST use only these pitch classes:\n"
            f"  {key} = {_note_names}"
        )
    else:
        scale_section = ""

    role_section = _instrument_role_section(harmony_instruments)
    part_template = _part_json_template(harmony_instruments)

    return f"""\
You are harmonising a fixed melody. Your job is to write the supporting parts only.

KEY:            {key}
TEMPO:          {tempo_bpm} BPM
TOTAL BEATS:    {total_beats:.1f}{style_section}{scale_section}{timesig_section}{chord_section}{forbidden_section}{harmony_style_section}{rhythm_section}

FIXED MELODY ({melody_instrument}) — do NOT include this part in your output:
{melody_line}

PARTS YOU MUST GENERATE: {part_names}
INSTRUMENT RANGES (MIDI note numbers):
{range_lines}

{role_section}

COVERAGE REQUIREMENT:
- Every part MUST reach beat {total_beats:.1f}. Do not stop early.
- Chordal instruments (piano/guitar): aim for one voicing per chord change (3–4 notes per change).
- Melodic/bass parts: aim for ~{target_notes} notes spread across the full duration.
- Do not abbreviate. Write every note explicitly.

Return ONLY this JSON — no other text:
{{
  "key": "{key}",
  "tempo": {tempo_bpm},
  "total_beats": {total_beats:.1f},
  "parts": {{
{part_template}
  }}
}}\
"""


def build_composition_prompt(
    key: str,
    tempo_bpm: float,
    total_beats: float,
    all_instruments: list[str],
    style_prompt: str = "",
    notes_per_beat: float = 1.0,
    chord_chart: str = "",
    chord_mode: str = "Strict",
    rhythm_style: str = "Free",
    forbidden_chords: list[str] | None = None,
    time_sig: str = "4/4",
    harmony_style: str = "Block chords",
    beats_per_chord: float = 4.0,
) -> str:
    """
    Build a prompt for composition from scratch — no fixed melody.
    The LLM generates all parts including the lead melodic voice.
    """
    melody_inst = all_instruments[0]
    range_lines = "\n".join(
        f"  {inst}: MIDI {INSTRUMENT_RANGES.get(inst, (36, 96))[0]}"
        f"–{INSTRUMENT_RANGES.get(inst, (36, 96))[1]}"
        for inst in all_instruments
    )
    part_names    = ", ".join(f'"{p}"' for p in all_instruments)
    style_section = f"\nSTYLE GUIDE:  {style_prompt}" if style_prompt.strip() else ""

    has_chordal = any(i in CHORDAL_INSTRUMENTS for i in all_instruments)
    harmony_style_section = (
        f"\n\n{HARMONY_STYLES[harmony_style]}" if has_chordal and HARMONY_STYLES.get(harmony_style) else ""
    )

    if chord_chart.strip():
        voicing_example = _chord_voicing_example(beats_per_chord)
        if chord_mode == "Strict":
            chord_section = (
                f"\n\nCHORD CHART — STRICT COMPLIANCE REQUIRED.\n"
                f"Every note in every part MUST be a chord tone listed below.\n"
                f"{chord_chart}\n\n"
                f"CHORDAL INSTRUMENT VOICING (piano, guitar):\n"
                f"Voice ALL listed tones simultaneously — one note object per tone, same start_beat.\n"
                f"{voicing_example}\n"
                f"  Melody rule:  use chord tones as structural notes; ornament sparingly.\n"
                f"  Bass rule:    use the MIDI note numbers shown in 'bass root MIDI' on beat 1."
            )
        else:
            chord_section = (
                f"\n\nHARMONIC PALETTE — improvise freely within these chords:\n"
                f"{chord_chart}\n\n"
                f"CHORDAL INSTRUMENT VOICING:\n"
                f"Voice chord tones simultaneously for piano/guitar (same start_beat).\n"
                f"{voicing_example}\n"
                f"  Melody:  use chord tones as structural pillars; passing tones welcome.\n"
                f"  Bass:    favour chord root on strong beats; walking lines welcome."
            )
    else:
        chord_section = ""

    forbidden_section = ""
    if forbidden_chords:
        lines = []
        for sym in forbidden_chords:
            try:
                tones = ", ".join(chord_tone_names(sym))
            except Exception:
                tones = sym
            lines.append(f"  {sym} ({tones})")
        forbidden_section = (
            f"\n\nFORBIDDEN CHORDS — never voice these combinations:\n"
            + "\n".join(lines)
        )

    rhythm_section  = f"\n\n{RHYTHM_STYLES[rhythm_style]}" if RHYTHM_STYLES.get(rhythm_style) else ""
    timesig_section = f"\n\n{TIME_SIGNATURES[time_sig]['prompt']}" if time_sig in TIME_SIGNATURES and time_sig != "4/4" else ""
    target_notes    = max(8, int(total_beats * notes_per_beat))

    parsed = parse_key_string(key)
    if parsed:
        _root, _intervals = parsed
        _note_names = ", ".join(scale_note_names(_root, _intervals))
        scale_section = (
            f"\n\nSCALE — use ONLY these pitch classes in ALL parts, including melody:\n"
            f"  {key} = {_note_names}\n"
            f"  Any note not in this list is FORBIDDEN regardless of octave."
        )
    else:
        scale_section = ""

    role_section  = _instrument_role_section(all_instruments)
    part_template = _part_json_template(all_instruments)

    return f"""\
You are composing an original piece from scratch. Generate ALL parts listed below.

KEY:            {key}
TEMPO:          {tempo_bpm} BPM
TOTAL BEATS:    {total_beats:.1f}{style_section}{scale_section}{timesig_section}{chord_section}{forbidden_section}{harmony_style_section}{rhythm_section}

PARTS TO GENERATE: {part_names}
INSTRUMENT RANGES (MIDI note numbers):
{range_lines}

{role_section}

COMPOSITION GUIDELINES:
- {melody_inst} carries the main melodic line. Give it a singable, memorable shape with a clear arc.
- Chordal instruments: voice full chords at every chord change (3–4 simultaneous notes).
- Bass instruments: single-note line, root on beat 1 of each chord.
- Create a complete musical phrase: opening, development, and cadence.
- Every part MUST reach beat {total_beats:.1f}. Do not stop early.
- Melodic/bass parts: aim for ~{target_notes} notes spread across the full duration.
- Do not abbreviate. Write every note explicitly.

Return ONLY this JSON — no other text:
{{
  "key": "{key}",
  "tempo": {tempo_bpm},
  "total_beats": {total_beats:.1f},
  "parts": {{
{part_template}
  }}
}}\
"""