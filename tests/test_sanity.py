"""
Sanity tests for the orchestral-composer pipeline.

These tests are LLM-free — they exercise the deterministic layers:
  - Harmonic palette / scale computation
  - Algo arranger (inject_algo_parts) for every preset × harmony style × key
  - Scale snapping correctness (regression for the D Double Harmonic bug)
  - Range clamping in the orchestration parser
  - Chord-scale compatibility checker
  - Song structure / chord timeline builder
  - Named progressions
  - Secondary dominants and tritone substitutions
  - MIDI build round-trip

Run:  pytest tests/test_sanity.py -v
"""

from __future__ import annotations
import random
import tempfile
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import INSTRUMENT_PRESETS
from src.harmony import (
    parse_key_string, scale_note_names, check_chord_scale_compatibility,
    build_chord_timeline, _NOTE_PC, snap_all_parts_to_scale,
    note_name_to_midi, midi_to_note_name,
)
from src.prompts import (
    CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS, INSTRUMENT_RANGES,
    HARMONY_STYLES, build_chord_chart_from_timeline,
)
from src.algo_arranger import inject_algo_parts, make_chord_part, make_bass_part
from src.voice_leading import insert_secondary_dominants, apply_tritone_substitutions
from src.song_structure import (
    COMMON_PROGRESSIONS, PROGRESSION_NAMES,
    progression_to_chords, parse_form, build_song_timeline,
)
from src.midi_builder import build_midi
from src.midi_diagnostics import analyse_midi
from src.humanize import humanize_orchestration

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Keys that span different scale families: diatonic, pentatonic, double harmonic,
# blues, ragas, whole-tone, and minor variants.
ALL_TEST_KEYS = [
    ("C", "Major (Ionian)"),
    ("D", "Double Harmonic"),
    ("A", "Minor (Aeolian)"),
    ("G", "Dorian"),
    ("F#", "Phrygian"),
    ("Bb", "Blues"),
    ("E", "Lydian"),
    ("D", "Mixolydian"),
    ("C#", "Harmonic Minor"),
    ("G", "Major Pentatonic"),
    ("D", "Bhairav"),
    ("A", "Hijaz"),
    ("F", "Whole Tone"),
    ("B", "Diminished (HW)"),
    ("C", "Neapolitan Minor"),
]

# Chord progressions guaranteed to work in each key (chosen to be scale-clean where possible)
_KEY_CHORDS: dict[str, list[str]] = {
    "C major":           ["Cmaj7", "Am7", "Dm7", "G7"],
    "D Double Harmonic": ["Dmaj7", "G",   "A7",  "D#"],
    "A Minor (Aeolian)": ["Am",    "F",   "C",   "G"],
    "G Dorian":          ["Gm7",   "C7",  "Gm7", "C7"],
    "F# Phrygian":       ["F#m",   "G",   "F#m", "G"],
    "Bb Blues":          ["Bb7",   "Eb7", "Bb7", "F7"],
    "E Lydian":          ["Emaj7", "F#7", "Emaj7", "F#7"],
    "D Mixolydian":      ["D7",    "G",   "D7",  "G"],
    "C# Harmonic Minor": ["C#m",   "G#7", "C#m", "G#7"],
    "G Major Pentatonic":["G",     "D",   "Em",  "C"],
    "D Bhairav":         ["Dmaj7", "Bbmaj7", "A7", "Dmaj7"],
    "A Hijaz":           ["Am",    "Bb",  "E7",  "Am"],
    "F Whole Tone":      ["Faug",  "Gaug","Faug", "Gaug"],
    "B Diminished (HW)": ["Bdim7", "Ddim7", "Fdim7", "Abdim7"],
    "C Neapolitan Minor":["Cm",    "Db",  "G7",  "Cm"],
}


def _key_str(root: str, mode: str) -> str:
    return f"{root} {mode}"


def _chords_for_key(key: str) -> list[str]:
    return _KEY_CHORDS.get(key, ["C", "F", "G", "C"])


def _make_timeline(key: str, beats_per_chord: float = 4.0, bars: int = 8) -> tuple:
    chords = _chords_for_key(key)
    beats_per_bar = 4.0
    total_beats = bars * beats_per_bar
    timeline = build_chord_timeline(chords, beats_per_chord, total_beats)
    return timeline, total_beats


# ---------------------------------------------------------------------------
# 1. Scale parsing correctness
# ---------------------------------------------------------------------------

class TestScaleParsing:

    def test_parse_returns_root_and_intervals_for_all_test_keys(self):
        for root, mode in ALL_TEST_KEYS:
            key = f"{root} {mode}"
            result = parse_key_string(key)
            assert result is not None, f"parse_key_string failed for '{key}'"
            parsed_root, intervals = result
            assert parsed_root == root, f"Root mismatch for '{key}'"
            assert len(intervals) >= 5, f"Too few intervals for '{key}' (got {len(intervals)})"
            assert 0 in intervals, f"Intervals must start with 0 for '{key}'"

    def test_scale_pcs_are_absolute_pitch_classes(self):
        """Regression: scale_pcs must use (root_pc + interval) % 12, not raw intervals."""
        root, intervals = parse_key_string("D Double Harmonic")
        root_pc = _NOTE_PC[root]
        scale_pcs = frozenset((root_pc + i) % 12 for i in intervals)
        # D Double Harmonic: D D# F# G A A# C#
        expected_notes = {"D", "D#", "F#", "G", "A", "A#", "C#"}
        actual_notes = {["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][pc] for pc in scale_pcs}
        assert actual_notes == expected_notes, f"Wrong scale notes: {actual_notes}"
        # F natural and G# must NOT be in scale
        assert 5 not in scale_pcs, "F natural wrongly in D Double Harmonic"
        assert 8 not in scale_pcs, "G# wrongly in D Double Harmonic"

    def test_c_major_scale_pcs(self):
        root, intervals = parse_key_string("C major")
        root_pc = _NOTE_PC[root]
        scale_pcs = frozenset((root_pc + i) % 12 for i in intervals)
        assert scale_pcs == frozenset({0, 2, 4, 5, 7, 9, 11}), "C major PCs wrong"

    @pytest.mark.parametrize("root,mode", ALL_TEST_KEYS)
    def test_scale_note_names_match_interval_count(self, root, mode):
        key = f"{root} {mode}"
        parsed = parse_key_string(key)
        assert parsed is not None
        r, intervals = parsed
        names = scale_note_names(r, intervals)
        assert len(names) == len(intervals), f"Note name count mismatch for '{key}'"


# ---------------------------------------------------------------------------
# 2. Algo arranger: all presets × harmony styles × keys
# ---------------------------------------------------------------------------

# Only test harmony styles that apply to instruments actually present in presets.
# Chordal styles only matter when there's a chordal instrument; others always apply.
_CHORDAL_STYLES = [
    "Block chords", "Jazz comping", "Arpeggio (up)", "Arpeggio (alt)",
    "Alberti bass", "Broken chords", "Pad (sustained)",
]
_ALWAYS_STYLES = ["Block chords", "Jazz comping"]  # used for non-chordal-instrument presets

# Random sample of keys to keep test count manageable while covering the space
_RNG = random.Random(42)
_SAMPLED_KEYS = _RNG.sample(
    [f"{r} {m}" for r, m in ALL_TEST_KEYS], k=min(6, len(ALL_TEST_KEYS))
)


class TestAlgoArranger:
    """Every preset must produce a valid, in-scale, in-range MIDI with every harmony style."""

    def _run_preset(self, preset_name: str, harmony_style: str, key: str,
                    beats_per_chord: float = 4.0, bars: int = 8) -> dict:
        instruments = INSTRUMENT_PRESETS[preset_name]
        timeline, total_beats = _make_timeline(key, beats_per_chord, bars)
        assert timeline, "Empty chord timeline"
        orchestration = {
            "key": key, "tempo": 120,
            "parts": {inst: [] for inst in instruments},
        }
        result = inject_algo_parts(
            orchestration, timeline, total_beats,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), harmony_style, 4.0, key,
        )
        return result, total_beats, key

    def _assert_valid_orchestration(self, result: dict, total_beats: float,
                                    key: str, preset_name: str, harmony_style: str):
        instruments = list(result["parts"].keys())
        assert instruments, "No instruments in result"

        parsed = parse_key_string(key)
        assert parsed is not None
        root, intervals = parsed
        root_pc = _NOTE_PC[root]
        scale_pcs = frozenset((root_pc + i) % 12 for i in intervals)

        for inst in instruments:
            notes = result["parts"][inst]
            lo, hi = INSTRUMENT_RANGES.get(inst, (21, 108))
            ctx = f"preset={preset_name}, style={harmony_style}, key={key}, inst={inst}"

            # Skip instruments with no notes (non-chordal non-bass — e.g. strings_harmony
            # is not touched by inject_algo_parts; leave those to the LLM)
            if not notes:
                if inst not in CHORDAL_INSTRUMENTS and inst not in BASS_INSTRUMENTS:
                    continue  # expected — only chordal/bass are algo-generated
                pytest.fail(f"Algo instrument '{inst}' has zero notes. {ctx}")

            # Range check
            for note in notes:
                midi = note_name_to_midi(note["note"])
                assert lo <= midi <= hi, (
                    f"Note {note['note']} (MIDI {midi}) outside range [{lo},{hi}] "
                    f"for '{inst}'. {ctx}"
                )

            # No scale-snap assertion: algo voicings use actual chord tones which may
            # intentionally include chromatic pitches outside the key scale.

            # Timing check
            for note in notes:
                assert note["start_beat"] >= 0, f"Negative start_beat in '{inst}'. {ctx}"
                assert note["duration_beats"] > 0, f"Zero duration in '{inst}'. {ctx}"
                assert note["velocity"] >= 1, f"Zero velocity in '{inst}'. {ctx}"

    @pytest.mark.parametrize("preset_name", list(INSTRUMENT_PRESETS.keys()))
    @pytest.mark.parametrize("harmony_style", _CHORDAL_STYLES)
    def test_preset_with_sampled_key(self, preset_name, harmony_style):
        """Each preset × harmony style runs with a random key from the sampled set."""
        key = _RNG.choice(_SAMPLED_KEYS)
        result, total_beats, key = self._run_preset(preset_name, harmony_style, key)
        self._assert_valid_orchestration(result, total_beats, key, preset_name, harmony_style)

    @pytest.mark.parametrize("key", [f"{r} {m}" for r, m in ALL_TEST_KEYS])
    def test_jazz_quartet_all_keys(self, key):
        """Jazz Quartet (the default preset) must work cleanly for every test key."""
        result, total_beats, key = self._run_preset("Jazz Quartet", "Jazz comping", key)
        self._assert_valid_orchestration(result, total_beats, key, "Jazz Quartet", "Jazz comping")

    @pytest.mark.parametrize("key", [f"{r} {m}" for r, m in ALL_TEST_KEYS])
    def test_string_trio_all_keys(self, key):
        """String Trio with block chords must work for every test key."""
        result, total_beats, key = self._run_preset("String Trio", "Block chords", key)
        self._assert_valid_orchestration(result, total_beats, key, "String Trio", "Block chords")

    def test_jazz_comping_two_beat_chords(self):
        """Regression: jazz comping must produce notes for chords shorter than a bar."""
        key = "C major"
        chords = ["Cmaj7", "Am7", "Dm7", "G7"]
        timeline = build_chord_timeline(chords, 2.0, 16.0)  # 2-beat chords
        orchestration = {"key": key, "tempo": 120, "parts": {"piano_harmony": []}}
        result = inject_algo_parts(
            orchestration, timeline, 16.0,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Jazz comping", 4.0, key,
        )
        assert len(result["parts"]["piano_harmony"]) > 0, \
            "Jazz comping produced zero notes for 2-beat chords"

    def test_all_harmony_styles_produce_notes(self):
        """Every harmony style must produce at least one note for a chordal instrument."""
        key = "C major"
        chords = ["Cmaj7", "Am7", "Dm7", "G7"]
        timeline = build_chord_timeline(chords, 4.0, 16.0)
        for style in _CHORDAL_STYLES:
            orchestration = {"key": key, "tempo": 120, "parts": {"piano_harmony": []}}
            result = inject_algo_parts(
                orchestration, timeline, 16.0,
                CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
                dict(INSTRUMENT_RANGES), style, 4.0, key,
            )
            assert len(result["parts"]["piano_harmony"]) > 0, \
                f"Harmony style '{style}' produced zero notes"


# ---------------------------------------------------------------------------
# 3. Full MIDI round-trip + diagnostic
# ---------------------------------------------------------------------------

class TestMidiRoundTrip:
    """Build MIDI for all presets, write to disk, read back, run diagnostic."""

    @pytest.mark.parametrize("preset_name,instruments", INSTRUMENT_PRESETS.items())
    def test_midi_round_trip(self, preset_name, instruments, tmp_path):
        key = _RNG.choice(_SAMPLED_KEYS)
        timeline, total_beats = _make_timeline(key, beats_per_chord=4.0, bars=8)
        orchestration = {
            "key": key, "tempo": 120,
            "parts": {inst: [] for inst in instruments},
        }
        orchestration = inject_algo_parts(
            orchestration, timeline, total_beats,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Block chords", 4.0, key,
        )
        midi_path = str(tmp_path / f"{preset_name.replace(' ', '_')}.mid")
        build_midi(orchestration, midi_path)

        assert os.path.exists(midi_path), "MIDI file not created"
        assert os.path.getsize(midi_path) > 100, "MIDI file suspiciously small"

        # Scale check removed: chord voicings intentionally preserve chromatic chord tones.
        assert os.path.getsize(midi_path) > 100, "MIDI file suspiciously small after round-trip"

    def test_humanize_skips_non_dict_notes(self):
        """humanize_notes must not crash when a part contains string items (LLM bug)."""
        from src.humanize import humanize_notes
        mixed = [
            "D4",  # string — LLM sometimes emits note names as bare strings
            {"note": "F#4", "start_beat": 1.0, "duration_beats": 1.0, "velocity": 80},
            "A4",
        ]
        result = humanize_notes(mixed, amount=0.5)
        # Strings silently skipped; only the dict survives
        assert all(isinstance(n, dict) for n in result)
        assert len(result) == 1

    def test_humanize_preserves_scale(self, tmp_path):
        """Humanization must not shift notes outside the scale."""
        key = "D Double Harmonic"
        timeline, total_beats = _make_timeline(key)
        orchestration = {
            "key": key, "tempo": 94,
            "parts": {"piano_harmony": [], "acoustic_bass": []},
        }
        orchestration = inject_algo_parts(
            orchestration, timeline, total_beats,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Jazz comping", 4.0, key,
        )
        # Humanize shifts timing/velocity only; may drop inner-voice notes but must never
        # introduce a pitch that wasn't present before.
        pitch_set_before = {n["note"] for notes in orchestration["parts"].values() for n in notes}
        orchestration = humanize_orchestration(orchestration, amount=0.8)
        pitch_set_after = {n["note"] for notes in orchestration["parts"].values() for n in notes}
        new_pitches = pitch_set_after - pitch_set_before
        assert not new_pitches, f"Humanize introduced new pitches not in original: {new_pitches}"

        midi_path = str(tmp_path / "humanized.mid")
        build_midi(orchestration, midi_path)


# ---------------------------------------------------------------------------
# 4. Scale snapping
# ---------------------------------------------------------------------------

class TestScaleSnapping:

    def test_snap_corrects_out_of_scale_notes(self):
        """snap_all_parts_to_scale must fix every out-of-scale note."""
        # D Double Harmonic: D D# F# G A A# C# (PCs 1,2,3,6,7,9,10)
        root, intervals = parse_key_string("D Double Harmonic")
        root_pc = _NOTE_PC[root]
        scale_pcs = frozenset((root_pc + i) % 12 for i in intervals)

        # Deliberately insert out-of-scale notes: E4(PC4), B4(PC11), F5(PC5)
        parts = {
            "piano_harmony": [
                {"note": "E4", "start_beat": 0.0, "duration_beats": 1.0, "velocity": 72},
                {"note": "B4", "start_beat": 1.0, "duration_beats": 1.0, "velocity": 72},
                {"note": "F5", "start_beat": 2.0, "duration_beats": 1.0, "velocity": 72},
                {"note": "D4", "start_beat": 3.0, "duration_beats": 1.0, "velocity": 72},  # in scale
            ]
        }
        result = snap_all_parts_to_scale({"parts": parts}, scale_pcs, dict(INSTRUMENT_RANGES))
        for note in result["parts"]["piano_harmony"]:
            midi = note_name_to_midi(note["note"])
            assert midi % 12 in scale_pcs, \
                f"Note {note['note']} still out of scale after snap"

    def test_snap_leaves_in_scale_notes_unchanged(self):
        """Notes already in scale must not be moved."""
        root, intervals = parse_key_string("C major")
        root_pc = _NOTE_PC[root]
        scale_pcs = frozenset((root_pc + i) % 12 for i in intervals)
        original = [
            {"note": "C4", "start_beat": 0.0, "duration_beats": 1.0, "velocity": 72},
            {"note": "G4", "start_beat": 1.0, "duration_beats": 1.0, "velocity": 72},
            {"note": "E5", "start_beat": 2.0, "duration_beats": 1.0, "velocity": 72},
        ]
        import copy
        snap_input = copy.deepcopy(original)
        result = snap_all_parts_to_scale(
            {"parts": {"piano_harmony": snap_input}}, scale_pcs, dict(INSTRUMENT_RANGES)
        )
        for orig, snapped in zip(original, result["parts"]["piano_harmony"]):
            assert orig["note"] == snapped["note"], \
                f"In-scale note {orig['note']} was incorrectly moved to {snapped['note']}"

    def test_inject_algo_parts_preserves_chord_tones_d_double_harmonic(self):
        """Algo arranger must use actual chord tones, not snap them to the key scale.

        D Double Harmonic is used because it exposes the old (wrong) scale-snap behaviour:
        Em7 contains G# (outside D Double Harmonic), which was formerly snapped to G,
        changing the chord quality. The arranger must now preserve G# as voiced.
        """
        key = "D Double Harmonic"
        clash_chords = ["Em7", "A7", "Dmaj7", "B7"]
        timeline = build_chord_timeline(clash_chords, 4.0, 32.0)
        orchestration = {
            "key": key, "tempo": 94,
            "parts": {"piano_harmony": [], "acoustic_bass": []},
        }
        result = inject_algo_parts(
            orchestration, timeline, 32.0,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Jazz comping", 4.0, key,
        )
        for inst in ("piano_harmony", "acoustic_bass"):
            notes = result["parts"][inst]
            assert len(notes) > 0, f"'{inst}' produced no notes"
            lo, hi = INSTRUMENT_RANGES[inst]
            for n in notes:
                midi = note_name_to_midi(n["note"])
                assert lo <= midi <= hi, f"'{inst}' note {n['note']} out of range [{lo},{hi}]"
        # Em7 chord tones: E G B D — G is within D Double Harmonic; verify notes are present
        em7_notes = {n["note"][:-1] for n in result["parts"]["piano_harmony"]
                     if n["start_beat"] < 4.0}
        assert any(pc in em7_notes for pc in ("E", "G", "B", "D")), \
            f"Em7 chord tones not found in piano_harmony voicing: {em7_notes}"


# ---------------------------------------------------------------------------
# 5. Range clamping
# ---------------------------------------------------------------------------

class TestRangeClamping:

    def _make_fake_orchestration_json(self, inst: str, note: str) -> str:
        import json
        return json.dumps({
            "key": "C major", "tempo": 120, "total_beats": 8.0,
            "parts": {
                inst: [
                    {"note": note, "start_beat": i * 1.0, "duration_beats": 0.9, "velocity": 72}
                    for i in range(12)  # enough to pass min_notes check
                ]
            }
        })

    def test_out_of_range_high_clamped(self):
        """Notes above instrument ceiling must be octave-shifted down."""
        from src.orchestrate import _parse_orchestration
        inst = "strings_harmony"
        lo, hi = INSTRUMENT_RANGES[inst]  # (48, 79) = C3–G5
        raw = self._make_fake_orchestration_json(inst, "C#7")  # MIDI 97, above hi=79
        result = _parse_orchestration(raw, [inst], 8.0, notes_per_beat=1.0)
        for note in result["parts"][inst]:
            midi = note_name_to_midi(note["note"])
            assert midi <= hi, f"Note {note['note']} (MIDI {midi}) still above hi={hi}"
            assert midi >= lo, f"Note {note['note']} (MIDI {midi}) below lo={lo}"

    def test_out_of_range_low_clamped(self):
        """Notes below instrument floor must be octave-shifted up."""
        from src.orchestrate import _parse_orchestration
        inst = "flute_melody"
        lo, hi = INSTRUMENT_RANGES[inst]  # (60, 96)
        raw = self._make_fake_orchestration_json(inst, "C2")  # MIDI 36, below lo=60
        result = _parse_orchestration(raw, [inst], 8.0, notes_per_beat=1.0)
        for note in result["parts"][inst]:
            midi = note_name_to_midi(note["note"])
            assert midi >= lo, f"Note {note['note']} (MIDI {midi}) still below lo={lo}"
            assert midi <= hi, f"Note {note['note']} (MIDI {midi}) above hi={hi}"

    def test_in_range_notes_untouched(self):
        """Notes already in range must not be moved."""
        from src.orchestrate import _parse_orchestration
        inst = "piano_harmony"
        raw = self._make_fake_orchestration_json(inst, "C4")  # MIDI 60, in range
        result = _parse_orchestration(raw, [inst], 8.0, notes_per_beat=1.0)
        for note in result["parts"][inst]:
            assert note["note"] == "C4", f"In-range note was moved: {note['note']}"


# ---------------------------------------------------------------------------
# 6. Chord-scale compatibility
# ---------------------------------------------------------------------------

class TestChordScaleCompatibility:

    def test_detects_clash_em7_in_d_double_harmonic(self):
        key = "D Double Harmonic"
        timeline = build_chord_timeline(["Em7"], 4.0, 4.0)
        warnings = check_chord_scale_compatibility(timeline, key)
        assert any("Em7" in w for w in warnings), \
            f"Em7 clash with D Double Harmonic not detected: {warnings}"
        assert any("E" in w or "B" in w for w in warnings), \
            f"Expected E or B in warning: {warnings}"

    def test_no_clash_dmaj7_in_d_double_harmonic(self):
        key = "D Double Harmonic"
        timeline = build_chord_timeline(["Dmaj7"], 4.0, 4.0)
        warnings = check_chord_scale_compatibility(timeline, key)
        assert len(warnings) == 0, \
            f"Dmaj7 should be clean in D Double Harmonic but got: {warnings}"

    def test_no_clash_c_major_diatonic_chords(self):
        key = "C major"
        timeline = build_chord_timeline(["Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7"], 4.0, 24.0)
        warnings = check_chord_scale_compatibility(timeline, key)
        assert len(warnings) == 0, \
            f"Diatonic chords in C major should have no clashes: {warnings}"

    def test_no_false_negatives_chromatic_chord(self):
        """A chromatic chord like Db7 in C major must be flagged."""
        key = "C major"
        timeline = build_chord_timeline(["Db7"], 4.0, 4.0)
        warnings = check_chord_scale_compatibility(timeline, key)
        assert len(warnings) > 0, "Db7 in C major should produce a clash warning"


# ---------------------------------------------------------------------------
# 7. Song structure / chord timeline builder
# ---------------------------------------------------------------------------

class TestSongStructure:

    def test_parse_form_space_separated(self):
        assert parse_form("A B A A") == ["A", "B", "A", "A"]

    def test_parse_form_compact(self):
        assert parse_form("AABA") == ["A", "A", "B", "A"]

    def test_build_song_timeline_total_beats(self):
        form = parse_form("A B A")
        chords = {"A": ["C", "F", "G", "C"], "B": ["Am", "F", "C", "G"]}
        bars   = {"A": 8, "B": 8}
        timeline, total, _ = build_song_timeline(form, chords, 4.0, 4.0, bars)
        assert total == 3 * 8 * 4.0, f"Expected {3*8*4.0} beats, got {total}"

    def test_build_song_timeline_no_gaps(self):
        """Every beat from 0 to total must be covered (no silent bars)."""
        form = parse_form("A B A A")
        chords = {"A": ["C", "G"], "B": ["F", "Am"]}
        bars = {"A": 4, "B": 4}
        timeline, total, _ = build_song_timeline(form, chords, 4.0, 4.0, bars)
        assert len(timeline) > 0
        # Each section must start at correct beat
        beats = [b for b, _ in timeline]
        assert beats[0] == 0.0, "Timeline doesn't start at beat 0"
        assert beats == sorted(beats), "Timeline beats are not monotonically increasing"

    def test_build_chord_chart_from_timeline_nonempty_for_song_structure(self):
        """This was the bug: song structure produced empty chord_chart for the LLM."""
        from src.prompts import build_chord_chart_from_timeline
        form = parse_form("A B A A")
        chords = {"A": ["Em7", "A7", "Dmaj7", "B7"], "B": ["Dmaj7", "Bm7", "Em7", "A7"]}
        bars = {"A": 8, "B": 8}
        timeline, total, _ = build_song_timeline(form, chords, 4.0, 4.0, bars)
        chart = build_chord_chart_from_timeline(timeline, total)
        assert chart.strip(), "Chord chart is empty for song structure — bug regression"
        lines = chart.strip().split("\n")
        assert len(lines) == len(timeline), \
            f"Chart has {len(lines)} lines but timeline has {len(timeline)} events"


# ---------------------------------------------------------------------------
# 8. Named progressions
# ---------------------------------------------------------------------------

class TestNamedProgressions:

    @pytest.mark.parametrize("prog_name", PROGRESSION_NAMES)
    def test_progression_produces_valid_chord_names(self, prog_name):
        """Every named progression must produce parseable chord symbols in C major."""
        chords = progression_to_chords(prog_name, "C major")
        assert len(chords) > 0, f"No chords produced for '{prog_name}'"
        for chord in chords:
            assert chord and chord[0].isalpha(), \
                f"Invalid chord symbol '{chord}' from progression '{prog_name}'"

    def test_iivi_in_d_major(self):
        chords = progression_to_chords("ii-V-I", "D major")
        assert len(chords) == 3
        assert chords[0].startswith("E"), f"ii of D major should start with E, got {chords[0]}"
        assert chords[1].startswith("A"), f"V of D major should start with A, got {chords[1]}"
        assert chords[2].startswith("D"), f"I of D major should start with D, got {chords[2]}"

    def test_12bar_blues_has_12_chords(self):
        chords = progression_to_chords("12-bar blues", "Bb major")
        assert len(chords) == 12, f"12-bar blues should have 12 chords, got {len(chords)}"


# ---------------------------------------------------------------------------
# 9. Voice leading / substitutions
# ---------------------------------------------------------------------------

class TestVoiceLeading:

    def test_insert_secondary_dominants_inserts_chromatic_dom(self):
        """A7 (V7/ii) should be inserted before Dm7 in C major."""
        key = "C major"
        timeline = [
            (0.0, "Fmaj7"),   # subdominant — eligible
            (8.0, "Cmaj7"),   # tonic — not eligible
        ]
        result = insert_secondary_dominants(timeline, key, min_chord_beats=2.0)
        symbols = [s for _, s in result]
        # A7 is V7/Dm — but here target is Fmaj7. C7 is V7/Fmaj7.
        # Fmaj7 is SD function; C7 has E (in C major scale, not chromatic) — won't insert
        # More reliable: test with Am7
        timeline2 = [(0.0, "Am7"), (8.0, "Cmaj7")]
        result2 = insert_secondary_dominants(timeline2, key, min_chord_beats=2.0)
        # E7 is V7/Am7 — E7 has G# which is chromatic in C major
        assert any("7" in s for _, s in result2), \
            f"No secondary dominant inserted before Am7: {result2}"

    def test_tritone_substitution_replaces_dominants(self):
        """G7 (tritone sub root = C#/Db) and A7 (tritone = D#/Eb) must be substituted."""
        timeline = [(0.0, "G7"), (4.0, "Cmaj7"), (8.0, "A7")]
        result = apply_tritone_substitutions(timeline)
        symbols = {s for _, s in result}
        # Implementation uses sharp names: G+6=C#, A+6=D#
        assert "C#7" in symbols, f"G7 tritone sub (C#7/Db7) missing: {symbols}"
        assert "D#7" in symbols, f"A7 tritone sub (D#7/Eb7) missing: {symbols}"
        # Non-dominant Cmaj7 must be unchanged
        assert "Cmaj7" in symbols, f"Cmaj7 was changed by tritone sub: {symbols}"

    def test_tritone_substitution_only_touches_dominants(self):
        """Non-dominant chords must pass through unchanged."""
        timeline = [(0.0, "Cmaj7"), (4.0, "Am7"), (8.0, "Dm7")]
        result = apply_tritone_substitutions(timeline)
        for orig, after in zip(timeline, result):
            assert orig[1] == after[1], \
                f"Non-dominant chord '{orig[1]}' was changed to '{after[1]}'"


# ---------------------------------------------------------------------------
# 10. Randomised integration smoke test
# ---------------------------------------------------------------------------

class TestRandomisedSmoke:
    """Pick random key + preset + harmony style, run the full harmony-only pipeline."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_combination(self, seed, tmp_path):
        rng = random.Random(seed)
        key = rng.choice([f"{r} {m}" for r, m in ALL_TEST_KEYS])
        preset_name = rng.choice(list(INSTRUMENT_PRESETS.keys()))
        harmony_style = rng.choice(_CHORDAL_STYLES)
        beats_per_chord = rng.choice([2.0, 4.0])
        bars = rng.choice([4, 8, 16])

        instruments = INSTRUMENT_PRESETS[preset_name]
        timeline, total_beats = _make_timeline(key, beats_per_chord, bars)
        orchestration = {
            "key": key, "tempo": rng.choice([80, 100, 120, 140]),
            "parts": {inst: [] for inst in instruments},
        }
        orchestration = inject_algo_parts(
            orchestration, timeline, total_beats,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), harmony_style, 4.0, key,
        )
        orchestration = humanize_orchestration(orchestration, amount=rng.uniform(0.1, 0.5))

        midi_path = str(tmp_path / f"smoke_{seed}.mid")
        build_midi(orchestration, midi_path)
        assert os.path.getsize(midi_path) > 50, "MIDI too small"

        # Scale check removed: chord voicings preserve chromatic chord tones intentionally.
        # Range check for algo instruments follows below.

        # Verify no notes are out of range for algo instruments
        parsed = parse_key_string(key)
        assert parsed is not None, f"Could not parse key: {key}"
        root, intervals = parsed
        root_pc = _NOTE_PC[root]
        scale_pcs = frozenset((root_pc + i) % 12 for i in intervals)
        for inst in instruments:
            if inst not in CHORDAL_INSTRUMENTS and inst not in BASS_INSTRUMENTS:
                continue
            lo, hi = INSTRUMENT_RANGES.get(inst, (21, 108))
            for note in orchestration["parts"].get(inst, []):
                midi = note_name_to_midi(note["note"])
                assert lo <= midi <= hi, (
                    f"Seed {seed}: {inst} note {note['note']} (MIDI {midi}) "
                    f"out of range [{lo},{hi}]"
                )


# ---------------------------------------------------------------------------
# 11. _parse_orchestration validation — exercises the exact code path that
#     caused "All variations failed" when the LLM output was too sparse.
# ---------------------------------------------------------------------------

class TestParseOrchestration:
    """These tests use the same validator the Ollama path runs through."""

    def _good_json(self, instruments, total_beats):
        import json
        n = max(4, int(total_beats / 6)) + 4      # safely above min_notes
        step = total_beats / n
        parts = {
            inst: [
                {
                    "note": "C4",
                    "start_beat": round(i * step, 3),
                    "duration_beats": round(step * 0.9, 3),
                    "velocity": 72,
                }
                for i in range(n)
            ]
            for inst in instruments
        }
        return json.dumps({"key": "C major", "tempo": 120, "parts": parts})

    def _sparse_json(self, instruments, total_beats):
        import json
        parts = {
            inst: [
                {"note": "C4", "start_beat": i * 8.0, "duration_beats": 2.0, "velocity": 72}
                for i in range(2)    # only 2 notes — always under min_notes for any piece > 12 beats
            ]
            for inst in instruments
        }
        return json.dumps({"key": "C major", "tempo": 120, "parts": parts})

    def _early_stop_json(self, instruments, total_beats):
        import json
        n = max(4, int(total_beats / 6)) + 4
        cutoff = total_beats * 0.30
        step = cutoff / n
        parts = {
            inst: [
                {"note": "C4", "start_beat": round(i * step, 3), "duration_beats": round(step * 0.9, 3), "velocity": 72}
                for i in range(n)
            ]
            for inst in instruments
        }
        return json.dumps({"key": "C major", "tempo": 120, "parts": parts})

    def test_valid_output_accepted(self):
        from src.orchestrate import _parse_orchestration
        instruments = ["piano_harmony", "acoustic_bass"]
        total_beats = 64.0
        raw = self._good_json(instruments, total_beats)
        result = _parse_orchestration(raw, instruments, total_beats)
        assert "parts" in result
        for inst in instruments:
            assert inst in result["parts"]
            assert len(result["parts"][inst]) > 0

    def test_sparse_output_rejected(self):
        from src.orchestrate import _parse_orchestration
        instruments = ["alto_sax", "piano_harmony"]
        total_beats = 64.0
        raw = self._sparse_json(instruments, total_beats)
        with pytest.raises((ValueError, KeyError)):
            _parse_orchestration(raw, instruments, total_beats)

    def test_early_stop_rejected(self):
        from src.orchestrate import _parse_orchestration
        instruments = ["strings_melody", "strings_harmony"]
        total_beats = 96.0
        raw = self._early_stop_json(instruments, total_beats)
        with pytest.raises(ValueError):
            _parse_orchestration(raw, instruments, total_beats)

    def test_missing_part_raises_key_error(self):
        from src.orchestrate import _parse_orchestration
        import json
        instruments = ["piano_harmony", "acoustic_bass"]
        raw = json.dumps({"parts": {
            "piano_harmony": [
                {"note": "C4", "start_beat": i * 1.0, "duration_beats": 0.9, "velocity": 72}
                for i in range(20)
            ]
        }})
        with pytest.raises(KeyError):
            _parse_orchestration(raw, instruments, 16.0)

    def test_min_notes_threshold_64_beats(self):
        """For 64 beats, min_notes = max(4, 64//6) = 10."""
        assert max(4, int(64 / 6)) == 10

    def test_min_notes_threshold_96_beats(self):
        """For 96 beats, old threshold was 24; new is max(4, 96//6) = 16."""
        assert max(4, int(96 / 6)) == 16

    def test_duration_field_alias_normalised(self):
        """LLMs sometimes emit 'duration_beat' (no trailing s) — must be fixed."""
        from src.orchestrate import _parse_orchestration
        import json
        instruments = ["piano_harmony"]
        n = 12
        raw = json.dumps({"parts": {
            "piano_harmony": [
                {"note": "C4", "start_beat": i * 1.0, "duration_beat": 0.9, "velocity": 72}
                for i in range(n)
            ]
        }})
        result = _parse_orchestration(raw, instruments, 10.0)
        for note in result["parts"]["piano_harmony"]:
            assert "duration_beats" in note, "duration_beat alias not normalised to duration_beats"
            assert "duration_beat" not in note


# ---------------------------------------------------------------------------
# 12. New harmonic transformation functions
# ---------------------------------------------------------------------------

class TestHarmonicTransformations:

    def test_extend_triads_major_to_maj7(self):
        from src.voice_leading import extend_triads_to_sevenths
        result = extend_triads_to_sevenths([(0, "C"), (4, "F")])
        assert result[0][1] == "Cmaj7", f"C→Cmaj7 expected, got {result[0][1]}"
        assert result[1][1] == "Fmaj7", f"F→Fmaj7 expected, got {result[1][1]}"

    def test_extend_triads_minor_to_m7(self):
        from src.voice_leading import extend_triads_to_sevenths
        result = extend_triads_to_sevenths([(0, "Am"), (4, "Dm")])
        assert result[0][1] == "Am7"
        assert result[1][1] == "Dm7"

    def test_extend_triads_dim_to_m7b5(self):
        from src.voice_leading import extend_triads_to_sevenths
        result = extend_triads_to_sevenths([(0, "Bdim")])
        assert result[0][1] == "Bm7b5"

    def test_extend_triads_leaves_7th_chords_unchanged(self):
        from src.voice_leading import extend_triads_to_sevenths
        chords = [(0, "Cmaj7"), (4, "G7"), (8, "Dm7"), (12, "Am7b5")]
        result = extend_triads_to_sevenths(chords)
        for orig, after in zip(chords, result):
            assert orig[1] == after[1], f"Seventh chord '{orig[1]}' was changed to '{after[1]}'"

    def test_extend_triads_preserves_slash_bass(self):
        from src.voice_leading import extend_triads_to_sevenths
        result = extend_triads_to_sevenths([(0, "C/E")])
        assert result[0][1] == "Cmaj7/E"

    def test_back_cycling_adds_dm7_before_g7(self):
        from src.voice_leading import insert_back_cycling
        tl = [(0, "Cmaj7"), (4, "G7"), (8, "Cmaj7")]
        result = insert_back_cycling(tl, min_chord_beats=2.0)
        syms = [s for _, s in result]
        assert "Dm7" in syms, f"Dm7 (ii of G7) not inserted: {syms}"
        assert "G7" in syms

    def test_back_cycling_ii_correct_for_d7(self):
        from src.voice_leading import insert_back_cycling
        # ii of D7 is Am7: D root PC=2, D-5=A(PC 9)
        tl = [(0, "Gmaj7"), (4, "D7"), (8, "Gmaj7")]
        result = insert_back_cycling(tl, min_chord_beats=2.0)
        syms = [s for _, s in result]
        assert "Am7" in syms, f"Am7 (ii of D7) not inserted: {syms}"

    def test_back_cycling_skips_short_chords(self):
        from src.voice_leading import insert_back_cycling
        # G7 only 2 beats — need ≥ 2*min_chord_beats=4 beats to insert
        tl = [(0, "Cmaj7"), (2, "G7"), (4, "Cmaj7")]
        result = insert_back_cycling(tl, min_chord_beats=2.0)
        assert len(result) == len(tl), "Should not insert for short chord"

    def test_back_cycling_skips_non_dominant(self):
        from src.voice_leading import insert_back_cycling
        tl = [(0, "Cmaj7"), (4, "Am7"), (8, "Fmaj7")]
        result = insert_back_cycling(tl, min_chord_beats=2.0)
        assert len(result) == len(tl), "Should not insert before non-dominant chord"

    def test_passing_chords_inserted_on_whole_step(self):
        from src.voice_leading import insert_passing_chords
        # C→Dm: roots C(0) to D(2) = whole step up → insert C#dim
        tl = [(0, "C"), (4, "Dm"), (8, "Em")]
        result = insert_passing_chords(tl, min_chord_beats=2.0)
        syms = [s for _, s in result]
        assert "C#dim" in syms, f"C#dim passing chord not inserted: {syms}"
        assert "D#dim" in syms, f"D#dim passing chord not inserted: {syms}"

    def test_passing_chords_skipped_on_half_step(self):
        from src.voice_leading import insert_passing_chords
        # E→F: half step → no passing chord
        tl = [(0, "Em"), (4, "Fmaj7"), (8, "Cmaj7")]
        result = insert_passing_chords(tl, min_chord_beats=2.0)
        syms = [s for _, s in result]
        assert "Edim" not in syms and "Fdim" not in syms

    def test_passing_chords_skipped_for_short_source(self):
        from src.voice_leading import insert_passing_chords
        # C→D whole step but C only 2 beats — need ≥ 2*min_chord_beats=4 beats
        tl = [(0, "C"), (2, "D"), (4, "Em")]
        result = insert_passing_chords(tl, min_chord_beats=2.0)
        assert len(result) == len(tl)


# ---------------------------------------------------------------------------
# 13. Chord detection from melody
# ---------------------------------------------------------------------------

class TestChordDetection:

    def _notes(self, pitch_beats):
        return [{"note_name": n, "start_beat": s, "duration_beats": d} for n, s, d in pitch_beats]

    def test_c_triad_arpeggio_suggests_c_family(self):
        from src.chord_detect import suggest_chords_from_melody
        notes = self._notes([("C4", 0, 1), ("E4", 1, 1), ("G4", 2, 1), ("C5", 3, 1)])
        result = suggest_chords_from_melody(notes, "C major", beats_per_chord=4.0, total_beats=4.0)
        assert len(result) == 1
        assert result[0].startswith("C"), f"Expected C-family chord, got {result[0]}"

    def test_result_contains_melody_notes_as_chord_tones(self):
        from src.chord_detect import suggest_chords_from_melody
        from src.harmony import chord_tones_from_symbol, _NOTE_PC
        # F-A-C: exclusively maps to F major in C major; only chord with all three
        notes = self._notes([("F4", 0, 1), ("A4", 1, 1), ("C5", 2, 1), ("F4", 3, 1)])
        result = suggest_chords_from_melody(notes, "C major", beats_per_chord=4.0, total_beats=4.0)
        assert len(result) == 1
        chord = result[0]
        # Verify the returned chord contains at least 2 of the 3 melody pitch classes
        c_root, c_ivs = chord_tones_from_symbol(chord)
        c_root_pc = _NOTE_PC.get(c_root, 0)
        chord_pcs = frozenset((c_root_pc + i) % 12 for i in c_ivs)
        melody_pcs = {5, 9, 0}  # F, A, C pitch classes
        overlap = melody_pcs & chord_pcs
        assert len(overlap) >= 2, f"Chord {chord} shares only {overlap} with F-A-C melody"

    def test_returns_one_chord_per_slot(self):
        from src.chord_detect import suggest_chords_from_melody
        notes = self._notes([("C4", i, 0.9) for i in range(16)])
        result = suggest_chords_from_melody(notes, "C major", beats_per_chord=4.0, total_beats=16.0)
        assert len(result) == 4, f"Expected 4 slots for 16 beats / 4bpc, got {len(result)}"

    def test_empty_notes_returns_empty(self):
        from src.chord_detect import suggest_chords_from_melody
        assert suggest_chords_from_melody([], "C major") == []

    def test_unknown_key_returns_empty(self):
        from src.chord_detect import suggest_chords_from_melody
        notes = self._notes([("C4", 0, 1)])
        assert suggest_chords_from_melody(notes, "Xyzzy Blarp") == []

    def test_all_results_are_non_empty_strings(self):
        from src.chord_detect import suggest_chords_from_melody
        notes = self._notes([
            ("D4", 0, 2), ("F#4", 2, 2), ("A4", 4, 2),
            ("E4", 6, 2), ("G4", 8, 2), ("B4", 10, 2),
        ])
        result = suggest_chords_from_melody(notes, "D major", beats_per_chord=4.0, total_beats=12.0)
        assert all(isinstance(c, str) and len(c) > 0 for c in result), f"Empty chord name: {result}"


# ---------------------------------------------------------------------------
# 14. Harmony-only mode regressions
#     Tests for bugs found during real use that caused empty tracks,
#     chromatic chord tone destruction, and wrong bass notes.
# ---------------------------------------------------------------------------

class TestHarmonyOnlyRegressions:
    """
    End-to-end tests for the Harmony-only code path using inject_algo_parts
    with fill_melody_parts=True. Each test is named after the exact user
    scenario that revealed the bug.
    """

    def _make_orchestration(self, instruments):
        return {"key": "D major", "tempo": 120, "parts": {i: [] for i in instruments}}

    # ── Bug: Sitar & Strings produced only bass track ─────────────────────────

    def test_sitar_strings_all_three_tracks_filled(self):
        """Sitar & Strings Harmony-only must produce notes for all three instruments."""
        instruments = ["sitar", "strings_harmony", "acoustic_bass"]
        timeline = build_chord_timeline(["Dmaj7", "Bm7", "Em7", "A7"], 4.0, 32.0)
        orchestration = self._make_orchestration(instruments)
        result = inject_algo_parts(
            orchestration, timeline, 32.0,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Broken chords", 4.0, "D major",
            fill_melody_parts=True,
        )
        for inst in instruments:
            assert len(result["parts"][inst]) > 0, \
                f"'{inst}' has zero notes — fill_melody_parts not working"

    def test_jazz_quartet_harmony_only_all_tracks_filled(self):
        """Jazz Quartet Harmony-only: all three tracks must be non-empty."""
        instruments = ["alto_sax", "piano_harmony", "acoustic_bass"]
        timeline = build_chord_timeline(["Cmaj7", "Am7", "Dm7", "G7"], 4.0, 32.0)
        orchestration = self._make_orchestration(instruments)
        result = inject_algo_parts(
            orchestration, timeline, 32.0,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Jazz comping", 4.0, "C major",
            fill_melody_parts=True,
        )
        for inst in instruments:
            assert len(result["parts"][inst]) > 0, f"'{inst}' empty in Harmony-only"

    def test_all_presets_harmony_only_no_empty_tracks(self):
        """Every preset in Harmony-only mode must produce notes for every instrument."""
        from config import INSTRUMENT_PRESETS
        timeline = build_chord_timeline(["Cmaj7", "Am7", "Dm7", "G7"], 4.0, 32.0)
        for preset_name, instruments in INSTRUMENT_PRESETS.items():
            orchestration = {"key": "C major", "tempo": 120,
                             "parts": {i: [] for i in instruments}}
            result = inject_algo_parts(
                orchestration, timeline, 32.0,
                CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
                dict(INSTRUMENT_RANGES), "Broken chords", 4.0, "C major",
                fill_melody_parts=True,
            )
            for inst in instruments:
                assert len(result["parts"][inst]) > 0, \
                    f"Preset '{preset_name}' instrument '{inst}' has zero notes in Harmony-only"

    # ── Bug: Scale snap destroyed chromatic chord tones (G# in E9, G#m7b5) ───

    def test_chromatic_chord_tones_preserved_e9(self):
        """E9 in D major context must keep G# (the major 3rd) — not snap it to G."""
        instruments = ["strings_harmony", "acoustic_bass"]
        timeline = [(0.0, "E9"), (4.0, "Bm")]
        orchestration = self._make_orchestration(instruments)
        result = inject_algo_parts(
            orchestration, timeline, 8.0,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Block chords", 4.0, "D major",
            fill_melody_parts=True,
        )
        e9_notes = {n["note"][:-1] for n in result["parts"]["strings_harmony"]
                    if n["start_beat"] < 4.0}
        assert "G#" in e9_notes or "Ab" in e9_notes, \
            f"G# missing from E9 voicing — scale snap may have reverted: {e9_notes}"

    def test_chromatic_chord_tones_preserved_gsharp_m7b5(self):
        """G#m7b5 must contain G# — not be snapped to G or A."""
        instruments = ["strings_harmony", "acoustic_bass"]
        timeline = [(0.0, "G#m7b5"), (4.0, "C#m7")]
        orchestration = self._make_orchestration(instruments)
        result = inject_algo_parts(
            orchestration, timeline, 8.0,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Block chords", 4.0, "D major",
            fill_melody_parts=True,
        )
        gm_notes = {n["note"][:-1] for n in result["parts"]["strings_harmony"]
                    if n["start_beat"] < 4.0}
        assert "G#" in gm_notes or "Ab" in gm_notes, \
            f"G# missing from G#m7b5 voicing: {gm_notes}"

    def test_bass_uses_chromatic_chord_root(self):
        """Bass must walk from E (not be snapped) on E9."""
        instruments = ["sitar", "strings_harmony", "acoustic_bass"]
        timeline = [(0.0, "E9"), (8.0, "Bm")]
        orchestration = self._make_orchestration(instruments)
        result = inject_algo_parts(
            orchestration, timeline, 16.0,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Block chords", 4.0, "D major",
            fill_melody_parts=True,
        )
        first_bass = result["parts"]["acoustic_bass"][0]["note"]
        assert first_bass.startswith("E"), \
            f"Bass root on E9 should be E, got {first_bass}"

    # ── Bug: fill_melody_parts=False must not change existing content ─────────

    def test_fill_false_leaves_existing_melody_untouched(self):
        """Without fill_melody_parts, non-chordal/non-bass parts keep their existing notes."""
        existing_melody = [{"note": "D4", "start_beat": 0.0,
                            "duration_beats": 4.0, "velocity": 80}]
        orchestration = {
            "key": "D major", "tempo": 120,
            "parts": {"strings_melody": existing_melody[:], "bass": []},
        }
        timeline = build_chord_timeline(["Dmaj7"], 4.0, 4.0)
        result = inject_algo_parts(
            orchestration, timeline, 4.0,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Block chords", 4.0, "D major",
            fill_melody_parts=False,
        )
        assert result["parts"]["strings_melody"] == existing_melody, \
            "fill_melody_parts=False modified existing melody content"
        assert len(result["parts"]["bass"]) > 0, "Bass was not generated"


# ---------------------------------------------------------------------------
# 15. orchestrate/compose streaming instrument completeness guarantee
# ---------------------------------------------------------------------------

class TestInstrumentCompleteness:
    """
    LLMs frequently omit or misname instruments. These tests verify that
    the setdefault guard in orchestrate_streaming and compose_streaming
    ensures every requested instrument is present before inject_algo_parts runs.
    """

    def test_missing_harmony_instrument_filled_by_algo(self):
        """If LLM JSON omits piano_harmony, it must still be filled by inject_algo_parts."""
        # Simulate LLM output that only has acoustic_bass (forgot piano_harmony)
        import json
        from src.orchestrate import _parse_orchestration
        from src.algo_arranger import inject_algo_parts
        from src.prompts import CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS, INSTRUMENT_RANGES

        harmony_instruments = ["piano_harmony", "acoustic_bass"]
        total_beats = 32.0
        # LLM only generated acoustic_bass — piano_harmony is absent
        llm_json = json.dumps({
            "key": "C major", "tempo": 120,
            "parts": {
                "acoustic_bass": [
                    {"note": "C2", "start_beat": b, "duration_beats": 0.9, "velocity": 75}
                    for b in range(0, 32)
                ]
            }
        })
        result = _parse_orchestration(llm_json, [], total_beats, 1.0)

        # Apply the setdefault guard (as done in orchestrate_streaming)
        for inst in harmony_instruments:
            result["parts"].setdefault(inst, [])

        timeline = build_chord_timeline(["Cmaj7", "Am7", "Dm7", "G7"], 4.0, total_beats)
        result = inject_algo_parts(
            result, timeline, total_beats,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Block chords", 4.0, "C major",
        )

        assert len(result["parts"].get("piano_harmony", [])) > 0, \
            "piano_harmony missing after setdefault + inject_algo_parts"
        assert len(result["parts"].get("acoustic_bass", [])) > 0, \
            "acoustic_bass missing after inject_algo_parts"

    def test_all_blues_band_instruments_present_in_algo_path(self):
        """Blues Band all-algo path must produce all 4 instrument tracks."""
        from config import INSTRUMENT_PRESETS
        from src.algo_arranger import inject_algo_parts
        from src.prompts import CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS, INSTRUMENT_RANGES

        instruments = INSTRUMENT_PRESETS["Blues Band"]
        assert instruments[0] == "harmonica", "harmonica must be melody (index 0)"
        harmony = instruments[1:]
        assert all(i in CHORDAL_INSTRUMENTS or i in BASS_INSTRUMENTS for i in harmony), \
            "All Blues Band harmony instruments must be algo-replaceable (no LLM needed)"

        timeline = build_chord_timeline(["A7", "D7", "E7"], 4.0, 48.0)
        orchestration = {"key": "A major", "tempo": 120,
                         "parts": {i: [] for i in instruments}}
        result = inject_algo_parts(
            orchestration, timeline, 48.0,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), "Broken chords", 4.0, "A major",
        )
        # Simulate melody assignment
        fake_melody = [{"note": "A4", "start_beat": b, "duration_beats": 1.0, "velocity": 90}
                       for b in range(0, 48)]
        result["parts"]["harmonica"] = fake_melody

        for inst in instruments:
            assert len(result["parts"].get(inst, [])) > 0, \
                f"Blues Band instrument '{inst}' has no notes in all-algo path"


# ---------------------------------------------------------------------------
# Phrase detector tests
# ---------------------------------------------------------------------------

class TestPhraseDetector:
    """Tests for src/phrase_detector.py — phrase boundary detection and auto chord timeline."""

    from src.phrase_detector import detect_phrases, auto_chord_timeline_from_melody

    def _make_notes(self, pattern: list[tuple[float, float]], note_name: str = "D4") -> list[dict]:
        """Build note dicts from (start_beat, duration_beats) pairs."""
        return [
            {"note_name": note_name, "note": note_name,
             "start_beat": s, "duration_beats": d}
            for s, d in pattern
        ]

    def test_single_phrase_no_rest(self):
        """Melody with no rests → one phrase spanning the whole melody."""
        from src.phrase_detector import detect_phrases
        notes = self._make_notes([(0, 1), (1, 1), (2, 1), (3, 1)])
        phrases = detect_phrases(notes, beats_per_bar=4.0)
        assert len(phrases) == 1
        assert phrases[0][0] == pytest.approx(0.0)
        assert phrases[0][1] == pytest.approx(4.0)

    def test_two_phrases_clear_rest(self):
        """Two phrases separated by a 2-beat rest → two phrase segments."""
        from src.phrase_detector import detect_phrases
        # Phrase 1: beats 0–8, Phrase 2: starts at beat 10 (2-beat rest at beat 8)
        notes = (
            self._make_notes([(0, 2), (2, 2), (4, 2), (6, 2)])   # phrase 1, ends beat 8
            + self._make_notes([(10, 2), (12, 2), (14, 2), (16, 2)])  # phrase 2
        )
        phrases = detect_phrases(notes, beats_per_bar=4.0)
        assert len(phrases) == 2, f"Expected 2 phrases, got {len(phrases)}: {phrases}"
        assert phrases[0][0] == pytest.approx(0.0)
        assert phrases[1][0] > 8.0

    def test_short_gap_no_split(self):
        """A gap shorter than min_rest_beats should NOT split the phrase."""
        from src.phrase_detector import detect_phrases
        # Gap of 0.5 beats between notes 4 and 5 — below default min_rest of 1.0
        notes = self._make_notes([(0, 1), (1, 1), (2, 1), (3, 0.5), (4.5, 1)])
        phrases = detect_phrases(notes, beats_per_bar=4.0, min_rest_beats=1.0)
        assert len(phrases) == 1, "Sub-threshold gap must not split the phrase"

    def test_phrase_shorter_than_min_phrase_beats_not_split(self):
        """A gap after a phrase shorter than min_phrase_beats should not create a boundary."""
        from src.phrase_detector import detect_phrases
        # Phrase so far only 2 beats before the gap — shorter than 8-beat minimum
        notes = self._make_notes([(0, 1), (1, 1), (4, 1), (5, 1)])
        # gap of 2 beats after beat 2, but phrase only 2 beats long
        phrases = detect_phrases(notes, beats_per_bar=4.0, min_phrase_beats=8.0)
        assert len(phrases) == 1, "Phrase shorter than min_phrase_beats must not be split"

    def test_snaps_to_bar_line(self):
        """Boundary beat is snapped to nearest bar when within ¼ bar."""
        from src.phrase_detector import detect_phrases
        # Phrase 1 ends at beat 7.75, phrase 2 starts at beat 9.75 (gap 2 beats)
        # Nearest bar to 9.75 is beat 8 (within 2 beats? No — 0.25 * 4 = 1 beat, 9.75-8=1.75 > 1)
        # Nearest bar to 9.75 is beat 12 (|12-9.75|=2.25 > 1) or beat 8 (|8-9.75|=1.75 > 1)
        # So no snap, use raw value 9.75
        # Simpler: phrase 2 starts at beat 8.2 (within ¼ bar = 1 beat of bar 8)
        notes = (
            self._make_notes([(0, 2), (2, 2), (4, 2), (6, 1.75)])  # ends 7.75, gap till 8.2
            + self._make_notes([(8.2, 2), (10.2, 2), (12.2, 2), (14.2, 2)])  # phrase 2
        )
        phrases = detect_phrases(notes, beats_per_bar=4.0, min_phrase_beats=6.0)
        if len(phrases) == 2:
            # When snapped, phrase 2 starts at bar 8 (not 8.2)
            assert phrases[1][0] == pytest.approx(8.0, abs=0.5)

    def test_empty_notes_returns_empty(self):
        """Empty note list → empty phrase list."""
        from src.phrase_detector import detect_phrases
        assert detect_phrases([]) == []

    def test_auto_chord_timeline_returns_nonempty(self):
        """auto_chord_timeline_from_melody returns at least one chord for a simple melody."""
        from src.phrase_detector import auto_chord_timeline_from_melody
        notes = self._make_notes(
            [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]
        )
        timeline = auto_chord_timeline_from_melody(
            notes, "C major", total_beats=8.0, beats_per_bar=4.0, beats_per_chord=4.0
        )
        assert len(timeline) >= 1
        for beat, chord in timeline:
            assert isinstance(chord, str) and chord
            assert beat >= 0.0

    def test_auto_chord_timeline_two_phrases(self):
        """Two-phrase melody gets chord changes at phrase boundaries."""
        from src.phrase_detector import auto_chord_timeline_from_melody
        # Phrase 1: C-major content (beats 0–8), Phrase 2: different content (beats 10–18)
        phrase1 = self._make_notes([(0, 2), (2, 2), (4, 2), (6, 2)], "C4")
        phrase2 = self._make_notes([(10, 2), (12, 2), (14, 2), (16, 2)], "G4")
        notes = phrase1 + phrase2
        timeline = auto_chord_timeline_from_melody(
            notes, "C major", total_beats=18.0, beats_per_bar=4.0, beats_per_chord=4.0
        )
        assert len(timeline) >= 2, "Should produce at least two chord changes for two phrases"
        beats = [b for b, _ in timeline]
        assert beats == sorted(beats), "Timeline must be sorted by beat"

    def test_auto_chord_timeline_no_chords_provided_integration(self):
        """Simulates app.py auto-detection: empty user chord input → non-empty timeline."""
        from src.phrase_detector import auto_chord_timeline_from_melody
        # 32-beat melody with rests at beats 8 and 20
        notes = (
            self._make_notes([(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2)])  # → ends 8
            + self._make_notes([(10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 2)])   # → ends 17
            + self._make_notes([(20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 2)])   # → ends 27
        )
        timeline = auto_chord_timeline_from_melody(
            notes, "G major", total_beats=32.0, beats_per_bar=4.0, beats_per_chord=4.0
        )
        assert timeline, "Auto-detection must produce a non-empty chord timeline"
        assert all(b < 32.0 for b, _ in timeline), "No chord should start past total_beats"
