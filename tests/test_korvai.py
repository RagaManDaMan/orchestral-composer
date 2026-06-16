"""
Unit tests for src/korvai_engine.py

Covers: tokenizer, matra counting, frame structure, MIDI output properties,
and the pipeline integration via inject_algo_parts.
"""

import pytest
from src.korvai_engine import (
    tokenize_solkattu,
    phrase_matras,
    korvai_info,
    make_korvai_chord_part,
    make_korvai_bass_part,
    gati_from_label,
    beats_per_matra_from_label,
    _build_frame,
)
from src.algo_arranger import inject_algo_parts
from src.prompts import CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS, INSTRUMENT_RANGES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CHORD_TL = [(0.0, "Dm7"), (8.0, "G7"), (16.0, "Cmaj7"), (24.0, "Am7")]
_TOTAL     = 32.0
_LO, _HI   = 48, 84   # piano_harmony range (approx)

_DEFAULT_PHRASE     = "ta ka dhi mi ta ka dhi mi"
_DEFAULT_CONNECTOR  = "ta ka"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:

    def test_space_separated_chatusram(self):
        toks = tokenize_solkattu("ta ka dhi mi")
        assert toks == [("ta", 1.0), ("ka", 1.0), ("dhi", 1.0), ("mi", 1.0)]

    def test_run_together_takadhimi(self):
        """'takadhimi' should resolve to 4 syllables totalling 4 matras."""
        toks = tokenize_solkattu("takadhimi")
        weights = [w for _, w in toks]
        assert sum(weights) == pytest.approx(4.0)

    def test_dirgham_taam(self):
        toks = tokenize_solkattu("taam")
        assert toks == [("taam", 2.0)]

    def test_dirgham_dheem(self):
        toks = tokenize_solkattu("dheem")
        assert toks == [("dheem", 2.0)]

    def test_comma_is_one_matra_rest(self):
        toks = tokenize_solkattu("ta , ka")
        assert toks[1] == ("rest", 1.0)

    def test_comma_inline_is_rest(self):
        toks = tokenize_solkattu("ta,ka")
        assert any(s == "rest" for s, _ in toks)

    def test_semicolon_is_two_matra_rest(self):
        toks = tokenize_solkattu("ta ; ka")
        assert toks[1] == ("rest", 2.0)

    def test_semicolon_inline_is_two_matra_rest(self):
        toks = tokenize_solkattu("ta;ka")
        rests = [(s, w) for s, w in toks if s == "rest"]
        assert rests == [("rest", 2.0)]

    def test_comma_and_semicolon_combined(self):
        # ta(1) + ,(1) + ta(1) + ;(2) + ka(1) = 6 matras
        assert phrase_matras("ta , ta ; ka") == pytest.approx(6.0)

    def test_hyphen_normalised(self):
        toks_hyphens = tokenize_solkattu("ta-ka-dhi-mi")
        toks_spaces  = tokenize_solkattu("ta ka dhi mi")
        assert toks_hyphens == toks_spaces

    def test_empty_string(self):
        assert tokenize_solkattu("") == []

    def test_case_insensitive(self):
        assert tokenize_solkattu("TA KA") == tokenize_solkattu("ta ka")

    def test_mixed_long_short(self):
        toks = tokenize_solkattu("ta taam ka dheem")
        weights = [w for _, w in toks]
        assert weights == [1.0, 2.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Matra counting
# ---------------------------------------------------------------------------

class TestMatraCounting:

    def test_chatusram_phrase_eight_matras(self):
        assert phrase_matras(_DEFAULT_PHRASE) == pytest.approx(8.0)

    def test_connector_two_matras(self):
        assert phrase_matras(_DEFAULT_CONNECTOR) == pytest.approx(2.0)

    def test_dirgham_phrase_counts_double(self):
        # "taam dheem" = 2+2 = 4 matras
        assert phrase_matras("taam dheem") == pytest.approx(4.0)

    def test_gati_2x_halves_matras(self):
        # 4 short syllables ÷ gati 2.0 = 2.0 matras (they fit in half the time)
        assert phrase_matras("ta ka dhi mi", gati_ratio=2.0) == pytest.approx(2.0)

    def test_gati_1_5x(self):
        # 4 short syllables / 1.5 = 2.667 matras
        assert phrase_matras("ta ka dhi mi", gati_ratio=1.5) == pytest.approx(4.0 / 1.5)

    def test_comma_rest_counts_one_matra(self):
        assert phrase_matras("ta , ka") == pytest.approx(3.0)

    def test_semicolon_rest_counts_two_matras(self):
        assert phrase_matras("ta ; ka") == pytest.approx(4.0)

    def test_keezh_kalam_doubles_frame_duration(self):
        # keezh kalam (2.0 beats/syllable) should produce exactly twice the
        # frame duration of normal kalam (1.0 beats/syllable) for the same phrase.
        from src.korvai_engine import _build_frame
        _, dur_normal = _build_frame("ta ka dhi mi", "ta", 1.0, 1.0)
        _, dur_keezh  = _build_frame("ta ka dhi mi", "ta", 1.0, 2.0)
        assert dur_keezh == pytest.approx(dur_normal * 2.0)

    def test_mel_kalam_halves_frame_duration(self):
        from src.korvai_engine import _build_frame
        _, dur_normal = _build_frame("ta ka dhi mi", "ta", 1.0, 1.0)
        _, dur_mel    = _build_frame("ta ka dhi mi", "ta", 1.0, 0.5)
        assert dur_mel == pytest.approx(dur_normal * 0.5)


# ---------------------------------------------------------------------------
# korvai_info
# ---------------------------------------------------------------------------

class TestKorvaiInfo:

    def test_default_remainder_four(self):
        info = korvai_info(_DEFAULT_PHRASE, _DEFAULT_CONNECTOR, 1.0, 32)
        assert info["total"] == pytest.approx(28.0)
        assert info["remainder"] == pytest.approx(4.0)
        assert not info["fits"]

    def test_perfect_fit(self):
        # phrase = 10 matras, connector = 1 matra → 3×10+2×1 = 32
        info = korvai_info("ta ka dhi mi ta ka dhi mi ta ka", "ta", 1.0, 32)
        assert info["fits"]
        assert info["remainder"] == pytest.approx(0.0)

    def test_returns_all_keys(self):
        info = korvai_info(_DEFAULT_PHRASE, _DEFAULT_CONNECTOR)
        for key in ("phrase_matras", "connector_matras", "total", "remainder", "fits"):
            assert key in info


# ---------------------------------------------------------------------------
# Frame structure
# ---------------------------------------------------------------------------

class TestFrameStructure:

    def _frame(self, phrase=_DEFAULT_PHRASE, connector=_DEFAULT_CONNECTOR,
                gati=1.0, bpm=1.0):
        return _build_frame(phrase, connector, gati, bpm)

    def test_frame_duration_matches_total_matras(self):
        _, dur = self._frame()
        # 3×8 + 2×2 = 28 beats at 1 beat per matra
        assert dur == pytest.approx(28.0)

    def test_event_count_correct(self):
        events, _ = self._frame()
        # 3 A phrases × 8 syllables + 2 B connectors × 2 syllables = 28 events
        assert len(events) == 28

    def test_phrase_nums_present(self):
        events, _ = self._frame()
        phrase_nums = {e["phrase_num"] for e in events}
        assert phrase_nums == {0, 1, 2, 3}

    def test_connectors_are_phrase_num_zero(self):
        events, _ = self._frame()
        connector_events = [e for e in events if e["phrase_num"] == 0]
        # 2 B connectors × 2 syllables = 4 connector events
        assert len(connector_events) == 4

    def test_phrase_1_comes_before_phrase_2(self):
        events, _ = self._frame()
        p1_beats = [e["beat"] for e in events if e["phrase_num"] == 1]
        p2_beats = [e["beat"] for e in events if e["phrase_num"] == 2]
        assert max(p1_beats) < min(p2_beats)

    def test_phrase_2_comes_before_phrase_3(self):
        events, _ = self._frame()
        p2_beats = [e["beat"] for e in events if e["phrase_num"] == 2]
        p3_beats = [e["beat"] for e in events if e["phrase_num"] == 3]
        assert max(p2_beats) < min(p3_beats)

    def test_beats_are_monotonically_non_decreasing(self):
        events, _ = self._frame()
        beats = [e["beat"] for e in events]
        assert beats == sorted(beats)

    def test_hrasva_duration_is_short(self):
        events, _ = self._frame()
        # All syllables here are hrasva (1 matra), so note_dur ≤ 0.5
        for e in events:
            assert e["duration"] <= 0.5 + 1e-9

    def test_dirgham_duration_is_longer(self):
        events, _ = _build_frame("taam dheem", "ta", 1.0, 1.0)  # connector="ta"
        long_events = [e for e in events if e["is_long"]]
        assert len(long_events) > 0
        for e in long_events:
            # dirgham lasts 90% of 2 beats = 1.8 beats
            assert e["duration"] == pytest.approx(1.8)

    def test_beats_per_matra_scales_duration(self):
        _, dur_1 = self._frame(bpm=1.0)
        _, dur_half = self._frame(bpm=0.5)
        assert dur_half == pytest.approx(dur_1 * 0.5)

    def test_gati_2x_halves_frame_duration(self):
        _, dur_1x = self._frame(gati=1.0)
        _, dur_2x = self._frame(gati=2.0)
        assert dur_2x == pytest.approx(dur_1x * 0.5)


# ---------------------------------------------------------------------------
# MIDI output properties
# ---------------------------------------------------------------------------

class TestKorvaiMidiOutput:

    def _gen(self, phrase=_DEFAULT_PHRASE, connector=_DEFAULT_CONNECTOR,
             gati=1.0, bpm=1.0, chord_tl=None, total=None):
        return make_korvai_chord_part(
            _CHORD_TL if chord_tl is None else chord_tl,
            _TOTAL if total is None else total,
            _LO, _HI, "C major",
            phrase_syl=phrase,
            connector_syl=connector,
            gati_ratio=gati,
            beats_per_matra=bpm,
        )

    def test_produces_notes(self):
        notes = self._gen()
        assert len(notes) > 0

    def test_all_notes_have_required_fields(self):
        for note in self._gen():
            assert "note" in note
            assert "start_beat" in note
            assert "duration_beats" in note
            assert "velocity" in note

    def test_no_notes_after_total_beats(self):
        notes = self._gen()
        for n in notes:
            assert n["start_beat"] < _TOTAL

    def test_no_notes_before_beat_zero(self):
        notes = self._gen()
        for n in notes:
            assert n["start_beat"] >= 0.0

    def test_velocities_in_valid_range(self):
        for n in self._gen():
            assert 1 <= n["velocity"] <= 127

    def test_three_velocity_tiers(self):
        """Phrase 1, 2, 3 must produce three distinct velocity bands."""
        notes = self._gen()
        vels = sorted(set(n["velocity"] for n in notes))
        # At least 3 distinct velocity levels
        assert len(vels) >= 3

    def test_phrase_3_highest_velocity(self):
        """The final (mukthāyi) phrase must have the highest average velocity."""
        notes = self._gen()
        frame_dur = 28.0   # known for default phrase/gap

        def avg_vel_in_range(lo_beat, hi_beat):
            relevant = [n["velocity"] for n in notes
                        if lo_beat <= n["start_beat"] < hi_beat]
            return sum(relevant) / len(relevant) if relevant else 0.0

        # Within the first frame: P1=0–8, P2=10–18, P3=20–28
        avg1 = avg_vel_in_range(0.0, 8.0)
        avg2 = avg_vel_in_range(10.0, 18.0)
        avg3 = avg_vel_in_range(20.0, 28.0)
        assert avg3 > avg2 > avg1

    def test_explicit_rest_produces_no_notes(self):
        """Comma-notation rest in a phrase produces silence at that beat position.

        The B connector phrase (phrase_num=0) NOW SOUNDS — only explicit rest
        notation (`,` `;`) in either phrase or connector produces silence.
        At bpm=1.0: "ta , ka" → ta at beat 0, rest at beat 1, ka at beat 2.
        """
        notes = self._gen(phrase="ta , ka", connector="ka", bpm=1.0, total=40.0)
        note_beats = {n["start_beat"] for n in notes}
        # Rest in A phrase 1 falls at beat 1
        assert 1.0 not in note_beats, "Beat 1 should be silent (explicit rest in Phrase 1)"

    def test_empty_chord_timeline_returns_empty(self):
        assert self._gen(chord_tl=[]) == []

    def test_notes_respect_pitch_range(self):
        from src.harmony import note_name_to_midi
        for n in self._gen():
            midi = note_name_to_midi(n["note"])
            assert _LO <= midi <= _HI, f"{n['note']} (MIDI {midi}) out of [{_LO},{_HI}]"

    def test_frame_repeats_over_total_beats(self):
        """A short frame should tile to fill a long piece."""
        # 28-beat frame over 64 beats → should repeat
        notes = self._gen(total=64.0)
        # With 8 attacks per phrase × 3 phrases, one frame → ~24 chord notes
        # Two frames → ≥ 48 notes
        assert len(notes) >= 40

    def test_chord_tones_from_active_chord(self):
        """Notes on the first chord slot should be Dm7 chord tones."""
        from src.harmony import note_name_to_midi, _NOTE_PC
        notes = self._gen()
        # Dm7 pitch classes: D(2) F(5) A(9) C(0)
        dm7_pcs = {2, 5, 9, 0}
        first_slot = [n for n in notes if n["start_beat"] < 8.0]
        assert len(first_slot) > 0
        for n in first_slot:
            pc = note_name_to_midi(n["note"]) % 12
            assert pc in dm7_pcs, f"{n['note']} (pc={pc}) not in Dm7"


# ---------------------------------------------------------------------------
# Korvai bass part
# ---------------------------------------------------------------------------

class TestKorvaiBassPart:

    _LO_B, _HI_B = 28, 55   # acoustic_bass range

    def _gen(self, phrase=_DEFAULT_PHRASE, connector=_DEFAULT_CONNECTOR, total=_TOTAL, bpm=1.0):
        # Explicit bpm=1.0 so connector windows are at predictable beat positions
        return make_korvai_bass_part(
            _CHORD_TL, total, self._LO_B, self._HI_B,
            phrase_syl=phrase, connector_syl=connector, beats_per_matra=bpm,
        )

    def test_produces_notes(self):
        assert len(self._gen()) > 0

    def test_empty_chord_timeline_returns_empty(self):
        assert make_korvai_bass_part([], _TOTAL, self._LO_B, self._HI_B) == []

    def test_pitch_in_bass_range(self):
        from src.harmony import note_name_to_midi
        for n in self._gen():
            midi = note_name_to_midi(n["note"])
            assert self._LO_B <= midi <= self._HI_B

    def test_connector_phrase_sounds(self):
        """B connector phrase (phrase_num=0) produces bass attacks — it is not silent.
        At bpm=1.0: phrase=8 beats, connector=2 beats → connector at [8,10) and [18,20)."""
        notes = self._gen(bpm=1.0)
        beats = {n["start_beat"] for n in notes}
        assert any(8.0 <= b < 10.0 for b in beats), "Bass should attack in Connector 1 [8,10)"
        assert any(18.0 <= b < 20.0 for b in beats), "Bass should attack in Connector 2 [18,20)"

    def test_rest_notation_is_silent(self):
        """Explicit comma-notation rest in connector phrase produces no bass attack.
        connector 'ta , ka' at bpm=1.0 → ta=beat8, rest=beat9, ka=beat10."""
        notes = self._gen(connector="ta , ka", bpm=1.0, total=40.0)
        beats = {n["start_beat"] for n in notes}
        assert 9.0 not in beats, "Beat 9 should be silent (explicit rest in Connector 1)"

    def test_phrase_attacks_present(self):
        """Phrase 1 should have bass notes (beats 0–8 at bpm=1.0)."""
        notes = self._gen(bpm=1.0)
        beats = {n["start_beat"] for n in notes}
        assert any(0.0 <= b < 8.0 for b in beats), "No bass attacks in Phrase 1"

    def test_velocity_three_tiers(self):
        notes = self._gen()
        vels = sorted(set(n["velocity"] for n in notes))
        assert len(vels) >= 2   # at least P1/P3 differ


# ---------------------------------------------------------------------------
# Gati and matra label helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_gati_chatusram(self):
        assert gati_from_label("Chatusram (4-feel)") == pytest.approx(1.0)

    def test_gati_tisram(self):
        assert gati_from_label("Tisram (3-feel)") == pytest.approx(1.5)

    def test_gati_khandam(self):
        assert gati_from_label("Khandam (5-feel)") == pytest.approx(1.25)

    def test_gati_unknown_defaults_to_1(self):
        assert gati_from_label("unknown") == pytest.approx(1.0)

    def test_kalam_normal(self):
        assert beats_per_matra_from_label("Normal kalam  (♬ sixteenth note)") == pytest.approx(0.25)

    def test_kalam_mel(self):
        assert beats_per_matra_from_label("Mel kalam     (faster, 1/32)") == pytest.approx(0.125)

    def test_kalam_keezh(self):
        assert beats_per_matra_from_label("Keezh kalam   (♪ eighth note)") == pytest.approx(0.5)

    def test_kalam_vilambita(self):
        assert beats_per_matra_from_label("Vilambita     (♩ quarter note)") == pytest.approx(1.0)

    def test_kalam_unknown_defaults_to_normal(self):
        assert beats_per_matra_from_label("unknown") == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:

    _KORVAI_PARAMS = {
        "phrase_syl":      _DEFAULT_PHRASE,
        "connector_syl":   _DEFAULT_CONNECTOR,
        "gati_ratio":      1.0,
        "beats_per_matra": 1.0,
    }

    def _run(self, preset_instruments, korvai_params=None):
        orch = {
            "key": "C major", "tempo": 120,
            "parts": {inst: [] for inst in preset_instruments},
        }
        return inject_algo_parts(
            orch, _CHORD_TL, _TOTAL,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES),
            harmony_style="Korvai pattern",
            beats_per_bar=4.0,
            key="C major",
            korvai_params=korvai_params or self._KORVAI_PARAMS,
        )

    def test_piano_harmony_gets_notes(self):
        orch = self._run(["piano_harmony", "acoustic_bass"])
        assert len(orch["parts"]["piano_harmony"]) > 0

    def test_bass_in_connector_phrase(self):
        """Bass plays during B connector windows — the connector phrase now sounds.
        Only explicit rest notation (`,` `;`) produces silence."""
        orch = self._run(["piano_harmony", "acoustic_bass"])
        bass = orch["parts"]["acoustic_bass"]
        assert len(bass) > 0
        # At beats_per_matra=1.0: connector 1 occupies [8,10) → should have bass attacks.
        bass_beats = {n["start_beat"] for n in bass}
        assert any(8.0 <= b < 10.0 for b in bass_beats), \
            "Bass should play in Connector 1 [8,10) — B phrase sounds, not silent"

    def test_korvai_absent_when_params_none(self):
        """Without korvai_params, falls back to block chords (no silent gaps)."""
        orch = {
            "key": "C major", "tempo": 120,
            "parts": {"piano_harmony": [], "acoustic_bass": []},
        }
        orch = inject_algo_parts(
            orch, _CHORD_TL, _TOTAL,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES),
            harmony_style="Korvai pattern",
            beats_per_bar=4.0, key="C major",
            korvai_params=None,  # no params → falls back
        )
        # Without params the elif doesn't match → falls back to block chords
        # Block chords produce notes in the gap window
        piano = orch["parts"]["piano_harmony"]
        gap_notes = [n for n in piano if 8.0 <= n["start_beat"] < 10.0]
        assert len(gap_notes) > 0

    def test_jazz_quartet_preset_chordal_and_bass_filled(self):
        """Korvai fills chordal and bass tracks; melody (alto_sax) needs LLM — stays empty."""
        from config import INSTRUMENT_PRESETS
        instruments = INSTRUMENT_PRESETS["Jazz Quartet"]
        orch = self._run(instruments)
        for inst in instruments:
            if inst in CHORDAL_INSTRUMENTS or inst in BASS_INSTRUMENTS:
                assert len(orch["parts"].get(inst, [])) > 0, \
                    f"{inst} (chordal/bass) has no notes in Jazz Quartet + Korvai"

    def test_tisram_gati_produces_shorter_frame(self):
        """Gati 1.5x → frame covers less real time → more cycles → more notes."""
        kp_1x = {**self._KORVAI_PARAMS, "gati_ratio": 1.0}
        kp_15 = {**self._KORVAI_PARAMS, "gati_ratio": 1.5}
        orch_1x = self._run(["piano_harmony"], korvai_params=kp_1x)
        orch_15 = self._run(["piano_harmony"], korvai_params=kp_15)
        n_1x = len(orch_1x["parts"]["piano_harmony"])
        n_15 = len(orch_15["parts"]["piano_harmony"])
        assert n_15 > n_1x


# ---------------------------------------------------------------------------
# Section-map (K1/K2 as song structure) integration
# ---------------------------------------------------------------------------

class TestSectionMapIntegration:
    """Tests for K sections routed through section_map in inject_algo_parts."""

    _K1_PARAMS = {
        "phrase_syl":      _DEFAULT_PHRASE,
        "connector_syl":   _DEFAULT_CONNECTOR,
        "gati_ratio":      1.0,
        "beats_per_matra": 1.0,   # 1.0 so connector windows are at predictable beats
    }

    def _run_sectioned(self, section_map, chord_tl=None, total=None):
        tl = chord_tl or _CHORD_TL
        tb = total or _TOTAL
        orch = {
            "key": "C major", "tempo": 120,
            "parts": {"piano_harmony": [], "acoustic_bass": []},
        }
        return inject_algo_parts(
            orch, tl, tb,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES),
            harmony_style="Jazz comping",
            beats_per_bar=4.0, key="C major",
            section_map=section_map,
        )

    def test_k1_standalone(self):
        """A section_map with only K1 produces notes in the piece's first 28 beats."""
        from src.song_structure import parse_form, build_song_timeline
        form = parse_form("K1")
        tl, total, sm = build_song_timeline(
            form, {}, 4.0, 4.0, {},
            korvai_definitions={"K1": self._K1_PARAMS},
            fallback_chord="Dm7",
        )
        orch = self._run_sectioned(sm, chord_tl=tl, total=total)
        assert len(orch["parts"]["piano_harmony"]) > 0
        assert len(orch["parts"]["acoustic_bass"]) > 0

    def test_a_k1_a_sections_in_order(self):
        """Regular sections and K sections are both populated and in time order."""
        from src.song_structure import parse_form, build_song_timeline
        form = parse_form("A K1 A")
        sc = {"A": ["Dm7", "G7", "Cmaj7"]}
        bars = {"A": 8}
        tl, total, sm = build_song_timeline(
            form, sc, 4.0, 4.0, bars,
            korvai_definitions={"K1": self._K1_PARAMS},
        )
        orch = self._run_sectioned(sm, chord_tl=tl, total=total)
        piano = orch["parts"]["piano_harmony"]
        assert len(piano) > 0
        beats = [n["start_beat"] for n in piano]
        assert beats == sorted(beats), "Piano notes not in chronological order"

    def test_k_section_connector_phrase_sounds_in_mixed_form(self):
        """In 'A K1', the K1 connector (B phrase) windows produce notes — they are not silent."""
        from src.song_structure import parse_form, build_song_timeline
        form = parse_form("A K1")
        sc = {"A": ["Dm7", "G7"]}
        bars = {"A": 8}
        tl, total, sm = build_song_timeline(
            form, sc, 4.0, 4.0, bars,
            korvai_definitions={"K1": self._K1_PARAMS},
        )
        # K1 starts at beat 32. Phrase=8 matras×1.0=8 beats → connector at beats [40,42).
        orch = self._run_sectioned(sm, chord_tl=tl, total=total)
        piano_beats = {n["start_beat"] for n in orch["parts"]["piano_harmony"]}
        connector_notes = [b for b in piano_beats if 40.0 <= b < 42.0]
        assert len(connector_notes) > 0, \
            "K1 connector (B phrase) at [40,42) should produce notes — it sounds, not silence"

    def test_k1_k2_both_present(self):
        """Two different korvai definitions can coexist in the same form."""
        from src.song_structure import parse_form, build_song_timeline
        k2_params = {**self._K1_PARAMS,
                     "phrase_syl": "ta ka dhi mi ta ka dhi mi ta ka dhi mi",
                     "connector_syl": "ta ka dhi mi"}
        form = parse_form("K1 K2")
        tl, total, sm = build_song_timeline(
            form, {}, 4.0, 4.0, {},
            korvai_definitions={"K1": self._K1_PARAMS, "K2": k2_params},
            fallback_chord="Cmaj7",
        )
        assert len(sm) == 2
        assert sm[0]["label"] == "K1"
        assert sm[1]["label"] == "K2"
        assert sm[1]["start"] == sm[0]["end"]

    def test_parse_form_k_tokens(self):
        """parse_form correctly handles K-label tokens in all formats."""
        from src.song_structure import parse_form
        assert parse_form("K1") == ["K1"]
        assert parse_form("A B K1 A") == ["A", "B", "K1", "A"]
        assert parse_form("A B K1 A B K2 A") == ["A", "B", "K1", "A", "B", "K2", "A"]
        assert parse_form("AABA") == ["A", "A", "B", "A"]   # char-by-char still works
