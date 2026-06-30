# -*- coding: utf-8 -*-
"""
Orchestral Composer
Gradio web app: supply a melody (or let AI generate one) → chord arrangement → MIDI.

Run:  python app.py
Then open http://127.0.0.1:7861 in your browser.
"""

import http.server
import os
import math
import re
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR  = Path(__file__).parent / "outputs"
PREVIEW_PORT = 7862
OUTPUTS_DIR.mkdir(exist_ok=True)


def _start_preview_server() -> None:
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(OUTPUTS_DIR), **kwargs)
        def log_message(self, *args, **kwargs):
            pass
        def handle_error(self, request, client_address):
            # Browser closes the connection as soon as it has what it needs;
            # suppress the resulting BrokenPipeError so it stays off the terminal.
            import sys
            exc = sys.exc_info()[1]
            if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                return
            super().handle_error(request, client_address)
    try:
        server = http.server.HTTPServer(("127.0.0.1", PREVIEW_PORT), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    except OSError:
        pass


_start_preview_server()

import gradio as gr

# Patch gradio_client bug: json_schema_to_python_type crashes on boolean schemas
# (additionalProperties: false/true) — causes health-check failure on launch.
try:
    import gradio_client.utils as _gcu
    _orig_j2p = _gcu.json_schema_to_python_type
    def _safe_j2p(schema, defs=None):
        try:
            return _orig_j2p(schema, defs)
        except (TypeError, AttributeError):
            return "any"
    _gcu.json_schema_to_python_type = _safe_j2p
except Exception:
    pass

from config import DEFAULT_TEMPO_BPM, INSTRUMENT_PRESETS, PRESET_STYLE_HINTS, DEFAULT_NOTES_PER_BEAT, DEFAULT_MAX_TOKENS
DEFAULT_TEMPERATURE = 0.7
from src.transcribe import transcribe_audio, quantize_to_grid, detect_key, detect_tempo
from src.orchestrate import orchestrate_streaming, compose_streaming, check_ollama
from src.midi_builder import build_midi, build_midi_sections
from src.prompts import build_chord_chart, build_chord_chart_from_timeline, RHYTHM_STYLES, TIME_SIGNATURES, HARMONY_STYLES
from src.harmony import (
    MODES, generate_scale_chords, generate_slash_variants,
    parse_mode_input, palette_html, suggest_progression,
    build_chord_timeline, check_chord_scale_compatibility,
    chord_tone_names, chord_tones_from_symbol,
)
from src.voice_leading import (
    insert_secondary_dominants, apply_tritone_substitutions,
    extend_triads_to_sevenths, insert_back_cycling, insert_passing_chords,
)
from src.chord_detect import suggest_chords_from_melody
from src.song_structure import (PROGRESSION_NAMES, progression_to_chords, parse_form,
                                build_song_timeline, is_roman_progression, parse_roman_progression)
from src.phrase_detector import auto_chord_timeline_from_melody
from src.humanize import humanize_orchestration
from src.prompts import CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS, INSTRUMENT_RANGES
from src.algo_arranger import inject_algo_parts
from src.korvai_engine import (
    korvai_info as _korvai_info,
    random_korvai as _random_korvai,
    GATI_OPTIONS, MATRA_OPTIONS,
)

# Indian audio engine -- loads lazily so missing samples don't block startup
try:
    from src.raga_engine import is_indian_preset, get_raga_for_key, generate_raga_melody
    from src.indian_audio_engine import render_indian_audio, setup_sample_folders
    setup_sample_folders()
    _INDIAN_ENGINE_AVAILABLE = True
except Exception as _ie:
    _INDIAN_ENGINE_AVAILABLE = False
    print(f"[indian] Engine not available: {_ie}")

_MELODY_INSTRUMENTS = frozenset({
    "strings_melody", "flute_melody", "trumpet_melody", "piano_melody",
    "alto_sax", "harmonica", "sitar", "oboe_melody", "electric_guitar",
})

SERVER_PORT = 7861

# ---------------------------------------------------------------------------
# Custom / user-editable modes  (persisted to data/custom_modes.json)
# ---------------------------------------------------------------------------

_CUSTOM_MODES_PATH = Path(__file__).parent / "data" / "custom_modes.json"

def _load_custom_modes() -> dict:
    if _CUSTOM_MODES_PATH.exists():
        try:
            import json as _j
            return _j.loads(_CUSTOM_MODES_PATH.read_text())
        except Exception:
            return {}
    return {}

def _persist_custom_modes(custom: dict) -> None:
    import json as _j
    _CUSTOM_MODES_PATH.parent.mkdir(exist_ok=True)
    _CUSTOM_MODES_PATH.write_text(_j.dumps(custom, indent=2, sort_keys=True))

# Merge custom modes into the live MODES dict at startup
_custom_modes_store = _load_custom_modes()
MODES.update(_custom_modes_store)

_MODE_NAMES = sorted(MODES.keys())

# ---------------------------------------------------------------------------
# Module-level context -- shared with reharmonize
def _write_section_midis(
    orchestration: dict,
    section_map: list[dict],
    stem_path: str,
    ts_info: dict,
    exclude_parts: set[str] | None = None,
    chord_timeline: list | None = None,
    key: str | None = None,
) -> str:
    """Write per-section MIDI files and return a status line, or '' if not applicable."""
    if not section_map:
        return ""
    unique_labels = list(dict.fromkeys(s["label"] for s in section_map))
    if len(unique_labels) < 2:
        return ""
    try:
        written = build_midi_sections(
            orchestration, section_map, stem_path,
            time_sig_num=ts_info["numerator"], time_sig_den=ts_info["denominator"],
            exclude_parts=exclude_parts,
            chord_timeline=chord_timeline,
            key=key,
        )
        if written:
            names = "  ".join(f"{lbl}" for lbl, _ in written)
            return f"\n\nSection files (drag into Logic): {names}"
    except Exception:
        pass
    return ""

# ---------------------------------------------------------------------------
_last_context: dict = {
    "chord_timeline": [], "key": "C major", "orchestration": {},
    "midi_path": "", "harmony_style": "Jazz comping", "beats_per_bar": 4.0,
    "time_sig": "4/4", "total_beats": 0.0, "style": "",
    "ts_info": {"numerator": 4, "denominator": 4, "beats_per_bar": 4.0},
    "section_map": None,
}

# ---------------------------------------------------------------------------
# Harmonic palette helpers
# ---------------------------------------------------------------------------

_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]
_MINOR_QUAL = {"min", "m", "m7", "min7", "m7b5", "min9", "m9"}
_DIM_QUAL   = {"dim", "dim7"}

# Semitone → (degree index 0-6, accidental prefix)
# Uses flat accidentals for non-diatonic tones (jazz/modal convention)
_ST_LABEL: list[tuple[int, str]] = [
    (0, ""),   # 0  → I
    (1, "♭"),  # 1  → ♭II
    (1, ""),   # 2  → II
    (2, "♭"),  # 3  → ♭III
    (2, ""),   # 4  → III
    (3, ""),   # 5  → IV
    (4, "♭"),  # 6  → ♭V
    (4, ""),   # 7  → V
    (5, "♭"),  # 8  → ♭VI
    (5, ""),   # 9  → VI
    (6, "♭"),  # 10 → ♭VII
    (6, ""),   # 11 → VII
]

_INTERVAL_NAMES = {
    0: "root", 1: "m2", 2: "M2", 3: "m3", 4: "M3",
    5: "P4",  6: "tri", 7: "P5", 8: "m6", 9: "M6",
    10: "m7", 11: "M7",
}

def _piano_svg(notes: list[str]) -> str:
    """1-octave SVG piano with highlighted keys for the given note names."""
    _ENHARMONIC = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
                   "Ab": "G#", "Bb": "A#", "Cb": "B"}
    pc_set = {_ENHARMONIC.get(n.strip(), n.strip()) for n in notes}
    whites = ["C", "D", "E", "F", "G", "A", "B"]
    blacks = {"C#": 9, "D#": 23, "F#": 51, "G#": 65, "A#": 79}
    WW, WH, BW, BH = 14, 40, 9, 25
    parts = []
    for i, k in enumerate(whites):
        fill = "#5ba3f5" if k in pc_set else "#f0f4ff"
        parts.append(f'<rect x="{i*WW}" y="0" width="{WW-1}" height="{WH}" '
                     f'fill="{fill}" stroke="#334" stroke-width="0.6" rx="2"/>')
    for k, x in blacks.items():
        fill = "#5ba3f5" if k in pc_set else "#1a1a2e"
        parts.append(f'<rect x="{x}" y="0" width="{BW}" height="{BH}" fill="{fill}" rx="1"/>')
    return f'<svg width="{WW*7}" height="{WH}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

def _degree_label(semitone: int, quality: str) -> str:
    """Return Roman numeral label for a chord whose root is `semitone` above the scale root."""
    deg_idx, acc = _ST_LABEL[semitone % 12]
    num = acc + _ROMAN[deg_idx]
    if quality in _DIM_QUAL:
        return num.lower() + "°"
    if quality in _MINOR_QUAL:
        return num.lower()
    return num


def generate_palette(root, mode_name, custom_intervals, beats_per_chord):
    try:
        intervals  = parse_mode_input(mode_name, custom_intervals or "")
        chords     = generate_scale_chords(root, intervals)
        slash      = generate_slash_variants(chords, root, intervals)
        suggestion = suggest_progression(chords, mode_name, beats_per_chord)
        if custom_intervals and custom_intervals.strip():
            matched = next(
                (name for name, ivs in MODES.items() if sorted(ivs) == sorted(intervals)),
                None,
            )
            label = matched if matched else f"custom:{','.join(str(i) for i in sorted(set(intervals)))}"
        else:
            label = mode_name if mode_name in MODES else "custom"

        seen: set[str] = set()
        # (display_label, value) tuples -- value is the plain symbol passed downstream
        all_choices: list[tuple[str, str]] = []
        for c in chords:
            if c.symbol not in seen:
                seen.add(c.symbol)
                chord_st = intervals[c.degree]   # semitone of this chord's root above scale root
                dlabel = f"{_degree_label(chord_st, c.quality)} · {c.symbol}"
                all_choices.append((dlabel, c.symbol))
        for s in slash:
            all_choices.append((s, s))   # slash variants have no degree

        pre_selected = [s for s in suggestion.split() if s in seen or "/" in s]

        import json as _json, html as _html
        chord_data: dict = {}
        for _, symbol in all_choices:
            try:
                notes = chord_tone_names(symbol)
                _, ivs = chord_tones_from_symbol(symbol)
                interval_labels = [_INTERVAL_NAMES.get(i, f"+{i}") for i in ivs]
                chord_data[symbol] = {
                    "notes": notes,
                    "intervals": interval_labels,
                    "piano": _piano_svg(notes),
                }
            except Exception:
                pass

        # Embed JSON as textContent of a hidden div so JS can read it safely
        json_escaped = _html.escape(_json.dumps(chord_data))
        data_html = f'<div id="chord-tooltip-data" style="display:none;height:0;overflow:hidden">{json_escaped}</div>'

        return (
            gr.CheckboxGroup(choices=all_choices, value=pre_selected),
            "Manual",
            f"{root} {label}",
            data_html,
        )
    except Exception:
        return gr.CheckboxGroup(choices=[], value=[]), gr.update(), gr.update(), ""


def use_selected_chords(selected):
    return "  ".join(selected)


def arrange_in_composer(selected, root, mode, tempo):
    """Bridge palette → composer: copy chords + key + tempo and lock palette as source."""
    chords = "  ".join(selected) if selected else ""
    key_str = f"{root} {mode}" if root and mode else (root or "")
    # use_flat_prog=True so the exact palette chords (not the text field) drive the MIDI
    return chords, "Manual", key_str, tempo, True


def crystallize_bridge(payload: str):
    """Live Recorder → Composer: parse JS payload and pre-fill composer inputs."""
    import json
    try:
        d = json.loads(payload)
    except Exception:
        return gr.update(), gr.update(), gr.update(), gr.update()
    chords = d.get("chords", "Dm7 G7 Cmaj7 Am7")
    root   = d.get("root", "C")
    mode   = d.get("mode", "Major (Ionian)")
    tempo  = float(d.get("bpm", 120))
    # Convert Live Recorder mode names to Composer key strings
    mode_map = {
        "Major (Ionian)": "major", "Natural Minor (Aeolian)": "minor",
        "Dorian": "dorian", "Phrygian": "phrygian", "Lydian": "lydian",
        "Mixolydian": "mixolydian", "Locrian": "locrian",
        "Harmonic Minor": "harmonic minor", "Melodic Minor": "melodic minor",
        "Major Pentatonic": "major pentatonic", "Minor Pentatonic": "minor pentatonic",
        "Blues Scale": "blues", "Whole Tone": "whole tone",
    }
    mode_str = mode_map.get(mode, "major")
    key_str  = f"{root} {mode_str}"
    return gr.update(value=chords), gr.update(value="Manual"), gr.update(value=key_str), gr.update(value=tempo)




# ---------------------------------------------------------------------------
# Mode editor helpers
# ---------------------------------------------------------------------------

_CHROMATIC_C = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
_NOTE_PC_MAP  = {n: i for i, n in enumerate(_CHROMATIC_C)}
_NOTE_PC_MAP.update({"Db":1,"D#":3,"E#":5,"Fb":4,"Gb":6,"G#":8,"A#":10,"Cb":11,"B#":0})

def _notes_for_root(root: str, intervals: list[int]) -> list[str]:
    pc = _NOTE_PC_MAP.get(root, 0)
    rotated = _CHROMATIC_C[pc:] + _CHROMATIC_C[:pc]
    return [rotated[i % 12] for i in intervals]

def _parse_intervals(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]

def fill_mode_editor(mode_name: str):
    ivs = MODES.get(mode_name, [])
    return mode_name, " ".join(str(i) for i in ivs)

def preview_mode_notes(root: str, intervals_str: str) -> str:
    try:
        ivs = _parse_intervals(intervals_str)
        if not ivs:
            return ""
        notes = _notes_for_root(root, ivs)
        return f"**{root}: {' – '.join(notes)}**  ({len(ivs)} notes)"
    except Exception:
        return "_Invalid intervals -- use space-separated integers 0–11_"

def save_mode(mode_name: str, intervals_str: str):
    mode_name = mode_name.strip()
    if not mode_name:
        return gr.update(), "⚠ Mode name cannot be empty."
    try:
        ivs = _parse_intervals(intervals_str)
    except ValueError:
        return gr.update(), "⚠ Invalid intervals -- use space-separated integers 0–11."
    if len(ivs) < 3:
        return gr.update(), "⚠ Need at least 3 intervals."
    if not all(0 <= i <= 11 for i in ivs):
        return gr.update(), "⚠ Intervals must be in range 0–11."

    MODES[mode_name] = ivs
    custom = _load_custom_modes()
    custom[mode_name] = ivs
    _persist_custom_modes(custom)

    new_names = sorted(MODES.keys())
    return gr.Dropdown(choices=new_names, value=mode_name), f"✓ '{mode_name}' saved."

def delete_mode(mode_name: str):
    custom = _load_custom_modes()
    if mode_name not in custom:
        return gr.update(), f"⚠ '{mode_name}' is a built-in mode and cannot be deleted. You can override it by saving with the same name."
    del custom[mode_name]
    _persist_custom_modes(custom)
    del MODES[mode_name]
    new_names = sorted(MODES.keys())
    return gr.Dropdown(choices=new_names, value="Major (Ionian)"), f"✓ '{mode_name}' deleted."


def _palette_to_section(selected, rotation=0):
    """Return palette chords for one section, rotated for variety."""
    if not selected:
        return gr.update()
    chords = list(selected)
    if rotation and len(chords) > 1:
        r = rotation % len(chords)
        chords = chords[r:] + chords[:r]
    return "  ".join(chords)


def _table_to_rows(table) -> list:
    """Convert a gr.Dataframe value (pandas DataFrame or list-of-lists) to [[label, bars, chords], …]."""
    if table is None:
        return []
    if hasattr(table, "iterrows"):  # pandas DataFrame
        return [
            [str(row.get("Label", "") or "").strip(),
             int(row.get("Bars", 8) or 8),
             str(row.get("Chords", "") or "").strip()]
            for _, row in table.iterrows()
        ]
    return [[str(r[0] or "").strip(), int(r[1] or 8), str(r[2] or "").strip()]
            for r in table if r and len(r) >= 3]


def _table_to_rows_korvai(table) -> list:
    """Convert korvai gr.Dataframe to [[label, phrase, connector, gati, kalam, target, chord], …]."""
    _H = ["Label", "A phrase", "Connector", "Gati (4/3/5)", "Kalam (n/k/v)", "Target", "Chord (opt)"]
    _DEF = ["", "", "", "4", "n", 32, ""]
    if table is None:
        return []
    if hasattr(table, "iterrows"):
        rows = []
        for _, row in table.iterrows():
            try:
                rows.append([
                    str(row.get(_H[0], "") or "").strip(),
                    str(row.get(_H[1], "") or "").strip(),
                    str(row.get(_H[2], "") or "").strip(),
                    str(row.get(_H[3], "4") or "4").strip(),
                    str(row.get(_H[4], "n") or "n").strip(),
                    int(row.get(_H[5], 32) or 32),
                    str(row.get(_H[6], "") or "").strip(),
                ])
            except Exception:
                pass
        return rows
    result = []
    for r in (table or []):
        if not r:
            continue
        result.append([
            str(r[i] if i < len(r) else _DEF[i]).strip() if i != 5
            else int(r[5] if len(r) > 5 else 32)
            for i in range(7)
        ])
    return result


def _parse_gati(s: str) -> float:
    """Flexible gati parser. '4'/chatusram→1.0 · '3'/tisram→1.5 · '5'/khandam→1.25."""
    s = str(s or "").strip().lower()
    if s in ("3", "ti", "tisram", "triplet"):   return 1.5
    if s in ("5", "kh", "khandam", "khanda"):   return 1.25
    return 1.0  # default: chatusram


def _parse_kalam(s: str) -> float:
    """Flexible kalam parser. 'n'/normal→0.25 · 'k'/keezh→0.5 · 'v'/vilambita→1.0 · 'm'/mel→0.125."""
    s = str(s or "").strip().lower()
    if s in ("k", "keezh", "1/8", "8", "eighth"):         return 0.5
    if s in ("v", "vil", "vilambita", "1/4", "4th"):       return 1.0
    if s in ("m", "mel", "1/32", "32", "faster", "fast"):  return 0.125
    return 0.25  # default: normal (1/16)


def _table_to_korvai_defs(table) -> dict:
    """Build a korvai_definitions dict from the korvai table."""
    defs: dict = {}
    for row in _table_to_rows_korvai(table):
        label, phrase, connector, gati_s, kalam_s, target = row[:6]
        chord_override = str(row[6]).strip() if len(row) > 6 else ""
        if not label or not phrase:
            continue
        defs[label] = {
            "phrase_syl":      phrase,
            "connector_syl":   connector or "ta ka",
            "gati_ratio":      _parse_gati(gati_s),
            "beats_per_matra": _parse_kalam(kalam_s),
            "chord_override":  chord_override,
        }
    return defs


def fill_all_sections_from_palette(selected, current_table):
    """Fill all section table rows with palette chords rotated for inter-section variety."""
    chords = list(selected or [])
    if not chords:
        return gr.update()
    rows = _table_to_rows(current_table) or [["A", 8, ""], ["B", 8, ""]]
    n = len(chords)
    new_rows = []
    for i, row in enumerate(rows):
        start = (i * max(1, n // max(1, len(rows)))) % n
        new_rows.append([row[0], row[1], "  ".join(chords[start:] + chords[:start])])
    return new_rows


# ---------------------------------------------------------------------------
# Duration calculator + feedback parser
# ---------------------------------------------------------------------------

def calc_duration(bars, bpm, time_sig="4/4"):
    beats_per_bar = TIME_SIGNATURES.get(time_sig, {}).get("beats_per_bar", 4)
    beats     = bars * beats_per_bar
    total_sec = beats / (bpm / 60)
    mins, secs = int(total_sec // 60), int(total_sec % 60)
    return f"**{bars} bars · {time_sig} · {bpm:.0f} BPM · {beats:.1f} beats · {mins}:{secs:02d}**"


_FEEDBACK_RULES = [
    (["too much harmonic", "too complex", "cluttered", "busy", "overwhelming"],
     "Disable secondary dominants and tritone substitutions. Lower note density."),
    (["too sparse", "plain", "boring", "empty", "thin", "not enough"],
     "Enable secondary dominants. Increase note density."),
    (["too mechanical", "robotic", "stiff", "quantised", "machine"],
     "Increase Humanize to 0.5–0.8."),
    (["too random", "too loose", "messy", "sloppy", "out of time"],
     "Reduce Humanize to 0.1–0.2."),
    (["wrong key", "out of key", "off key"],
     "Double-check the Key field or try Auto-detect."),
    (["too fast", "rushed"],
     "Lower BPM or increase beats-per-chord."),
    (["too slow", "dragging"],
     "Increase BPM or switch to Arpeggio style."),
    (["monotonous", "repetitive"],
     "Add a B section in Song structure, or use a turnaround progression."),
    (["no bass", "bass missing"],
     "Jazz Quartet and Piano Trio presets include acoustic bass."),
]

def parse_feedback(text):
    if not (text or "").strip():
        return ""
    low = text.lower()
    hits = [f"→ {s}" for kws, s in _FEEDBACK_RULES if any(k in low for k in kws)]
    return ("Suggested adjustments:\n" + "\n".join(hits)) if hits else \
        "No match. Try: 'too mechanical', 'too sparse', 'too complex'."

# ---------------------------------------------------------------------------
# Shared chord-timeline builder (used by all pipeline modes)
# ---------------------------------------------------------------------------


def _build_chord_timeline(
    chord_input, beats_per_chord, total_beats, beats_per_bar,
    key, use_sec_dom, use_tritone_sub,
    form_str="", section_rows=None,
    palette_chords=None, use_flat_prog=False,
    use_auto_seventh=False, use_back_cycle=False, use_passing_chords=False,
    korvai_definitions=None,
):
    """Return (chord_timeline, total_beats, chord_chart_str, summary_line, section_map)."""
    def _parse(s):
        s = (s or "").strip()
        if s and is_roman_progression(s):
            return parse_roman_progression(s, key or "C major")
        return [c for c in re.split(r"[\s|,\-]+", s) if c]

    form = parse_form(form_str or "")
    _k_defs = korvai_definitions or {}
    k_in_form = any(lbl in _k_defs for lbl in form)
    sections_defined = bool(section_rows)
    k_sections_active = bool(k_in_form and _k_defs)
    # K sections always force structured mode; use_flat_prog only skips structure for non-K forms
    using_structure = bool(form and (k_sections_active or (not use_flat_prog and sections_defined)))

    section_map: list[dict] = []

    if using_structure:
        palette_list = list(palette_chords) if (use_flat_prog and palette_chords) else []
        section_chords, bars_per_sec = {}, {}
        for row in (section_rows or []):
            if not row or not str(row[0] or "").strip():
                continue
            lbl  = str(row[0]).strip()
            bars = int(row[1] or 8)
            raw  = str(row[2] or "")
            chords = _parse(raw)
            if not chords and palette_list:
                chords = palette_list
            if chords:
                section_chords[lbl] = chords
                bars_per_sec[lbl]   = max(1, bars)
        raw_global = _parse(chord_input)
        # When use_flat_prog is set, palette takes priority as fallback chord for K sections
        if use_flat_prog and palette_list:
            fallback = palette_list[0]
        else:
            fallback = raw_global[0] if raw_global else "Cmaj7"

        # For K-only forms, tile the form to fill the requested duration
        effective_form = form
        k_only_form = bool(form) and all(lbl in _k_defs for lbl in form)
        if k_only_form and _k_defs:
            from src.korvai_engine import korvai_frame_beats as _kfb
            single_pass = sum(
                _kfb(_k_defs[lbl]["phrase_syl"], _k_defs[lbl]["connector_syl"],
                     _k_defs[lbl]["gati_ratio"], _k_defs[lbl]["beats_per_matra"])
                for lbl in form if lbl in _k_defs
            )
            if single_pass > 0 and total_beats > single_pass:
                reps = max(1, math.ceil(total_beats / single_pass))
                effective_form = form * reps

        chord_timeline, total_beats, section_map = build_song_timeline(
            effective_form, section_chords, beats_per_chord, beats_per_bar, bars_per_sec,
            korvai_definitions=_k_defs, fallback_chord=fallback,
        )
        # Inject per-korvai chord overrides into the timeline
        for sec in section_map:
            kp = sec.get("korvai_params") or {}
            override = (kp.get("chord_override") or "").strip()
            if override:
                beat = sec["start"]
                chord_timeline = [(b, c) for b, c in chord_timeline if b != beat]
                chord_timeline.append((beat, override))
                chord_timeline.sort(key=lambda x: x[0])
        undefined = [l for l in set(form) if l not in section_chords and l not in _k_defs]
        val_warn = (f"\n  ⚠ Undefined: {', '.join(sorted(undefined))} -- add to section table or korvai slot"
                    if undefined else "")
        summary = f"  Form: {' → '.join(form)}{val_warn}\n" + "".join(
            f"  {l} ({bars_per_sec[l]} bars): {' '.join(section_chords[l])}\n"
            for l in sorted(section_chords.keys()))
        for sec in section_map:
            if sec.get("korvai_params"):
                kp = sec["korvai_params"]
                from src.korvai_engine import korvai_frame_beats
                kdur = korvai_frame_beats(
                    kp["phrase_syl"], kp["connector_syl"],
                    kp["gati_ratio"], kp["beats_per_matra"],
                )
                summary += (
                    f"  {sec['label']}: phrase={kp['phrase_syl']} | "
                    f"connector={kp['connector_syl']}  ({kdur:.1f} beats)\n"
                )
    else:
        if use_flat_prog and palette_chords:
            raw_chords = list(palette_chords)
            summary = f"  Palette chords (flat): {' → '.join(raw_chords)}\n"
        else:
            raw_chords = _parse(chord_input)
            summary = f"  Chords: {' → '.join(raw_chords)}\n" if raw_chords else ""
        chord_timeline = build_chord_timeline(raw_chords, beats_per_chord, total_beats) if raw_chords else []

    # Harmonic transformations (order matters: extend triads before inserting extra chords)
    if chord_timeline and use_auto_seventh:
        chord_timeline = extend_triads_to_sevenths(chord_timeline, key)
    if chord_timeline and use_sec_dom:
        chord_timeline = insert_secondary_dominants(chord_timeline, key)
    if chord_timeline and use_back_cycle:
        chord_timeline = insert_back_cycling(chord_timeline, key)
    if chord_timeline and use_passing_chords:
        chord_timeline = insert_passing_chords(chord_timeline)
    if chord_timeline and use_tritone_sub:
        chord_timeline = apply_tritone_substitutions(chord_timeline)

    chord_chart = build_chord_chart_from_timeline(chord_timeline, total_beats) if chord_timeline else ""
    clash_warnings = check_chord_scale_compatibility(chord_timeline, key)
    if clash_warnings:
        summary += "  ⚠ Chord/scale clashes:\n" + "".join(f"    • {w}\n" for w in clash_warnings)

    return chord_timeline, total_beats, chord_chart, summary, section_map


_ROLL_CSS = """
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:#0a0d12;font-family:'Segoe UI',system-ui,sans-serif;font-size:11px;color:#8899aa;user-select:none;}
  #toolbar{display:flex;gap:4px;align-items:center;padding:5px 6px;background:#0d1018;border-bottom:1px solid #1a2030;flex-wrap:wrap;}
  .tb{padding:4px 11px;border:1px solid #1e2a3a;border-radius:4px;background:#141c28;color:#6688aa;cursor:pointer;font-size:10px;font-family:inherit;transition:all .12s;}
  .tb:hover{background:#1a2538;color:#99ccff;border-color:#2a3a50;}
  .tb.on{background:#1a3a6a;border-color:#4477cc;color:#88bbff;box-shadow:0 0 6px #2255aa44;}
  .tb-sep{width:1px;height:18px;background:#1a2030;margin:0 3px;}
  #hint{font-size:9px;color:#2a3a4a;margin-left:4px;flex:1;}
  #snap-label{font-size:9px;color:#445566;margin-right:2px;}
  #roll-area{display:flex;flex-direction:column;border:1px solid #131b26;border-radius:4px;overflow:hidden;}
  #roll-wrap{overflow:auto;cursor:default;flex:1;position:relative;}
  #roll-wrap.draw{cursor:crosshair;}
  #roll-wrap.erase{cursor:cell;}
  canvas{display:block;}
  #vel-wrap{height:52px;overflow:hidden;border-top:1px solid #0f1820;background:#07090f;position:relative;}
  #vel-cv{display:block;}
  #vel-label{position:absolute;left:4px;top:3px;font-size:8px;color:#223344;pointer-events:none;}
  #tracks-legend{display:flex;gap:10px;flex-wrap:wrap;padding:4px 6px;font-size:9px;background:#090d14;border-top:1px solid #0f1820;}
  .track-dot{display:inline-block;width:7px;height:7px;border-radius:2px;margin-right:3px;vertical-align:middle;}
"""

_DRUM_SYNTH_JS = r"""
/* Web-Audio drum synthesizer using Tone.js's own AudioContext for tight sync */
(function(){
  function _ctx(){ return Tone.getContext().rawContext; }

  function _kick(t, vel){
    var ctx=_ctx(), g=ctx.createGain();
    g.gain.setValueAtTime(vel*2.0, t);
    g.gain.exponentialRampToValueAtTime(0.001, t+0.45);
    g.connect(ctx.destination);
    var o=ctx.createOscillator(); o.type='sine';
    o.frequency.setValueAtTime(180, t);
    o.frequency.exponentialRampToValueAtTime(28, t+0.35);
    o.connect(g); o.start(t); o.stop(t+0.46);
    /* punch layer */
    var pg=ctx.createGain(); pg.gain.setValueAtTime(vel*0.8,t); pg.gain.exponentialRampToValueAtTime(0.001,t+0.05);
    pg.connect(ctx.destination);
    var buf=ctx.createBuffer(1,Math.ceil(ctx.sampleRate*0.05),ctx.sampleRate);
    var d=buf.getChannelData(0); for(var i=0;i<d.length;i++) d[i]=Math.random()*2-1;
    var ns=ctx.createBufferSource(); ns.buffer=buf;
    var lp=ctx.createBiquadFilter(); lp.type='lowpass'; lp.frequency.value=200;
    ns.connect(lp); lp.connect(pg); ns.start(t);
  }

  function _snare(t, vel){
    var ctx=_ctx();
    /* noise burst */
    var dur=0.18;
    var ng=ctx.createGain(); ng.gain.setValueAtTime(vel*1.2,t); ng.gain.exponentialRampToValueAtTime(0.001,t+dur);
    ng.connect(ctx.destination);
    var buf=ctx.createBuffer(1,Math.ceil(ctx.sampleRate*dur),ctx.sampleRate);
    var d=buf.getChannelData(0); for(var i=0;i<d.length;i++) d[i]=Math.random()*2-1;
    var ns=ctx.createBufferSource(); ns.buffer=buf;
    var hp=ctx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=1500;
    var bp=ctx.createBiquadFilter(); bp.type='bandpass'; bp.frequency.value=3000; bp.Q.value=0.7;
    ns.connect(hp); hp.connect(bp); bp.connect(ng);
    ns.start(t);
    /* body tone */
    var og=ctx.createGain(); og.gain.setValueAtTime(vel*0.6,t); og.gain.exponentialRampToValueAtTime(0.001,t+0.09);
    og.connect(ctx.destination);
    var o=ctx.createOscillator(); o.type='triangle'; o.frequency.value=185;
    o.connect(og); o.start(t); o.stop(t+0.09);
  }

  function _hihat(t, vel, open){
    var ctx=_ctx(), dur=open?0.35:0.045;
    var g=ctx.createGain(); g.gain.setValueAtTime(vel*0.7,t); g.gain.exponentialRampToValueAtTime(0.001,t+dur);
    g.connect(ctx.destination);
    var buf=ctx.createBuffer(1,Math.ceil(ctx.sampleRate*Math.max(dur,0.05)),ctx.sampleRate);
    var d=buf.getChannelData(0); for(var i=0;i<d.length;i++) d[i]=Math.random()*2-1;
    var ns=ctx.createBufferSource(); ns.buffer=buf;
    var hp=ctx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=8000;
    ns.connect(hp); hp.connect(g); ns.start(t);
  }

  function _ride(t, vel){
    var ctx=_ctx(), dur=0.6;
    var g=ctx.createGain(); g.gain.setValueAtTime(vel*0.5,t); g.gain.exponentialRampToValueAtTime(0.001,t+dur);
    g.connect(ctx.destination);
    var buf=ctx.createBuffer(1,Math.ceil(ctx.sampleRate*dur),ctx.sampleRate);
    var d=buf.getChannelData(0); for(var i=0;i<d.length;i++) d[i]=Math.random()*2-1;
    var ns=ctx.createBufferSource(); ns.buffer=buf;
    var bp=ctx.createBiquadFilter(); bp.type='bandpass'; bp.frequency.value=6000; bp.Q.value=1.5;
    ns.connect(bp); bp.connect(g); ns.start(t);
  }

  function _tom(t, vel, freq){
    var ctx=_ctx(), g=ctx.createGain();
    g.gain.setValueAtTime(vel*1.5,t); g.gain.exponentialRampToValueAtTime(0.001,t+0.35);
    g.connect(ctx.destination);
    var o=ctx.createOscillator(); o.type='sine';
    o.frequency.setValueAtTime(freq,t); o.frequency.exponentialRampToValueAtTime(freq*0.35,t+0.25);
    o.connect(g); o.start(t); o.stop(t+0.36);
  }

  window._drumHit = function(midiNote, t, vel){
    var v=Math.max(0.15, Math.min(1.0, vel));
    if     (midiNote===36||midiNote===35)              _kick(t,v);
    else if(midiNote===38||midiNote===40||midiNote===37) _snare(t,v);
    else if(midiNote===42||midiNote===44)              _hihat(t,v,false);
    else if(midiNote===46)                             _hihat(t,v,true);
    else if(midiNote===49||midiNote===57||midiNote===55) _hihat(t,v,true);
    else if(midiNote===51||midiNote===53||midiNote===59) _ride(t,v);
    else if(midiNote===50)                             _tom(t,v,210);
    else if(midiNote===47||midiNote===48)              _tom(t,v,155);
    else if(midiNote===43||midiNote===45)              _tom(t,v,105);
    else                                               _snare(t,v*0.4);
  };
})();
"""

_GARCIA_JS = r"""
/* ═══════════════════════════════════════════════════════════════════════
   GarSIa — Gesture Live Looper  v2
   ═══════════════════════════════════════════════════════════════════════ */
(function(){

/* Pre-load Tone.js at page-load so it's ready before first click */
(function(){
  if(window.Tone) return;
  var s=document.createElement('script');
  s.src='https://cdn.jsdelivr.net/npm/tone@14.7.77/build/Tone.js';
  document.head.appendChild(s);
})();

/* ── State ── */
var _video, _canvas, _ctx;
var _hands, _camera;
var _running = false;

/* BPM */
var _bpmHistory=[], _currentBPM=120, _lastTapTime=0;
var _lastWristY=null, _tapCooldown=0, _tapFlash=0;

/* Swipe */
var _lastWristX=null, _swipeCooldown=0;

/* Gesture */
var _currentGesture=null, _prevGestureName=null;
var _gestureHoldFrames=0, _holdProgress=0;
var HOLD_FRAMES=4, _gestureCooldown=0, COOLDOWN_FRAMES=30;

/* Layers: each has on/off + style index */
var _layers = {
  drums: {on:false, style:0},
  bass:  {on:false, style:0},
  keys:  {on:false, style:0},
  pad:   {on:false, style:0},
};

/* Audio objects */
var _audioReady = false;
var _bassSynth = null;
var _keysSynth = null;
var _padSynth  = null;
var _drumLoop  = null;
var _bassLoop  = null;
var _keysLoop  = null;
var _padLoop   = null;

/* Chord preset */
var _currentChords = ['Cmaj7','Am7','Dm7','G7'];
var _chordIdx = 0;

/* ── Pattern / style libraries ── */
var _DRUM_PATTERNS = [
  {name:'Basic 4/4', steps:[36,42,38,42, 36,42,38,46]},
  {name:'Funk',      steps:[36,42,36,42, 38,42,36,42]},
  {name:'Jazz',      steps:[42,42,42,46, 38,42,42,46]},
  {name:'Half-time', steps:[36,42,42,42, 38,42,42,42]},
  {name:'Latin',     steps:[36,42,38,42, 36,46,38,42]},
];
var _BASS_STYLES = ['Root','Walking','Syncopated'];
var _KEYS_STYLES = ['Block','Arpeggio','Comping'];
var _PAD_STYLES  = ['Whole','Half','Swell'];

/* active swipe target */
var _activeLayer = 'drums'; /* which layer swipe gesture affects */

/* ── BPM snap ── */
var _BPMS = [60,65,70,72,75,80,85,90,95,100,105,110,115,120,125,130,140,150,160,170,180];
function _snapBPM(raw) {
  return _BPMS.reduce(function(a,b){ return Math.abs(b-raw)<Math.abs(a-raw)?b:a; });
}

/* ── Finger states ── */
function _fingers(lm, isRight) {
  var thumb = isRight ? (lm[4].x < lm[3].x) : (lm[4].x > lm[3].x);
  return {
    t: thumb,
    i: lm[8].y  < lm[6].y,
    m: lm[12].y < lm[10].y,
    r: lm[16].y < lm[14].y,
    p: lm[20].y < lm[18].y
  };
}

/* ── Gesture table ── */
var _G = [
  {t:0,i:0,m:0,r:0,p:0, name:'Fist',      layer:'drums',      color:'#ff5555', icon:'✊'},
  {t:0,i:1,m:0,r:0,p:0, name:'Point',     layer:'bass',       color:'#ff9933', icon:'☝'},
  {t:0,i:1,m:1,r:0,p:0, name:'Peace',     layer:'keys',       color:'#ffee44', icon:'✌'},
  {t:0,i:1,m:1,r:1,p:0, name:'Three',     layer:'lead',       color:'#44ff88', icon:'🖖'},
  {t:0,i:1,m:1,r:1,p:1, name:'Four',      layer:'pad',        color:'#44aaff', icon:'🖐'},
  {t:1,i:1,m:1,r:1,p:1, name:'Open Palm', layer:'all',        color:'#ffffff', icon:'🖐'},
  {t:1,i:0,m:0,r:0,p:0, name:'Thumb Up',  layer:'bpm_up',     color:'#aaffaa', icon:'👍'},
  {t:1,i:0,m:0,r:0,p:1, name:'Shaka',     layer:'bpm_down',   color:'#ffaaaa', icon:'🤙'},
  {t:0,i:0,m:0,r:0,p:1, name:'Pinky',     layer:'chord_next', color:'#cc88ff', icon:'🤙'},
];

function _classify(lm, isRight) {
  var f = _fingers(lm, isRight);
  for (var i=0; i<_G.length; i++) {
    var g = _G[i];
    if ((g.t?1:0)===(f.t?1:0) && (g.i?1:0)===(f.i?1:0) &&
        (g.m?1:0)===(f.m?1:0) && (g.r?1:0)===(f.r?1:0) && (g.p?1:0)===(f.p?1:0))
      return g;
  }
  return {name:'--', layer:null, color:'#555', icon:'?'};
}

/* ── Wrist tap → BPM ── */
function _detectTap(wristY, ts) {
  if (_lastWristY === null) { _lastWristY = wristY; return; }
  var dy = (wristY - _lastWristY) * 10;
  _lastWristY = wristY;
  if (dy > 0.14 && _tapCooldown <= 0) {
    var interval = (ts - _lastTapTime) / 1000;
    if (_lastTapTime > 0 && interval > 0.2 && interval < 2.5) {
      _bpmHistory.push(60 / interval);
      if (_bpmHistory.length > 8) _bpmHistory.shift();
      if (_bpmHistory.length >= 2) {
        var sorted = _bpmHistory.slice().sort(function(a,b){return a-b;});
        var median = sorted[Math.floor(sorted.length/2)];
        var snapped = _snapBPM(Math.round(median));
        if (snapped !== _currentBPM) {
          _currentBPM = snapped;
          if (window.Tone) Tone.Transport.bpm.rampTo(_currentBPM, 0.1);
          var el = document.getElementById('garcia-bpm-val');
          if (el) el.textContent = _currentBPM;
        }
      }
    }
    _lastTapTime = ts;
    _tapCooldown = 12;
    _tapFlash = 8;
  }
  if (_tapCooldown > 0) _tapCooldown--;
  if (_tapFlash > 0)    _tapFlash--;
}

/* ── Actions ── */
function _styleCount(k) {
  if(k==='drums') return _DRUM_PATTERNS.length;
  if(k==='bass')  return _BASS_STYLES.length;
  if(k==='keys')  return _KEYS_STYLES.length;
  if(k==='pad')   return _PAD_STYLES.length;
  return 1;
}
function _styleName(k) {
  if(k==='drums') return _DRUM_PATTERNS[_layers.drums.style].name;
  if(k==='bass')  return _BASS_STYLES[_layers.bass.style];
  if(k==='keys')  return _KEYS_STYLES[_layers.keys.style];
  if(k==='pad')   return _PAD_STYLES[_layers.pad.style];
  return '';
}

function _swipeStyle(dir) {
  /* dir: +1 = next, -1 = prev; affects _activeLayer */
  var k = _activeLayer;
  if(!_layers.hasOwnProperty(k)) return;
  var n = _styleCount(k);
  _layers[k].style = (_layers[k].style + dir + n) % n;
  /* rebuild drum loop immediately so the new pattern plays */
  if(k==='drums' && _drumLoop) {
    _drumLoop.dispose();
    _drumLoop = null;
    _buildDrumLoop();
  }
  _refreshLayers();
}

function _trigger(layer) {
  if (!layer) return;
  if (layer === 'all') {
    var anyOn = Object.keys(_layers).some(function(k){return _layers[k].on;});
    Object.keys(_layers).forEach(function(k){ _layers[k].on=!anyOn; });
  } else if (layer === 'bpm_up') {
    var idx = _BPMS.indexOf(_currentBPM);
    if (idx < _BPMS.length-1) {
      _currentBPM = _BPMS[idx+1];
      if (window.Tone) Tone.Transport.bpm.rampTo(_currentBPM, 0.2);
      var el = document.getElementById('garcia-bpm-val'); if(el) el.textContent=_currentBPM;
    }
  } else if (layer === 'bpm_down') {
    var idx = _BPMS.indexOf(_currentBPM);
    if (idx > 0) {
      _currentBPM = _BPMS[idx-1];
      if (window.Tone) Tone.Transport.bpm.rampTo(_currentBPM, 0.2);
      var el = document.getElementById('garcia-bpm-val'); if(el) el.textContent=_currentBPM;
    }
  } else if (layer === 'chord_next') {
    _chordIdx = (_chordIdx+1) % _currentChords.length;
    var el = document.getElementById('garcia-chord-now'); if(el) el.textContent=_currentChords[_chordIdx]||'--';
  } else if (_layers.hasOwnProperty(layer)) {
    _layers[layer].on = !_layers[layer].on;
    _activeLayer = layer; /* swipes now target this layer */
  }
  _refreshLayers();
}

function _refreshLayers() {
  Object.keys(_layers).forEach(function(k) {
    var el = document.getElementById('garcia-layer-'+k);
    if (!el) return;
    el.classList.toggle('active', _layers[k].on);
    var sn = el.querySelector('.garcia-style-name');
    if(sn) sn.textContent = _styleName(k);
    el.classList.toggle('garcia-active-target', k===_activeLayer);
  });
}

/* ── Chord → MIDI notes ── */
var _ROOT_PC = {C:0,'C#':1,Db:1,D:2,'D#':3,Eb:3,E:4,F:5,'F#':6,Gb:6,G:7,'G#':8,Ab:8,A:9,'A#':10,Bb:10,B:11};
var _NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
function _chordNotes(sym, octave) {
  var rm = sym.match(/^([A-G][#b]?)/); if (!rm) return ['C'+octave];
  var root = rm[1]; var rpc = _ROOT_PC[root]||0;
  var qual = sym.slice(root.length);
  var ivs = qual.indexOf('maj7')>=0?[0,4,7,11]:qual==='m7'?[0,3,7,10]:qual==='7'?[0,4,7,10]:qual==='m'?[0,3,7]:qual==='dim7'?[0,3,6,9]:[0,4,7];
  var base = 12*(octave+1);
  return ivs.map(function(iv){ var m=base+rpc+iv; return _NOTE_NAMES[m%12]+Math.floor(m/12-1); });
}

/* ── Self-contained drum synth (mirrors _DRUM_SYNTH_JS but scoped to GarSIa) ── */
function _garciaKick(t,v){
  var ctx=Tone.getContext().rawContext;
  var g=ctx.createGain(); g.gain.setValueAtTime(v,t); g.gain.exponentialRampToValueAtTime(0.001,t+0.4);
  g.connect(ctx.destination);
  var o=ctx.createOscillator(); o.frequency.setValueAtTime(180,t);
  o.frequency.exponentialRampToValueAtTime(28,t+0.35);
  o.connect(g); o.start(t); o.stop(t+0.45);
}
function _garciaSnare(t,v){
  var ctx=Tone.getContext().rawContext;
  var buf=ctx.createBuffer(1,ctx.sampleRate*0.22,ctx.sampleRate);
  var d=buf.getChannelData(0); for(var i=0;i<d.length;i++) d[i]=(Math.random()*2-1);
  var src=ctx.createBufferSource(); src.buffer=buf;
  var hp=ctx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=1200;
  var g=ctx.createGain(); g.gain.setValueAtTime(v*0.7,t); g.gain.exponentialRampToValueAtTime(0.001,t+0.18);
  src.connect(hp); hp.connect(g); g.connect(ctx.destination); src.start(t); src.stop(t+0.22);
}
function _garciaHihat(t,v,open){
  var ctx=Tone.getContext().rawContext;
  var buf=ctx.createBuffer(1,ctx.sampleRate*0.1,ctx.sampleRate);
  var d=buf.getChannelData(0); for(var i=0;i<d.length;i++) d[i]=(Math.random()*2-1);
  var src=ctx.createBufferSource(); src.buffer=buf;
  var hp=ctx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=8000;
  var dur=open?0.32:0.04;
  var g=ctx.createGain(); g.gain.setValueAtTime(v*0.45,t); g.gain.exponentialRampToValueAtTime(0.001,t+dur);
  src.connect(hp); hp.connect(g); g.connect(ctx.destination); src.start(t); src.stop(t+dur+0.01);
}
function _garciaDrumHit(note,t,v){
  if(note===36||note===35) _garciaKick(t,v);
  else if(note===38||note===40) _garciaSnare(t,v);
  else if(note===42||note===44) _garciaHihat(t,v,false);
  else if(note===46) _garciaHihat(t,v,true);
}

/* ── Drum loop builder (swappable pattern) ── */
function _buildDrumLoop() {
  var pat = _DRUM_PATTERNS[_layers.drums.style].steps;
  _drumLoop = new Tone.Sequence(function(time, note){
    if(!_layers.drums.on) return;
    _garciaDrumHit(note, time, note===36?0.9:note===38?0.82:0.45);
  }, pat, '8n');
  _drumLoop.start(0);
}

/* ── Audio init (called after Tone.start() resolves) ── */
function _buildAudio() {
  Tone.Transport.bpm.value = _currentBPM;

  _bassSynth = new Tone.MonoSynth({oscillator:{type:'sawtooth'},
    envelope:{attack:0.02,decay:0.1,sustain:0.3,release:0.4},
    filter:{frequency:500,type:'lowpass'}}).toDestination();
  _bassSynth.volume.value = -4;

  _keysSynth = new Tone.PolySynth(Tone.Synth,{oscillator:{type:'triangle'},
    envelope:{attack:0.02,decay:0.3,sustain:0.4,release:0.8}}).toDestination();
  _keysSynth.volume.value = -10;

  _padSynth = new Tone.PolySynth(Tone.Synth,{oscillator:{type:'sine'},
    envelope:{attack:1.0,decay:0.2,sustain:0.9,release:2.5}}).toDestination();
  _padSynth.volume.value = -14;

  _buildDrumLoop();

  _bassLoop = new Tone.Sequence(function(time, step){
    if(!_layers.bass.on||!_bassSynth) return;
    var ch = _currentChords[_chordIdx]||'C';
    var rm = ch.match(/^([A-G][#b]?)/); if(!rm) return;
    var root = rm[1];
    var style = _layers.bass.style;
    if(style===0){ /* Root: root on 1, root on 3 */
      if(step===0||step===2) _bassSynth.triggerAttackRelease(root+'2','8n',time,0.7);
    } else if(style===1){ /* Walking: root, 3rd, 5th, 7th down */
      var ns2=_chordNotes(ch,2);
      _bassSynth.triggerAttackRelease(ns2[step%ns2.length]||root+'2','8n',time,0.65);
    } else { /* Syncopated: hits on 0,1,3 */
      if(step===0||step===1||step===3) _bassSynth.triggerAttackRelease(root+'2','16n',time,0.75);
    }
  },[0,1,2,3],'4n');
  _bassLoop.start(0);

  _keysLoop = new Tone.Loop(function(time){
    if(!_layers.keys.on||!_keysSynth) return;
    var ch = _currentChords[_chordIdx]||'C';
    var style = _layers.keys.style;
    if(style===0){ /* Block chord */
      _keysSynth.triggerAttackRelease(_chordNotes(ch,4),'2n',time,0.45);
    } else if(style===1){ /* Arpeggio — schedule each note offset */
      var ns2=_chordNotes(ch,4);
      ns2.forEach(function(n,i){
        _keysSynth.triggerAttackRelease(n,'8n',time+Tone.Time('8n').toSeconds()*i,0.4);
      });
    } else { /* Comping — offbeat */
      _keysSynth.triggerAttackRelease(_chordNotes(ch,4),'8n',time+Tone.Time('8n').toSeconds(),0.38);
    }
  },'1m');
  _keysLoop.start(0);

  _padLoop = new Tone.Loop(function(time){
    if(!_layers.pad.on||!_padSynth) return;
    var ch = _currentChords[_chordIdx]||'C';
    var style = _layers.pad.style;
    var dur = style===0?'2m':style===1?'1m':'2m';
    var vel = style===2?0.2:0.3;
    _padSynth.triggerAttackRelease(_chordNotes(ch,3),dur,time,vel);
  },'2m');
  _padLoop.start(0);

  Tone.Transport.start();
}

/* ── Audio init: called from button click (synchronous user-gesture context) ── */
function _initAudio() {
  if(_audioReady) return;
  if(!window.Tone) { console.warn('GarSIa: Tone.js not loaded yet'); return; }
  _audioReady = true;
  /* Tone.start() / AudioContext.resume() MUST be called synchronously from
     a user-gesture handler — Promise callbacks are too late for browser policy */
  Tone.start();
  _buildAudio();
}

/* ── Canvas drawing ── */
var _CONN = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],[5,9],[9,13]
];

function _drawSkeleton(lm, color, W, H) {
  _ctx.save();
  _ctx.scale(-1,1); _ctx.translate(-W,0);
  _ctx.strokeStyle=color; _ctx.lineWidth=2.5;
  _ctx.shadowColor=color; _ctx.shadowBlur=8;
  _CONN.forEach(function(c){
    var a=lm[c[0]],b=lm[c[1]];
    _ctx.beginPath(); _ctx.moveTo(a.x*W,a.y*H); _ctx.lineTo(b.x*W,b.y*H); _ctx.stroke();
  });
  _ctx.shadowBlur=3;
  lm.forEach(function(pt,i){
    var r=[0,4,8,12,16,20].indexOf(i)>=0?6:3;
    _ctx.beginPath(); _ctx.arc(pt.x*W,pt.y*H,r,0,Math.PI*2);
    _ctx.fillStyle=color; _ctx.fill();
  });
  _ctx.restore();
}

function _drawHUD(gesture, W, H) {
  /* BPM — top left, flashes white on tap */
  var bpmCol = _tapFlash>0 ? '#ffffff' : '#33cc66';
  _ctx.font='bold 40px monospace';
  _ctx.fillStyle=bpmCol; _ctx.shadowColor=bpmCol; _ctx.shadowBlur=_tapFlash>0?22:8;
  _ctx.fillText('♪ '+_currentBPM+' BPM', 20, 52);

  /* Chord — below BPM */
  _ctx.font='bold 26px monospace';
  _ctx.fillStyle='#88ccff'; _ctx.shadowColor='#4488ff'; _ctx.shadowBlur=6;
  _ctx.fillText(_currentChords[_chordIdx]||'--', 20, 88);
  _ctx.shadowBlur=0;

  /* Hold-progress ring — centre screen */
  if (_holdProgress>0 && gesture && gesture.name!=='--') {
    _ctx.beginPath();
    _ctx.arc(W/2,H/2,54,-Math.PI/2,-Math.PI/2+_holdProgress*Math.PI*2);
    _ctx.strokeStyle=gesture.color; _ctx.lineWidth=5;
    _ctx.shadowColor=gesture.color; _ctx.shadowBlur=14; _ctx.stroke();
    _ctx.shadowBlur=0;
  }

  /* Active layer + style hint — top right */
  _ctx.font='bold 16px monospace';
  _ctx.fillStyle='#aaffcc'; _ctx.shadowColor='#33cc66'; _ctx.shadowBlur=6;
  _ctx.textAlign='right';
  _ctx.fillText('▶ '+_activeLayer.toUpperCase()+': '+_styleName(_activeLayer), W-16, 30);
  _ctx.font='12px monospace'; _ctx.fillStyle='#4a8a5a'; _ctx.shadowBlur=0;
  _ctx.fillText('← swipe wrist → to change style', W-16, 50);
  _ctx.textAlign='left';

  /* Gesture label — bottom left */
  if (gesture && gesture.name!=='--') {
    _ctx.font='bold 30px monospace';
    _ctx.fillStyle=gesture.color; _ctx.shadowColor=gesture.color; _ctx.shadowBlur=10;
    _ctx.fillText(gesture.icon+'  '+gesture.name, 20, H-56);
    _ctx.shadowBlur=0;
    _ctx.font='12px monospace'; _ctx.fillStyle='#4a8a5a';
    if (gesture.layer) _ctx.fillText('→ '+gesture.layer.toUpperCase().replace('_',' '), 20, H-34);
  }

  /* Beat dots — bottom right */
  if (window.Tone && Tone.Transport) {
    try {
      var beat = parseInt(Tone.Transport.position.split(':')[1])||0;
      for (var i=0;i<4;i++) {
        var active=(i===beat%4);
        _ctx.beginPath(); _ctx.arc(W-112+i*28, H-22, active?8:5, 0, Math.PI*2);
        _ctx.fillStyle=active?'#33cc66':'#1a4a2a'; _ctx.fill();
      }
    } catch(e){}
  }
}

function _onResults(results, ts) {
  var W=_canvas.width, H=_canvas.height;
  _ctx.clearRect(0,0,W,H);

  /* Mirrored camera feed */
  _ctx.save(); _ctx.scale(-1,1); _ctx.translate(-W,0);
  _ctx.drawImage(results.image,0,0,W,H); _ctx.restore();

  /* Vignette */
  var vg=_ctx.createRadialGradient(W/2,H/2,H*0.28,W/2,H/2,H*0.88);
  vg.addColorStop(0,'rgba(0,0,0,0.08)'); vg.addColorStop(1,'rgba(0,0,0,0.62)');
  _ctx.fillStyle=vg; _ctx.fillRect(0,0,W,H);

  var gesture=null;
  if (results.multiHandLandmarks&&results.multiHandLandmarks.length) {
    for (var h=0;h<results.multiHandLandmarks.length;h++) {
      var lm=results.multiHandLandmarks[h];
      var isRight=results.multiHandedness&&results.multiHandedness[h]&&
                  results.multiHandedness[h].label==='Right';
      var g=_classify(lm,isRight);
      if(h===0){
        gesture=g;
        _detectTap(lm[0].y,ts);
        /* ── Horizontal swipe detection (wrist X) ── */
        var wx=lm[0].x;
        if(_lastWristX!==null && _swipeCooldown===0) {
          var dx=wx-_lastWristX;
          if(Math.abs(dx)>0.12) {
            _swipeStyle(dx>0?-1:1); /* mirrored: move right = wrist goes left in raw */
            _swipeCooldown=25;
          }
        }
        _lastWristX=wx;
        if(_swipeCooldown>0) _swipeCooldown--;
      }
      _drawSkeleton(lm,g.color,W,H);
    }
  } else {
    _lastWristY=null; _lastWristX=null;
  }

  /* Hold gesture logic */
  if(_gestureCooldown>0) _gestureCooldown--;
  if (gesture&&gesture.name!=='--') {
    if (gesture.name===_prevGestureName) _gestureHoldFrames++;
    else { _gestureHoldFrames=0; }
    if (_gestureHoldFrames===HOLD_FRAMES && _gestureCooldown===0) {
      _trigger(gesture.layer);
      _gestureHoldFrames=0;
      _gestureCooldown=COOLDOWN_FRAMES;
    }
    _holdProgress=Math.min(1,_gestureHoldFrames/HOLD_FRAMES);
  } else {
    _gestureHoldFrames=0; _holdProgress=0;
  }
  _prevGestureName=gesture?gesture.name:null;
  _currentGesture=gesture;

  _drawHUD(gesture,W,H);
}

/* ── Dynamic script loader ── */
function _loadScript(src,cb) {
  var s=document.createElement('script'); s.src=src; s.onload=cb;
  document.head.appendChild(s);
}

/* ── Public API ── */
window.garciaStart = function() {
  if (_running) return;
  /* ↓ Must be FIRST — AudioContext.resume() requires synchronous user-gesture context */
  _initAudio();

  var btn=document.getElementById('garcia-start-btn');
  var status=document.getElementById('garcia-status');
  if(btn){btn.textContent='Starting…';btn.disabled=true;}
  if(status) status.textContent='Loading hand tracking…';

  _video=document.getElementById('garcia-video');
  _canvas=document.getElementById('garcia-canvas');
  if(!_video||!_canvas){console.error('GarSIa: DOM elements not found');return;}
  _ctx=_canvas.getContext('2d');

  function _resize(){
    var vp=document.getElementById('garcia-viewport');
    if(!vp)return;
    _canvas.width=vp.offsetWidth; _canvas.height=vp.offsetHeight;
  }
  _resize(); window.addEventListener('resize',_resize);

  function _boot() {
    _hands=new Hands({locateFile:window._garciaLocate});
    _hands.setOptions({maxNumHands:2,modelComplexity:1,
      minDetectionConfidence:0.75,minTrackingConfidence:0.6});
    _hands.onResults(function(r){_onResults(r,performance.now());});
    _camera=new Camera(_video,{
      onFrame:async function(){await _hands.send({image:_video});},
      width:1280,height:720
    });
    _camera.start().then(function(){
      _running=true;
      if(btn){btn.style.display='none';}
      var ov=document.getElementById('garcia-start-overlay'); if(ov) ov.style.display='none';
      if(status) status.textContent='✓ Camera live — show a hand to begin';
    }).catch(function(e){
      if(status) status.textContent='✗ Camera error: '+e.message+' (allow camera in browser)';
      if(btn){btn.textContent='► Start Camera';btn.disabled=false;}
    });
  }

  /* Pinned CDN versions — unpinned @0.4 resolves inconsistently */
  var _MP_VER  = '0.4.1675469240';
  var _CU_VER  = '0.3.1675466862';
  var _MP_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@'+_MP_VER+'/';
  var _CU_URL  = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@'+_CU_VER+'/camera_utils.js';

  /* Override locateFile so the Hands constructor finds its WASM at the pinned path */
  window._garciaLocate = function(f){ return _MP_BASE+f; };

  function _err(msg) {
    if(status) status.textContent = '✗ '+msg+' — check DevTools console';
    if(btn){ btn.textContent='► Retry'; btn.disabled=false; }
  }

  function _loadScript(src,cb,errcb) {
    var s=document.createElement('script'); s.src=src; s.crossOrigin='anonymous';
    s.onload=cb; s.onerror=errcb||function(){ _err('Failed to load '+src.split('/').pop()); };
    document.head.appendChild(s);
  }

  function _bootWhenReady() {
    /* Hands constructor may not be available on the same tick as onload */
    if(typeof Hands==='undefined' || typeof Camera==='undefined') {
      setTimeout(_bootWhenReady, 80); return;
    }
    if(status) status.textContent = 'Initialising model…';
    _boot();
  }

  var _loadTimeout = setTimeout(function(){
    if(!_running) _err('Timed out — CDN may be blocked, try a different network');
  }, 35000);

  if(typeof Hands==='undefined') {
    if(status) status.textContent = 'Downloading camera utils…';
    _loadScript(_CU_URL, function(){
      if(status) status.textContent = 'Downloading hand-tracking model (≈8 MB)…';
      _loadScript(_MP_BASE+'hands.js', function(){
        clearTimeout(_loadTimeout);
        _bootWhenReady();
      }, function(){ _err('hands.js failed to load'); });
    }, function(){ _err('camera_utils.js failed to load'); });
  } else {
    clearTimeout(_loadTimeout);
    _bootWhenReady();
  }
};

window.garciaStop = function() {
  _running=false;
  if(_camera){try{_camera.stop();}catch(e){}}
  if(window.Tone&&Tone.Transport) Tone.Transport.stop();
  Object.keys(_layers).forEach(function(k){_layers[k].on=false;});
  _refreshLayers();
  var btn=document.getElementById('garcia-start-btn');
  var ov=document.getElementById('garcia-start-overlay');
  if(btn){btn.textContent='► Start Camera';btn.disabled=false;btn.style.display='';}
  if(ov) ov.style.display='';
  var status=document.getElementById('garcia-status');
  if(status) status.textContent='Stopped';
};

window.garciaBPMSet = function(val) {
  var v=parseInt(val); if(isNaN(v)||v<40||v>200) return;
  _currentBPM=_snapBPM(v); _bpmHistory=[];
  if(window.Tone) Tone.Transport.bpm.value=_currentBPM;
  var el=document.getElementById('garcia-bpm-val'); if(el) el.textContent=_currentBPM;
};

window.garciaChordSet = function(val) {
  var ch=val.split(',').map(function(s){return s.trim();}).filter(Boolean);
  if(ch.length){_currentChords=ch;_chordIdx=0;
    var el=document.getElementById('garcia-chord-now'); if(el) el.textContent=ch[0];}
};

})();
"""

_ROLL_JS = r"""
(function(){
  /* -- constants -- */
  const NOTE_H=12, KEY_W=44, VEL_H=48;
  const SNAP_VALS=[0.0625,0.125,0.25,0.5,1.0]; // 1/16 1/8 1/4 1/2 1
  const SNAP_LABELS=['1/16','1/8','1/4','1/2','1'];
  const BLACK_PCS=[1,3,6,8,10];
  const NOTE_NAMES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
  const TRACK_COLS=['#3d85f5','#f5523d','#2dc98a','#c93df5','#f5a623','#23dfc9','#f523a3','#a3c923'];

  /* -- state -- */
  let notes=[], totalTime=0, bpm=120, ns=null;
  let zoom=1.5, pxSec=80;
  let snapIdx=2; // default 1/4
  let tool='draw'; // draw | erase | select
  let drawing=null, selNote=null, resizing=null;
  let minP=36, maxP=84;
  let velDrag=null;
  let MIDI_URL=window.__MIDI_URL__;

  /* -- helpers -- */
  function snapTime(t){
    const spb=60/bpm, grid=SNAP_VALS[snapIdx]*spb;
    return Math.round(t/grid)*grid;
  }
  function tToX(t){ return KEY_W + t*pxSec*zoom; }
  function xToT(x){ return (x-KEY_W)/(pxSec*zoom); }
  function pToY(p){ return (maxP-p)*NOTE_H; }
  function yToP(y){ return maxP - Math.floor(y/NOTE_H); }
  function noteColor(n){ return TRACK_COLS[(n.program||n.instrument||0)%TRACK_COLS.length]; }

  /* -- canvas event coords (correct for scroll) -- */
  function evXY(e, cv){
    const r=cv.getBoundingClientRect();
    const wrap=document.getElementById('roll-wrap');
    return { x: e.clientX - r.left + wrap.scrollLeft,
             y: e.clientY - r.top  + wrap.scrollTop };
  }

  /* -- status helper -- */
  function setStatus(msg, col){
    const el=document.getElementById('roll-status');
    if(el){ el.textContent=msg; el.style.color=col||'#5577aa'; }
  }

  /* -- MIDI load (uses @tonejs/midi -- tiny, fast) -- */
  async function load(url){
    if(url) MIDI_URL=url;
    if(!MIDI_URL){ setStatus('No MIDI URL','#cc4444'); return; }
    setStatus('Loading...','#557799');
    try{
      const resp=await fetch(MIDI_URL);
      if(!resp.ok){ setStatus('Fetch failed: '+resp.status,'#cc4444'); return; }
      const buf=await resp.arrayBuffer();
      const midi=new Midi(buf);
      notes=[];
      midi.tracks.forEach((track,ti)=>{
        const prog=track.instrument?track.instrument.number:0;
        track.notes.forEach(n=>{
          notes.push({
            pitch:n.midi, startTime:n.time, endTime:n.time+n.duration,
            velocity:Math.round(n.velocity*127)||80,
            instrument:ti, program:prog
          });
        });
      });
      notes.sort((a,b)=>a.startTime-b.startTime);
      totalTime=midi.duration||4;
      bpm=midi.header.tempos&&midi.header.tempos.length ? midi.header.tempos[0].bpm : 120;
      if(notes.length){
        const ps=notes.map(n=>n.pitch);
        minP=Math.max(0,  Math.min(...ps)-3);
        maxP=Math.min(127,Math.max(...ps)+3);
      }
      setStatus(notes.length+' notes  |  BPM '+Math.round(bpm)+'  |  Draw = add  Right-click = delete','#44aa66');
      buildLegend();
      render(); renderVel();
    }catch(e){
      setStatus('Error: '+e.message,'#cc4444');
      console.error('roll load:',e);
    }
  }

  /* -- simple MIDI export (no Magenta needed) -- */
  function writeMidi(noteList, bpmVal){
    const tpq=480;
    const uspb=Math.round(60000000/bpmVal);
    function varlen(n){
      if(n<128) return [n];
      const out=[];
      out.unshift(n&0x7f); n>>=7;
      while(n>0){out.unshift((n&0x7f)|0x80);n>>=7;}
      return out;
    }
    // group by instrument/channel
    const tracks={};
    noteList.forEach(n=>{
      const ch=n.instrument%16;
      if(!tracks[ch]) tracks[ch]=[];
      tracks[ch].push(n);
    });
    const trackChunks=[];
    // tempo track
    const tempoEvt=[0,0xFF,0x51,0x03,(uspb>>16)&0xFF,(uspb>>8)&0xFF,uspb&0xFF,0,0xFF,0x2F,0];
    const tData=new Uint8Array(tempoEvt);
    const tLen=[(tData.length>>24)&0xFF,(tData.length>>16)&0xFF,(tData.length>>8)&0xFF,tData.length&0xFF];
    trackChunks.push([0x4D,0x54,0x72,0x6B,...tLen,...tData]);
    // note tracks
    Object.keys(tracks).forEach(ch=>{
      const ch_=parseInt(ch);
      const evts=[];
      tracks[ch_].sort((a,b)=>a.startTime-b.startTime);
      tracks[ch_].forEach(n=>{
        const t0=Math.round(n.startTime*tpq*bpmVal/60);
        const t1=Math.round(n.endTime*tpq*bpmVal/60);
        evts.push({t:t0,d:[0x90|ch_,n.pitch,n.velocity||80]});
        evts.push({t:t1,d:[0x80|ch_,n.pitch,0]});
      });
      evts.sort((a,b)=>a.t-b.t);
      let prev=0; const raw=[];
      evts.forEach(ev=>{
        raw.push(...varlen(ev.t-prev)); prev=ev.t;
        raw.push(...ev.d);
      });
      raw.push(0,0xFF,0x2F,0);
      const len=[(raw.length>>24)&0xFF,(raw.length>>16)&0xFF,(raw.length>>8)&0xFF,raw.length&0xFF];
      trackChunks.push([0x4D,0x54,0x72,0x6B,...len,...raw]);
    });
    // header
    const numTracks=trackChunks.length;
    const header=[0x4D,0x54,0x68,0x64,0,0,0,6,0,1,(numTracks>>8)&0xFF,numTracks&0xFF,(tpq>>8)&0xFF,tpq&0xFF];
    const all=[...header,...trackChunks.flat()];
    return new Uint8Array(all);
  }

  /* -- main render -- */
  function render(){
    const cv=document.getElementById('rc');
    if(!cv) return;
    const ctx=cv.getContext('2d');
    const rows=maxP-minP+1;
    const W=Math.max(600, Math.ceil(tToX(totalTime))+60);
    const H=rows*NOTE_H;
    if(cv.width!==W||cv.height!==H){ cv.width=W; cv.height=H; }

    /* rows + piano keys */
    for(let p=minP; p<=maxP; p++){
      const y=pToY(p), pc=p%12, isB=BLACK_PCS.includes(pc);
      /* grid row */
      ctx.fillStyle=isB ? '#080c12' : '#0d1220';
      ctx.fillRect(KEY_W, y, W-KEY_W, NOTE_H);
      /* C marker row */
      if(pc===0){
        ctx.fillStyle='#131f30'; ctx.fillRect(KEY_W,y,W-KEY_W,1);
      }
      /* piano key */
      ctx.fillStyle=isB ? '#181c24' : '#2a3848';
      ctx.fillRect(0, y, KEY_W, NOTE_H);
      if(isB){
        ctx.fillStyle='#0e1018';
        ctx.fillRect(0, y, KEY_W*0.65, NOTE_H);
      }
      if(pc===0){
        ctx.fillStyle='#6699bb'; ctx.font='bold 8px sans-serif';
        ctx.fillText('C'+(Math.floor(p/12)-1), 2, y+NOTE_H-2);
      } else if(pc===5){
        ctx.fillStyle='#3a5566'; ctx.font='7px sans-serif';
        ctx.fillText('F'+(Math.floor(p/12)-1), 2, y+NOTE_H-2);
      }
      /* key border */
      ctx.fillStyle='#0d1520';
      ctx.fillRect(0, y+NOTE_H-1, KEY_W, 1);
    }

    /* beat/bar grid */
    const spb=60/bpm;
    const totalBeats=Math.ceil(totalTime/spb)+2;
    for(let b=0; b<=totalBeats; b++){
      const x=tToX(b*spb);
      const isBar=b%4===0;
      ctx.fillStyle=isBar ? '#1c3050' : '#0e1825';
      ctx.fillRect(x, 0, isBar?2:1, H);
      if(isBar){
        ctx.fillStyle='#2a4a70'; ctx.font='8px sans-serif';
        ctx.fillText('Bar '+(b/4+1), x+3, 9);
      }
    }

    /* sub-grid (snap lines) */
    const spbSnap=(60/bpm)*SNAP_VALS[snapIdx];
    if(spbSnap < spb){
      const steps=Math.ceil(totalTime/spbSnap)+2;
      ctx.fillStyle='#0a1018';
      for(let s=0;s<=steps;s++){
        const x=tToX(s*spbSnap);
        if(Math.abs(x-tToX(Math.round(s*spbSnap/spb)*spb))<1) continue;
        ctx.fillRect(x,0,1,H);
      }
    }

    /* notes */
    notes.forEach(n=>{
      const x=tToX(n.startTime);
      const w=Math.max(4,(n.endTime-n.startTime)*pxSec*zoom-1);
      const y=pToY(n.pitch)+1, h=NOTE_H-2;
      const col=noteColor(n);
      /* main body */
      ctx.globalAlpha=1;
      ctx.fillStyle=col;
      ctx.fillRect(x,y,w,h);
      /* bright top sheen */
      ctx.fillStyle='rgba(255,255,255,0.45)';
      ctx.fillRect(x,y,w,2);
      /* right resize grip */
      ctx.fillStyle='rgba(255,255,255,0.5)';
      ctx.fillRect(x+w-4,y+2,3,h-4);
      /* selected outline */
      if(n===selNote){
        ctx.strokeStyle='#ffffff'; ctx.lineWidth=1.5;
        ctx.strokeRect(x+0.5,y+0.5,w-1,h-1);
      }
    });

    /* ghost note while drawing */
    if(drawing){
      const x=tToX(drawing.startTime);
      const w=Math.max(4,tToX(drawing.endTime)-x);
      const y=pToY(drawing.pitch)+1;
      ctx.fillStyle='rgba(120,190,255,0.6)';
      ctx.fillRect(x,y,w,NOTE_H-2);
    }
  }

  /* -- velocity lane -- */
  function renderVel(){
    const cv=document.getElementById('vel-cv');
    if(!cv) return;
    const W=document.getElementById('rc').width||600;
    cv.width=W; cv.height=VEL_H;
    const ctx=cv.getContext('2d');
    ctx.fillStyle='#07090f';
    ctx.fillRect(0,0,W,VEL_H);
    notes.forEach(n=>{
      const x=tToX(n.startTime);
      const h=Math.round((n.velocity/127)*(VEL_H-4));
      const col=noteColor(n);
      ctx.fillStyle=col+'99';
      ctx.fillRect(x, VEL_H-h, 4, h);
      /* stem */
      ctx.fillStyle=col;
      ctx.fillRect(x, VEL_H-h, 4, 2);
    });
  }

  /* -- hit test -- */
  function hitNote(x, y){
    for(let i=notes.length-1;i>=0;i--){
      const n=notes[i];
      const nx=tToX(n.startTime), nw=Math.max(3,(n.endTime-n.startTime)*pxSec*zoom-1);
      const ny=pToY(n.pitch)+1, nh=NOTE_H-2;
      if(x>=nx&&x<=nx+nw&&y>=ny&&y<=ny+nh) return {note:n,handle:x>nx+nw-6};
    }
    return null;
  }

  /* -- mouse -- */
  function setupMouse(){
    const cv=document.getElementById('rc');
    const wrap=document.getElementById('roll-wrap');
    if(!cv) return;

    cv.addEventListener('contextmenu',e=>e.preventDefault());

    cv.addEventListener('mousedown',e=>{
      e.preventDefault();
      const {x,y}=evXY(e,cv);
      const pitch=yToP(y), time=xToT(x);
      if(pitch<0||pitch>127) return;

      /* right-click always erases */
      if(e.button===2){
        const hit=hitNote(x,y);
        if(hit){ notes.splice(notes.indexOf(hit.note),1); render(); renderVel(); }
        return;
      }

      if(tool==='erase'){
        const hit=hitNote(x,y);
        if(hit){ notes.splice(notes.indexOf(hit.note),1); selNote=null; render(); renderVel(); }
        return;
      }

      if(tool==='select'){
        const hit=hitNote(x,y);
        if(hit){
          selNote=hit.note;
          if(hit.handle) resizing={note:hit.note};
          else { /* drag move -- store offset */ resizing=null; drawing={move:hit.note, ox:x-tToX(hit.note.startTime), oy:y-pToY(hit.note.pitch)}; }
        } else { selNote=null; }
        render(); return;
      }

      /* draw tool */
      if(x<KEY_W) return;
      const st=Math.max(0,snapTime(time));
      drawing={pitch, startTime:st, endTime:st+(60/bpm)*SNAP_VALS[snapIdx]};
    });

    cv.addEventListener('mousemove',e=>{
      const {x,y}=evXY(e,cv);
      const time=xToT(x), pitch=yToP(y);

      if(resizing){
        const n=resizing.note;
        const et=Math.max(n.startTime+(60/bpm)*SNAP_VALS[snapIdx], snapTime(time));
        n.endTime=et;
        totalTime=Math.max(totalTime,et+0.5);
        render(); renderVel(); return;
      }

      if(drawing){
        if(drawing.move){
          /* moving a selected note */
          const n=drawing.move;
          const newSt=Math.max(0,snapTime(xToT(x-drawing.ox)));
          const dur=n.endTime-n.startTime;
          n.startTime=newSt; n.endTime=newSt+dur;
          n.pitch=Math.max(minP,Math.min(maxP,yToP(y-drawing.oy+NOTE_H/2)));
          render(); renderVel(); return;
        }
        /* ghost note */
        const et=Math.max(drawing.startTime+(60/bpm)*SNAP_VALS[snapIdx], snapTime(time));
        drawing.endTime=et;
        render();
      }
    });

    function endDraw(e){
      if(drawing && !drawing.move && drawing.pitch>=0 && drawing.pitch<=127){
        const dur=drawing.endTime-drawing.startTime;
        if(dur>0.01){
          notes.push({pitch:drawing.pitch, startTime:drawing.startTime,
            endTime:drawing.endTime, velocity:80, instrument:0, program:0});
          notes.sort((a,b)=>a.startTime-b.startTime);
          totalTime=Math.max(totalTime, drawing.endTime+0.5);
        }
      }
      drawing=null; resizing=null;
      render(); renderVel();
    }
    cv.addEventListener('mouseup', endDraw);
    document.addEventListener('mouseup', ()=>{ if(drawing||resizing){ endDraw(); }});

    /* velocity drag */
    const vcv=document.getElementById('vel-cv');
    if(vcv){
      vcv.addEventListener('mousedown',e=>{
        e.preventDefault();
        const r=vcv.getBoundingClientRect();
        const x=e.clientX-r.left;
        let closest=null, dist=Infinity;
        notes.forEach(n=>{ const d=Math.abs(tToX(n.startTime)-x); if(d<dist){dist=d;closest=n;} });
        if(closest&&dist<12) velDrag=closest;
      });
      vcv.addEventListener('mousemove',e=>{
        if(!velDrag) return;
        const r=vcv.getBoundingClientRect();
        const yRel=r.bottom-e.clientY;
        velDrag.velocity=Math.max(1,Math.min(127,Math.round((yRel/VEL_H)*127)));
        render(); renderVel();
      });
      vcv.addEventListener('mouseup',()=>{ velDrag=null; });
    }
  }

  /* -- legend -- */
  function buildLegend(){
    const el=document.getElementById('tracks-legend');
    if(!el) return;
    const progs={};
    notes.forEach(n=>{ const k=n.program||0; progs[k]=(progs[k]||0)+1; });
    const NAMES=['Piano','Chromatic Perc','Organ','Guitar','Bass',
      'Strings','Ensemble','Brass','Reed','Pipe','Synth Lead','Synth Pad',
      'Synth FX','Ethnic','Percussive','Sound FX'];
    el.innerHTML=Object.keys(progs).map(p=>{
      const pi=parseInt(p), col=TRACK_COLS[pi%TRACK_COLS.length];
      const fam=NAMES[Math.floor(pi/8)]||'Prog '+p;
      return `<span><span class="track-dot" style="background:${col}"></span>${fam} (${progs[p]})</span>`;
    }).join('');
  }

  /* -- export -- */
  function exportMidi(){
    if(!notes.length){ alert('Nothing to export.'); return; }
    try{
      const bytes=writeMidi(notes, bpm);
      const a=document.createElement('a');
      a.href=URL.createObjectURL(new Blob([bytes],{type:'audio/midi'}));
      a.download='edited_arrangement.mid'; a.click();
    }catch(e){ alert('Export error: '+e.message); }
  }

  /* -- toolbar wiring (called after load) -- */
  function wireTool(id, t){
    const btn=document.getElementById(id);
    if(!btn) return;
    btn.onclick=function(){
      tool=t;
      ['draw-btn','erase-btn','sel-btn'].forEach(b=>{
        const el=document.getElementById(b);
        if(el) el.classList.remove('on');
      });
      this.classList.add('on');
      const wrap=document.getElementById('roll-wrap');
      wrap.className='';
      if(t==='draw') wrap.classList.add('draw');
      if(t==='erase') wrap.classList.add('erase');
      document.getElementById('hint').textContent=
        t==='draw'  ? 'Click+drag = draw note  |  Right-click = erase' :
        t==='erase' ? 'Click a note to delete it' :
                      'Click to select  |  Drag body = move  |  Drag right edge = resize';
    };
  }

  /* -- init -- */
  function init(){
    setupMouse();
    wireTool('draw-btn','draw');
    wireTool('erase-btn','erase');
    wireTool('sel-btn','select');
    /* default active */
    const db=document.getElementById('draw-btn');
    if(db){ db.classList.add('on'); document.getElementById('roll-wrap').classList.add('draw'); }

    const snap=document.getElementById('snap-sel');
    if(snap){
      snap.value=snapIdx;
      snap.onchange=()=>{ snapIdx=parseInt(snap.value); render(); };
    }
    document.getElementById('zi').onclick=()=>{ zoom=Math.min(5,zoom*1.35); render(); renderVel(); };
    document.getElementById('zo').onclick=()=>{ zoom=Math.max(0.3,zoom/1.35); render(); renderVel(); };
    document.getElementById('dl-btn').onclick=exportMidi;
  }

  /* -- playhead needle -- */
  function startPlayhead(){
    const wrap=document.getElementById('roll-wrap');
    const ph=document.getElementById('ph');
    if(!wrap||!ph) return;

    function frame(){
      requestAnimationFrame(frame);
      const cv=document.getElementById('rc');
      if(!cv) return;
      const H=cv.height;
      // resize overlay to match visible viewport
      if(ph.width!==wrap.clientWidth||ph.height!==H){
        ph.width=wrap.clientWidth; ph.height=H;
      }
      const ctx=ph.getContext('2d');
      ctx.clearRect(0,0,ph.width,ph.height);
      const t=(typeof Tone!=='undefined'&&Tone.Transport.state==='started')?Tone.Transport.seconds:0;
      if(t<=0) return;
      const xCanvas=tToX(t);           // position in full canvas coords
      const xView=xCanvas-wrap.scrollLeft; // position in viewport
      // auto-scroll: keep needle 30% from left edge
      const target=xCanvas-wrap.clientWidth*0.3;
      if(xView>wrap.clientWidth*0.75||xView<KEY_W)
        wrap.scrollLeft=Math.max(0,target);
      // draw needle
      const xDraw=xCanvas-wrap.scrollLeft;
      if(xDraw<KEY_W||xDraw>wrap.clientWidth) return;
      ctx.fillStyle='rgba(255,255,255,0.15)';
      ctx.fillRect(xDraw,0,2,H);
      ctx.fillStyle='#ffffff';
      ctx.fillRect(xDraw,0,1.5,H);
      // glow
      const grad=ctx.createLinearGradient(xDraw-6,0,xDraw+6,0);
      grad.addColorStop(0,'rgba(100,180,255,0)');
      grad.addColorStop(0.5,'rgba(100,180,255,0.35)');
      grad.addColorStop(1,'rgba(100,180,255,0)');
      ctx.fillStyle=grad;
      ctx.fillRect(xDraw-6,0,12,H);
    }
    frame();
  }

  /* wait for @tonejs/midi then load */
  function waitAndLoad(url){
    if(typeof Midi!=='undefined'){ load(url); }
    else { setTimeout(()=>waitAndLoad(url), 100); }
  }

  window._rollReload=function(url){ waitAndLoad(url); };
  window.addEventListener('load',()=>{ init(); waitAndLoad(MIDI_URL); startPlayhead(); });
})();
"""

def _make_simple_player(midi_path: str) -> str:
    """Build a single-file MIDI player iframe and return the iframe HTML."""
    midi_fname   = Path(midi_path).name
    player_fname = midi_fname.replace(".mid", "_player.html")
    midi_url     = f"http://127.0.0.1:{PREVIEW_PORT}/{midi_fname}"
    (OUTPUTS_DIR / player_fname).write_text(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/tone@14.7.77/build/Tone.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tonejs/midi@2.0.27/build/Midi.js"></script>
<script>{_DRUM_SYNTH_JS}</script>
<style>{_ROLL_CSS}
  #play-btn{{background:#1a3a80;color:#aaccff;border:1px solid #2244aa;}}
  #stop-btn{{background:#1a1a2a;color:#778899;border:1px solid #222;}}
</style></head><body>
<div id="toolbar">
  <button class="tb" id="play-btn" onclick="tonePlay()">&#9654; Play</button>
  <button class="tb" id="stop-btn" onclick="toneStop()">&#9646;&#9646; Stop</button>
  <span class="tb-sep"></span>
  <button class="tb on" id="draw-btn">Draw</button>
  <button class="tb" id="erase-btn">Erase</button>
  <button class="tb" id="sel-btn">Select</button>
  <span class="tb-sep"></span>
  <span id="snap-label">Snap</span>
  <select id="snap-sel" style="background:#141c28;color:#6688aa;border:1px solid #1e2a3a;border-radius:3px;font-size:10px;padding:2px 4px;">
    <option value="0">1/16</option><option value="1">1/8</option>
    <option value="2" selected>1/4</option><option value="3">1/2</option><option value="4">1</option>
  </select>
  <span class="tb-sep"></span>
  <button class="tb" id="zi">+</button><button class="tb" id="zo">-</button>
  <span class="tb-sep"></span>
  <button class="tb" id="dl-btn">Export MIDI</button>
  <span id="hint">Click+drag = draw note  |  Right-click = erase</span>
</div>
<div id="roll-status" style="padding:3px 6px;font-size:9px;color:#557799;background:#090d14;">Loading...</div>
<div id="roll-area">
  <div id="roll-wrap" class="draw" style="position:relative;"><canvas id="rc"></canvas><canvas id="ph" style="position:absolute;top:0;left:0;pointer-events:none;z-index:3;"></canvas></div>
  <div id="vel-wrap"><canvas id="vel-cv"></canvas><span id="vel-label">vel</span></div>
</div>
<div id="tracks-legend"></div>
<script>
window.__MIDI_URL__="{midi_url}";
var _toneSynth=null,_toneParts=[],_midiData=null;
function _initSynth(){{if(!_toneSynth){{_toneSynth=new Tone.PolySynth(Tone.Synth,{{oscillator:{{type:'triangle'}},envelope:{{attack:0.02,decay:0.1,sustain:0.5,release:0.8}}}}).toDestination();}}}}
function toneStop(){{Tone.Transport.stop();Tone.Transport.cancel(0);_toneParts.forEach(p=>{{try{{p.dispose();}}catch(e){{}}}});_toneParts=[];}}
function tonePlay(){{
  if(!_midiData)return;
  toneStop();_initSynth();
  var tempos=_midiData.header.tempos;
  Tone.Transport.bpm.value=tempos&&tempos.length?tempos[0].bpm:120;
  _midiData.tracks.forEach(function(track){{
    var isDrum=(track.instrument&&track.instrument.percussion)||track.name==='drums';
    var evts=track.notes.map(n=>[n.time,n]);
    if(!evts.length)return;
    if(isDrum){{
      var p=new Tone.Part(function(time,n){{
        if(window._drumHit) window._drumHit(n.midi, time, n.velocity);
      }},evts);
      p.start(0);_toneParts.push(p);
    }}else{{
      var p=new Tone.Part(function(time,n){{_toneSynth.triggerAttackRelease(n.name,Math.max(n.duration,0.08),time,n.velocity);}},evts);
      p.start(0);_toneParts.push(p);
    }}
  }});
  Tone.Transport.start('+0.05');
}}
window._rollReload=function(url){{
  toneStop();_midiData=null;
  fetch(url).then(r=>r.arrayBuffer()).then(b=>{{_midiData=new Midi(b);}});
  if(window._rollReloadCb)window._rollReloadCb(url);
}};
fetch("{midi_url}").then(r=>r.arrayBuffer()).then(b=>{{_midiData=new Midi(b);}}).catch(()=>{{}});
window._rollReloadCb=null;
{_ROLL_JS}
</script>
</body></html>""", encoding="utf-8")
    player_url = f"http://127.0.0.1:{PREVIEW_PORT}/{player_fname}"
    return f'<iframe src="{player_url}" style="width:100%;height:420px;border:none;border-radius:8px;" loading="lazy"></iframe>'


def _make_sync_player(midi_path: str, audio_path: str, start_offset: float) -> str:
    """Build a player that syncs MIDI + original audio, return iframe HTML."""
    midi_fname   = Path(midi_path).name
    stem_name    = Path(midi_path).stem
    audio_suffix = Path(audio_path).suffix
    audio_fname  = stem_name + "_input" + audio_suffix
    shutil.copy2(audio_path, OUTPUTS_DIR / audio_fname)

    midi_url   = f"http://127.0.0.1:{PREVIEW_PORT}/{midi_fname}"
    audio_url  = f"http://127.0.0.1:{PREVIEW_PORT}/{audio_fname}"
    player_fname = stem_name + "_player.html"
    (OUTPUTS_DIR / player_fname).write_text(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/tone@14.7.77/build/Tone.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tonejs/midi@2.0.27/build/Midi.js"></script>
<script>{_DRUM_SYNTH_JS}</script>
<style>{_ROLL_CSS}
  #btns{{display:flex;gap:6px;margin-bottom:6px;}}
  #play-btn{{background:#1a3a80;color:#aaccff;border:1px solid #2244aa;}} #stop-btn{{background:#1a1a2a;color:#778899;border:1px solid #222;}}
  #orig{{width:100%;height:28px;display:block;margin-bottom:5px;}}
  #timeinfo{{font-size:9px;color:#334455;margin-bottom:4px;}}
</style></head><body>
<div id="btns">
  <button class="tb" id="play-btn">&#9654; Play Both</button>
  <button class="tb" id="stop-btn">&#9646;&#9646; Stop</button>
</div>
<audio id="orig" src="{audio_url}" controls style="width:100%;height:28px;margin-bottom:4px;"></audio>
<div id="timeinfo"></div>
<div id="toolbar">
  <button class="tb on" id="draw-btn">Draw</button>
  <button class="tb" id="erase-btn">Erase</button>
  <button class="tb" id="sel-btn">Select</button>
  <span class="tb-sep"></span>
  <span id="snap-label">Snap</span>
  <select id="snap-sel" style="background:#141c28;color:#6688aa;border:1px solid #1e2a3a;border-radius:3px;font-size:10px;padding:2px 4px;">
    <option value="0">1/16</option><option value="1">1/8</option>
    <option value="2" selected>1/4</option><option value="3">1/2</option><option value="4">1</option>
  </select>
  <span class="tb-sep"></span>
  <button class="tb" id="zi">+</button><button class="tb" id="zo">-</button>
  <span class="tb-sep"></span>
  <button class="tb" id="dl-btn">Export MIDI</button>
  <span id="hint">Click+drag = draw note  |  Right-click = erase</span>
</div>
<div id="roll-status" style="padding:3px 6px;font-size:9px;color:#557799;background:#090d14;">Loading...</div>
<div id="roll-area">
  <div id="roll-wrap" class="draw" style="position:relative;"><canvas id="rc"></canvas><canvas id="ph" style="position:absolute;top:0;left:0;pointer-events:none;z-index:3;"></canvas></div>
  <div id="vel-wrap"><canvas id="vel-cv"></canvas><span id="vel-label">vel</span></div>
</div>
<div id="tracks-legend"></div>
<script>
window.__MIDI_URL__="{midi_url}";
var _toneSynth=null,_toneParts=[],_midiData=null;
function _initSynth(){{if(!_toneSynth){{_toneSynth=new Tone.PolySynth(Tone.Synth,{{oscillator:{{type:'triangle'}},envelope:{{attack:0.02,decay:0.1,sustain:0.5,release:0.8}}}}).toDestination();}}}}
function toneStop(){{Tone.Transport.stop();Tone.Transport.cancel(0);_toneParts.forEach(p=>{{try{{p.dispose();}}catch(e){{}}}});_toneParts=[];}}
function tonePlay(){{
  if(!_midiData)return;
  toneStop();_initSynth();
  var tempos=_midiData.header.tempos;
  Tone.Transport.bpm.value=tempos&&tempos.length?tempos[0].bpm:120;
  _midiData.tracks.forEach(function(track){{
    var isDrum=(track.instrument&&track.instrument.percussion)||track.name==='drums';
    var evts=track.notes.map(n=>[n.time,n]);
    if(!evts.length)return;
    if(isDrum){{
      var p=new Tone.Part(function(time,n){{
        if(window._drumHit) window._drumHit(n.midi, time, n.velocity);
      }},evts);
      p.start(0);_toneParts.push(p);
    }}else{{
      var p=new Tone.Part(function(time,n){{_toneSynth.triggerAttackRelease(n.name,Math.max(n.duration,0.08),time,n.velocity);}},evts);
      p.start(0);_toneParts.push(p);
    }}
  }});
  Tone.Transport.start('+0.05');
}}
fetch("{midi_url}").then(r=>r.arrayBuffer()).then(b=>{{_midiData=new Midi(b);}}).catch(()=>{{}});
const orig=document.getElementById('orig');
const START={start_offset:.3f};
const info=document.getElementById('timeinfo');
function fmt(s){{const m=Math.floor(s/60);return m+':'+(Math.floor(s%60)+'').padStart(2,'0');}}
orig.addEventListener('timeupdate',()=>{{
  const t=Math.max(0,orig.currentTime-START);
  info.textContent='Audio '+fmt(orig.currentTime)+'  ·  Arrangement '+fmt(t);
}});
document.getElementById('play-btn').onclick=()=>{{orig.currentTime=START;orig.play();tonePlay();}};
document.getElementById('stop-btn').onclick=()=>{{orig.pause();toneStop();}};
window._rollReload=function(url){{toneStop();_midiData=null;fetch(url).then(r=>r.arrayBuffer()).then(b=>{{_midiData=new Midi(b);}});}};
{_ROLL_JS}
</script></body></html>""", encoding="utf-8")
    player_url = f"http://127.0.0.1:{PREVIEW_PORT}/{player_fname}"
    return f'<iframe src="{player_url}" style="width:100%;height:500px;border:none;border-radius:8px;" loading="lazy"></iframe>'


def _make_variation_player(paths: list[str], stem_name: str, num_variations: int) -> str:
    """Multi-variation player for compose mode."""
    var_urls = []
    for i, vpath in enumerate(paths):
        mf  = Path(vpath).name
        mu  = f"http://127.0.0.1:{PREVIEW_PORT}/{mf}"
        var_urls.append(mu)
    urls_js = str(var_urls).replace("'", '"')
    labels  = [f"Variation {i+1}" if num_variations > 1 else "Composition" for i in range(len(paths))]
    tabs_html = "".join(
        f'<button class="tb vtab" onclick="selectVar({i})" id="vtab{i}">{lbl}</button>'
        for i, lbl in enumerate(labels)
    )
    player_fname = stem_name + "_player.html"
    (OUTPUTS_DIR / player_fname).write_text(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/tone@14.7.77/build/Tone.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tonejs/midi@2.0.27/build/Midi.js"></script>
<script>{_DRUM_SYNTH_JS}</script>
<style>{_ROLL_CSS}
  .vtab.on{{background:#1a3a6a;border-color:#4477cc;color:#88bbff;}}
  #play-btn{{background:#1a3a80;color:#aaccff;border:1px solid #2244aa;}}
  #stop-btn{{background:#1a1a2a;color:#778899;border:1px solid #222;}}
</style></head><body>
<div id="toolbar">
  {tabs_html}
  <span class="tb-sep"></span>
  <button class="tb" id="play-btn" onclick="tonePlay()">&#9654; Play</button>
  <button class="tb" id="stop-btn" onclick="toneStop()">&#9646;&#9646; Stop</button>
  <span class="tb-sep"></span>
  <button class="tb on" id="draw-btn">Draw</button>
  <button class="tb" id="erase-btn">Erase</button>
  <button class="tb" id="sel-btn">Select</button>
  <span class="tb-sep"></span>
  <span id="snap-label">Snap</span>
  <select id="snap-sel" style="background:#141c28;color:#6688aa;border:1px solid #1e2a3a;border-radius:3px;font-size:10px;padding:2px 4px;">
    <option value="0">1/16</option><option value="1">1/8</option>
    <option value="2" selected>1/4</option><option value="3">1/2</option><option value="4">1</option>
  </select>
  <span class="tb-sep"></span>
  <button class="tb" id="zi">+</button><button class="tb" id="zo">-</button>
  <span class="tb-sep"></span>
  <button class="tb" id="dl-btn">Export MIDI</button>
  <span id="hint">Click+drag = draw note  |  Right-click = erase</span>
</div>
<div id="roll-status" style="padding:3px 6px;font-size:9px;color:#557799;background:#090d14;">Loading...</div>
<div id="roll-area">
  <div id="roll-wrap" class="draw" style="position:relative;"><canvas id="rc"></canvas><canvas id="ph" style="position:absolute;top:0;left:0;pointer-events:none;z-index:3;"></canvas></div>
  <div id="vel-wrap"><canvas id="vel-cv"></canvas><span id="vel-label">vel</span></div>
</div>
<div id="tracks-legend"></div>
<script>
var _toneSynth=null,_toneParts=[],_midiData=null;
function _initSynth(){{if(!_toneSynth){{_toneSynth=new Tone.PolySynth(Tone.Synth,{{oscillator:{{type:'triangle'}},envelope:{{attack:0.02,decay:0.1,sustain:0.5,release:0.8}}}}).toDestination();}}}}
function toneStop(){{Tone.Transport.stop();Tone.Transport.cancel(0);_toneParts.forEach(p=>{{try{{p.dispose();}}catch(e){{}}}});_toneParts=[];}}
function tonePlay(){{
  if(!_midiData)return;
  toneStop();_initSynth();
  var tempos=_midiData.header.tempos;
  Tone.Transport.bpm.value=tempos&&tempos.length?tempos[0].bpm:120;
  _midiData.tracks.forEach(function(track){{
    var isDrum=(track.instrument&&track.instrument.percussion)||track.name==='drums';
    var evts=track.notes.map(n=>[n.time,n]);
    if(!evts.length)return;
    if(isDrum){{
      var p=new Tone.Part(function(time,n){{
        if(window._drumHit) window._drumHit(n.midi, time, n.velocity);
      }},evts);
      p.start(0);_toneParts.push(p);
    }}else{{
      var p=new Tone.Part(function(time,n){{_toneSynth.triggerAttackRelease(n.name,Math.max(n.duration,0.08),time,n.velocity);}},evts);
      p.start(0);_toneParts.push(p);
    }}
  }});
  Tone.Transport.start('+0.05');
}}
const URLS={urls_js};
window.__MIDI_URL__=URLS[0];
function _loadMidi(url){{toneStop();_midiData=null;fetch(url).then(r=>r.arrayBuffer()).then(b=>{{_midiData=new Midi(b);}}).catch(()=>{{}});}}
function selectVar(i){{
  window.__MIDI_URL__=URLS[i];
  document.querySelectorAll('.vtab').forEach((b,j)=>b.classList.toggle('on',i===j));
  if(window._rollReload) window._rollReload(URLS[i]);
  _loadMidi(URLS[i]);
}}
window._rollReload=function(url){{_loadMidi(url);}};
_loadMidi(URLS[0]);
selectVar(0);
{_ROLL_JS}
</script></body></html>""", encoding="utf-8")
    player_url    = f"http://127.0.0.1:{PREVIEW_PORT}/{player_fname}"
    iframe_height = min(700, 460 + max(0, len(paths) - 1) * 20)
    return f'<iframe src="{player_url}" style="width:100%;height:{iframe_height}px;border:none;border-radius:8px;" loading="lazy"></iframe>'


# ---------------------------------------------------------------------------
# Unified pipeline
# ---------------------------------------------------------------------------

def _make_audio_player(wav_path: str, mp3_path) -> str:
    """Build an HTML audio player for Indian engine renders."""
    audio_path = mp3_path if mp3_path and Path(mp3_path).exists() else wav_path
    if not audio_path or not Path(audio_path).exists():
        return "<p style='color:#e55;'>Audio render failed -- check samples folder.</p>"
    fname = Path(audio_path).name
    dest  = OUTPUTS_DIR / fname
    if str(Path(audio_path).resolve()) != str(dest.resolve()):
        shutil.copy2(audio_path, dest)
    audio_url = f"http://127.0.0.1:{PREVIEW_PORT}/{fname}"
    ext = "mpeg" if fname.endswith(".mp3") else "wav"
    return (
        f'<audio controls style="width:100%;margin:8px 0;">'
        f'<source src="{audio_url}" type="audio/{ext}"></audio>'
        f'<p style="font-size:11px;color:#6b7280;">Rendered via Indian audio engine -- real instrument samples</p>'
    )


def run_pipeline(
    melody_source: str,          # "Upload / Record" | "AI-generated" | "Harmony only"
    audio_input,
    pitch_tracker: str,
    tempo_mode: str,
    manual_tempo: float,
    duration_bars: int,
    key_mode: str,
    manual_key: str,
    preset_name: str,
    user_style: str,
    chord_input: str,
    beats_per_chord: float,
    chord_mode: str,
    rhythm_style: str,
    harmony_style: str,
    forbidden_input: str,
    notes_per_beat: float,
    max_tokens: int,
    temperature: float,
    time_sig: str,
    use_sec_dom: bool,
    use_tritone_sub: bool,
    humanize_amount: float,
    num_variations: int,
    form_str: str,
    section_table = None,
    chord_picker: list | None = None,
    use_flat_prog: bool = False,
    use_auto_seventh: bool = False,
    use_back_cycle: bool = False,
    use_passing_chords: bool = False,
    korvai_table = None,
    reverb_preset: str = "studio",
    reverb_mix: float = 0.18,
    pitch_strength: float = 0.8,
    noise_strength: float = 0.7,
    melody_contour: str = "Arch",
    motif_input: str = "",
):
    ts_info      = TIME_SIGNATURES.get(time_sig, TIME_SIGNATURES["4/4"])
    beats_per_bar = ts_info["beats_per_bar"]
    instruments  = INSTRUMENT_PRESETS[preset_name]
    user_style     = user_style or ""
    forbidden_input = forbidden_input or ""
    combined_style = PRESET_STYLE_HINTS.get(preset_name, "")
    if user_style.strip():
        combined_style = f"{combined_style}; {user_style.strip()}" if combined_style else user_style.strip()
    forbidden_chords = [c for c in re.split(r"[\s|,]+", forbidden_input.strip()) if c]
    num_variations   = max(1, int(num_variations))
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_preset = re.sub(r"[^\w]+", "_", preset_name).strip("_")

    korvai_definitions = _table_to_korvai_defs(korvai_table)
    section_rows = _table_to_rows(section_table)
    _korvai_status_block = ""   # filled after section_map is built below
    _korvai_suffix = ""

    # -- Mode 1: Upload / Record (transcription) ----------------------------
    if melody_source == "Upload / Record":
        if audio_input is None:
            yield None, None,"No audio provided. Record or upload a melody first."
            return

        audio_path = audio_input if isinstance(audio_input, str) else audio_input.name
        backend = "crepe" if pitch_tracker == "CREPE (neural)" else "pyin"
        tracker_label = "CREPE neural" if backend == "crepe" else "pyin"
        yield None, None,f"Step 1/4 -- Transcribing with {tracker_label} pitch detector…"
        try:
            raw_notes = transcribe_audio(
                audio_path, backend=backend,
                key=manual_key,
                reverb_preset=reverb_preset,
                reverb_mix=reverb_mix,
                pitch_strength=pitch_strength,
                noise_reduce_strength=noise_strength,
            )
        except Exception as e:
            yield None, None,f"Transcription failed: {e}"; return
        if not raw_notes:
            yield None, None,"No pitched notes detected. Hum clearly and try again."; return

        if tempo_mode == "Auto-detect":
            tempo = detect_tempo(audio_path)
        elif tempo_mode == "Auto-detect (÷2)":
            tempo = round(detect_tempo(audio_path) / 2.0, 1)
        else:
            tempo = manual_tempo
        yield None, None,f"Step 2/4 -- {tempo} BPM. Quantising…"
        notes      = quantize_to_grid(raw_notes, tempo)
        key        = detect_key(notes) if key_mode == "Auto-detect" else manual_key
        # Use the audio file's actual duration, not just the last pyin-detected pitch.
        # pyin misses trailing long notes / soft passages, so last-note-end can be much
        # shorter than the recording. Harmony must cover the full recording.
        import librosa as _librosa
        audio_dur_beats = _librosa.get_duration(path=audio_path) * (tempo / 60.0)
        last_note_beats = max(n["start_beat"] + n["duration_beats"] for n in notes)
        total_beats = max(last_note_beats, audio_dur_beats)
        # Round up to the next complete bar so the arrangement doesn't cut off abruptly
        total_beats = math.ceil(total_beats / beats_per_bar) * beats_per_bar

        melody_track = [{"note": n["note_name"], "start_beat": n["start_beat"],
                         "duration_beats": n["duration_beats"], "velocity": 90} for n in notes]
        note_summary = f"  {len(notes)} notes · {key} · {tempo} BPM · {total_beats:.1f} beats"

        melody_instrument   = instruments[0]
        harmony_instruments = instruments[1:]

        # Always clamp to melody length -- song structure must not extend beyond the recording.
        melody_total_beats = total_beats
        chord_timeline, _, chord_chart, chord_summary, section_map = _build_chord_timeline(
            chord_input, beats_per_chord, melody_total_beats, beats_per_bar, key,
            use_sec_dom, use_tritone_sub, form_str,
            section_rows,
            palette_chords=chord_picker or [], use_flat_prog=use_flat_prog,
            use_auto_seventh=use_auto_seventh, use_back_cycle=use_back_cycle,
            use_passing_chords=use_passing_chords,
            korvai_definitions=korvai_definitions,
        )
        # Trim any chords the structure may have added beyond the melody
        chord_timeline = [(b, c) for b, c in chord_timeline if b < melody_total_beats]
        total_beats = melody_total_beats

        # If no chords were provided, detect phrase boundaries from the melody
        # and suggest chords per phrase -- no rigid bar-grid required.
        if not chord_timeline and notes:
            chord_timeline = auto_chord_timeline_from_melody(
                notes, key, total_beats, beats_per_bar, beats_per_chord
            )
            chord_chart = build_chord_chart_from_timeline(chord_timeline, total_beats) if chord_timeline else ""
            unique_chords = list(dict.fromkeys(c for _, c in chord_timeline))
            chord_summary = (
                f"  Auto-detected {len(chord_timeline)} chord changes from melody\n"
                f"  Chords: {' → '.join(unique_chords)}\n"
            )

        safe_key = re.sub(r"[^\w]+", "_", key).strip("_")

        # If every harmony instrument is algo-replaceable, skip the LLM entirely.
        all_algo_harmony = all(
            inst in CHORDAL_INSTRUMENTS or inst in BASS_INSTRUMENTS or inst == "drums"
            for inst in harmony_instruments
        )

        if all_algo_harmony and chord_timeline:
            yield None, None,f"Step 3/4 -- Building harmony (algorithmic)…\n{note_summary}\n{chord_summary}"
            orchestration = {
                "key": key, "tempo": tempo, "preset": preset_name,
                "parts": {inst: [] for inst in instruments},
            }
            orchestration = inject_algo_parts(
                orchestration, chord_timeline, total_beats,
                CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
                dict(INSTRUMENT_RANGES), harmony_style, beats_per_bar, key,
                section_map=section_map or None,
            )
        else:
            yield None, None,f"Step 3/4 -- Generating harmony…\n{note_summary}\n{chord_summary}Starting…"
            try:
                orchestration = None
                for progress, result in orchestrate_streaming(
                    notes, key, tempo, total_beats,
                    melody_instrument=melody_instrument,
                    harmony_instruments=harmony_instruments,
                    style_prompt=combined_style,
                    notes_per_beat=notes_per_beat,
                    max_tokens=int(max_tokens),
                    temperature=temperature,
                    chord_chart=chord_chart,
                    chord_mode=chord_mode,
                    rhythm_style=rhythm_style,
                    harmony_style=harmony_style,
                    beats_per_chord=beats_per_chord,
                    forbidden_chords=forbidden_chords,
                    chord_timeline=chord_timeline,
                    time_sig=time_sig,
                ):
                    if result is not None:
                        orchestration = result
                    else:
                        yield None, None,f"Step 3/4 -- {progress}\n{note_summary}"
            except RuntimeError as e:
                yield None, None,f"Orchestration failed:\n{e}"; return

        # Overwrite melody part with the user's actual recording (never use LLM/algo melody).
        orchestration["parts"][melody_instrument] = melody_track
        if humanize_amount > 0:
            orchestration = humanize_orchestration(orchestration, amount=humanize_amount,
                                                   melody_instruments=_MELODY_INSTRUMENTS)

        # -- Indian preset: render to audio via sample engine instead of MIDI --
        if _INDIAN_ENGINE_AVAILABLE and is_indian_preset(preset_name):
            yield None, None, "Step 4/4 -- Rendering Indian audio (real samples)…"
            try:
                output_stem = str(OUTPUTS_DIR / f"{timestamp}_{safe_preset}_{safe_key}")
                wav_path, mp3_path = render_indian_audio(
                    melody_notes=melody_track, key=key, tempo_bpm=tempo,
                    total_beats=total_beats, preset_name=preset_name,
                    output_stem=output_stem, beats_per_bar=beats_per_bar,
                )
            except Exception as e:
                yield None, None, f"Indian audio render failed: {e}"; return
            part_summary = "\n".join(
                f"  • {p}: {len(orchestration['parts'].get(p, []))} notes"
                for p in instruments if p in orchestration["parts"]
            )
            _last_context.update({"chord_timeline": chord_timeline, "key": key,
                "orchestration": orchestration, "midi_path": "",
                "harmony_style": harmony_style, "beats_per_bar": beats_per_bar,
                "time_sig": time_sig, "total_beats": total_beats,
                "style": combined_style, "ts_info": ts_info, "section_map": section_map})
            yield _make_audio_player(wav_path, mp3_path), mp3_path or wav_path, (
                f"Done (Indian audio engine).\n{note_summary}\n\n{preset_name}:\n{part_summary}"
            )
            return

        # Inject drums algorithmically (LLM never generates drum notes)
        if "drums" in instruments:
            from src.algo_arranger import make_drum_part, _DRUM_STYLE_MAP
            drum_style = _DRUM_STYLE_MAP.get(preset_name, "pop")
            if "parts" not in orchestration:
                orchestration["parts"] = {}
            orchestration["parts"]["drums"] = make_drum_part(total_beats, beats_per_bar, drum_style)

        yield None, None,"Step 4/4 -- Building MIDI…"
        output_path = str(OUTPUTS_DIR / f"{timestamp}_{safe_preset}_{safe_key}{_korvai_suffix}.mid")
        harmony_midi_path = str(OUTPUTS_DIR / f"{timestamp}_{safe_preset}_{safe_key}{_korvai_suffix}_harmony.mid")
        try:
            build_midi(orchestration, output_path,
                       time_sig_num=ts_info["numerator"], time_sig_den=ts_info["denominator"],
                       chord_timeline=chord_timeline, key=key, section_map=section_map)
            build_midi(orchestration, harmony_midi_path,
                       time_sig_num=ts_info["numerator"], time_sig_den=ts_info["denominator"],
                       exclude_parts={melody_instrument},
                       chord_timeline=chord_timeline, key=key, section_map=section_map)
        except Exception as e:
            yield None, None,f"MIDI build failed: {e}"; return

        stem_path = output_path[:-4]  # strip .mid
        section_note = _write_section_midis(orchestration, section_map, stem_path, ts_info,
                                            chord_timeline=chord_timeline, key=key)
        player_html = _make_sync_player(harmony_midi_path, audio_path,
                                        raw_notes[0]["start_sec"] if raw_notes else 0.0)
        part_summary = "\n".join(f"  • {p}: {len(orchestration['parts'].get(p,[]))} notes"
                                 for p in instruments if p in orchestration["parts"])
        _last_context.update({"chord_timeline": chord_timeline, "key": key,
            "orchestration": orchestration, "midi_path": output_path,
            "harmony_style": harmony_style, "beats_per_bar": beats_per_bar,
            "time_sig": time_sig, "total_beats": total_beats,
            "style": combined_style, "ts_info": ts_info, "section_map": section_map})
        harmony_names = [p for p in instruments if p != melody_instrument]
        yield player_html, output_path, (
            f"Done.\n{note_summary}\n\n{preset_name}:\n{part_summary}\n\n"
            f"Player: {', '.join(harmony_names)} only.\n"
            f"Full MIDI (download) also includes {melody_instrument} -- your transcribed melody as MIDI."
            f"{_korvai_status_block}{section_note}"
        )
        return

    # -- Mode 2: Harmony only (no LLM -- instant) ---------------------------
    if melody_source == "Harmony only":
        tempo = manual_tempo
        key   = manual_key if key_mode == "Manual" else "C major"
        total_beats_raw = duration_bars * beats_per_bar
        chord_timeline, total_beats, _, chord_summary, section_map = _build_chord_timeline(
            chord_input, beats_per_chord, total_beats_raw, beats_per_bar, key,
            use_sec_dom, use_tritone_sub, form_str,
            section_rows,
            palette_chords=chord_picker or [], use_flat_prog=use_flat_prog,
            use_auto_seventh=use_auto_seventh, use_back_cycle=use_back_cycle,
            use_passing_chords=use_passing_chords,
            korvai_definitions=korvai_definitions,
        )
        if not chord_timeline:
            yield None, None,"Harmony only mode requires a chord progression or K section. Add chords or a korvai form above."; return

        yield None, None,f"Building harmony arrangement…\n  {key} · {tempo} BPM\n{chord_summary}"
        orchestration = {"key": key, "tempo": tempo, "preset": preset_name, "parts": {inst: [] for inst in instruments}}
        orchestration = inject_algo_parts(
            orchestration, chord_timeline, total_beats,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), harmony_style, beats_per_bar, key,
            fill_melody_parts=True, section_map=section_map or None,
        )

        # Synfire-style: overlay contour melody or motif on any empty melody parts
        _mel_insts = [i for i in instruments if i in _MELODY_INSTRUMENTS]
        if _mel_insts:
            from src.algo_arranger import make_contour_melody, make_motif_melody
            lo_m, hi_m = dict(INSTRUMENT_RANGES).get(_mel_insts[0], (55, 88))
            if motif_input and motif_input.strip():
                mel_notes = make_motif_melody(
                    chord_timeline, total_beats, lo_m, hi_m,
                    motif_str=motif_input.strip(), notes_per_beat=2.0,
                )
            elif melody_contour and melody_contour != "None":
                mel_notes = make_contour_melody(
                    chord_timeline, total_beats, lo_m, hi_m,
                    contour=melody_contour, notes_per_beat=2.0,
                )
            else:
                mel_notes = []
            if mel_notes:
                orchestration["parts"][_mel_insts[0]] = mel_notes

        if humanize_amount > 0:
            orchestration = humanize_orchestration(orchestration, amount=humanize_amount,
                                                   melody_instruments=_MELODY_INSTRUMENTS)

        # Inject drums algorithmically
        if "drums" in instruments:
            from src.algo_arranger import make_drum_part, _DRUM_STYLE_MAP
            drum_style = _DRUM_STYLE_MAP.get(preset_name, "pop")
            if "parts" not in orchestration:
                orchestration["parts"] = {}
            orchestration["parts"]["drums"] = make_drum_part(total_beats, beats_per_bar, drum_style)

        safe_key    = re.sub(r"[^\w]+", "_", key).strip("_")
        output_path = str(OUTPUTS_DIR / f"{timestamp}_{safe_preset}_{safe_key}{_korvai_suffix}_harmony.mid")

        # -- Indian preset: generate raga melody and render to audio ----------
        if _INDIAN_ENGINE_AVAILABLE and is_indian_preset(preset_name):
            yield None, None, "Rendering Indian audio (real samples)…"
            try:
                raga = get_raga_for_key(key)
                melody_notes = generate_raga_melody(
                    key=key, tempo_bpm=tempo, total_beats=total_beats,
                    raga=raga, temperature=0.8,
                )
                output_stem = str(OUTPUTS_DIR / f"{timestamp}_{safe_preset}_{safe_key}")
                wav_path, mp3_path = render_indian_audio(
                    melody_notes=melody_notes, key=key, tempo_bpm=tempo,
                    total_beats=total_beats, preset_name=preset_name,
                    output_stem=output_stem, beats_per_bar=beats_per_bar,
                )
            except Exception as e:
                yield None, None, f"Indian audio render failed: {e}"; return
            _last_context.update({"chord_timeline": chord_timeline, "key": key,
                "orchestration": orchestration, "midi_path": "",
                "harmony_style": harmony_style, "beats_per_bar": beats_per_bar,
                "time_sig": time_sig, "total_beats": total_beats,
                "style": combined_style, "ts_info": ts_info, "section_map": section_map})
            yield _make_audio_player(wav_path, mp3_path), mp3_path or wav_path, (
                f"Indian audio ready.\n  {preset_name} · {key} · {tempo} BPM\n{chord_summary}"
            )
            return

        try:
            build_midi(orchestration, output_path,
                       time_sig_num=ts_info["numerator"], time_sig_den=ts_info["denominator"],
                       chord_timeline=chord_timeline, key=key, section_map=section_map)
        except Exception as e:
            yield None, None,f"MIDI build failed: {e}"; return

        stem_path = output_path[:-4]
        section_note = _write_section_midis(orchestration, section_map, stem_path, ts_info,
                                            chord_timeline=chord_timeline, key=key)
        _last_context.update({"chord_timeline": chord_timeline, "key": key,
            "orchestration": orchestration, "midi_path": output_path,
            "harmony_style": harmony_style, "beats_per_bar": beats_per_bar,
            "time_sig": time_sig, "total_beats": total_beats,
            "style": combined_style, "ts_info": ts_info, "section_map": section_map,
            "chord_input": chord_input, "beats_per_chord": beats_per_chord})
        yield _make_simple_player(output_path), output_path, (
            f"Harmony-only arrangement ready (no LLM used).\n  {key} · {tempo} BPM\n{chord_summary}"
            f"{_korvai_status_block}{section_note}"
        )
        return

    # -- Mode 3: AI-generated melody (compose_streaming) -------------------
    tempo = manual_tempo
    key   = manual_key if key_mode == "Manual" else "C major"
    total_beats_raw = duration_bars * beats_per_bar
    chord_timeline, total_beats, chord_chart, chord_summary, section_map = _build_chord_timeline(
        chord_input, beats_per_chord, total_beats_raw, beats_per_bar, key,
        use_sec_dom, use_tritone_sub, form_str,
        section_rows,
        palette_chords=chord_picker or [], use_flat_prog=use_flat_prog,
        use_auto_seventh=use_auto_seventh, use_back_cycle=use_back_cycle,
        use_passing_chords=use_passing_chords,
        korvai_definitions=korvai_definitions,
    )
    safe_key  = re.sub(r"[^\w]+", "_", key).strip("_")
    stem_name = f"{timestamp}_{safe_preset}_{safe_key}{_korvai_suffix}"
    summary   = f"  {key} · {tempo} BPM · {total_beats:.0f} beats\n{chord_summary}"
    temp_offsets = [0.0, 0.15, -0.1, 0.25]
    variation_paths, variation_summaries, variation_errors = [], [], []

    for var_idx in range(num_variations):
        var_temp  = min(1.5, max(0.1, temperature + temp_offsets[var_idx % len(temp_offsets)]))
        var_label = f"Variation {var_idx+1}/{num_variations}"
        remaining = num_variations - var_idx - 1

        # -- Indian preset: raga engine → audio, skip LLM entirely -----------
        if _INDIAN_ENGINE_AVAILABLE and is_indian_preset(preset_name):
            yield None, None, f"{var_label} -- generating raga melody…\n{summary}"
            try:
                raga = get_raga_for_key(key)
                melody_notes = generate_raga_melody(
                    key=key, tempo_bpm=tempo, total_beats=total_beats,
                    raga=raga, temperature=var_temp,
                )
                _stem = f"{stem_name}_v{var_idx+1}" if num_variations > 1 else stem_name
                output_stem = str(OUTPUTS_DIR / _stem)
                wav_path, mp3_path = render_indian_audio(
                    melody_notes=melody_notes, key=key, tempo_bpm=tempo,
                    total_beats=total_beats, preset_name=preset_name,
                    output_stem=output_stem, beats_per_bar=beats_per_bar,
                )
                out_audio = mp3_path or wav_path
                variation_paths.append(out_audio)
                variation_summaries.append(
                    f"Var {var_idx+1} -- raga melody ({len(melody_notes)} notes)"
                )
            except Exception as e:
                variation_errors.append(f"{var_label}: {e}")
                yield None, None, f"{var_label} failed: {e}"; continue
            next_msg = f" -- generating variation {var_idx+2}…" if remaining > 0 else ""
            yield (
                _make_audio_player(wav_path, mp3_path), out_audio,
                f"✓ {var_label} ready{next_msg}\n\n{summary}\n\n" + "\n\n".join(variation_summaries),
            )
            continue

        yield None, None,f"{var_label} -- generating (temp={var_temp:.2f})…\n{summary}"

        try:
            orchestration = None
            for progress, result in compose_streaming(
                key, tempo, total_beats, instruments,
                style_prompt=combined_style, notes_per_beat=notes_per_beat,
                max_tokens=int(max_tokens), temperature=var_temp,
                chord_chart=chord_chart, chord_mode=chord_mode,
                rhythm_style=rhythm_style, harmony_style=harmony_style,
                beats_per_chord=beats_per_chord, forbidden_chords=forbidden_chords,
                chord_timeline=chord_timeline, time_sig=time_sig,
            ):
                if result is not None:
                    orchestration = result
                else:
                    yield None, None,f"{var_label} -- {progress}\n{summary}"
        except RuntimeError as e:
            variation_errors.append(f"{var_label}: {e}")
            yield None, None,f"{var_label} failed (details below).\n\n{e}"; continue

        if humanize_amount > 0:
            orchestration = humanize_orchestration(orchestration, amount=humanize_amount,
                                                   melody_instruments=_MELODY_INSTRUMENTS, seed=var_idx)
        if "drums" in instruments:
            from src.algo_arranger import make_drum_part, _DRUM_STYLE_MAP
            if "parts" not in orchestration:
                orchestration["parts"] = {}
            orchestration["parts"]["drums"] = make_drum_part(
                total_beats, beats_per_bar, _DRUM_STYLE_MAP.get(preset_name, "pop"))
        try:
            suffix      = f"_v{var_idx+1}" if num_variations > 1 else ""
            output_path = str(OUTPUTS_DIR / f"{stem_name}{suffix}.mid")
            build_midi(orchestration, output_path,
                       time_sig_num=ts_info["numerator"], time_sig_den=ts_info["denominator"],
                       chord_timeline=chord_timeline, key=key, section_map=section_map)
            section_note = _write_section_midis(orchestration, section_map, output_path[:-4], ts_info,
                                                chord_timeline=chord_timeline, key=key)
            variation_paths.append(output_path)
            part_summary = "\n".join(f"  • {p}: {len(orchestration['parts'].get(p,[]))} notes"
                                     for p in instruments if p in orchestration["parts"])
            variation_summaries.append(f"Var {var_idx+1} (temp={var_temp:.2f}):\n{part_summary}")
        except Exception as e:
            variation_errors.append(f"{var_label} MIDI: {e}")
            yield None, None,f"{var_label} MIDI failed: {e}"; continue

        _last_context.update({"chord_timeline": chord_timeline, "key": key,
            "orchestration": orchestration, "midi_path": output_path,
            "harmony_style": harmony_style, "beats_per_bar": beats_per_bar,
            "time_sig": time_sig, "total_beats": total_beats,
            "style": combined_style, "ts_info": ts_info, "section_map": section_map})

        next_msg = f" -- generating variation {var_idx+2}…" if remaining > 0 else ""
        yield (
            _make_variation_player(variation_paths, stem_name, num_variations),
            output_path,
            f"✓ {var_label} ready{next_msg}\n\n{summary}\n\n" + "\n\n".join(variation_summaries),
        )

    if not variation_paths:
        error_detail = "\n\n".join(variation_errors) if variation_errors else "Unknown error."
        yield None, None,(
            f"All {num_variations} variation(s) failed.\n\n"
            f"Failure reason(s):\n{error_detail}\n\n"
            f"Tip: if the LLM is generating but failing validation, try reducing Duration bars, "
            f"switching to a simpler preset (Piano Trio), or lowering Max tokens."
        ); return
    yield (
        _make_variation_player(variation_paths, stem_name, num_variations),
        variation_paths[-1],
        f"Done -- {len(variation_paths)} variation(s).\n\n{summary}\n\nAll MIDIs in outputs/{_korvai_status_block}{section_note}",
    )

# ---------------------------------------------------------------------------
# Reharmonize (no LLM)
# ---------------------------------------------------------------------------

def run_reharmonize(
    use_auto_seventh: bool, use_sec_dom: bool, use_back_cycle: bool,
    use_tritone_sub: bool, use_passing_chords: bool,
):
    import copy
    ctx = _last_context
    if not ctx["chord_timeline"]:
        yield None, None,"Generate something first, then click Re-arrange."; return
    if not any([use_auto_seventh, use_sec_dom, use_back_cycle, use_tritone_sub, use_passing_chords]):
        yield None, None,"Check at least one option, then click Re-arrange."; return

    yield None, None,"Applying reharmonization…"
    enhanced = list(ctx["chord_timeline"])
    changes  = []
    if use_auto_seventh:
        before = [s for _, s in enhanced]
        enhanced = extend_triads_to_sevenths(enhanced, ctx["key"])
        after = [s for _, s in enhanced]
        upgrades = [(b, a) for b, a in zip(before, after) if b != a]
        if upgrades:
            changes.append("Auto 7ths: " + ", ".join(f"{b}→{a}" for b, a in upgrades[:4]))
    if use_sec_dom:
        before   = [s for _, s in enhanced]
        enhanced = insert_secondary_dominants(enhanced, ctx["key"])
        after    = [s for _, s in enhanced]
        if before != after:
            changes.append(f"Secondary dominants ({len(after)-len(before)} inserted)")
    if use_back_cycle:
        before   = [s for _, s in enhanced]
        enhanced = insert_back_cycling(enhanced, ctx["key"])
        after    = [s for _, s in enhanced]
        if before != after:
            changes.append(f"Back-cycling ({len(after)-len(before)} ii chords inserted)")
    if use_tritone_sub:
        before   = [s for _, s in enhanced]
        enhanced = apply_tritone_substitutions(enhanced)
        after    = [s for _, s in enhanced]
        subs = [(b, a) for b, a in zip(before, after) if b != a]
        if subs:
            changes.append("Tritone subs: " + ", ".join(f"{b}→{a}" for b, a in subs[:4]))
    if use_passing_chords:
        before   = [s for _, s in enhanced]
        enhanced = insert_passing_chords(enhanced)
        after    = [s for _, s in enhanced]
        if before != after:
            changes.append(f"Passing chords ({len(after)-len(before)} inserted)")

    explanation = "\n".join(f"  • {c}" for c in changes) if changes else "  • No changes"
    yield None, None,f"Rebuilding…\n{explanation}"

    orchestration = copy.deepcopy(ctx["orchestration"])
    orchestration = inject_algo_parts(
        orchestration, enhanced, ctx["total_beats"],
        CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
        dict(INSTRUMENT_RANGES), ctx["harmony_style"], ctx["beats_per_bar"], ctx["key"],
        section_map=ctx.get("section_map") or None,
    )
    base_path = ctx["midi_path"].replace(".mid", "_reharm.mid")
    try:
        ts = ctx["ts_info"]
        build_midi(orchestration, base_path, time_sig_num=ts["numerator"], time_sig_den=ts["denominator"],
                   chord_timeline=enhanced, key=ctx.get("key"),
                   section_map=ctx.get("section_map"))
    except Exception as e:
        yield None, None,f"MIDI build failed: {e}"; return

    yield _make_simple_player(base_path), base_path, \
        f"Reharmonization applied:\n{explanation}\n\nDownload MIDI above."

# ---------------------------------------------------------------------------
# MIDI diagnostic
# ---------------------------------------------------------------------------

def run_diagnostics():
    from src.midi_diagnostics import analyse_midi
    midi_path = _last_context.get("midi_path", "")
    if not midi_path or not Path(midi_path).exists():
        return "No MIDI generated yet. Run Generate first."
    key  = _last_context.get("key", "")
    total = _last_context.get("total_beats", 0.0)
    return analyse_midi(midi_path, key=key, total_beats=total)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_TOOLTIP_JS = """
(function() {
  const tip = document.createElement('div');
  tip.id = 'chord-hover-tip';
  Object.assign(tip.style, {
    display: 'none', position: 'fixed', zIndex: '99999',
    pointerEvents: 'none',
    background: 'rgba(10,25,47,0.97)',
    border: '1.5px solid #3a72b8',
    borderRadius: '14px',
    padding: '14px 18px 12px',
    color: '#e8f4ff',
    fontSize: '13px',
    boxShadow: '0 8px 32px rgba(0,0,0,0.7)',
    minWidth: '120px',
    textAlign: 'center',
  });
  document.body.appendChild(tip);

  // -- Audio synthesis --------------------------------------------------------
  var _audioCtx    = null;
  var _activeOscs  = [];
  var _masterGain  = null, _bassGain = null, _chordGain = null, _drumGain = null;
  var _reverbNode  = null, _reverbSend = null;
  var _melodyGain  = null;
  var _noiseBuffer = null;

  function _makeReverb(ctx, decaySec, roomSize) {
    // Synthetic impulse response: exponentially decaying stereo noise
    var sr = ctx.sampleRate;
    var len = Math.floor(sr * decaySec);
    var ir = ctx.createBuffer(2, len, sr);
    for (var ch = 0; ch < 2; ch++) {
      var d = ir.getChannelData(ch);
      for (var i = 0; i < len; i++) {
        // Early reflections boost for first 30ms
        var earlyBoost = i < sr * 0.03 ? 1.6 : 1.0;
        d[i] = (Math.random() * 2 - 1) * earlyBoost * Math.pow(1 - i / len, roomSize);
      }
    }
    var conv = ctx.createConvolver();
    conv.buffer = ir;
    return conv;
  }

  window._ensureGainNodes = function() { _ensureGainNodes(); };
  function _ensureGainNodes() {
    if (!_audioCtx || _masterGain) return;
    window._audioCtx = _audioCtx;
    _masterGain = _audioCtx.createGain(); _masterGain.gain.value = 0.5;
    window._masterGain = _masterGain;
    _bassGain   = _audioCtx.createGain(); _bassGain.gain.value   = 0.5;
    _chordGain  = _audioCtx.createGain(); _chordGain.gain.value  = 0.5;
    _drumGain   = _audioCtx.createGain(); _drumGain.gain.value   = 0.5;

    // Room reverb: small send bus so instruments sit in a space, not a void
    _reverbNode = _makeReverb(_audioCtx, 1.8, 2.2); // 1.8s decay, medium room
    _reverbSend = _audioCtx.createGain(); _reverbSend.gain.value = 0.18; // wet level
    var reverbOut = _audioCtx.createGain(); reverbOut.gain.value = 0.72;
    _reverbNode.connect(reverbOut); reverbOut.connect(_masterGain);

    _melodyGain = _audioCtx.createGain(); _melodyGain.gain.value = 0.0; // off by default
    _bassGain.connect(_masterGain);
    _chordGain.connect(_masterGain);
    _drumGain.connect(_masterGain);
    _melodyGain.connect(_masterGain);
    // Send each bus to reverb (bass gets less room -- keeps it tight)
    var bassRevSend = _audioCtx.createGain(); bassRevSend.gain.value = 0.45;
    _bassGain.connect(bassRevSend); bassRevSend.connect(_reverbSend);
    _chordGain.connect(_reverbSend);
    _drumGain.connect(_reverbSend);
    _melodyGain.connect(_reverbSend);
    _reverbSend.connect(_reverbNode);

    _masterGain.connect(_audioCtx.destination);
    // Pre-bake noise buffer for drum synthesis (reused every bar)
    var sr = _audioCtx.sampleRate;
    _noiseBuffer = _audioCtx.createBuffer(1, Math.floor(sr * 0.5), sr);
    var nd = _noiseBuffer.getChannelData(0);
    for (var i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
    // Sync slider positions to match initial gain values
    var _initVols = {master:0.45, bass:0.6, chord:0.6, drum:0.6};
    ['master','bass','chord','drum'].forEach(function(t) {
      var el = document.getElementById('mix-'+t);
      if (el) { el.value = _initVols[t]; var node = {master:_masterGain,bass:_bassGain,chord:_chordGain,drum:_drumGain}[t]; if(node) node.gain.value = _initVols[t]; }
    });
    _updateMixLabels();
  }

  function _updateMixLabels() {
    var map = {master:_masterGain, bass:_bassGain, chord:_chordGain, drum:_drumGain};
    Object.keys(map).forEach(function(t) {
      var el = document.getElementById('mix-'+t+'-val');
      if (el && map[t]) el.textContent = Math.round(map[t].gain.value * 100) + '%';
    });
  }

  window.setAccompVolume = function(track, value) {
    value = parseFloat(value);
    if (!_audioCtx) return;
    _ensureGainNodes();
    var map = {master:_masterGain, bass:_bassGain, chord:_chordGain, drum:_drumGain};
    var node = map[track];
    if (node) node.gain.setTargetAtTime(value, _audioCtx.currentTime, 0.01);
    _updateMixLabels();
  };

  var _NOTE_PC = {
    'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'Fb':4,
    'F':5,'E#':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,
    'A':9,'A#':10,'Bb':10,'B':11,'Cb':11
  };

  // Spread chord tones across octaves for a natural piano voicing.
  // Root lands in C3-G3 range; inner voices build upward, each at least a minor 3rd above the previous.
  // Style-aware chord voicing -- adds 7ths/9ths for jazz/Rhodes styles,
  // shell voicings (3rd+7th only) for 'Jazz Shell', close voicings for pop.
  function _voiceNotesToMidi(notes, style) {
    var pcs = notes.map(function(n){ return _NOTE_PC[n] || 0; });
    if (!pcs.length) return [];

    // For jazz/Rhodes styles, add a 7th above the top note if not already present
    var jazzStyle = style === 'Jazz' || style === 'Rhodes Jazz' || style === 'Vibraphone' ||
                    style === 'Brushed Trio' || style === 'Jazz Shell';
    var pop9 = style === 'Ballad' || style === 'Pad' || style === 'Lo-Fi' ||
               style === 'Acoustic' || style === 'Disco Pop';

    // Root in C3-G3 range
    var rootPc = pcs[0];
    var root = 48 + ((rootPc + 12) % 12);
    if (root > 55) root -= 12;

    var midi = [root];
    for (var i = 1; i < pcs.length; i++) {
      var pc = pcs[i], prev = midi[midi.length - 1];
      var semAbove = ((pc - (prev % 12)) + 12) % 12;
      if (semAbove === 0) semAbove = 12;
      var next = prev + semAbove;
      if (semAbove < 3) next += 12;
      midi.push(next);
    }

    // Jazz shell: drop middle notes, keep root + 3rd + 7th (more open)
    if (jazzStyle && midi.length >= 3) {
      var third = midi[1], seventh = midi[midi.length - 1];
      // Add a 9th above the 7th for richness
      var ninthPc = (rootPc + 2) % 12;
      var ninthAbove = ((ninthPc - (seventh % 12)) + 12) % 12;
      if (ninthAbove === 0) ninthAbove = 12;
      var ninth = seventh + ninthAbove;
      if (ninth - root > 24) ninth -= 12; // don't spread too wide
      midi = [root, third, seventh, ninth];
      // Keep it in a comfortable register -- shift whole voicing up if too low
      if (midi[midi.length-1] < 60) midi = midi.map(function(m){ return m+12; });
    }

    // Pop 9th: add a 9th colour on top for richness (add9 sound)
    if (pop9 && midi.length >= 2) {
      var top = midi[midi.length - 1];
      var n9pc = (pcs[0] + 2) % 12;
      var diff = ((n9pc - (top % 12)) + 12) % 12;
      if (diff === 0) diff = 12;
      var n9 = top + diff;
      if (n9 - root <= 26) midi.push(n9);
    }

    return midi;
  }

  // Swing offset: pushes "and" beats (beat + 0.5) to beat + 0.667 (triplet feel)
  // Applies only to swing/jazz styles. amt 0=straight, 1=full triplet swing.
  function _swingBeat(rawBeat, swingAmt) {
    if (swingAmt <= 0) return rawBeat;
    var floor = Math.floor(rawBeat);
    var frac  = rawBeat - floor;
    // Straight 8th "and" = 0.5 → swing to 0.667 (2/3 of a beat)
    if (Math.abs(frac - 0.5) < 0.05) {
      return floor + 0.5 + (0.167 * swingAmt);
    }
    return rawBeat;
  }

  function _stopActive() {
    var t = _audioCtx ? _audioCtx.currentTime : 0;
    _activeOscs.forEach(function(o) {
      try {
        o.gain.gain.cancelScheduledValues(t);
        o.gain.gain.setTargetAtTime(0, t, 0.015);
        o.osc.stop(t + 0.08);
      } catch(e) {}
    });
    _activeOscs = [];
  }

  // Hover-preview chord (short pluck, goes to destination directly)
  function playChord(notes) {
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    if (!_audioCtx) { _audioCtx = new AC(); window._audioCtx = _audioCtx; }
    if (_audioCtx.state === 'suspended') _audioCtx.resume();
    _stopActive();
    var ctx = _audioCtx, now = ctx.currentTime;
    var lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 3200;
    lp.connect(ctx.destination);
    var mg = ctx.createGain(); mg.gain.value = Math.max(0.12, 0.5 / notes.length);
    mg.connect(lp);
    notes.forEach(function(note) {
      var pc = _NOTE_PC[note]; if (pc === undefined) return;
      var freq = 440 * Math.pow(2, (60 + pc - 69) / 12);
      [[1,'triangle',1.0],[2,'sine',0.25],[3,'sine',0.07]].forEach(function(h) {
        var osc = ctx.createOscillator(), g = ctx.createGain();
        osc.type = h[1]; osc.frequency.value = freq * h[0];
        g.gain.setValueAtTime(0, now);
        g.gain.linearRampToValueAtTime(h[2], now + 0.01);
        g.gain.exponentialRampToValueAtTime(h[2] * 0.35, now + 0.25);
        g.gain.exponentialRampToValueAtTime(0.0001, now + 2.2);
        osc.connect(g); g.connect(mg); osc.start(now); osc.stop(now + 2.3);
        _activeOscs.push({osc:osc, gain:g});
      });
    });
  }

  // -- Data & rendering -------------------------------------------------------
  function getChordData() {
    const el = document.getElementById('chord-tooltip-data');
    if (!el) return {};
    try { return JSON.parse(el.textContent || '{}'); } catch(e) { return {}; }
  }

  function renderTip(symbol, d) {
    const badges = d.notes.map(function(n) {
      return '<span style="display:inline-block;background:#1e5ca8;color:#fff;border-radius:50%;' +
        'width:30px;height:30px;line-height:30px;text-align:center;font-weight:700;font-size:12px;margin:2px">' +
        n + '</span>';
    }).join('');
    return '<div style="font-size:15px;font-weight:700;color:#7eb8f7;margin-bottom:8px">' + symbol + '</div>' +
      '<div>' + badges + '</div>' +
      '<div style="font-size:10px;color:#8ab4e8;margin-top:6px">' + d.intervals.join(' · ') + '</div>' +
      '<div style="margin-top:8px;line-height:0">' + d.piano + '</div>';
  }

  // -- Event delegation -- attached once, reads live data on every hover -------
  // Per-label handlers would capture stale closures when the palette regenerates
  // (Svelte reuses DOM nodes, so data-tip-attached persists across palette changes).
  // Delegation on the container always reads the current chord-tooltip-data div.
  function initDelegation() {
    const picker = document.querySelector('#chord_picker');
    if (!picker) { setTimeout(initDelegation, 400); return; }

    var lastSymbol = null;

    picker.addEventListener('mousemove', function(e) {
      const label = e.target.closest('label');
      if (!label) {
        tip.style.display = 'none';
        lastSymbol = null;
        return;
      }
      tip.style.left = (e.clientX + 16) + 'px';
      tip.style.top = Math.max(8, e.clientY - 120) + 'px';

      const span = label.querySelector('span');
      if (!span) return;
      const parts = span.textContent.trim().split('·');
      const symbol = (parts.length > 1 ? parts[parts.length - 1] : parts[0]).trim();

      if (symbol === lastSymbol) return; // still on the same chord -- just reposition
      lastSymbol = symbol;

      const d = getChordData()[symbol];
      if (!d) { tip.style.display = 'none'; return; }

      tip.innerHTML = renderTip(symbol, d);
      tip.style.display = 'block';
      if (_hoverAudioEnabled) playChord(d.notes);
    });

    picker.addEventListener('mouseleave', function() {
      tip.style.display = 'none';
      lastSymbol = null;
    });

    // Hover audio + tooltip for the JS tile grid (the visual palette overlay)
    var lastTileSym = null;
    function initTileHover() {
      var grid = document.getElementById('pal-tile-grid');
      if (!grid) { setTimeout(initTileHover, 400); return; }
      grid.addEventListener('mouseover', function(e) {
        var tile = e.target.closest('[data-sym]');
        if (!tile) return;
        var sym = tile.dataset.sym;
        if (!sym || sym === lastTileSym) return;
        lastTileSym = sym;
        if (!_hoverAudioEnabled) return;
        var d = getChordData()[sym];
        var notes = d ? d.notes : (window._chordNotesFromSym ? window._chordNotesFromSym(sym) : null);
        if (notes && notes.length) playChord(notes);
      });
      grid.addEventListener('mouseleave', function() { lastTileSym = null; });
    }
    initTileHover();
  }

  window.attachChordTooltips = function() {}; // kept so .then() call is harmless

  var _hoverAudioEnabled = true;
  window.toggleHoverAudio = function() {
    _hoverAudioEnabled = !_hoverAudioEnabled;
    var btn = document.getElementById('hover-audio-btn');
    if (btn) {
      btn.textContent = _hoverAudioEnabled ? '🔊 Palette preview' : '🔇 Palette preview';
      btn.style.opacity = _hoverAudioEnabled ? '1' : '0.45';
    }
  };

  // -- Live accompaniment engine -----------------------------------------------
  var _accompRunning   = false;
  var _accompTimer     = null;
  var _accompBeatsLeft = 0;   // float beat counter (supports fractional bars/chord)
  var _accompState     = { chords:[], idx:0, nextBarTime:0, bpb:4, beatDur:0.5, barsPerChord:2, style:'Ballad' };
  var _bassRangeLo = 36, _bassRangeHi = 55;   // C2–G3
  var _chordRangeLo = 48, _chordRangeHi = 79; // C3–G5

  // ── Melodic response state ────────────────────────────────────────────────
  var _phraseArmed    = false;
  var _phraseCapturing = false;
  var _capturedPhrase  = [];     // [{midi, t, dur}] normalized to t=0 at phrase start
  var _phraseGapTimer  = null;
  var _micLastMidi     = -1;
  var _micLastT        = 0;
  var _micPitchHist    = [];   // 3-frame median for pitch smoothing
  var _melodyGain      = null;
  var PHRASE_GAP_SEC   = 1.2;
  // density: 1=sparse 2=moderate 3=dense 0=off
  var _density = { bass:2, chord:2, drum:1 };
  // humanize (0=robotic, 1=loose feel)
  var _humanize        = 0.0;
  // Bar counter and dynamic state for "human band" feel
  var _barCount        = 0;
  var _dynamicLevel    = 1.0;   // 0.65–1.0, drifts slowly bar-by-bar
  var _lastFillType    = -1;    // avoid repeating same fill twice in a row
  var _sparseBarsLeft  = 0;     // countdown for a "pull back" moment
  // Shared clock drift -- all parts follow this together; corrects at chord boundaries
  var _clockDrift      = 0.0;   // current offset in seconds (applied to all parts)
  var _driftTarget     = 0.0;   // where the drift is slowly heading
  // MIDI output port for streaming to DAW (Logic Pro via IAC)
  var _midiOutput      = null;
  var _midiAccess2     = null;
  var _audioCtxStartMs = 0;   // performance.now() at AudioContext t=0
  // Passing chords
  var _passProb        = 0.0;  // 0–1 probability of inserting a passing chord at each transition
  var _passType        = 'sec_dom'; // 'sec_dom' | 'dim' | 'chromatic'
  var _pendingIdx      = -1;   // index of destination chord while passing chord plays (-1 = none)
  var _passingChord    = null; // {symbol, label, notes} of the current passing chord
  // Reverse pitch-class map for building passing chord notes
  var _PC_NOTE = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'];
  // MIDI clock -- uses setInterval (not WebAudio scheduling) so 0xF8 arrives in real-time
  var _midiClockTimer  = null;
  var _midiClockNextMs = 0;     // performance.now() ms when next 0xF8 is due
  var _midiClockIvMs   = 20.83; // pulse interval in ms (updated on every BPM change)

  function _startMidiClock() {
    _stopMidiClock();
    if (!_midiOutput) return;
    var bpm = parseFloat((document.getElementById('ctrl-bpm')||{}).value) || 120;
    _midiClockIvMs   = 60000 / (bpm * 24);
    _midiClockNextMs = performance.now();
    _midiOutput.send([0xFA]); // MIDI Start
    _midiClockTimer = setInterval(function() {
      if (!_midiOutput || !_accompRunning) return;
      var now = performance.now();
      while (_midiClockNextMs <= now) {
        _midiOutput.send([0xF8]); // Timing Clock -- sent immediately, no future timestamp
        _midiClockNextMs += _midiClockIvMs;
      }
    }, 2); // 2ms poll -- tight enough to keep jitter well under 1 BPM at any tempo
  }

  function _stopMidiClock() {
    if (_midiClockTimer) { clearInterval(_midiClockTimer); _midiClockTimer = null; }
    if (_midiOutput) _midiOutput.send([0xFC]); // MIDI Stop
  }

  window.setAccompDensity = function(track, value) { _density[track] = parseInt(value); };
  window._setDens = function(track, value, btn) {
    _density[track] = parseInt(value);
    var row = btn.parentElement;
    row.querySelectorAll('.dens-btn').forEach(function(b){ b.classList.remove('active'); });
    btn.classList.add('active');
  };

  // Chord instrument override (empty string = use style default)
  var _chordInstOverride = '';
  window.setChordInstrument = function(v) { _chordInstOverride = v; window._chordInstOverride = v; };
  window._chordInstOverride = '';


  // Melody track (gain node created in _ensureGainNodes -- do NOT redeclare here)
  var _melodyInst = 'piano2', _melodyVol = 0;
  window.setMelodyVolume = function(v) {
    _melodyVol = parseFloat(v);
    if (_melodyGain) _melodyGain.gain.value = _melodyVol;
  };
  window.setMelodyInstrument = function(v) { _melodyInst = v; };

  // Reverb wet level
  window.setReverbLevel = function(v) {
    if (_reverbSend) _reverbSend.gain.value = parseFloat(v);
  };
  var _REVERB_PARAMS = { sm:[0.8,3.5], md:[1.8,2.2], lg:[3.5,1.6], pl:[1.2,4.5] };
  window.setReverbRoom = function(v) {
    if (!_audioCtx) return;
    var p = _REVERB_PARAMS[v] || _REVERB_PARAMS.md;
    var newRev = _makeReverb(_audioCtx, p[0], p[1]);
    var reverbOut = _audioCtx.createGain(); reverbOut.gain.value = 0.72;
    newRev.connect(reverbOut); reverbOut.connect(_masterGain);
    if (_reverbSend) { _reverbSend.disconnect(); _reverbSend.connect(newRev); }
    if (_reverbNode) { try { _reverbNode.disconnect(); } catch(e){} }
    _reverbNode = newRev;
  };
  window.setHumanize  = function(v) { _humanize  = Math.max(0, Math.min(1, parseFloat(v) || 0)); };
  window.setPassProb  = function(v) { _passProb  = Math.max(0, Math.min(1, parseFloat(v) || 0)); };
  window.setPassType  = function(v) { _passType  = v || 'sec_dom'; };

  // -- Instrument synthesizers -------------------------------------------------

  function _playPianoNote(midiNote, t, dur, vel) {
    // Identical signal chain to the palette hover: triangle fundamental → lowpass → chordGain
    var freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    var ctx = _audioCtx, dec = Math.min(dur, 2.5);
    var lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 3200;
    lp.connect(_chordGain);
    [[1,'triangle',1.0],[2,'sine',0.25],[3,'sine',0.07]].forEach(function(h) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = h[1]; osc.frequency.value = freq * h[0];
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * h[2], t + 0.01);
      g.gain.exponentialRampToValueAtTime(vel * h[2] * 0.35, t + 0.25);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dec);
      osc.connect(g); g.connect(lp);
      osc.start(t); osc.stop(t + dec + 0.05);
    });
  }

  // -- Style-specific chord synthesizers --------------------------------------

  // Helper: midi note → freq, with fallback to note-string + octave offset
  function _midiFreq(midiNote, note, octaveBase) {
    if (midiNote !== undefined) return 440 * Math.pow(2, (midiNote - 69) / 12);
    var pc = _NOTE_PC[note]; if (pc === undefined) return 440;
    return 440 * Math.pow(2, ((octaveBase || 60) + pc - 69) / 12);
  }

  // Strings: slow-attack detuned ensemble
  function _playStringsNote(note, t, dur, vel, midiNote) {
    var freq = _midiFreq(midiNote, note, 60);
    var ctx = _audioCtx;
    var att = 0.22, dec = Math.max(dur, 1.8);
    [-6, 0, 6].forEach(function(cents) {
      var osc = ctx.createOscillator(), g = ctx.createGain(), filt = ctx.createBiquadFilter();
      osc.type = 'sawtooth';
      osc.frequency.value = freq * Math.pow(2, cents / 1200);
      filt.type = 'lowpass'; filt.frequency.setValueAtTime(500, t);
      filt.frequency.linearRampToValueAtTime(2000, t + att); filt.Q.value = 0.4;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * 0.38, t + att);
      g.gain.setValueAtTime(vel * 0.32, t + dec - 0.15);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dec);
      osc.connect(filt); filt.connect(g); g.connect(_chordGain);
      osc.start(t); osc.stop(t + dec + 0.1);
    });
  }

  // Guitar: plucked triangle harmonics
  function _playGuitarNote(note, t, dur, vel, bossa, midiNote) {
    var freq = _midiFreq(midiNote, note, 60);
    var ctx = _audioCtx;
    var decay = bossa ? Math.min(dur, 0.7) : Math.min(dur, 1.2);
    var lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 2800; lp.connect(_chordGain);
    [[1, 0.8], [2, 0.28], [3, 0.1]].forEach(function(h, i) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = 'triangle'; osc.frequency.value = freq * h[0];
      var d = decay / (1 + i * 0.5);
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * h[1], t + 0.004);
      g.gain.exponentialRampToValueAtTime(vel * h[1] * 0.22, t + 0.07);
      g.gain.exponentialRampToValueAtTime(0.0001, t + d);
      osc.connect(g); g.connect(lp);
      osc.start(t); osc.stop(t + d + 0.05);
    });
  }

  // Synth pad: slow attack, detuned layers
  function _playPadNote(note, t, dur, vel, midiNote) {
    var freq = _midiFreq(midiNote, note, 60);
    var ctx = _audioCtx;
    var att = Math.min(0.45, dur * 0.35);
    var dec = Math.max(dur * 1.1, 3.0);
    [[-8,'sine',0.45],[0,'triangle',0.4],[8,'sine',0.4],[1200,'sine',0.1]].forEach(function(h) {
      var cents = h[0], type = h[1], amp = h[2];
      var osc = ctx.createOscillator(), g = ctx.createGain(), filt = ctx.createBiquadFilter();
      osc.type = type;
      osc.frequency.value = freq * (cents === 1200 ? 2 : Math.pow(2, cents / 1200));
      filt.type = 'lowpass'; filt.frequency.value = 1600; filt.Q.value = 0.35;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * amp, t + att);
      g.gain.setValueAtTime(vel * amp * 0.88, t + dec * 0.75);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dec);
      osc.connect(filt); filt.connect(g); g.connect(_chordGain);
      osc.start(t); osc.stop(t + dec + 0.1);
    });
  }

  // Honky-tonk piano: two detuned voices
  function _playHonkyTonkNote(note, t, dur, vel, midiNote) {
    var freq = _midiFreq(midiNote, note, 60);
    var ctx = _audioCtx;
    [0, 15].forEach(function(cents) {
      var f = freq * Math.pow(2, cents / 1200);
      [[1.000, 0.55, dur * 0.65], [2.001, 0.25, dur * 0.32], [3.004, 0.10, dur * 0.18]].forEach(function(p) {
        var osc = ctx.createOscillator(), g = ctx.createGain();
        osc.type = 'sine'; osc.frequency.value = f * p[0];
        g.gain.setValueAtTime(0, t);
        g.gain.linearRampToValueAtTime(vel * p[1] * 0.55, t + 0.004);
        g.gain.exponentialRampToValueAtTime(vel * p[1] * 0.20, t + 0.07);
        g.gain.exponentialRampToValueAtTime(0.0001, t + p[2]);
        osc.connect(g); g.connect(_chordGain);
        osc.start(t); osc.stop(t + p[2] + 0.05);
      });
    });
  }

  // Accordion: beating sawtooth reeds
  function _playAccordionNote(note, t, dur, vel, midiNote) {
    var freq = _midiFreq(midiNote, note, 60);
    var ctx = _audioCtx;
    var d = Math.min(dur, 2.2);
    [-8, 8].forEach(function(cents) {
      var osc = ctx.createOscillator(), g = ctx.createGain(), filt = ctx.createBiquadFilter();
      osc.type = 'sawtooth'; osc.frequency.value = freq * Math.pow(2, cents / 1200);
      filt.type = 'lowpass'; filt.frequency.value = 2200; filt.Q.value = 0.5;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * 0.38, t + 0.04);
      g.gain.setValueAtTime(vel * 0.35, t + d - 0.04);
      g.gain.exponentialRampToValueAtTime(0.0001, t + d);
      osc.connect(filt); filt.connect(g); g.connect(_chordGain);
      osc.start(t); osc.stop(t + d + 0.05);
    });
  }

  // Rhodes: FM bell-tone with key click, velocity brightness, pitch droop
  function _playRhodesNote(midiNote, t, dur, vel) {
    // Each tine is slightly detuned -- gives that warm, slightly imperfect character
    var detuneCents = (Math.random() - 0.5) * 6;
    var freq = 440 * Math.pow(2, (midiNote - 69 + detuneCents / 100) / 12);
    var ctx = _audioCtx, dec = Math.min(dur, 3.5);
    // Velocity scales both amplitude AND FM depth (harder = brighter bark)
    var fmDepth = freq * (vel * 2.2 + 0.4);
    // FM modulator
    var mod = ctx.createOscillator(), modEnv = ctx.createGain();
    mod.type = 'sine'; mod.frequency.value = freq * 1.975;
    modEnv.gain.setValueAtTime(fmDepth, t);
    modEnv.gain.exponentialRampToValueAtTime(freq * 0.008, t + 0.22);
    mod.connect(modEnv);
    // Carrier with slight pitch droop at onset (real tine behavior)
    var car = ctx.createOscillator(), carEnv = ctx.createGain();
    car.type = 'sine';
    car.frequency.setValueAtTime(freq * 1.006, t); // starts 6 cents sharp
    car.frequency.exponentialRampToValueAtTime(freq, t + 0.04); // settles to pitch
    modEnv.connect(car.frequency);
    carEnv.gain.setValueAtTime(0, t);
    carEnv.gain.linearRampToValueAtTime(vel * 0.90, t + 0.004);
    carEnv.gain.exponentialRampToValueAtTime(vel * 0.48, t + 0.12);
    carEnv.gain.exponentialRampToValueAtTime(0.0001, t + dec);
    // Tremolo LFO (deeper for harder hits)
    var lfo = ctx.createOscillator(), lfoG = ctx.createGain();
    lfo.type = 'sine'; lfo.frequency.value = 4.8 + vel * 0.6;
    lfoG.gain.value = vel * 0.065;
    lfo.connect(lfoG); lfoG.connect(carEnv.gain);
    // Key click: mechanical noise at note-on (very short, scaled by vel)
    if (_noiseBuffer && vel > 0.3) {
      var kc = ctx.createBufferSource(), kcG = ctx.createGain(), kcF = ctx.createBiquadFilter();
      kc.buffer = _noiseBuffer; kcF.type = 'bandpass'; kcF.frequency.value = 1100; kcF.Q.value = 2.5;
      kcG.gain.setValueAtTime(vel * 0.18, t); kcG.gain.exponentialRampToValueAtTime(0.0001, t + 0.009);
      kc.connect(kcF); kcF.connect(kcG); kcG.connect(_chordGain); kc.start(t); kc.stop(t + 0.012);
    }
    var lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 3600 + vel * 1200;
    car.connect(carEnv); carEnv.connect(lp); lp.connect(_chordGain);
    [mod, car, lfo].forEach(function(o){ o.start(t); o.stop(t + dec + 0.1); });
  }

  // Vibraphone: mallet thwack + metallic sines + spinning-fan tremolo
  function _playVibesNote(midiNote, t, dur, vel) {
    var freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    var ctx = _audioCtx, dec = Math.min(dur * 1.4 + 0.6, 5.0);
    var master = ctx.createGain(); master.gain.setValueAtTime(vel * 0.88, t); master.connect(_chordGain);
    // Tremolo (fan motor)
    var lfo = ctx.createOscillator(), lfoG = ctx.createGain();
    lfo.type = 'sine'; lfo.frequency.value = 6.0 + Math.random() * 0.4; lfoG.gain.value = 0.12;
    lfo.connect(lfoG); lfoG.connect(master.gain);
    // Metallic partials: vibraphone has inharmonic partials at ~1×, 3.9×, 10.4×
    [[1, 1.00], [3.92, 0.12], [10.4, 0.04]].forEach(function(h) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = 'sine'; osc.frequency.value = freq * h[0];
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(h[1], t + 0.002);
      g.gain.exponentialRampToValueAtTime(h[1] * 0.50, t + 0.06);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dec * (h[0] > 3 ? 0.4 : 1.0));
      osc.connect(g); g.connect(master);
      osc.start(t); osc.stop(t + dec + 0.1);
    });
    // Mallet thwack: bright noise burst at onset (softens with softer velocity)
    if (_noiseBuffer) {
      var ns = ctx.createBufferSource(), ng = ctx.createGain(), nf = ctx.createBiquadFilter();
      ns.buffer = _noiseBuffer; nf.type = 'bandpass'; nf.frequency.value = freq * 5; nf.Q.value = 2.0;
      ng.gain.setValueAtTime(vel * 0.22, t); ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.008);
      ns.connect(nf); nf.connect(ng); ng.connect(_chordGain); ns.start(t); ns.stop(t + 0.012);
    }
    lfo.start(t); lfo.stop(t + dec + 0.1);
  }

  // Clavinet: bright plucked sawtooth with bandpass colour
  function _playClav(midiNote, t, dur, vel) {
    var freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    var ctx = _audioCtx, dec = Math.min(dur, 0.55);
    var bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = freq * 2.5; bp.Q.value = 1.2;
    bp.connect(_chordGain);
    [[1, 0.70], [2, 0.30], [3, 0.12], [4, 0.05]].forEach(function(h) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = 'sawtooth'; osc.frequency.value = freq * h[0];
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * h[1], t + 0.002);
      g.gain.exponentialRampToValueAtTime(vel * h[1] * 0.18, t + 0.04);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dec);
      osc.connect(g); g.connect(bp);
      osc.start(t); osc.stop(t + dec + 0.05);
    });
  }

  // Acoustic guitar: plucked Karplus-Strong-ish via bandpass noise + harmonic decay
  function _playAcousticNote(midiNote, t, dur, vel) {
    var freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    var ctx = _audioCtx, dec = Math.min(dur * 1.1 + 0.3, 3.2);
    var master = ctx.createGain(); master.connect(_chordGain);
    // Body resonance: bandpass around string freq
    var bp = ctx.createBiquadFilter(); bp.type = 'bandpass';
    bp.frequency.value = freq; bp.Q.value = 18;
    // Attack transient (pluck noise)
    if (_noiseBuffer) {
      var ns = ctx.createBufferSource(), ng = ctx.createGain();
      ns.buffer = _noiseBuffer;
      ng.gain.setValueAtTime(vel * 0.35, t);
      ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.025);
      ns.connect(bp); bp.connect(ng); ng.connect(master);
      ns.start(t); ns.stop(t + 0.03);
    }
    // Sustain: fundamental + 2nd + 3rd harmonics decaying naturally
    [[1, 0.70, 1.8], [2, 0.18, 0.9], [3, 0.07, 0.5]].forEach(function(h) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = 'triangle'; osc.frequency.value = freq * h[0];
      osc.detune.value = (Math.random() - 0.5) * 3;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * h[1], t + 0.005);
      g.gain.exponentialRampToValueAtTime(vel * h[1] * 0.3, t + h[2]);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dec);
      osc.connect(g); g.connect(master);
      osc.start(t); osc.stop(t + dec + 0.05);
    });
    master.gain.setValueAtTime(0.7, t);
  }

  // Disco/pop synth: punchy piano + synth layer with tight gate
  function _playDiscoNote(midiNote, t, dur, vel) {
    var freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    var ctx = _audioCtx, dec = Math.min(dur, 0.30);
    // Bright piano layer
    var master = ctx.createGain(); master.connect(_chordGain);
    var lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 4200; lp.Q.value = 0.5;
    lp.connect(master);
    [[1, 0.65, 'sawtooth'], [2, 0.22, 'square'], [3, 0.08, 'sine']].forEach(function(h) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = h[2]; osc.frequency.value = freq * h[0];
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * h[1], t + 0.003);
      g.gain.exponentialRampToValueAtTime(vel * h[1] * 0.15, t + 0.04);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dec);
      osc.connect(g); g.connect(lp);
      osc.start(t); osc.stop(t + dec + 0.05);
    });
    master.gain.setValueAtTime(0.85, t);
  }

  // Hammond B3 organ: additive drawbar synthesis + Leslie tremolo
  function _playOrganNote(midiNote, t, dur, vel) {
    var freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    var ctx = _audioCtx, dec = Math.min(dur + 0.04, 6.0);
    var master = ctx.createGain(); master.gain.setValueAtTime(vel * 0.72, t);
    master.gain.setValueAtTime(vel * 0.72, t + dec - 0.01);
    master.gain.linearRampToValueAtTime(0.0001, t + dec); // hard stop = key-off click
    // Leslie cabinet: dual LFO for rotary speaker (fast/slow modes)
    var leslie = ctx.createOscillator(), leslieG = ctx.createGain();
    leslie.type = 'sine'; leslie.frequency.value = 6.3; // fast rotor
    leslieG.gain.value = vel * 0.055;
    leslie.connect(leslieG); leslieG.connect(master.gain);
    master.connect(_chordGain);
    // Drawbar partials: 16', 8', 5⅓', 4', 3⅕', 2⅔', 2', 1⅗', 1'
    var drawbars = [[0.5,0.6],[1,1.0],[1.5,0.5],[2,0.7],[2.5,0.3],[3,0.4],[4,0.3],[5,0.1],[8,0.06]];
    drawbars.forEach(function(db) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = 'sine'; osc.frequency.value = freq * db[0];
      osc.detune.value = (Math.random() - 0.5) * 2.5; // slight tuning drift
      g.gain.value = db[1];
      osc.connect(g); g.connect(master);
      osc.start(t); osc.stop(t + dec + 0.05);
    });
    leslie.start(t); leslie.stop(t + dec + 0.05);
  }

  // Acoustic Piano: bright hammer strike + inharmonic string partials
  function _playPianoNote2(midiNote, t, dur, vel) {
    var freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    var ctx = _audioCtx, dec = Math.min(dur * 1.1 + 0.5, 4.5);
    var master = ctx.createGain(); master.connect(_chordGain);
    master.gain.setValueAtTime(vel * 0.88, t);
    // Hammer knock at attack
    if (_noiseBuffer) {
      var hn = ctx.createBufferSource(), hg = ctx.createGain(), hf = ctx.createBiquadFilter();
      hn.buffer = _noiseBuffer; hf.type = 'bandpass';
      hf.frequency.value = freq * 3.5 + 800; hf.Q.value = 1.8;
      hg.gain.setValueAtTime(vel * 0.28, t); hg.gain.exponentialRampToValueAtTime(0.0001, t + 0.012);
      hn.connect(hf); hf.connect(hg); hg.connect(master);
      hn.start(t); hn.stop(t + 0.016);
    }
    // Inharmonic string partials (real piano strings are slightly sharp on overtones)
    var inharmonic = 0.00015;
    [[1,1.00],[2,0.50],[3,0.28],[4,0.16],[5,0.09],[6,0.05]].forEach(function(h, hi) {
      var partialFreq = freq * h[0] * (1 + inharmonic * h[0] * h[0]);
      var detune = (Math.random() - 0.5) * 4;
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = 'sine'; osc.frequency.value = partialFreq + detune;
      var decScale = Math.pow(0.55, hi);
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * h[1], t + 0.003);
      g.gain.exponentialRampToValueAtTime(vel * h[1] * 0.25, t + 0.12);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dec * decScale + 0.3);
      osc.connect(g); g.connect(master);
      osc.start(t); osc.stop(t + dec + 0.1);
    });
  }

  // Strings ensemble: detuned sawtooths, slow attack, lush pad
  function _playStringsEnsemble(midiNote, t, dur, vel) {
    var freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    var ctx = _audioCtx, dec = Math.min(dur + 0.8, 6.0);
    var attack = 0.18; // slow bow-like attack
    var master = ctx.createGain(); master.connect(_chordGain);
    var lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 3200; lp.Q.value = 0.6;
    lp.connect(master);
    // 5 detuned saws per note -- ensemble shimmer
    var detunes = [-8, -3, 0, 3, 8];
    detunes.forEach(function(d) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = 'sawtooth'; osc.frequency.value = freq; osc.detune.value = d;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * 0.18, t + attack);
      g.gain.setValueAtTime(vel * 0.18, t + dec - 0.12);
      g.gain.linearRampToValueAtTime(0.0001, t + dec);
      osc.connect(g); g.connect(lp);
      osc.start(t); osc.stop(t + dec + 0.05);
    });
    master.gain.value = 1.0;
  }

  // Clean electric guitar: warm pluck with chorus (think John Mayer, smooth jazz)
  function _playCleanGuitar(midiNote, t, dur, vel) {
    var freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    var ctx = _audioCtx, dec = Math.min(dur * 0.9 + 0.2, 2.5);
    var master = ctx.createGain(); master.connect(_chordGain);
    // Chorus: two slightly detuned copies
    [-5, 0, 5].forEach(function(detuneCents, vi) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = vi === 1 ? 'triangle' : 'sine';
      osc.frequency.value = freq; osc.detune.value = detuneCents;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * (vi === 1 ? 0.55 : 0.22), t + 0.004);
      g.gain.exponentialRampToValueAtTime(vel * (vi === 1 ? 0.18 : 0.07), t + 0.08);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dec);
      osc.connect(g); g.connect(master);
      osc.start(t); osc.stop(t + dec + 0.05);
    });
    // Pluck transient
    if (_noiseBuffer) {
      var ns = ctx.createBufferSource(), ng = ctx.createGain(), nf = ctx.createBiquadFilter();
      ns.buffer = _noiseBuffer; nf.type = 'bandpass'; nf.frequency.value = freq * 3; nf.Q.value = 3;
      ng.gain.setValueAtTime(vel * 0.15, t); ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.015);
      ns.connect(nf); nf.connect(ng); ng.connect(master);
      ns.start(t); ns.stop(t + 0.02);
    }
    // 2nd harmonic for warmth
    var h2 = ctx.createOscillator(), h2g = ctx.createGain();
    h2.type = 'sine'; h2.frequency.value = freq * 2;
    h2g.gain.setValueAtTime(0, t);
    h2g.gain.linearRampToValueAtTime(vel * 0.12, t + 0.003);
    h2g.gain.exponentialRampToValueAtTime(0.0001, t + dec * 0.5);
    h2.connect(h2g); h2g.connect(master);
    h2.start(t); h2.stop(t + dec + 0.05);
    master.gain.value = 0.82;
  }

  // Brass stab: punchy filtered sawtooth, fast attack/release
  function _playBrassNote(midiNote, t, dur, vel) {
    var freq = 440 * Math.pow(2, (midiNote - 69) / 12);
    var ctx = _audioCtx, dec = Math.min(dur, 0.45);
    var master = ctx.createGain(); master.connect(_chordGain);
    // Brass formant filter: sweeps from dark to bright on attack
    var lp = ctx.createBiquadFilter(); lp.type = 'lowpass';
    lp.frequency.setValueAtTime(freq * 1.5, t);
    lp.frequency.exponentialRampToValueAtTime(freq * 6, t + 0.025);
    lp.frequency.exponentialRampToValueAtTime(freq * 2.5, t + 0.09);
    lp.Q.value = 2.8; lp.connect(master);
    [[1,0.70,'sawtooth'],[2,0.35,'sawtooth'],[3,0.18,'sine'],[4,0.08,'sine']].forEach(function(h) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = h[2]; osc.frequency.value = freq * h[0];
      osc.detune.value = (Math.random() - 0.5) * 8; // ensemble spread
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * h[1], t + 0.012);
      g.gain.exponentialRampToValueAtTime(vel * h[1] * 0.55, t + 0.09);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dec);
      osc.connect(g); g.connect(lp);
      osc.start(t); osc.stop(t + dec + 0.05);
    });
    master.gain.value = 0.92;
  }

  // -- Style → instrument routing ----------------------------------------------
  var _STYLE_INSTR = {
    'Ballad':        'strings',
    'Jazz':          'guitar',
    'Blues':         'honkyton',
    'Bossa Nova':    'guitar_bossa',
    'Waltz':         'accordion',
    'Honky Tonk':   'honkyton',
    'Pad':           'pad',
    'Arpeggio Up':   'piano',
    'Arpeggio Down': 'piano',
    'Rhodes':        'rhodes',
    'Rhodes Jazz':   'rhodes',
    'Vibraphone':    'vibes',
    'Funk Chop':     'clav',
    'Lo-Fi':         'rhodes',
    'Acoustic':      'acoustic',
    'Disco Pop':     'disco',
    'Brushed Trio':  'rhodes',
    'Jazz Shell':    'rhodes',
    // New styles
    'R&B':           'piano2',
    'Neo Soul':      'rhodes',
    'Gospel':        'organ',
    'Soul':          'organ',
    'Pop':           'piano2',
    'Country':       'guitar',
    'Reggae':        'organ',
    'Latin':         'piano2',
    'Funk':          'clav',
    'Singer-Songwriter': 'acoustic',
    'Indie Pop':     'piano2',
    'Smooth Jazz':   'guitar_clean',
    'Cinematic':     'strings_ens',
    'Worship':       'organ',
    'Tropical':      'vibes',
    'Samba':         'guitar_bossa',
    'Swing':         'vibes',
    'Motown':        'organ',
    'New Soul':      'strings_ens',
    'Brass':         'brass',
  };

  var _INSTR_LABEL = {
    strings:'Strings', guitar:'Guitar', guitar_bossa:'Guitar (Nylon)',
    pad:'Synth Pad', honkyton:'Honky Tonk Piano', accordion:'Accordion', piano:'Piano',
  };

  // midiNote: pre-voiced MIDI note number (from _voiceNotesToMidi); falls back to 60+pc
  function _playChordInst(note, t, dur, vel, midiNote) {
    var pc = _NOTE_PC[note];
    var midi = (midiNote !== undefined) ? midiNote : (pc !== undefined ? 60 + pc : 60);
    if (pc !== undefined) _logNote(t, Math.min(dur, 4.0), midi, vel, 0);
    var inst = _chordInstOverride || _STYLE_INSTR[_accompState.style] || 'piano';
    if      (inst === 'strings')      _playStringsNote(note, t, dur, vel, midi);
    else if (inst === 'strings_ens')  _playStringsEnsemble(midi, t, dur, vel);
    else if (inst === 'guitar')       _playGuitarNote(note, t, dur, vel, false, midi);
    else if (inst === 'guitar_bossa') _playGuitarNote(note, t, dur, vel, true,  midi);
    else if (inst === 'guitar_clean') _playCleanGuitar(midi, t, dur, vel);
    else if (inst === 'pad')          _playPadNote(note, t, dur, vel, midi);
    else if (inst === 'honkyton')     _playHonkyTonkNote(note, t, dur, vel, midi);
    else if (inst === 'accordion')    _playAccordionNote(note, t, dur, vel, midi);
    else if (inst === 'organ')        _playOrganNote(midi, t, dur, vel);
    else if (inst === 'piano2')       _playPianoNote2(midi, t, dur, vel);
    else if (inst === 'brass')        _playBrassNote(midi, t, dur, vel);
    else if (inst === 'rhodes')       _playRhodesNote(midi, t, dur, vel);
    else if (inst === 'vibes')        _playVibesNote(midi, t, dur, vel);
    else if (inst === 'clav')         _playClav(midi, t, dur, vel);
    else if (inst === 'acoustic')     _playAcousticNote(midi, t, dur, vel);
    else if (inst === 'disco')        _playDiscoNote(midi, t, dur, vel);
    else                              _playPianoNote(midi, t, dur, vel);
  }

  function _sustainedStyle() {
    var inst = _STYLE_INSTR[_accompState.style] || 'piano';
    return inst === 'strings' || inst === 'strings_ens' || inst === 'pad' ||
           inst === 'accordion' || inst === 'organ';
  }

  // -- MIDI recording ----------------------------------------------------------
  var _recording    = false;
  var _recordStart  = 0;
  var _recordEvents = [];
  var _recordBPM    = 120;
  var _recordTimeSig = '4/4';
  var _recordTimer  = null;

  function _logNote(t, dur, midiNote, vel, ch) {
    // Stream to DAW via MIDI output port (IAC / Logic Pro)
    if (_midiOutput) {
      var vel127 = Math.max(1, Math.min(127, Math.round(vel * 127)));
      _midiOutput.send([0x90 | (ch & 0xF), midiNote & 0x7F, vel127],
                       _audioCtxStartMs + t * 1000);
      _midiOutput.send([0x80 | (ch & 0xF), midiNote & 0x7F, 0],
                       _audioCtxStartMs + (t + dur) * 1000);
    }
    if (!_recording) return;
    _recordEvents.push({t:t-_recordStart, type:'on',  ch:ch, note:midiNote&0x7F, vel:vel});
    _recordEvents.push({t:t-_recordStart+dur, type:'off', ch:ch, note:midiNote&0x7F, vel:0});
  }

  // GM drum note map
  var _DRUM_GM = {
    kick:36, snare:38, ghost:38, hihat:42, hihat_open:46,
    ride:51, crash:49, clap:39, tom_hi:50, tom_mid:47, tom_lo:45
  };

  function _logDrumGM(t, type, vel) {
    var note = _DRUM_GM[type]; if (note === undefined) return;
    var dur  = type === 'crash' ? 0.5 : type === 'hihat_open' ? 0.18 : 0.07;
    var vel127 = Math.max(1, Math.min(127, Math.round(vel * 127)));
    if (_midiOutput) {
      _midiOutput.send([0x99, note, vel127], _audioCtxStartMs + t * 1000);
      _midiOutput.send([0x89, note, 0],      _audioCtxStartMs + (t + dur) * 1000);
    }
    if (!_recording) return;
    _recordEvents.push({t:t-_recordStart,     type:'on',  ch:9, note:note, vel:vel});
    _recordEvents.push({t:t-_recordStart+dur, type:'off', ch:9, note:note, vel:0});
  }

  // -- Drum synthesis -- multi-layer, velocity-sensitive timbre -----------------

  function _playKick(t, vel) {
    _logDrumGM(t, 'kick', vel);
    var ctx = _audioCtx;
    // Sub layer: deep sine with pitch sweep -- the "thump"
    var sub = ctx.createOscillator(), subG = ctx.createGain();
    sub.type = 'sine';
    var f0 = 52 + vel * 18;  // harder hit = slightly higher fundamental
    sub.frequency.setValueAtTime(f0 * 3.2, t);
    sub.frequency.exponentialRampToValueAtTime(f0, t + 0.055 + vel * 0.025);
    subG.gain.setValueAtTime(0, t);
    subG.gain.linearRampToValueAtTime(vel * 1.0, t + 0.002);
    subG.gain.exponentialRampToValueAtTime(vel * 0.3, t + 0.08);
    subG.gain.exponentialRampToValueAtTime(0.0001, t + 0.28 + vel * 0.08);
    sub.connect(subG); subG.connect(_drumGain); sub.start(t); sub.stop(t + 0.40);
    // Body layer: mid-range punch (adds thwack character on harder hits)
    var body = ctx.createOscillator(), bodyG = ctx.createGain();
    body.type = 'sine'; body.frequency.setValueAtTime(160 + vel * 40, t);
    body.frequency.exponentialRampToValueAtTime(70, t + 0.04);
    bodyG.gain.setValueAtTime(0, t);
    bodyG.gain.linearRampToValueAtTime(vel * 0.50, t + 0.001);
    bodyG.gain.exponentialRampToValueAtTime(0.0001, t + 0.06);
    body.connect(bodyG); bodyG.connect(_drumGain); body.start(t); body.stop(t + 0.08);
    // Click transient: beater impact -- brighter on harder hits
    if (_noiseBuffer) {
      var ns = ctx.createBufferSource(), ng = ctx.createGain(), nf = ctx.createBiquadFilter();
      ns.buffer = _noiseBuffer;
      nf.type = 'bandpass'; nf.frequency.value = 120 + vel * 80; nf.Q.value = 0.6;
      ng.gain.setValueAtTime(vel * 0.38, t); ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.012);
      ns.connect(nf); nf.connect(ng); ng.connect(_drumGain); ns.start(t); ns.stop(t + 0.015);
    }
  }

  function _playSnare(t, vel) {
    _logDrumGM(t, 'snare', vel);
    if (!_noiseBuffer) return;
    var ctx = _audioCtx;
    // Crack -- high noise burst, filter opens wider on harder hits (brightness follows velocity)
    var ns1 = ctx.createBufferSource(), ng1 = ctx.createGain(), nf1 = ctx.createBiquadFilter();
    ns1.buffer = _noiseBuffer;
    nf1.type = 'bandpass'; nf1.frequency.value = 800 + vel * 1400; nf1.Q.value = 0.7 + vel * 0.5;
    ng1.gain.setValueAtTime(0, t);
    ng1.gain.linearRampToValueAtTime(vel * 0.90, t + 0.001);
    ng1.gain.exponentialRampToValueAtTime(0.0001, t + 0.06 + vel * 0.06);
    ns1.connect(nf1); nf1.connect(ng1); ng1.connect(_drumGain); ns1.start(t); ns1.stop(t + 0.16);
    // Wire buzz -- longer tail, filtered separately
    var ns2 = ctx.createBufferSource(), ng2 = ctx.createGain(), nf2 = ctx.createBiquadFilter();
    ns2.buffer = _noiseBuffer;
    nf2.type = 'highpass'; nf2.frequency.value = 3500;
    ng2.gain.setValueAtTime(0, t + 0.004);
    ng2.gain.linearRampToValueAtTime(vel * 0.30, t + 0.012);
    ng2.gain.exponentialRampToValueAtTime(0.0001, t + 0.10 + vel * 0.05);
    ns2.connect(nf2); nf2.connect(ng2); ng2.connect(_drumGain); ns2.start(t); ns2.stop(t + 0.18);
    // Body resonance -- pitch varies slightly per hit (no two snares sound identical)
    var bodyF = 175 + Math.random() * 25;
    var osc = ctx.createOscillator(), og = ctx.createGain();
    osc.type = 'triangle'; osc.frequency.setValueAtTime(bodyF * 1.3, t);
    osc.frequency.exponentialRampToValueAtTime(bodyF, t + 0.015);
    og.gain.setValueAtTime(0, t); og.gain.linearRampToValueAtTime(vel * 0.42, t + 0.001);
    og.gain.exponentialRampToValueAtTime(0.0001, t + 0.055 + vel * 0.02);
    osc.connect(og); og.connect(_drumGain); osc.start(t); osc.stop(t + 0.10);
  }

  function _playGhost(t, vel) {
    _logDrumGM(t, 'ghost', vel * 0.28);
    if (!_noiseBuffer) return;
    var ctx = _audioCtx;
    // Ghost: wire buzz only -- no crack, no body. Very soft, high freq.
    var ns = ctx.createBufferSource(), ng = ctx.createGain(), nf = ctx.createBiquadFilter();
    ns.buffer = _noiseBuffer;
    nf.type = 'bandpass'; nf.frequency.value = 2200 + Math.random() * 600; nf.Q.value = 1.1;
    ng.gain.setValueAtTime(0, t); ng.gain.linearRampToValueAtTime(vel * 0.14, t + 0.001);
    ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.045 + Math.random() * 0.02);
    ns.connect(nf); nf.connect(ng); ng.connect(_drumGain); ns.start(t); ns.stop(t + 0.07);
  }

  function _playHihat(t, vel, open) {
    _logDrumGM(t, open ? 'hihat_open' : 'hihat', vel);
    if (!_noiseBuffer) return;
    var ctx = _audioCtx;
    var dur = open ? (0.14 + vel * 0.12) : (0.018 + vel * 0.022);
    // Multi-band metallic character: three filtered noise layers
    [[7200, 0.8, 0.40], [10500, 0.5, 0.30], [4800, 1.4, 0.18]].forEach(function(band) {
      var ns = ctx.createBufferSource(), ng = ctx.createGain(), nf = ctx.createBiquadFilter();
      ns.buffer = _noiseBuffer;
      nf.type = 'bandpass'; nf.frequency.value = band[0]; nf.Q.value = band[1];
      ng.gain.setValueAtTime(0, t);
      ng.gain.linearRampToValueAtTime(vel * band[2], t + 0.0008);
      ng.gain.exponentialRampToValueAtTime(0.0001, t + dur * (open ? 1.0 : 0.9));
      ns.connect(nf); nf.connect(ng); ng.connect(_drumGain); ns.start(t); ns.stop(t + dur + 0.01);
    });
    // "Chick" transient: very short click right at t=0
    var tc = ctx.createBufferSource(), tg = ctx.createGain(), tf = ctx.createBiquadFilter();
    tc.buffer = _noiseBuffer; tf.type = 'highpass'; tf.frequency.value = 9000;
    tg.gain.setValueAtTime(vel * 0.55, t); tg.gain.exponentialRampToValueAtTime(0.0001, t + 0.006);
    tc.connect(tf); tf.connect(tg); tg.connect(_drumGain); tc.start(t); tc.stop(t + 0.008);
  }

  function _playRide(t, vel) {
    _logDrumGM(t, 'ride', vel);
    if (!_noiseBuffer) return;
    var ctx = _audioCtx;
    // Bell: pitched component at ~1000-1200 Hz -- sine + slight harmonics
    var bellFreq = 1020 + Math.random() * 60;
    [[1, 0.28], [2.76, 0.10], [5.4, 0.04]].forEach(function(h) {
      var osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = 'sine'; osc.frequency.value = bellFreq * h[0];
      g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(vel * h[1], t + 0.001);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.35 + vel * 0.20);
      osc.connect(g); g.connect(_drumGain); osc.start(t); osc.stop(t + 0.60);
    });
    // Wash: broadband shimmer after the bell
    var ns = ctx.createBufferSource(), ng = ctx.createGain(), nf = ctx.createBiquadFilter();
    ns.buffer = _noiseBuffer; nf.type = 'bandpass'; nf.frequency.value = 5500; nf.Q.value = 0.4;
    ng.gain.setValueAtTime(0, t + 0.003);
    ng.gain.linearRampToValueAtTime(vel * 0.16, t + 0.018);
    ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);
    ns.connect(nf); nf.connect(ng); ng.connect(_drumGain); ns.start(t); ns.stop(t + 0.25);
  }

  function _playCrash(t, vel) {
    _logDrumGM(t, 'crash', vel);
    if (!_noiseBuffer) return;
    var ctx = _audioCtx;
    // Crash: two noise layers -- initial explosion + long wash
    [[3800, 0.45, 0.008, 0.90], [6500, 0.38, 0.003, 0.55]].forEach(function(b) {
      var ns = ctx.createBufferSource(), ng = ctx.createGain(), nf = ctx.createBiquadFilter();
      ns.buffer = _noiseBuffer; nf.type = 'highpass'; nf.frequency.value = b[0];
      ng.gain.setValueAtTime(0, t); ng.gain.linearRampToValueAtTime(vel * b[1], t + b[2]);
      ng.gain.exponentialRampToValueAtTime(vel * b[1] * 0.25, t + 0.08);
      ng.gain.exponentialRampToValueAtTime(0.0001, t + b[3] + vel * 0.4);
      ns.connect(nf); nf.connect(ng); ng.connect(_drumGain); ns.start(t); ns.stop(t + 1.4);
    });
    // Bell shimmer at impact
    var osc = ctx.createOscillator(), og = ctx.createGain();
    osc.type = 'sine'; osc.frequency.value = 680 + Math.random() * 120;
    og.gain.setValueAtTime(vel * 0.18, t); og.gain.exponentialRampToValueAtTime(0.0001, t + 0.12);
    osc.connect(og); og.connect(_drumGain); osc.start(t); osc.stop(t + 0.15);
  }

  function _playTom(t, vel, pitch) {
    var type = 'tom_' + pitch;
    _logDrumGM(t, type, vel);
    var freqs = {hi:150, mid:105, lo:76};
    var f0 = (freqs[pitch] || 105) + Math.random() * 6; // slight pitch variation per hit
    var ctx = _audioCtx;
    // Sub sine: main tom body
    var osc = ctx.createOscillator(), g = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(f0 * 1.6, t);
    osc.frequency.exponentialRampToValueAtTime(f0, t + 0.04 + vel * 0.02);
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(vel * 0.88, t + 0.002);
    g.gain.exponentialRampToValueAtTime(vel * 0.25, t + 0.08);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.28 + vel * 0.06);
    osc.connect(g); g.connect(_drumGain); osc.start(t); osc.stop(t + 0.38);
    // Attack transient: stick hit
    if (_noiseBuffer) {
      var ns = ctx.createBufferSource(), ng = ctx.createGain(), nf = ctx.createBiquadFilter();
      ns.buffer = _noiseBuffer; nf.type = 'bandpass'; nf.frequency.value = f0 * 3; nf.Q.value = 1.2;
      ng.gain.setValueAtTime(vel * 0.30, t); ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.018);
      ns.connect(nf); nf.connect(ng); ng.connect(_drumGain); ns.start(t); ns.stop(t + 0.022);
    }
  }

  function _playClap(t, vel) {
    _logDrumGM(t, 'clap', vel);
    if (!_noiseBuffer) return;
    var ctx = _audioCtx;
    // Three staggered hand slaps + body resonance = real clap texture
    [0, 0.008, 0.018, 0.030].forEach(function(off, i) {
      var ns = ctx.createBufferSource(), ng = ctx.createGain(), nf = ctx.createBiquadFilter();
      ns.buffer = _noiseBuffer;
      nf.type = 'bandpass'; nf.frequency.value = 900 + vel * 600; nf.Q.value = 1.0 + i * 0.3;
      ng.gain.setValueAtTime(0, t+off); ng.gain.linearRampToValueAtTime(vel * (0.50 - i * 0.08), t+off+0.002);
      ng.gain.exponentialRampToValueAtTime(0.0001, t+off + 0.04 + i * 0.01);
      ns.connect(nf); nf.connect(ng); ng.connect(_drumGain); ns.start(t+off); ns.stop(t+off+0.06);
    });
  }

  function _playDrumEvent(type, t, vel) {
    if      (type==='kick')       _playKick(t, vel);
    else if (type==='snare')      _playSnare(t, vel);
    else if (type==='ghost')      _playGhost(t, vel);
    else if (type==='hihat')      _playHihat(t, vel, false);
    else if (type==='hihat_open') _playHihat(t, vel, true);
    else if (type==='ride')       _playRide(t, vel);
    else if (type==='crash')      _playCrash(t, vel);
    else if (type==='tom_hi')     _playTom(t, vel, 'hi');
    else if (type==='tom_mid')    _playTom(t, vel, 'mid');
    else if (type==='tom_lo')     _playTom(t, vel, 'lo');
    else if (type==='clap')       _playClap(t, vel);
  }

  // -- Drum fills -- called at end-of-phrase bars --------------------------------
  function _playDrumFill(startTime, beatDur, bpb, style) {
    var pick = Math.floor(Math.random() * 4);
    while (pick === _lastFillType) pick = Math.floor(Math.random() * 4);
    _lastFillType = pick;
    var t = startTime;
    if (pick === 0) {
      // Snare build: 8th notes on beat 3→4, getting louder
      [2.5, 3, 3.25, 3.5, 3.75].forEach(function(b, i) {
        if (b < bpb) _playDrumEvent('snare', t + b * beatDur, 0.42 + i * 0.11);
      });
    } else if (pick === 1) {
      // Tom cascade: hi → mid → lo → kick, timed to fall perfectly on beat 1
      [[2.5,'tom_hi',0.62],[3,'tom_hi',0.68],[3.33,'tom_mid',0.72],[3.67,'tom_lo',0.78],[3.92,'kick',0.88]].forEach(function(e) {
        if (e[0] < bpb) _playDrumEvent(e[1], t + e[0] * beatDur, e[2]);
      });
    } else if (pick === 2) {
      // Crash + snare accent on beat 3, then a tight ghost roll into beat 4
      _playDrumEvent('crash', t + 2 * beatDur, 0.70);
      [2.5, 2.75, 3, 3.25, 3.5, 3.75].forEach(function(b, i) {
        if (b < bpb) _playDrumEvent(i < 2 ? 'ghost' : 'snare', t + b * beatDur, 0.28 + i * 0.09);
      });
    } else {
      // Kick stutter + snare: unexpected syncopated push into the next bar
      [[2.5,'kick',0.75],[2.75,'kick',0.55],[3,'snare',0.80],[3.5,'snare',0.65],[3.75,'ghost',0.35]].forEach(function(e) {
        if (e[0] < bpb) _playDrumEvent(e[1], t + e[0] * beatDur, e[2]);
      });
    }
  }

  // -- Bass fill -- walking run toward next chord root ---------------------------
  function _playBassFill(root, nextRoot, startTime, beatDur, bpb) {
    var pc = _NOTE_PC[root]; if (pc === undefined) return;
    var base = 48 + pc;
    var nextPc = _NOTE_PC[nextRoot]; if (nextPc === undefined) return;
    var target = 48 + nextPc; if (target <= base) target += 12;
    // Chromatic run from current root toward target across beats 3-4
    var span = target - base;
    var steps = Math.min(4, Math.abs(span));
    var dir = span > 0 ? 1 : -1;
    for (var i = 0; i < steps; i++) {
      var midi = base + dir * i;
      var beat = (bpb - steps + i);
      if (beat < 0) continue;
      (function(m, b) {
        var f = 440 * Math.pow(2, (m - 69) / 12), ctx = _audioCtx;
        var g = ctx.createGain(), o = ctx.createOscillator();
        o.type = 'sine'; o.frequency.value = f;
        var bt = startTime + b * beatDur;
        g.gain.setValueAtTime(0, bt); g.gain.linearRampToValueAtTime(0.75, bt + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, bt + beatDur * 0.82);
        _logNote(bt, beatDur * 0.82, m, 0.75, 1);
        o.connect(g); g.connect(_bassGain); o.start(bt); o.stop(bt + beatDur);
      })(midi, beat);
    }
  }

  function _varLen(n) {
    n = Math.max(0, Math.round(n));
    if (n === 0) return [0];
    var b = [];
    while (n > 0) { b.unshift(n & 0x7F); n >>>= 7; }
    for (var i=0; i<b.length-1; i++) b[i] |= 0x80;
    return b;
  }

  function _midiProgForStyle() {
    var map = {piano:0, strings:48, guitar:25, guitar_bossa:24, pad:89, honkyton:3, accordion:21};
    return map[_STYLE_INSTR[(_accompState||{}).style] || 'piano'] || 0;
  }

  function _buildMIDI(events, bpm, tpb) {
    tpb = tpb || 480;
    var tps = tpb * bpm / 60; // ticks per second
    var us  = Math.round(60000000 / bpm);
    var timeSigNum = parseInt((_recordTimeSig||'4/4').split('/')[0]) || 4;

    // Group events by channel
    var byChannel = {};
    events.forEach(function(ev) {
      if (ev.t < 0) return; // before recording start
      if (!byChannel[ev.ch]) byChannel[ev.ch] = [];
      byChannel[ev.ch].push(ev);
    });
    var chans = Object.keys(byChannel).map(Number).sort(function(a,b){return a-b;});

    // Track builder helper
    function buildTrack(bytes) {
      bytes = bytes.concat([0x00, 0xFF, 0x2F, 0x00]); // end-of-track
      var len = bytes.length;
      return [0x4D,0x54,0x72,0x6B,
        (len>>24)&0xFF,(len>>16)&0xFF,(len>>8)&0xFF,len&0xFF].concat(bytes);
    }

    // Track 0: tempo + time signature
    var t0 = [
      0x00, 0xFF, 0x58, 0x04, timeSigNum, 0x02, 0x18, 0x08, // time sig
      0x00, 0xFF, 0x51, 0x03, (us>>16)&0xFF, (us>>8)&0xFF, us&0xFF, // tempo
    ];

    var chNames = {0:'Chords', 1:'Bass', 9:'Drums'};
    var chProg  = {0:_midiProgForStyle(), 1:32}; // 32=Acoustic Bass; ch9 needs no program

    var nTracks = 1 + chans.length;
    // MThd
    var mthd = [0x4D,0x54,0x68,0x64, 0,0,0,6,
      0,(nTracks>1?1:0), (nTracks>>8)&0xFF,nTracks&0xFF,
      (tpb>>8)&0xFF, tpb&0xFF];

    var bytes = mthd.concat(buildTrack(t0));

    chans.forEach(function(ch) {
      var evs = byChannel[ch].slice().sort(function(a,b){return a.t-b.t;});
      var tb  = [];
      // Track name
      var nm = (chNames[ch]||('Ch'+ch));
      tb = tb.concat([0x00, 0xFF, 0x03, nm.length]
                     .concat(nm.split('').map(function(c){return c.charCodeAt(0);})));
      // Program change (not for drums)
      if (ch !== 9 && chProg[ch] !== undefined) {
        tb = tb.concat([0x00, 0xC0|ch, chProg[ch]&0x7F]);
      }
      var prev = 0;
      evs.forEach(function(ev) {
        var tick  = Math.max(0, Math.round(ev.t * tps));
        var delta = Math.max(0, tick - prev); prev = tick;
        tb = tb.concat(_varLen(delta));
        var vel127 = Math.max(1, Math.min(127, Math.round(ev.vel * 127)));
        if (ev.type === 'on') tb.push(0x90|ch, ev.note, vel127);
        else                  tb.push(0x80|ch, ev.note, 0);
      });
      bytes = bytes.concat(buildTrack(tb));
    });

    return new Uint8Array(bytes);
  }

  window.toggleRecording = function() {
    if (_recording) {
      _recording = false;
      clearInterval(_recordTimer);
      var btn = document.getElementById('rec-btn');
      var dur = document.getElementById('rec-dur');
      if (btn) { btn.textContent = '⏺ Record'; btn.style.background = '#3d1a1a'; }
      if (dur) dur.textContent = '';

      if (_recordEvents.length > 0) {
        var mid = _buildMIDI(_recordEvents, _recordBPM, 480);
        var blob = new Blob([mid], {type:'audio/midi'});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href = url;
        var ts = new Date().toISOString().slice(0,19).replace(/[T:]/g,'-');
        a.download = 'accompaniment_' + ts + '.mid';
        document.body.appendChild(a); a.click();
        document.body.removeChild(a); URL.revokeObjectURL(url);
      }
      _recordEvents = [];
    } else {
      if (!_audioCtx) { alert('Start accompaniment first'); return; }
      _recording     = true;
      _recordStart   = _audioCtx.currentTime;
      _recordEvents  = [];
      _recordBPM     = parseFloat((document.getElementById('ctrl-bpm')||{}).value) || 120;
      _recordTimeSig = (document.getElementById('ctrl-timesig')||{}).value || '4/4';

      var t0 = Date.now();
      var btn = document.getElementById('rec-btn');
      var dur = document.getElementById('rec-dur');
      if (btn) { btn.textContent = '⏹ Stop & save'; btn.style.background = '#8b0000'; }
      _recordTimer = setInterval(function() {
        var s = Math.floor((Date.now()-t0)/1000), m = Math.floor(s/60); s %= 60;
        if (dur) dur.textContent = m+':'+(s<10?'0':'')+s;
      }, 1000);
    }
  };

  function _updateMixChordLabel() {
    var inst = _STYLE_INSTR[_accompState.style] || 'piano';
    var el = document.getElementById('mix-chord-label');
    if (el) el.textContent = _INSTR_LABEL[inst] || 'Piano';
  }

  function _playBassNote(note, t, dur) {
    var pc = _NOTE_PC[note]; if (pc === undefined) return;
    // Hard-cap duration so notes don't bleed into each other at faster tempos
    var d = Math.min(dur, 1.6);
    _logNote(t, d, 48 + pc, 0.9, 1);
    var midi = 48 + pc;
    var freq = 440 * Math.pow(2, (midi - 69) / 12);
    var ctx  = _audioCtx;
    // Master gain -- all layers go through this so they share the amplitude envelope
    var master = ctx.createGain();
    master.gain.setValueAtTime(0, t);
    master.gain.linearRampToValueAtTime(0.92, t + 0.020);
    master.gain.exponentialRampToValueAtTime(0.62, t + 0.18);
    master.gain.exponentialRampToValueAtTime(0.0001, t + d);
    master.connect(_bassGain);
    // Fundamental: sine with fretless pitch slide from 2.5% below
    var osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq * 0.975, t);
    osc.frequency.exponentialRampToValueAtTime(freq, t + 0.035);
    osc.connect(master);
    // 2nd harmonic: adds warmth, fades fast (like a plucked string)
    var osc2 = ctx.createOscillator(), g2 = ctx.createGain();
    osc2.type = 'sine'; osc2.frequency.value = freq * 2;
    g2.gain.setValueAtTime(0.14, t);
    g2.gain.exponentialRampToValueAtTime(0.0001, t + Math.min(d * 0.5, 0.5));
    osc2.connect(g2); g2.connect(master);
    // Pluck transient -- short click, modest level
    if (_noiseBuffer) {
      var ns = ctx.createBufferSource(), ng2 = ctx.createGain(), nf = ctx.createBiquadFilter();
      ns.buffer = _noiseBuffer; nf.type = 'bandpass'; nf.frequency.value = freq * 3.5; nf.Q.value = 2.5;
      ng2.gain.setValueAtTime(0.22, t); ng2.gain.exponentialRampToValueAtTime(0.0001, t + 0.010);
      ns.connect(nf); nf.connect(ng2); ng2.connect(_bassGain); ns.start(t); ns.stop(t + 0.013);
    }
    [osc, osc2].forEach(function(o){ o.start(t); o.stop(t + d + 0.04); });
  }

  // Walking bass: root + passing tones toward next chord root
  function _playWalkingBass(chord, nextChord, startTime, beatDur, bpb) {
    var root = (chord.notes || [])[0]; if (!root) return;
    var pc = _NOTE_PC[root]; if (pc === undefined) return;
    var rootMidi = 48 + pc;
    // Beat 1: root
    _playBassNote(root, startTime, beatDur * 0.88);
    if (bpb < 3) return;
    // Beat 2: 5th above (or down an octave if too high)
    var fifthMidi = rootMidi + 7; if (fifthMidi > 57) fifthMidi -= 12;
    var fNode = 440 * Math.pow(2, (fifthMidi - 69) / 12);
    (function(f, t2) {
      var ctx = _audioCtx, g = ctx.createGain(), o = ctx.createOscillator();
      o.type = 'sine'; o.frequency.value = f;
      g.gain.setValueAtTime(0,t2); g.gain.linearRampToValueAtTime(0.72,t2+0.025);
      g.gain.exponentialRampToValueAtTime(0.0001,t2+beatDur*0.85);
      _logNote(t2, beatDur*0.85, fifthMidi, 0.72, 1);
      o.connect(g); g.connect(_bassGain); o.start(t2); o.stop(t2+beatDur);
    })(fNode, startTime + beatDur);
    if (bpb < 4) return;
    // Beat 3: chord 3rd or return to root an octave up
    var thirdMidi = rootMidi + 4; if (thirdMidi > 57) thirdMidi -= 12;
    (function(midi, t3) {
      var f = 440 * Math.pow(2, (midi-69)/12), ctx = _audioCtx;
      var g = ctx.createGain(), o = ctx.createOscillator();
      o.type = 'sine'; o.frequency.value = f;
      g.gain.setValueAtTime(0,t3); g.gain.linearRampToValueAtTime(0.65,t3+0.025);
      g.gain.exponentialRampToValueAtTime(0.0001,t3+beatDur*0.82);
      _logNote(t3, beatDur*0.82, midi, 0.65, 1);
      o.connect(g); g.connect(_bassGain); o.start(t3); o.stop(t3+beatDur);
    })(thirdMidi, startTime + beatDur * 2);
    // Beat 4: chromatic approach to next chord root
    var nextRoot = nextChord && nextChord.notes ? nextChord.notes[0] : root;
    var nextPc = _NOTE_PC[nextRoot]; if (nextPc === undefined) nextPc = pc;
    var nextMidi = 48 + nextPc; if (nextMidi <= rootMidi) nextMidi += 12;
    var approachMidi = nextMidi - 1; // half-step below
    (function(midi, t4) {
      var f = 440 * Math.pow(2, (midi-69)/12), ctx = _audioCtx;
      var g = ctx.createGain(), o = ctx.createOscillator();
      o.type = 'sine'; o.frequency.value = f;
      g.gain.setValueAtTime(0,t4); g.gain.linearRampToValueAtTime(0.68,t4+0.020);
      g.gain.exponentialRampToValueAtTime(0.0001,t4+beatDur*0.78);
      _logNote(t4, beatDur*0.78, midi, 0.68, 1);
      o.connect(g); g.connect(_bassGain); o.start(t4); o.stop(t4+beatDur);
    })(approachMidi, startTime + beatDur * 3);
  }

  function _playBrush(t, vel, type) {
    _logDrum(t, type);
    if (!_noiseBuffer) return;
    var ctx = _audioCtx;
    var src = ctx.createBufferSource(); src.buffer = _noiseBuffer;
    var flt = ctx.createBiquadFilter();
    var g   = ctx.createGain();
    var stopAt;
    if (type === 'snare') {
      flt.type = 'bandpass'; flt.frequency.value = 900; flt.Q.value = 0.6;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel, t + 0.003);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.07);
      stopAt = t + 0.09;
    } else {
      flt.type = 'bandpass'; flt.frequency.value = 280; flt.Q.value = 0.5;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel, t + 0.008);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.14);
      stopAt = t + 0.18;
    }
    src.connect(flt); flt.connect(g); g.connect(_drumGain);
    src.start(t); src.stop(stopAt);
  }

  // -- Rhythm patterns ---------------------------------------------------------
  // d = minimum density level to include (1=sparse, 2=moderate, 3=dense)

  var _PATTERNS = {
    'Pad': function(bpb) { return [
      {beat:0, type:'bass',  vel:0.75, d:1},
      {beat:0, type:'chord', vel:0.50, d:1},
      {beat:2, type:'bass',  vel:0.62, d:2},
      {beat:3, type:'bass',  vel:0.50, d:3}]; },
    'Blues': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.85, d:1},
      {beat:0.5, type:'chord', vel:0.55, d:2},
      {beat:1,   type:'bass',  vel:0.70, d:3},
      {beat:1.5, type:'chord', vel:0.60, d:1},
      {beat:2,   type:'bass',  vel:0.80, d:2},
      {beat:2.5, type:'chord', vel:0.50, d:2},
      {beat:3,   type:'bass',  vel:0.70, d:3},
      {beat:3.5, type:'chord', vel:0.58, d:1}]; },
    'Honky Tonk': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.88, d:1},
      {beat:0.5, type:'chord', vel:0.68, d:1},
      {beat:1,   type:'chord', vel:0.55, d:2},
      {beat:1.5, type:'chord', vel:0.62, d:1},
      {beat:1.5, type:'bass',  vel:0.68, d:3},
      {beat:2,   type:'bass',  vel:0.82, d:2},
      {beat:2.5, type:'chord', vel:0.68, d:1},
      {beat:3,   type:'chord', vel:0.52, d:2},
      {beat:3,   type:'bass',  vel:0.62, d:3},
      {beat:3.5, type:'chord', vel:0.62, d:1}]; },
    'Ballad': function(bpb) {
      if (bpb === 3) return [
        {beat:0, type:'bass',  vel:0.85, d:1},
        {beat:1, type:'chord', vel:0.55, d:1},
        {beat:2, type:'bass',  vel:0.65, d:3},
        {beat:2, type:'chord', vel:0.45, d:2}];
      return [
        {beat:0,   type:'bass',  vel:0.82, d:1},
        {beat:0.5, type:'chord', vel:0.42, d:2},
        {beat:1.5, type:'chord', vel:0.52, d:1},
        {beat:2,   type:'bass',  vel:0.72, d:2},
        {beat:2.5, type:'chord', vel:0.42, d:2},
        {beat:3,   type:'bass',  vel:0.60, d:3},
        {beat:3.5, type:'chord', vel:0.52, d:1}];
    },
    'Jazz': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.80, d:1},
      {beat:0.5, type:'chord', vel:0.48, d:2},
      {beat:1,   type:'bass',  vel:0.65, d:3},
      {beat:1.5, type:'chord', vel:0.55, d:1},
      {beat:2,   type:'bass',  vel:0.72, d:2},
      {beat:2.5, type:'chord', vel:0.50, d:2},
      {beat:3,   type:'bass',  vel:0.60, d:3},
      {beat:3,   type:'chord', vel:0.42, d:3},
      {beat:3.5, type:'chord', vel:0.55, d:1}]; },
    'Waltz': function(bpb) { return [
      {beat:0, type:'bass',  vel:0.88, d:1},
      {beat:1, type:'chord', vel:0.52, d:1},
      {beat:1, type:'bass',  vel:0.60, d:2},
      {beat:2, type:'chord', vel:0.42, d:1},
      {beat:2, type:'bass',  vel:0.52, d:3}]; },
    'Bossa Nova': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.82, d:1},
      {beat:0.5, type:'chord', vel:0.48, d:2},
      {beat:1,   type:'chord', vel:0.42, d:3},
      {beat:1.5, type:'chord', vel:0.55, d:1},
      {beat:2,   type:'bass',  vel:0.70, d:3},
      {beat:2.5, type:'bass',  vel:0.68, d:2},
      {beat:3,   type:'chord', vel:0.42, d:2},
      {beat:3.5, type:'chord', vel:0.55, d:1}]; },
    // Rhodes: jazz comp feel -- chord on 2 and 4 with added anticipation
    'Rhodes': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.78, d:1},
      {beat:1,   type:'chord', vel:0.55, d:1},
      {beat:1.5, type:'chord', vel:0.42, d:2},
      {beat:2,   type:'bass',  vel:0.65, d:2},
      {beat:2.5, type:'chord', vel:0.38, d:3},
      {beat:3,   type:'bass',  vel:0.55, d:3},
      {beat:3,   type:'chord', vel:0.58, d:1},
      {beat:3.5, type:'chord', vel:0.44, d:2}]; },
    'Rhodes Jazz': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.75, d:1},
      {beat:0.5, type:'chord', vel:0.44, d:2},
      {beat:1,   type:'bass',  vel:0.60, d:3},
      {beat:1.5, type:'chord', vel:0.58, d:1},
      {beat:2,   type:'bass',  vel:0.65, d:2},
      {beat:2.5, type:'chord', vel:0.40, d:3},
      {beat:3,   type:'bass',  vel:0.55, d:3},
      {beat:3.5, type:'chord', vel:0.62, d:1}]; },
    // Vibraphone: sparse and open -- lots of ring
    'Vibraphone': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.70, d:1},
      {beat:1,   type:'chord', vel:0.60, d:1},
      {beat:2,   type:'bass',  vel:0.58, d:2},
      {beat:2.5, type:'chord', vel:0.50, d:1},
      {beat:3,   type:'bass',  vel:0.50, d:3},
      {beat:3.5, type:'chord', vel:0.42, d:2}]; },
    // Funk Chop: tight 16th-note stabs, no sustain
    'Funk Chop': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.90, d:1},
      {beat:0.5,  type:'chord', vel:0.72, d:1},
      {beat:1,    type:'chord', vel:0.38, d:2},
      {beat:1,    type:'bass',  vel:0.68, d:3},
      {beat:1.5,  type:'chord', vel:0.68, d:1},
      {beat:2,    type:'bass',  vel:0.82, d:2},
      {beat:2.25, type:'chord', vel:0.30, d:3},
      {beat:2.5,  type:'chord', vel:0.65, d:1},
      {beat:3,    type:'chord', vel:0.36, d:2},
      {beat:3,    type:'bass',  vel:0.60, d:3},
      {beat:3.5,  type:'chord', vel:0.70, d:1},
      {beat:3.75, type:'chord', vel:0.28, d:3}]; },
    // Lo-Fi: slow, spaced-out Rhodes hits with laid-back feel
    'Lo-Fi': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.72, d:1},
      {beat:0.5, type:'chord', vel:0.50, d:2},
      {beat:1.5, type:'bass',  vel:0.55, d:3},
      {beat:2,   type:'bass',  vel:0.60, d:2},
      {beat:2.5, type:'chord', vel:0.62, d:1},
      {beat:3,   type:'bass',  vel:0.48, d:3},
      {beat:3.5, type:'chord', vel:0.44, d:2}]; },
    // Acoustic: fingerpick pattern -- bass on 1, plucked arpeggiated chord fills
    'Acoustic': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.82, d:1},
      {beat:0.5,  type:'chord', vel:0.52, d:1},
      {beat:1,    type:'chord', vel:0.42, d:2},
      {beat:1.5,  type:'chord', vel:0.56, d:1},
      {beat:2,    type:'bass',  vel:0.72, d:2},
      {beat:2.5,  type:'chord', vel:0.50, d:1},
      {beat:3,    type:'chord', vel:0.44, d:2},
      {beat:3.5,  type:'chord', vel:0.58, d:1}]; },
    // Disco Pop: four-on-the-floor bass, punchy off-beat chords
    // d:1=sparse(beat 0 only), d:2=moderate(beat 0+2), d:3=dense(all four beats)
    'Disco Pop': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.90, d:1},
      {beat:0.5,  type:'chord', vel:0.70, d:1},
      {beat:1,    type:'bass',  vel:0.80, d:3},
      {beat:1.5,  type:'chord', vel:0.62, d:1},
      {beat:2,    type:'bass',  vel:0.88, d:2},
      {beat:2.5,  type:'chord', vel:0.72, d:1},
      {beat:3,    type:'bass',  vel:0.78, d:3},
      {beat:3.5,  type:'chord', vel:0.68, d:1}]; },
    // Brushed Trio: sparse jazz feel, chord on 2+4 only, walking bass
    'Brushed Trio': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.72, d:1},
      {beat:1,   type:'chord', vel:0.50, d:1},
      {beat:1,   type:'bass',  vel:0.58, d:3},
      {beat:2,   type:'bass',  vel:0.60, d:2},
      {beat:3,   type:'chord', vel:0.55, d:1},
      {beat:3,   type:'bass',  vel:0.52, d:3}]; },
    // Jazz Shell: very open comping -- long spaces, shell voicings, comp on "and" beats
    'Jazz Shell': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.70, d:1},
      {beat:0.5,  type:'chord', vel:0.42, d:1},
      {beat:1.5,  type:'chord', vel:0.50, d:2},
      {beat:2,    type:'bass',  vel:0.58, d:2},
      {beat:2.5,  type:'chord', vel:0.45, d:1},
      {beat:3,    type:'bass',  vel:0.50, d:3},
      {beat:3.5,  type:'chord', vel:0.55, d:2}]; },
    // R&B: chord on 2+4, 16th-note bass push, soulful
    'R&B': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.88, d:1},
      {beat:0.75, type:'chord', vel:0.62, d:2},
      {beat:1,    type:'bass',  vel:0.72, d:3},
      {beat:1.5,  type:'chord', vel:0.78, d:1},
      {beat:2,    type:'bass',  vel:0.80, d:2},
      {beat:2.75, type:'chord', vel:0.58, d:2},
      {beat:3,    type:'bass',  vel:0.68, d:3},
      {beat:3.5,  type:'chord', vel:0.82, d:1}]; },
    // Neo Soul: complex syncopation, off-beat chords (D'Angelo, Frank Ocean feel)
    'Neo Soul': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.82, d:1},
      {beat:0.5,  type:'chord', vel:0.50, d:2},
      {beat:0.75, type:'chord', vel:0.38, d:3},
      {beat:1.5,  type:'bass',  vel:0.65, d:3},
      {beat:1.75, type:'chord', vel:0.62, d:1},
      {beat:2.25, type:'bass',  vel:0.72, d:2},
      {beat:2.5,  type:'chord', vel:0.45, d:2},
      {beat:3,    type:'bass',  vel:0.58, d:3},
      {beat:3.25, type:'chord', vel:0.68, d:1},
      {beat:3.75, type:'chord', vel:0.40, d:2}]; },
    // Gospel: big chords on all beats, bass walking (Hezekiah Walker feel)
    'Gospel': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.90, d:1},
      {beat:0,    type:'chord', vel:0.82, d:1},
      {beat:1,    type:'bass',  vel:0.75, d:2},
      {beat:1,    type:'chord', vel:0.72, d:1},
      {beat:2,    type:'bass',  vel:0.85, d:1},
      {beat:2,    type:'chord', vel:0.78, d:1},
      {beat:2.5,  type:'chord', vel:0.55, d:2},
      {beat:3,    type:'bass',  vel:0.70, d:2},
      {beat:3,    type:'chord', vel:0.68, d:1},
      {beat:3.5,  type:'chord', vel:0.80, d:1}]; },
    // Soul: Motown/Stax feel -- bass push, chord on 2+4
    'Soul': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.90, d:1},
      {beat:0.5,  type:'chord', vel:0.48, d:3},
      {beat:1,    type:'bass',  vel:0.68, d:3},
      {beat:1.5,  type:'chord', vel:0.80, d:1},
      {beat:2,    type:'bass',  vel:0.82, d:2},
      {beat:2.5,  type:'chord', vel:0.50, d:3},
      {beat:3,    type:'bass',  vel:0.65, d:3},
      {beat:3.5,  type:'chord', vel:0.84, d:1}]; },
    // Pop: simple 4-on-the-floor feel (Ed Sheeran, Taylor Swift)
    'Pop': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.85, d:1},
      {beat:0.5,  type:'chord', vel:0.60, d:2},
      {beat:1,    type:'chord', vel:0.52, d:1},
      {beat:1.5,  type:'chord', vel:0.48, d:2},
      {beat:2,    type:'bass',  vel:0.80, d:2},
      {beat:2.5,  type:'chord', vel:0.62, d:1},
      {beat:3,    type:'chord', vel:0.50, d:2},
      {beat:3,    type:'bass',  vel:0.62, d:3},
      {beat:3.5,  type:'chord', vel:0.58, d:1}]; },
    // Country: boom-chick pattern
    'Country': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.90, d:1},
      {beat:1,   type:'chord', vel:0.70, d:1},
      {beat:2,   type:'bass',  vel:0.82, d:2},
      {beat:2.5, type:'bass',  vel:0.58, d:3},
      {beat:3,   type:'chord', vel:0.68, d:1},
      {beat:3.5, type:'chord', vel:0.42, d:2}]; },
    // Reggae: skank on the off-beats (beats 2+4 and their "and"s)
    'Reggae': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.88, d:1},
      {beat:1,    type:'chord', vel:0.72, d:1},
      {beat:1.5,  type:'chord', vel:0.45, d:2},
      {beat:2,    type:'bass',  vel:0.70, d:2},
      {beat:3,    type:'chord', vel:0.75, d:1},
      {beat:3.5,  type:'chord', vel:0.42, d:2},
      {beat:3.75, type:'bass',  vel:0.55, d:3}]; },
    // Motown: punchy, bass-forward
    'Motown': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.92, d:1},
      {beat:0.75, type:'bass',  vel:0.60, d:3},
      {beat:1,    type:'chord', vel:0.65, d:1},
      {beat:1.5,  type:'bass',  vel:0.72, d:2},
      {beat:2,    type:'bass',  vel:0.85, d:1},
      {beat:2.75, type:'bass',  vel:0.58, d:3},
      {beat:3,    type:'chord', vel:0.68, d:1},
      {beat:3.5,  type:'bass',  vel:0.65, d:2}]; },
    // Latin: clave feel (3-2 son clave)
    'Latin': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.88, d:1},
      {beat:0.5,  type:'chord', vel:0.55, d:2},
      {beat:0.75, type:'chord', vel:0.42, d:3},
      {beat:1.5,  type:'bass',  vel:0.70, d:2},
      {beat:2,    type:'chord', vel:0.62, d:1},
      {beat:2.5,  type:'bass',  vel:0.75, d:3},
      {beat:3,    type:'chord', vel:0.50, d:2},
      {beat:3.5,  type:'bass',  vel:0.60, d:2}]; },
    // Singer-Songwriter: intimate fingerpick, very sparse
    'Singer-Songwriter': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.80, d:1},
      {beat:0.5,  type:'chord', vel:0.48, d:1},
      {beat:1,    type:'chord', vel:0.38, d:2},
      {beat:1.5,  type:'chord', vel:0.52, d:1},
      {beat:2,    type:'bass',  vel:0.70, d:2},
      {beat:2.5,  type:'chord', vel:0.44, d:1},
      {beat:3,    type:'chord', vel:0.40, d:2},
      {beat:3.5,  type:'chord', vel:0.56, d:1}]; },
    // Smooth Jazz: laid-back comp, chord on 2 and 4
    'Smooth Jazz': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.75, d:1},
      {beat:0.5,  type:'chord', vel:0.40, d:2},
      {beat:1.5,  type:'chord', vel:0.62, d:1},
      {beat:2,    type:'bass',  vel:0.62, d:2},
      {beat:2.5,  type:'chord', vel:0.38, d:2},
      {beat:3,    type:'bass',  vel:0.52, d:3},
      {beat:3.5,  type:'chord', vel:0.65, d:1}]; },
    // Cinematic: slow, spacious strings
    'Cinematic': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.78, d:1},
      {beat:0,    type:'chord', vel:0.60, d:1},
      {beat:2,    type:'bass',  vel:0.65, d:2},
      {beat:2,    type:'chord', vel:0.50, d:2}]; },
    // Worship: big sustained chords, simple bass
    'Worship': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.85, d:1},
      {beat:0,    type:'chord', vel:0.75, d:1},
      {beat:2,    type:'bass',  vel:0.72, d:2},
      {beat:2,    type:'chord', vel:0.65, d:2},
      {beat:3.5,  type:'chord', vel:0.70, d:1}]; },
    // Indie Pop: strummed feel with passing notes
    'Indie Pop': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.82, d:1},
      {beat:0.5,  type:'chord', vel:0.55, d:1},
      {beat:1,    type:'chord', vel:0.48, d:2},
      {beat:2,    type:'bass',  vel:0.75, d:2},
      {beat:2.5,  type:'chord', vel:0.60, d:1},
      {beat:3,    type:'bass',  vel:0.62, d:3},
      {beat:3.5,  type:'chord', vel:0.52, d:1}]; },
    // Funk: tight 16ths, syncopated
    'Funk': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.92, d:1},
      {beat:0.25, type:'chord', vel:0.35, d:3},
      {beat:0.5,  type:'chord', vel:0.70, d:1},
      {beat:1,    type:'bass',  vel:0.65, d:3},
      {beat:1.5,  type:'chord', vel:0.72, d:2},
      {beat:2,    type:'bass',  vel:0.88, d:2},
      {beat:2.5,  type:'chord', vel:0.38, d:3},
      {beat:3,    type:'bass',  vel:0.60, d:3},
      {beat:3.5,  type:'chord', vel:0.75, d:1}]; },
    // Tropical: marimba/vibes feel
    'Tropical': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.80, d:1},
      {beat:0.5,  type:'chord', vel:0.58, d:1},
      {beat:1,    type:'chord', vel:0.45, d:2},
      {beat:1.5,  type:'bass',  vel:0.60, d:3},
      {beat:2,    type:'chord', vel:0.62, d:1},
      {beat:2.5,  type:'chord', vel:0.50, d:2},
      {beat:3,    type:'bass',  vel:0.72, d:2},
      {beat:3.5,  type:'chord', vel:0.55, d:1}]; },
    // Brass: punchy stabs, short hits
    'Brass': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.90, d:1},
      {beat:0.5,  type:'chord', vel:0.85, d:1},
      {beat:1.5,  type:'chord', vel:0.72, d:2},
      {beat:2,    type:'bass',  vel:0.82, d:2},
      {beat:2.5,  type:'chord', vel:0.80, d:1},
      {beat:3.5,  type:'chord', vel:0.68, d:2}]; },
    // Samba: Brazilian bounce
    'Samba': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.88, d:1},
      {beat:0.5,  type:'chord', vel:0.52, d:2},
      {beat:0.75, type:'chord', vel:0.40, d:3},
      {beat:1.5,  type:'bass',  vel:0.72, d:2},
      {beat:2,    type:'bass',  vel:0.80, d:1},
      {beat:2.5,  type:'chord', vel:0.55, d:2},
      {beat:3,    type:'chord', vel:0.42, d:3},
      {beat:3.5,  type:'bass',  vel:0.65, d:3}]; },
    // Swing: big band feel
    'Swing': function(bpb) { return [
      {beat:0,   type:'bass',  vel:0.85, d:1},
      {beat:0.5, type:'chord', vel:0.55, d:2},
      {beat:1,   type:'bass',  vel:0.65, d:3},
      {beat:2,   type:'bass',  vel:0.80, d:1},
      {beat:2.5, type:'chord', vel:0.60, d:2},
      {beat:3,   type:'bass',  vel:0.60, d:3},
      {beat:3,   type:'chord', vel:0.50, d:1},
      {beat:3.5, type:'chord', vel:0.68, d:2}]; },
    // New Soul: strings + bass, cinematic soul (like Lauryn Hill)
    'New Soul': function(bpb) { return [
      {beat:0,    type:'bass',  vel:0.82, d:1},
      {beat:0.5,  type:'chord', vel:0.55, d:2},
      {beat:1.5,  type:'chord', vel:0.68, d:1},
      {beat:2,    type:'bass',  vel:0.72, d:2},
      {beat:2.5,  type:'chord', vel:0.48, d:2},
      {beat:3,    type:'bass',  vel:0.60, d:3},
      {beat:3.5,  type:'chord', vel:0.72, d:1}]; },
  };

  // -- Drum pattern library -----------------------------------------------------
  // Events: beat (0-based), type (see _DRUM_GM), vel (0-1), d (min density 1-3)
  // Events with beat >= bpb are skipped for shorter time signatures automatically.

  var _activeDrumPattern = 'Rock Basic';
  window.setDrumPattern = function(name) { _activeDrumPattern = name; };

  var _DRUM_LIBRARY = {

  'Ballad': { label:'Ballad', cat:'Soft', events: [
    {beat:0,   type:'kick',  vel:0.70,d:1},{beat:2,   type:'kick',  vel:0.50,d:2},
    {beat:1,   type:'snare', vel:0.55,d:1},{beat:3,   type:'snare', vel:0.55,d:1},
    {beat:0,   type:'hihat', vel:0.32,d:1},{beat:0.5, type:'hihat', vel:0.20,d:2},
    {beat:1,   type:'hihat', vel:0.28,d:1},{beat:1.5, type:'hihat', vel:0.18,d:2},
    {beat:2,   type:'hihat', vel:0.28,d:1},{beat:2.5, type:'hihat', vel:0.18,d:2},
    {beat:3,   type:'hihat', vel:0.26,d:1},{beat:3.5, type:'hihat', vel:0.16,d:2},
  ]},

  'Rock Basic': { label:'Rock Basic', cat:'Rock', events: [
    // d:1 sparse — kick+snare + quarter hi-hats
    {beat:0,   type:'kick',  vel:0.88,d:1},{beat:2,   type:'kick',  vel:0.82,d:1},
    {beat:1,   type:'snare', vel:0.82,d:1},{beat:3,   type:'snare', vel:0.82,d:1},
    {beat:0,   type:'hihat', vel:0.55,d:1},{beat:1,   type:'hihat', vel:0.50,d:1},
    {beat:2,   type:'hihat', vel:0.52,d:1},{beat:3,   type:'hihat', vel:0.48,d:1},
    // d:2 moderate — 8th hi-hats
    {beat:0.5, type:'hihat', vel:0.42,d:2},{beat:1.5, type:'hihat', vel:0.40,d:2},
    {beat:2.5, type:'hihat', vel:0.40,d:2},{beat:3.5, type:'hihat', vel:0.38,d:2},
    // d:3 dense — add syncopated kick + ghost snares to fill
    {beat:2.5, type:'kick',  vel:0.55,d:3},{beat:0.5, type:'ghost', vel:0.22,d:3},
    {beat:1.5, type:'ghost', vel:0.20,d:3},{beat:2.5, type:'ghost', vel:0.22,d:3},
    {beat:3.5, type:'ghost', vel:0.20,d:3},
  ]},

  'Rock Groove': { label:'Rock Groove', cat:'Rock', events: [
    {beat:0,    type:'kick',      vel:0.90,d:1},{beat:0.75, type:'kick',  vel:0.62,d:2},
    {beat:2,    type:'kick',      vel:0.88,d:1},{beat:3.5,  type:'kick',  vel:0.58,d:2},
    {beat:1,    type:'snare',     vel:0.85,d:1},{beat:3,    type:'snare', vel:0.85,d:1},
    {beat:1.5,  type:'ghost',     vel:0.28,d:3},{beat:2.5,  type:'ghost', vel:0.25,d:3},
    {beat:0,    type:'hihat',     vel:0.55,d:1},{beat:0.5,  type:'hihat', vel:0.48,d:1},
    {beat:1,    type:'hihat',     vel:0.52,d:1},{beat:1.5,  type:'hihat', vel:0.42,d:1},
    {beat:2,    type:'hihat',     vel:0.52,d:1},{beat:2.5,  type:'hihat_open',vel:0.58,d:2},
    {beat:3,    type:'hihat',     vel:0.50,d:1},{beat:3.5,  type:'hihat', vel:0.40,d:1},
    {beat:0.25, type:'hihat',     vel:0.28,d:3},{beat:0.75, type:'hihat', vel:0.24,d:3},
    {beat:1.25, type:'hihat',     vel:0.28,d:3},{beat:1.75, type:'hihat', vel:0.24,d:3},
    {beat:2.25, type:'hihat',     vel:0.28,d:3},{beat:2.75, type:'hihat', vel:0.24,d:3},
    {beat:3.25, type:'hihat',     vel:0.28,d:3},{beat:3.75, type:'hihat', vel:0.22,d:3},
  ]},

  'Rock Heavy': { label:'Rock Heavy', cat:'Rock', events: [
    {beat:0,    type:'crash',     vel:0.78,d:2},
    {beat:0,    type:'kick',      vel:0.92,d:1},{beat:0.5,  type:'kick',  vel:0.52,d:3},
    {beat:1.75, type:'kick',      vel:0.68,d:2},{beat:2,    type:'kick',  vel:0.88,d:1},
    {beat:2.75, type:'kick',      vel:0.62,d:2},{beat:3.5,  type:'kick',  vel:0.58,d:2},
    {beat:1,    type:'snare',     vel:0.88,d:1},{beat:3,    type:'snare', vel:0.88,d:1},
    {beat:0.5,  type:'ghost',     vel:0.30,d:2},{beat:1.5,  type:'ghost', vel:0.28,d:2},
    {beat:2.5,  type:'ghost',     vel:0.28,d:2},{beat:3.25, type:'ghost', vel:0.25,d:3},
    {beat:3.75, type:'tom_hi',    vel:0.70,d:2},{beat:3.875,type:'tom_mid',vel:0.62,d:3},
    {beat:0,    type:'hihat',     vel:0.55,d:1},{beat:0.5,  type:'hihat', vel:0.45,d:1},
    {beat:1,    type:'hihat',     vel:0.52,d:1},{beat:1.5,  type:'hihat', vel:0.42,d:1},
    {beat:2,    type:'hihat',     vel:0.52,d:1},{beat:2.5,  type:'hihat_open',vel:0.60,d:2},
    {beat:3,    type:'hihat',     vel:0.50,d:1},{beat:3.5,  type:'hihat', vel:0.40,d:1},
  ]},

  'Half-Time': { label:'Half-Time', cat:'Half-Time', events: [
    {beat:0,   type:'kick',  vel:0.88,d:1},{beat:1,   type:'kick',  vel:0.62,d:2},
    {beat:2.5, type:'kick',  vel:0.72,d:1},
    {beat:2,   type:'snare', vel:0.85,d:1}, // snare on beat 3 only = half-time feel
    {beat:0,   type:'hihat', vel:0.50,d:1},{beat:0.5, type:'hihat', vel:0.36,d:1},
    {beat:1,   type:'hihat', vel:0.44,d:1},{beat:1.5, type:'hihat', vel:0.34,d:1},
    {beat:2,   type:'hihat', vel:0.46,d:1},{beat:2.5, type:'hihat', vel:0.36,d:1},
    {beat:3,   type:'hihat', vel:0.44,d:1},{beat:3.5, type:'hihat', vel:0.33,d:1},
  ]},

  'Half-Time Heavy': { label:'Half-Time Heavy', cat:'Half-Time', events: [
    {beat:0,    type:'kick',      vel:0.90,d:1},{beat:1.5,  type:'kick',  vel:0.70,d:2},
    {beat:2.5,  type:'kick',      vel:0.80,d:1},{beat:3,    type:'kick',  vel:0.58,d:2},
    {beat:2,    type:'snare',     vel:0.88,d:1},
    {beat:1,    type:'ghost',     vel:0.28,d:2},{beat:3.5,  type:'ghost', vel:0.25,d:2},
    {beat:2.75, type:'tom_hi',    vel:0.68,d:3},{beat:3,    type:'tom_mid',vel:0.62,d:3},
    {beat:3.5,  type:'hihat_open',vel:0.55,d:2},
    {beat:0,    type:'hihat',     vel:0.50,d:1},{beat:0.5,  type:'hihat', vel:0.36,d:1},
    {beat:1,    type:'hihat',     vel:0.44,d:1},{beat:1.5,  type:'hihat', vel:0.34,d:1},
    {beat:2,    type:'hihat',     vel:0.46,d:1},{beat:2.5,  type:'hihat', vel:0.36,d:1},
    {beat:3,    type:'hihat',     vel:0.42,d:1},{beat:3.5,  type:'hihat', vel:0.32,d:1},
  ]},

  'Funk Light': { label:'Funk Light', cat:'Funk', events: [
    {beat:0,    type:'kick',  vel:0.85,d:1},{beat:0.75, type:'kick',  vel:0.58,d:2},
    {beat:2.5,  type:'kick',  vel:0.78,d:1},
    {beat:1,    type:'snare', vel:0.82,d:1},{beat:3,    type:'snare', vel:0.82,d:1},
    {beat:0.5,  type:'ghost', vel:0.22,d:2},{beat:1.75, type:'ghost', vel:0.20,d:2},
    {beat:2.25, type:'ghost', vel:0.22,d:3},{beat:3.25, type:'ghost', vel:0.20,d:2},
    {beat:0,    type:'hihat', vel:0.52,d:1},{beat:0.25, type:'hihat', vel:0.26,d:2},
    {beat:0.5,  type:'hihat', vel:0.45,d:1},{beat:0.75, type:'hihat', vel:0.24,d:2},
    {beat:1,    type:'hihat', vel:0.50,d:1},{beat:1.25, type:'hihat', vel:0.26,d:2},
    {beat:1.5,  type:'hihat', vel:0.44,d:1},{beat:1.75, type:'hihat', vel:0.22,d:2},
    {beat:2,    type:'hihat', vel:0.50,d:1},{beat:2.25, type:'hihat', vel:0.26,d:2},
    {beat:2.5,  type:'hihat', vel:0.44,d:1},{beat:2.75, type:'hihat', vel:0.22,d:2},
    {beat:3,    type:'hihat', vel:0.50,d:1},{beat:3.25, type:'hihat', vel:0.26,d:2},
    {beat:3.5,  type:'hihat', vel:0.44,d:1},{beat:3.75, type:'hihat', vel:0.20,d:2},
  ]},

  'Funk Heavy': { label:'Funk Heavy', cat:'Funk', events: [
    {beat:0,    type:'kick',  vel:0.90,d:1},{beat:0.5,  type:'kick',  vel:0.52,d:3},
    {beat:0.75, type:'kick',  vel:0.68,d:2},{beat:2,    type:'kick',  vel:0.86,d:1},
    {beat:2.5,  type:'kick',  vel:0.72,d:1},{beat:3.5,  type:'kick',  vel:0.52,d:2},
    {beat:1,    type:'snare', vel:0.88,d:1},{beat:2.25, type:'snare', vel:0.62,d:2},
    {beat:3,    type:'snare', vel:0.88,d:1},
    {beat:0.25, type:'ghost', vel:0.20,d:3},{beat:1.5,  type:'ghost', vel:0.25,d:2},
    {beat:1.75, type:'ghost', vel:0.20,d:3},{beat:2.75, type:'ghost', vel:0.22,d:2},
    {beat:1,    type:'clap',  vel:0.52,d:2},{beat:3,    type:'clap',  vel:0.52,d:2},
    {beat:0,    type:'hihat', vel:0.55,d:1},{beat:0.25, type:'hihat', vel:0.28,d:2},
    {beat:0.5,  type:'hihat', vel:0.48,d:1},{beat:0.75, type:'hihat', vel:0.24,d:2},
    {beat:1,    type:'hihat', vel:0.52,d:1},{beat:1.25, type:'hihat', vel:0.28,d:2},
    {beat:1.5,  type:'hihat', vel:0.44,d:1},{beat:1.75, type:'hihat', vel:0.22,d:2},
    {beat:2,    type:'hihat', vel:0.52,d:1},{beat:2.25, type:'hihat', vel:0.28,d:2},
    {beat:2.5,  type:'hihat', vel:0.44,d:1},{beat:2.75, type:'hihat', vel:0.22,d:2},
    {beat:3,    type:'hihat', vel:0.50,d:1},{beat:3.25, type:'hihat', vel:0.28,d:2},
    {beat:3.5,  type:'hihat', vel:0.44,d:1},{beat:3.75, type:'hihat', vel:0.20,d:2},
  ]},

  'Jazz Swing': { label:'Jazz Swing', cat:'Jazz', events: [
    // Triplet ride: quarter on each beat + 8th-triplet on the "and"
    {beat:0,    type:'ride',  vel:0.55,d:1},{beat:0.667,type:'ride', vel:0.38,d:1},
    {beat:1,    type:'ride',  vel:0.50,d:1},{beat:1.667,type:'ride', vel:0.38,d:1},
    {beat:2,    type:'ride',  vel:0.52,d:1},{beat:2.667,type:'ride', vel:0.38,d:1},
    {beat:3,    type:'ride',  vel:0.50,d:1},{beat:3.667,type:'ride', vel:0.38,d:1},
    // Hi-hat foot pedal on 2 and 4
    {beat:1,    type:'hihat', vel:0.32,d:1},{beat:3,    type:'hihat', vel:0.32,d:1},
    // Sparse kick and brushed snare
    {beat:0,    type:'kick',  vel:0.52,d:2},{beat:2,    type:'kick',  vel:0.42,d:3},
    {beat:1,    type:'snare', vel:0.42,d:2},{beat:3,    type:'snare', vel:0.42,d:2},
  ]},

  'Bossa Nova': { label:'Bossa Nova', cat:'Latin', events: [
    {beat:0,    type:'kick',  vel:0.72,d:1},{beat:2,    type:'kick',  vel:0.65,d:1},
    {beat:2.5,  type:'kick',  vel:0.52,d:2},
    // Rim click (snare) on off-beats -- the clave feel
    {beat:0.5,  type:'snare', vel:0.42,d:1},{beat:1.5,  type:'snare', vel:0.38,d:1},
    {beat:2.5,  type:'snare', vel:0.40,d:2},{beat:3,    type:'snare', vel:0.35,d:1},
    // Hi-hat avoids downbeats
    {beat:0.5,  type:'hihat', vel:0.38,d:1},{beat:1,    type:'hihat', vel:0.32,d:1},
    {beat:1.5,  type:'hihat', vel:0.40,d:1},{beat:2.5,  type:'hihat', vel:0.38,d:1},
    {beat:3,    type:'hihat', vel:0.32,d:2},{beat:3.5,  type:'hihat', vel:0.36,d:1},
    {beat:0.75, type:'hihat', vel:0.25,d:3},{beat:1.75, type:'hihat', vel:0.22,d:3},
    {beat:2.75, type:'hihat', vel:0.25,d:3},{beat:3.75, type:'hihat', vel:0.22,d:3},
  ]},

  'Double Time': { label:'Double Time', cat:'Groove', events: [
    // d:1 sparse — skeleton + quarter hi-hats (clear breathing room)
    {beat:0,    type:'kick',  vel:0.88,d:1},{beat:2,    type:'kick',  vel:0.82,d:1},
    {beat:1,    type:'snare', vel:0.85,d:1},{beat:3,    type:'snare', vel:0.85,d:1},
    {beat:0,    type:'hihat', vel:0.55,d:1},{beat:1,    type:'hihat', vel:0.50,d:1},
    {beat:2,    type:'hihat', vel:0.52,d:1},{beat:3,    type:'hihat', vel:0.48,d:1},
    // d:2 moderate — 8th hi-hats fill gaps + syncopated kicks + ghost snares
    {beat:0.5,  type:'hihat', vel:0.42,d:2},{beat:1.5,  type:'hihat', vel:0.38,d:2},
    {beat:2.5,  type:'hihat', vel:0.42,d:2},{beat:3.5,  type:'hihat', vel:0.36,d:2},
    {beat:1.75, type:'kick',  vel:0.58,d:2},{beat:3.5,  type:'kick',  vel:0.52,d:2},
    {beat:0.5,  type:'ghost', vel:0.22,d:2},{beat:2.5,  type:'ghost', vel:0.20,d:2},
    // d:3 dense — fill remaining 16th positions; every slot covered, no pauses
    {beat:0.25, type:'hihat', vel:0.33,d:3},{beat:0.75, type:'hihat', vel:0.28,d:3},
    {beat:1.25, type:'hihat', vel:0.33,d:3},{beat:1.75, type:'hihat', vel:0.27,d:3},
    {beat:2.25, type:'hihat', vel:0.33,d:3},{beat:2.75, type:'hihat', vel:0.27,d:3},
    {beat:3.25, type:'hihat', vel:0.33,d:3},{beat:3.75, type:'hihat', vel:0.25,d:3},
  ]},

  'Reggae': { label:'Reggae', cat:'Groove', events: [
    // Rockers pattern: kick on all 4 beats, snare on 3
    {beat:0,   type:'kick',  vel:0.82,d:1},{beat:1,   type:'kick',  vel:0.75,d:1},
    {beat:2,   type:'kick',  vel:0.85,d:1},{beat:3,   type:'kick',  vel:0.72,d:1},
    {beat:2,   type:'snare', vel:0.78,d:1},
    {beat:0,   type:'hihat', vel:0.42,d:1},{beat:0.5, type:'hihat', vel:0.35,d:1},
    {beat:1,   type:'hihat', vel:0.40,d:1},{beat:1.5, type:'hihat', vel:0.32,d:1},
    {beat:2,   type:'hihat', vel:0.40,d:1},{beat:2.5, type:'hihat', vel:0.35,d:1},
    {beat:3,   type:'hihat', vel:0.38,d:1},{beat:3.5, type:'hihat', vel:0.30,d:1},
    {beat:1,   type:'clap',  vel:0.50,d:2},{beat:3,   type:'clap',  vel:0.50,d:2},
  ]},

  'Acoustic': { label:'Acoustic', cat:'Soft', events: [
    {beat:0,   type:'kick',  vel:0.60,d:1},{beat:2,   type:'kick',  vel:0.45,d:2},
    {beat:1,   type:'snare', vel:0.48,d:1},{beat:3,   type:'snare', vel:0.48,d:1},
    {beat:0,   type:'hihat', vel:0.28,d:1},{beat:0.5, type:'hihat', vel:0.16,d:2},
    {beat:1,   type:'hihat', vel:0.24,d:1},{beat:1.5, type:'hihat', vel:0.14,d:2},
    {beat:2,   type:'hihat', vel:0.24,d:1},{beat:2.5, type:'hihat', vel:0.14,d:2},
    {beat:3,   type:'hihat', vel:0.22,d:1},{beat:3.5, type:'hihat', vel:0.12,d:2},
  ]},

  'Disco Pop': { label:'Disco Pop', cat:'Pop', events: [
    // Four-on-the-floor kick, clap on 2+4, open hi-hat on 8th off-beats
    {beat:0,    type:'kick',      vel:0.90,d:1},{beat:1,    type:'kick',  vel:0.85,d:1},
    {beat:2,    type:'kick',      vel:0.92,d:1},{beat:3,    type:'kick',  vel:0.88,d:1},
    {beat:1,    type:'clap',      vel:0.80,d:1},{beat:3,    type:'clap',  vel:0.82,d:1},
    {beat:1,    type:'snare',     vel:0.55,d:2},{beat:3,    type:'snare', vel:0.55,d:2},
    {beat:0.5,  type:'hihat_open',vel:0.50,d:1},{beat:1.5,  type:'hihat_open',vel:0.46,d:1},
    {beat:2.5,  type:'hihat_open',vel:0.50,d:1},{beat:3.5,  type:'hihat_open',vel:0.46,d:1},
    {beat:0,    type:'hihat',     vel:0.40,d:2},{beat:1,    type:'hihat', vel:0.38,d:2},
    {beat:2,    type:'hihat',     vel:0.40,d:2},{beat:3,    type:'hihat', vel:0.38,d:2},
  ]},

  'Brushed Trio': { label:'Brushed Trio', cat:'Jazz', events: [
    // Ride on triplet grid, brushed snare on 2+4, very light kick
    {beat:0,     type:'ride',  vel:0.48,d:1},{beat:0.667,type:'ride', vel:0.32,d:1},
    {beat:1,     type:'ride',  vel:0.44,d:1},{beat:1.667,type:'ride', vel:0.32,d:1},
    {beat:2,     type:'ride',  vel:0.46,d:1},{beat:2.667,type:'ride', vel:0.32,d:1},
    {beat:3,     type:'ride',  vel:0.44,d:1},{beat:3.667,type:'ride', vel:0.32,d:1},
    {beat:1,     type:'hihat', vel:0.28,d:1},{beat:3,    type:'hihat',vel:0.28,d:1},
    {beat:0,     type:'kick',  vel:0.38,d:2},{beat:2,    type:'kick', vel:0.32,d:3},
    {beat:1,     type:'snare', vel:0.35,d:1},{beat:3,    type:'snare',vel:0.35,d:1},
  ]},

  'Jazz Shell': { label:'Jazz Shell', cat:'Jazz', events: [
    {beat:0,     type:'ride',  vel:0.52,d:1},{beat:0.667,type:'ride', vel:0.35,d:1},
    {beat:1,     type:'ride',  vel:0.48,d:1},{beat:1.667,type:'ride', vel:0.35,d:1},
    {beat:2,     type:'ride',  vel:0.50,d:1},{beat:2.667,type:'ride', vel:0.35,d:1},
    {beat:3,     type:'ride',  vel:0.48,d:1},{beat:3.667,type:'ride', vel:0.35,d:1},
    {beat:1,     type:'hihat', vel:0.30,d:1},{beat:3,    type:'hihat',vel:0.30,d:1},
    {beat:0,     type:'kick',  vel:0.45,d:2},
    {beat:1,     type:'ghost', vel:0.18,d:2},{beat:2.5,  type:'ghost',vel:0.16,d:3},
    {beat:3,     type:'snare', vel:0.38,d:1},
  ]},

  }; // end _DRUM_LIBRARY

  // -- Passing chord builder ----------------------------------------------------

  function _buildPassingChord(destChord) {
    var root0 = destChord.notes[0];
    var destPc = _NOTE_PC[root0];
    if (destPc === undefined) return null;

    var rootPc, ivs, suffix;
    if (_passType === 'sec_dom') {
      // V7 of the destination: a fifth above its root (= dom7 chord)
      rootPc = (destPc + 7) % 12;
      ivs    = [0, 4, 7, 10];
      suffix = '7';
    } else if (_passType === 'dim') {
      // Diminished 7th a half-step below the destination root
      rootPc = ((destPc - 1) + 12) % 12;
      ivs    = [0, 3, 6, 9];
      suffix = '°7';
    } else {
      // Chromatic: same voicing a half-step below
      rootPc = ((destPc - 1) + 12) % 12;
      // Reconstruct quality intervals from destination notes
      var destPcs = destChord.notes.map(function(n){ return _NOTE_PC[n] || 0; });
      ivs = destPcs.map(function(p){ return ((p - destPc) + 12) % 12; });
      suffix = destChord.symbol.substring(root0.length);
    }

    var rootName = _PC_NOTE[rootPc];
    var notes    = ivs.map(function(iv){ return _PC_NOTE[(rootPc + iv) % 12]; });
    return { symbol: rootName + suffix, label: '→ ' + rootName + suffix, notes: notes, isPassing: true };
  }

  // -- Bar scheduler -----------------------------------------------------------

  function _accompScheduleBar(startTime) {
    var st    = _accompState;
    var chord = (_pendingIdx >= 0 && _passingChord) ? _passingChord : st.chords[st.idx];
    if (!chord) return;
    var notes = chord.notes, root = notes[0];
    var barDur = st.bpb * st.beatDur;
    var bd = _density.bass, cd = _density.chord, dd = _density.drum;
    var sustained = _sustainedStyle();
    _updateMixChordLabel();

    // -- Shared clock drift: all parts follow this together ----------------------
    // At Feel=0, drift is hard zero -- everything perfectly on the grid.
    // At Feel>0, drift wanders slowly (Ornstein-Uhlenbeck) and snaps back toward
    // zero at chord boundaries ("appear to make mistakes, then correct yourself").
    if (_humanize > 0) {
      _driftTarget += (Math.random() - 0.5) * _humanize * 0.024;
      var maxDrift = _humanize * 0.052;
      _driftTarget = Math.max(-maxDrift, Math.min(maxDrift, _driftTarget));
      _clockDrift += (_driftTarget - _clockDrift) * 0.38; // smooth approach to target
    } else {
      _clockDrift = 0; _driftTarget = 0;
    }
    // Per-instrument character on top of shared drift (computed once per bar so
    // all chord events anticipate together, all bass events sit back together)
    var bassChar  = _humanize * Math.random() * 0.013;           // pocket: bass slightly late
    var chordAnti = _humanize * (0.006 + Math.random() * 0.013); // anticipation: chords slightly early

    // Pre-compute voiced MIDI notes with style-aware extensions
    var voicedMidi = _voiceNotesToMidi(notes, st.style);

    // Swing amount: 1.0 = full triplet swing, 0 = straight
    var swingAmt = 0;
    if (st.style === 'Jazz' || st.style === 'Rhodes Jazz' || st.style === 'Brushed Trio' ||
        st.style === 'Jazz Shell' || st.style === 'Blues' || st.style === 'Swing' ||
        st.style === 'Smooth Jazz') swingAmt = 0.88;
    else if (st.style === 'Lo-Fi' || st.style === 'Neo Soul' || st.style === 'R&B') swingAmt = 0.40;
    else if (st.style === 'Gospel' || st.style === 'Soul' || st.style === 'Motown') swingAmt = 0.28;

    var nextChord = st.chords[(st.idx + 1) % st.chords.length];
    var walkingBass = (st.style === 'Rhodes Jazz' || st.style === 'Jazz' || st.style === 'Vibraphone' ||
                       st.style === 'Brushed Trio' || st.style === 'Jazz Shell');

    if (st.style === 'Arpeggio Up' || st.style === 'Arpeggio Down') {
      if (cd > 0) {
        var seq     = st.style === 'Arpeggio Down' ? notes.slice().reverse() : notes;
        var seqMidi = st.style === 'Arpeggio Down' ? voicedMidi.slice().reverse() : voicedMidi;
        var step = st.beatDur / seq.length;
        for (var b = 0; b < st.bpb; b++) {
          seq.forEach(function(n, ni) {
            var jitter = _clockDrift + (Math.random() - 0.5) * _humanize * 0.009;
            _playChordInst(n, startTime + b * st.beatDur + ni * step + jitter,
                           step * 0.85, 0.5 + Math.random() * (0.2 + _humanize * 0.15), seqMidi[ni]);
          });
        }
      }
      if (bd > 0) _playBassNote(root, startTime + _clockDrift + bassChar, barDur * 0.9);
    } else {
      var pattern = (_PATTERNS[st.style] || _PATTERNS['Ballad'])(st.bpb);
      var dropIdx = (cd >= 2 && Math.random() < 0.22 && notes.length > 2)
                    ? 1 + Math.floor(Math.random() * (notes.length - 2)) : -1;

      pattern.forEach(function(evt) {
        var swungBeat = _swingBeat(evt.beat, swingAmt);
        var t = startTime + swungBeat * st.beatDur
                + _clockDrift
                + (evt.type === 'bass'  ?  bassChar  : 0)
                + (evt.type === 'chord' ? -chordAnti : 0);
        var dur = sustained
          ? barDur * (st.barsPerChord > 1 ? 1.6 : 0.95)
          : (st.style === 'Pad' ? barDur * st.barsPerChord : st.beatDur * 0.88);
        var v   = (evt.vel || 0.6) * _dynamicLevel * (0.88 + Math.random() * (0.24 + _humanize * 0.18));

        if (evt.type === 'bass' && bd >= evt.d && evt.beat === 0 && walkingBass) {
          _playWalkingBass(chord, nextChord, startTime + _clockDrift + bassChar, st.beatDur, st.bpb);
        } else if (evt.type === 'bass' && bd >= evt.d && !walkingBass) {
          _playBassNote(root, t, Math.min(dur, st.beatDur * 1.9));
        }
        if (evt.type === 'chord' && cd >= evt.d && _melodyVol > 0.01 && evt.beat % 2 === 0) {
          // Melody: play top voiced note (melody voice) through separate lead instrument
          var topMidi = voicedMidi[voicedMidi.length - 1] + 12; // up an octave for lead
          var prevChordGain = _chordGain;
          _chordGain = _melodyGain;
          var melInst = _melodyInst || 'piano2';
          if (melInst === 'piano2') _playPianoNote2(topMidi, t, Math.min(dur, st.beatDur * 1.5), v * 0.9);
          else if (melInst === 'rhodes') _playRhodesNote(topMidi, t, Math.min(dur, st.beatDur * 1.5), v * 0.9);
          else if (melInst === 'vibes') _playVibesNote(topMidi, t, Math.min(dur, st.beatDur * 1.8), v * 0.85);
          else if (melInst === 'guitar_clean') _playCleanGuitar(topMidi, t, Math.min(dur, st.beatDur * 1.2), v * 0.9);
          else if (melInst === 'strings_ens') _playStringsEnsemble(topMidi, t, Math.min(dur, st.beatDur * 2.0), v * 0.8);
          else if (melInst === 'organ') _playOrganNote(topMidi, t, Math.min(dur, st.beatDur * 1.5), v * 0.85);
          _chordGain = prevChordGain;
        }
        if (evt.type === 'chord' && cd >= evt.d) {
          // Style-aware strum width: acoustic/funk strums wide, piano rolls tight
          var strumWidth = st.style === 'Acoustic'   ? 0.055 :
                           st.style === 'Funk Chop'  ? 0.012 :
                           st.style === 'Disco Pop'  ? 0.018 :
                           (st.style === 'Jazz' || st.style === 'Rhodes Jazz' ||
                            st.style === 'Jazz Shell' || st.style === 'Brushed Trio') ? 0.022 :
                           st.style === 'Ballad'     ? 0.035 :
                           0.028; // default piano/rhodes roll
          var nNotes = notes.length;
          notes.forEach(function(n, ni) {
            if (ni === dropIdx) return;
            // Strum: each note offset from bottom to top (adds physical realism)
            var strumOffset = ni * (strumWidth / Math.max(nNotes - 1, 1))
                              + (Math.random() - 0.5) * 0.004;
            // Voice velocity: top note (melody) loudest, inner voices sit back
            var voiceVelScale = (ni === nNotes - 1) ? 1.08   // top = melody, slightly forward
                              : (ni === 0)           ? 0.88   // bottom inner voice
                              : 0.78;                         // middle inner voices recede
            // Duration jitter: inner voices release slightly early (pianist lifting fingers)
            var durScale = 1.0 - (nNotes - 1 - ni) * 0.04 + (Math.random() - 0.5) * 0.08;
            _playChordInst(n, t + strumOffset,
                           Math.min(dur * Math.max(0.7, durScale), barDur * 1.8),
                           v * voiceVelScale * (0.90 + Math.random() * 0.14),
                           voicedMidi[ni]);
          });
        }
      });
    }

    // -- Dynamic level: drifts slowly like a band breathing in and out ----------
    _barCount++;
    _dynamicLevel += (Math.random() - 0.48) * 0.09; // band breathes bar to bar
    _dynamicLevel = Math.max(0.62, Math.min(1.0, _dynamicLevel));
    // Pull back every ~7-9 bars for 1-2 bars, then swell back
    if (_sparseBarsLeft > 0) {
      _dynamicLevel = Math.max(_dynamicLevel - 0.1, 0.55);
      _sparseBarsLeft--;
    } else if (_barCount % 8 === 7 && Math.random() < 0.40) {
      _sparseBarsLeft = 1 + Math.floor(Math.random() * 2); // 1 or 2 bars of pullback
    }
    // Snap back toward full after a pullback
    if (_sparseBarsLeft === 0 && _dynamicLevel < 0.85) _dynamicLevel += 0.08;

    var isFillBar  = (dd > 0 && _barCount % 4 === 0 && !chord.isPassing);
    var isOpenBar  = (_sparseBarsLeft > 0); // "everyone takes it down"
    var isCrashBar = (_barCount % 8 === 1 && _barCount > 1); // crash on 1 of a new 8-bar phrase

    if (dd > 0) {
      var drumPat = (_DRUM_LIBRARY[_activeDrumPattern] || _DRUM_LIBRARY['Rock Basic']).events;
      drumPat.forEach(function(evt) {
        if (dd < evt.d || evt.beat >= st.bpb) return;
        // Open bars: drums pull way back (only beat 1 kick + sparse hat)
        if (isOpenBar && evt.type !== 'kick' && evt.beat !== 0) return;
        var drumT = startTime + _swingBeat(evt.beat, swingAmt) * st.beatDur
                    + _clockDrift + (Math.random() - 0.5) * _humanize * 0.012;
        // Occasional hi-hat type swap on the off-beat (human drummer variation)
        var etype = evt.type;
        if (etype === 'hihat' && evt.beat % 1 === 0.5 && Math.random() < 0.12) etype = 'hihat_open';
        // Occasionally skip a ghost note (space = taste)
        if (etype === 'ghost' && Math.random() < 0.30) return;
        var velMult = _dynamicLevel * (0.85 + Math.random() * 0.28);
        _playDrumEvent(etype, drumT, evt.vel * velMult);
      });
      // Fill at end of every 4-bar phrase
      if (isFillBar) _playDrumFill(startTime, st.beatDur, st.bpb, _activeDrumPattern);
      // Crash accent at top of new 8-bar phrase
      if (isCrashBar) _playDrumEvent('crash', startTime + _clockDrift, 0.65 * _dynamicLevel);
    }

    // -- Occasional bass fill: chromatic run toward next chord (every ~6 bars) --
    var nextCh = nextChord;
    if (bd > 0 && !walkingBass && isFillBar && nextCh && Math.random() < 0.55) {
      var bRoot = (chord.notes || [])[0], nRoot = (nextCh.notes || [])[0];
      if (bRoot && nRoot && bRoot !== nRoot) {
        _playBassFill(bRoot, nRoot, startTime, st.beatDur, st.bpb);
      }
    }

    // -- Chord voicing rotation -- every ~3 bars, rotate inversion for variety ---
    // (already scheduled above; this is a no-op placeholder for future extension)

    var disp = document.getElementById('accomp-now-playing');
    if (disp) {
      if (chord.isPassing) {
        var dest = st.chords[_pendingIdx];
        disp.textContent = chord.symbol + ' (passing → ' + (dest ? dest.symbol : '?') + ')';
      } else {
        disp.textContent = chord.symbol + '  (' + (chord.label.split('·')[0]||'').trim() + ')';
      }
    }

  }

  // ── Phrase capture ─────────────────────────────────────────────────────────

  function _melStatus(msg) { var el=document.getElementById('mel-status'); if(el) el.textContent=msg; }

  function _updateArmBtn(active) {
    var btn = document.getElementById('mel-arm-btn');
    if (!btn) return;
    btn.style.background = active ? '#2a7040' : '#1a3050';
    btn.textContent = active ? '📡 Listening…' : '📡 Arm';
  }

  function _captureNoteOn(midi) {
    if (_phraseArmed) {
      _capturedPhrase = []; _phraseArmed = false; _phraseCapturing = true;
      _updateArmBtn(false);
    }
    clearTimeout(_phraseGapTimer);
    var now = _audioCtx ? _audioCtx.currentTime : 0;
    _capturedPhrase.push({midi:midi, t:now, dur:0});
  }

  function _captureNoteOff(midi) {
    if (!_phraseCapturing) return;
    var now = _audioCtx ? _audioCtx.currentTime : 0;
    for (var i = _capturedPhrase.length-1; i >= 0; i--) {
      if (_capturedPhrase[i].midi === midi && _capturedPhrase[i].dur === 0) {
        _capturedPhrase[i].dur = Math.max(0.05, now - _capturedPhrase[i].t);
        break;
      }
    }
    if (_capturedPhrase.every(function(n){return n.dur > 0;}))
      _phraseGapTimer = setTimeout(_phraseComplete, PHRASE_GAP_SEC * 1000);
  }

  function _phraseComplete() {
    if (!_capturedPhrase.length) { _phraseCapturing = false; return; }
    var now = _audioCtx ? _audioCtx.currentTime : 0;
    _capturedPhrase.forEach(function(n){ if(n.dur===0) n.dur = Math.max(0.05, now-n.t); });
    var t0 = _capturedPhrase[0].t;
    _capturedPhrase.forEach(function(n){ n.t -= t0; });
    _capturedPhrase = _capturedPhrase.filter(function(n){ return n.dur >= 0.06; });
    _phraseCapturing = false; _micLastMidi = -1;
    _updateArmBtn(false);
    _melStatus('Captured ' + _capturedPhrase.length + ' note(s) — click ▶ Generate');
    var autoEl = document.getElementById('mel-auto-respond');
    if (autoEl && autoEl.checked) setTimeout(function(){ window.generateMelodicResponse&&window.generateMelodicResponse(); }, 200);
  }

  // ── Melodic response generators ────────────────────────────────────────────

  // Snap captured phrase to 8th-note grid; collapse duplicate onsets
  function _quantizePhrase(phrase, beatDur) {
    if (!phrase.length) return phrase;
    var grid = beatDur / 2; // 8th note
    var sorted = phrase.slice().sort(function(a,b){return a.t-b.t;});
    var result = [];
    sorted.forEach(function(n) {
      var tQ   = Math.round(n.t / grid) * grid;
      var durQ = Math.max(grid * 0.5, Math.round(n.dur / grid) * grid);
      if (result.length && Math.abs(tQ - result[result.length-1].t) < grid * 0.1) return; // deduplicate
      result.push({midi:n.midi, t:tQ, dur:durQ});
    });
    return result;
  }

  function _freePhrase(scale, anchor, targetBeats, beatDur) {
    var sorted = scale.slice().sort(function(a,b){return a-b;});
    var result = [], current = _snapToScale(anchor, sorted);
    var totalSec = targetBeats * beatDur, beat = 0, half = totalSec / 2;
    var ascending = true;
    var lo = _chordRangeLo + 7, hi = _chordRangeHi;
    while (beat < totalSec - 0.01) {
      var r = Math.random();
      var noteDur = r<0.40 ? beatDur : r<0.70 ? beatDur/2 : r<0.85 ? beatDur*1.5 : beatDur*2;
      noteDur = Math.min(noteDur, totalSec - beat);
      if (beat >= half) ascending = false;
      // Walk with direction bias; near end converge to anchor
      var steps;
      if (totalSec - beat < beatDur * 2) {
        var dist = _scaleDegDist(current, anchor, sorted);
        steps = dist > 0 ? 1 : dist < 0 ? -1 : 0;
      } else {
        var mr = Math.random();
        if (mr < 0.55) steps = ascending ? 1 : -1;
        else if (mr < 0.75) steps = 0;
        else if (mr < 0.88) steps = ascending ? 2 : -2;
        else steps = ascending ? -1 : 1;
      }
      current = _scaleStep(current, sorted, steps);
      while (current < lo) current += 12;
      while (current > hi) current -= 12;
      result.push({midi:current, beat:beat, dur:noteDur * 0.88});
      beat += noteDur;
    }
    if (result.length) result[result.length-1].midi = _snapToScale(anchor, sorted);
    return result;
  }

  function _imitatePhrase(input, scale, anchor, targetBeats, beatDur) {
    var sorted = scale.slice().sort(function(a,b){return a-b;});
    var shift = 2 + Math.floor(Math.random() * 3);
    var current = _scaleStep(_snapToScale(anchor, sorted), sorted, shift);
    var lo = _chordRangeLo + 7, hi = _chordRangeHi;
    while (current < lo) current += 12; while (current > hi) current -= 12;
    var inputSpan = (input[input.length-1].t + input[input.length-1].dur) || 1;
    var stretch = (targetBeats * beatDur) / inputSpan;
    var result = [];
    // Use input onset times directly to preserve rhythm
    for (var i = 0; i < input.length; i++) {
      var beat = input[i].t * stretch;
      var dur  = input[i].dur * stretch;
      result.push({midi:current, beat:beat, dur:dur * 0.88});
      if (i + 1 < input.length) {
        var steps = _scaleDegDist(input[i].midi, input[i+1].midi, sorted);
        steps = Math.max(-4, Math.min(4, steps));
        current = _scaleStep(current, sorted, steps);
        while (current < lo) current += 12; while (current > hi) current -= 12;
      }
    }
    return result;
  }

  function _answerPhrase(input, scale, anchor, targetBeats, beatDur) {
    var sorted = scale.slice().sort(function(a,b){return a-b;});
    var current = _snapToScale(input[input.length-1].midi, sorted);
    var lo = _chordRangeLo + 7, hi = _chordRangeHi;
    while (current < lo) current += 12; while (current > hi) current -= 12;
    var inputSpan = (input[input.length-1].t + input[input.length-1].dur) || 1;
    var stretch = (targetBeats * beatDur) / inputSpan;
    var result = [];
    for (var i = 0; i < input.length; i++) {
      var beat = input[i].t * stretch;  // onset time preserved
      var dur  = input[i].dur * stretch;
      result.push({midi:current, beat:beat, dur:dur * 0.88});
      var stepsToTonic = _scaleDegDist(current, anchor, sorted);
      var move;
      if (i >= input.length - 2) {
        move = stepsToTonic > 0 ? Math.min(stepsToTonic,1) : stepsToTonic < 0 ? Math.max(stepsToTonic,-1) : 0;
      } else {
        var dir = stepsToTonic > 0 ? 1 : -1;
        move = Math.random() < 0.65 ? dir : (Math.random() < 0.5 ? dir*2 : 0);
      }
      current = _scaleStep(current, sorted, move);
      while (current < lo) current += 12; while (current > hi) current -= 12;
    }
    if (result.length) result[result.length-1].midi = _snapToScale(anchor, sorted);
    return result;
  }

  function _generateMelody(input, scalePCs, targetBeats, style) {
    if (!scalePCs.length) scalePCs = [0,2,4,5,7,9,11];
    var scale = scalePCs.slice().sort(function(a,b){return a-b;});
    var lp = _readLiveParams(), beatDur = lp.beatDur;
    var mid = _chordRangeLo + Math.floor((_chordRangeHi - _chordRangeLo) * 0.55);
    var anchor = _snapToScale(mid, scale);
    while (anchor < _chordRangeLo + 7) anchor += 12;
    while (anchor > _chordRangeHi)     anchor -= 12;
    // Always quantize captured input to beat grid before using it
    var q = input.length ? _quantizePhrase(input, beatDur) : [];
    if (q.length >= 2) {
      if (style === 'imitate') return _imitatePhrase(q, scale, anchor, targetBeats, beatDur);
      if (style === 'answer')  return _answerPhrase(q, scale, anchor, targetBeats, beatDur);
    }
    return _freePhrase(scale, anchor, targetBeats, beatDur);
  }

  // ── Markov chord transitions ────────────────────────────────────────────────

  function _chordFn(label) {
    var r = (label.split('·')[0]||'').trim().replace(/[♭♯°✶]/g,'').toUpperCase();
    if (['I','III','VI'].indexOf(r) >= 0) return 'tonic';
    if (['II','IV'].indexOf(r) >= 0) return 'subdom';
    if (['V','VII'].indexOf(r) >= 0) return 'dominant';
    return 'other';
  }

  function _nextChord() {
    var st = _accompState;
    if (st.chords.length <= 1) return 0;
    var fn   = _chordFn(st.chords[st.idx].label);
    var pool = st.chords.map(function(_,i){return i;}).filter(function(i){return i!==st.idx;});
    var w    = pool.map(function(i) {
      var cf = _chordFn(st.chords[i].label);
      if (fn==='dominant') return cf==='tonic' ? 4 : cf==='subdom' ? 1 : 1.5;
      if (fn==='subdom')   return cf==='dominant' ? 3 : cf==='tonic' ? 2 : 1;
      if (fn==='tonic')    return cf==='subdom' ? 2 : cf==='dominant' ? 2.5 : 1;
      return 1;
    });
    var total = w.reduce(function(a,b){return a+b;},0), r = Math.random()*total;
    for (var i=0; i<pool.length; i++) { r-=w[i]; if(r<=0) return pool[i]; }
    return pool[pool.length-1];
  }

  // Style → default drum pattern (only auto-selects if user hasn't manually changed drum)
  var _STYLE_DEFAULT_DRUM = {
    'Jazz':         'Jazz Swing', 'Rhodes Jazz':  'Jazz Swing',
    'Jazz Shell':   'Jazz Shell', 'Brushed Trio': 'Brushed Trio',
    'Vibraphone':   'Jazz Swing', 'Bossa Nova':   'Bossa Nova',
    'Blues':        'Funk Light', 'Funk Chop':    'Funk Heavy',
    'Disco Pop':    'Disco Pop',  'Acoustic':     'Acoustic',
    'Ballad':       'Ballad',     'Pad':          'Ballad',
    'Lo-Fi':        'Ballad',     'Waltz':        'Ballad',
    'Rhodes':       'Rock Basic', 'Honky Tonk':   'Rock Basic',
    'R&B':          'Funk Heavy', 'Neo Soul':     'Funk Light',
    'Gospel':       'Rock Groove','Soul':         'Funk Heavy',
    'Pop':          'Rock Basic', 'Country':      'Rock Basic',
    'Reggae':       'Ballad',     'Motown':       'Funk Heavy',
    'Latin':        'Bossa Nova', 'Singer-Songwriter': 'Acoustic',
    'Smooth Jazz':  'Jazz Swing', 'Cinematic':    'Ballad',
    'Worship':      'Ballad',     'Indie Pop':    'Rock Groove',
    'Funk':         'Funk Heavy', 'Tropical':     'Bossa Nova',
    'Brass':        'Funk Heavy', 'Samba':        'Bossa Nova',
    'Swing':        'Jazz Swing', 'New Soul':     'Ballad',
  };
  var _drumUserOverride = false;
  window._setDrumUserOverride = function(v) { _drumUserOverride = v; };

  function _readLiveParams() {
    var bpm = parseFloat(document.getElementById('ctrl-bpm') && document.getElementById('ctrl-bpm').value) || 120;
    var ts  = (document.getElementById('ctrl-timesig') && document.getElementById('ctrl-timesig').value) || '4/4';
    var sty = (document.getElementById('ctrl-style')   && document.getElementById('ctrl-style').value)   || 'Ballad';
    var bpc = parseFloat((document.getElementById('ctrl-bpc') && document.getElementById('ctrl-bpc').value) || 2);
    // Auto-select drum pattern based on style (unless user has explicitly picked one)
    if (!_drumUserOverride && _STYLE_DEFAULT_DRUM[sty]) {
      _activeDrumPattern = _STYLE_DEFAULT_DRUM[sty];
      var drumSel = document.getElementById('drum-pattern-sel');
      if (drumSel && drumSel.value !== _activeDrumPattern) drumSel.value = _activeDrumPattern;
    }
    return { bpm:bpm, bpb:parseInt(ts.split('/')[0])||4, beatDur:60/bpm, style:sty, barsPerChord:Math.max(0.5, bpc) };
  }

  // Re-read the live palette selection; called at every chord boundary so palette
  // changes (new root/mode, different checkboxes) take effect without stopping.
  function _refreshChordPool() {
    var fresh = _readAccompChords();
    if (!fresh.length) return; // nothing selected yet -- keep current pool
    var curSym = (_accompState.chords[_accompState.idx] || {}).symbol;
    _accompState.chords = fresh;
    // Stay on the same chord if it still exists in the new pool
    var found = -1;
    fresh.forEach(function(c, i) { if (c.symbol === curSym) found = i; });
    _accompState.idx = found >= 0 ? found : Math.floor(Math.random() * fresh.length);
  }

  function _accompTick() {
    if (!_accompRunning) return;
    var st = _accompState;
    // Apply live param changes at each bar boundary
    var lp = _readLiveParams();
    st.bpb  = lp.bpb;
    st.beatDur = lp.beatDur;
    st.style = lp.style;
    if (lp.barsPerChord !== st.barsPerChord) {
      st.barsPerChord = lp.barsPerChord;
      _accompBeatsLeft = Math.max(st.bpb, Math.min(_accompBeatsLeft, lp.barsPerChord * st.bpb));
    }
    while (st.nextBarTime < _audioCtx.currentTime + 0.25) {
      try { _accompScheduleBar(st.nextBarTime); } catch(e) { console.error('[accomp]', e); }
      st.nextBarTime += st.bpb * st.beatDur;
      _accompBeatsLeft -= st.bpb;
      if (_accompBeatsLeft <= 0) {
        _driftTarget *= 0.18; // drift correction at every chord boundary
        if (_pendingIdx >= 0) {
          // Passing chord just finished -- land on the destination
          st.idx      = _pendingIdx;
          _pendingIdx = -1;
          _passingChord = null;
          _accompBeatsLeft = lp.barsPerChord * lp.bpb + _accompBeatsLeft;
        } else {
          _refreshChordPool(); // absorb live palette changes
          var nextIdx = _nextChord();
          // Roll dice for passing chord (skip if pool has only 1 chord)
          if (_passProb > 0 && Math.random() < _passProb && st.chords.length > 1) {
            var pc = _buildPassingChord(st.chords[nextIdx]);
            if (pc) {
              _pendingIdx   = nextIdx;
              _passingChord = pc;
              _accompBeatsLeft = lp.bpb + _accompBeatsLeft; // passing chord = 1 bar
            } else {
              st.idx = nextIdx;
              _accompBeatsLeft = lp.barsPerChord * lp.bpb + _accompBeatsLeft;
            }
          } else {
            st.idx = nextIdx;
            _accompBeatsLeft = lp.barsPerChord * lp.bpb + _accompBeatsLeft;
          }
        }
      }
    }
    // Keep MIDI clock interval in sync with any BPM slider change
    if (_midiClockTimer) _midiClockIvMs = 60000 / (lp.bpm * 24);

    _accompTimer = setTimeout(_accompTick, 25);
  }

  // Build note list for a chord symbol from JS -- maps symbol to MIDI pitch classes
  function _chordNotesFromSym(sym) {
    // Parse the symbol: root + quality suffix
    var rootPat = /^([A-G][b#]?)/;
    var m = sym.match(rootPat); if (!m) return [sym];
    var root = m[1]; var qual = sym.slice(root.length);
    var rootPc = _NOTE_PC[root]; if (rootPc === undefined) return [sym];
    var intervals;
    if (qual==='')        intervals=[0,4,7];
    else if (qual==='m') intervals=[0,3,7];
    else if (qual==='dim') intervals=[0,3,6];
    else if (qual==='aug') intervals=[0,4,8];
    else if (qual==='maj7') intervals=[0,4,7,11];
    else if (qual==='7')   intervals=[0,4,7,10];
    else if (qual==='m7')  intervals=[0,3,7,10];
    else if (qual==='m7b5') intervals=[0,3,6,10];
    else if (qual==='dim7') intervals=[0,3,6,9];
    else if (qual==='maj9') intervals=[0,4,7,11,14];
    else if (qual==='m9')  intervals=[0,3,7,10,14];
    else if (qual==='9')   intervals=[0,4,7,10,14];
    else if (qual==='7#11') intervals=[0,4,7,10,18];
    else if (qual==='sus4') intervals=[0,5,7];
    else if (qual==='sus2') intervals=[0,2,7];
    else intervals=[0,4,7];
    var _CHROM2=['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'];
    return intervals.map(function(i){ return _CHROM2[(rootPc+i)%12]; });
  }
  window._chordNotesFromSym = _chordNotesFromSym;

  function _readAccompChords() {
    // 1. Explicit user selection (clicked tiles in sequence order)
    if (window._seqOrder && window._seqOrder.length > 0) {
      return window._seqOrder.map(function(sym){
        return {symbol:sym, label:sym, notes:_chordNotesFromSym(sym)};
      });
    }
    if (window._selectedChords) {
      var sel = Object.keys(window._selectedChords).filter(function(k){return window._selectedChords[k];});
      if (sel.length > 0) {
        return sel.map(function(sym){
          return {symbol:sym, label:sym, notes:_chordNotesFromSym(sym)};
        });
      }
    }
    // 2. JS-built tile grid — always reflects the current root/mode/custom intervals.
    //    Must come before the Gradio CheckboxGroup, which is only updated on explicit
    //    backend generate calls and will be stale whenever the user changes root/mode in JS.
    if (window._currentPaletteTiles && window._currentPaletteTiles.length) {
      return window._currentPaletteTiles;
    }
    // 3. Last resort: hidden Gradio CheckboxGroup (may be stale)
    var data = getChordData();
    var result = [];
    document.querySelectorAll('#chord_picker label').forEach(function(label) {
      var inp = label.querySelector('input[type=checkbox]'), span = label.querySelector('span');
      if (!inp||!span) return;
      var text = span.textContent.trim(), parts = text.split('·');
      var sym = (parts.length>1 ? parts[parts.length-1] : parts[0]).trim();
      var notes = (data[sym] && data[sym].notes) || _chordNotesFromSym(sym);
      result.push({symbol:sym, label:text, notes:notes, checked:inp.checked});
    });
    var checked = result.filter(function(c){return c.checked;});
    if (checked.length > 0) return checked.map(function(c){ return {symbol:c.symbol, label:c.label, notes:c.notes}; });
    if (result.length > 0) return result.map(function(c){ return {symbol:c.symbol, label:c.label, notes:c.notes}; });
  }

  // -- Tap Tempo -----------------------------------------------------------
  var _tapTimes = [];
  window.tapTempo = function() {
    var now = performance.now();
    // Reset if last tap was more than 3 seconds ago
    if (_tapTimes.length > 0 && now - _tapTimes[_tapTimes.length - 1] > 3000) {
      _tapTimes = [];
    }
    _tapTimes.push(now);
    if (_tapTimes.length > 8) _tapTimes.shift();  // keep last 8 taps
    if (_tapTimes.length >= 2) {
      var intervals = [];
      for (var i = 1; i < _tapTimes.length; i++) {
        intervals.push(_tapTimes[i] - _tapTimes[i-1]);
      }
      var avg = intervals.reduce(function(a,b){return a+b;},0) / intervals.length;
      var bpm = Math.round(60000 / avg);
      bpm = Math.max(20, Math.min(240, bpm));
      var sl = document.getElementById('ctrl-bpm');
      var vl = document.getElementById('ctrl-bpm-val');
      if (sl) sl.value = bpm;
      if (vl) vl.textContent = bpm;
      // Flash the button green briefly
      var btn = document.getElementById('tap-btn');
      if (btn) {
        btn.style.background = '#1a5c30';
        btn.textContent = bpm + ' BPM';
        setTimeout(function(){
          btn.style.background = '#1a3050';
          btn.textContent = 'Tap';
        }, 800);
      }
    }
  };

  // Generate a default I-IV-V-I (or equivalent) when palette has no selection
  function _autoChords() {
    var root = (window._selRoot) || 'C';
    var mode = (window._selMode) || 'Major (Ionian)';
    var modeIntervals = {
      'Major (Ionian)':[0,2,4,5,7,9,11],'Natural Minor (Aeolian)':[0,2,3,5,7,8,10],
      'Dorian':[0,2,3,5,7,9,10],'Phrygian':[0,1,3,5,7,8,10],
      'Lydian':[0,2,4,6,7,9,11],'Mixolydian':[0,2,4,5,7,9,10],
      'Harmonic Minor':[0,2,3,5,7,8,11],'Melodic Minor':[0,2,3,5,7,9,11],
      'Major Pentatonic':[0,2,4,7,9],'Minor Pentatonic':[0,3,5,7,10],
      'Blues Scale':[0,3,5,6,7,10],
    };
    var ivs = modeIntervals[mode] || modeIntervals['Major (Ionian)'];
    var _CHROM=['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'];
    var _NPC={C:0,'C#':1,Db:1,D:2,'D#':3,Eb:3,E:4,F:5,'F#':6,Gb:6,G:7,'G#':8,Ab:8,A:9,'A#':10,Bb:10,B:11};
    var rootPc = _NPC[root] || 0;
    // Pick degrees 0,3,4,0 (I IV V I) from scale, or fewer for pentatonic
    var degIdxs = ivs.length >= 5 ? [0, 3, 4, 0] : [0, 2, 3, 0];
    var chords = [];
    var seen = {};
    degIdxs.forEach(function(di) {
      if (di >= ivs.length) return;
      var chordRootPc = (rootPc + ivs[di]) % 12;
      var chordRoot = _CHROM[chordRootPc];
      var avail = {};
      ivs.forEach(function(s){ avail[(s - ivs[di] + 12) % 12] = true; });
      var sym = avail[4] && avail[7] ? chordRoot : (avail[3] && avail[7] ? chordRoot+'m' : chordRoot);
      var key = sym + di;
      if (seen[key]) return; seen[key] = true;
      chords.push({symbol:sym, label:sym, notes:_chordNotesFromSym(sym)});
    });
    return chords;
  }

  window.startAccompaniment = function() {
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) { alert('Web Audio not supported'); return; }
    if (!_audioCtx) { _audioCtx = new AC(); window._audioCtx = _audioCtx; }
    if (_audioCtx.state === 'suspended') _audioCtx.resume();
    _stopActive(); _ensureGainNodes();
    var chords = _readAccompChords();
    if (!chords.length) chords = _autoChords();
    if (!chords.length) {
      var d = document.getElementById('accomp-now-playing');
      if (d) d.textContent = 'Select chords in the palette first.'; return;
    }
    _audioCtxStartMs = performance.now() - _audioCtx.currentTime * 1000;
    _clockDrift = 0; _driftTarget = 0;
    _pendingIdx = -1; _passingChord = null;
    _startMidiClock();
    var lp = _readLiveParams();
    _accompState = { chords:chords, idx:Math.floor(Math.random()*chords.length),
      nextBarTime:_audioCtx.currentTime+0.08,
      bpb:lp.bpb, beatDur:lp.beatDur, barsPerChord:lp.barsPerChord, style:lp.style };
    window._accompState = _accompState;
    _accompBeatsLeft = _accompState.barsPerChord * _accompState.bpb;
    _accompRunning   = true; window._accompRunning = true;
    _accompTick();
    var d = document.getElementById('accomp-now-playing');
    if (d) d.textContent = chords.map(function(c){return c.symbol;}).join(' · ');
  };

  window.stopAccompaniment = function() {
    _accompRunning = false; window._accompRunning = false;
    if (_accompTimer) { clearTimeout(_accompTimer); _accompTimer = null; }
    _stopMidiClock();
    var d = document.getElementById('accomp-now-playing'); if (d) d.textContent = '—';
    // Silence any pre-scheduled melody notes immediately
    if (_melodyGain && _audioCtx) _melodyGain.gain.setTargetAtTime(0, _audioCtx.currentTime, 0.04);
  };

  // -- MIDI output -- stream to DAW (Logic Pro via IAC driver) ------------------
  function _initMIDIOut() {
    if (!navigator.requestMIDIAccess) return;
    navigator.requestMIDIAccess({sysex:false}).then(function(access) {
      _midiAccess2 = access;
      _populateMIDIOutputs(access);
      access.onstatechange = function() { _populateMIDIOutputs(access); };
    }, function() { console.warn('[MIDI out] access denied'); });
  }

  function _populateMIDIOutputs(access) {
    var sel = document.getElementById('ctrl-midi-out');
    if (!sel) return;
    var cur = sel.value;
    sel.innerHTML = '<option value="">Off</option>';
    access.outputs.forEach(function(out) {
      var opt = document.createElement('option');
      opt.value = out.id; opt.textContent = out.name;
      sel.appendChild(opt);
    });
    if (cur) sel.value = cur;
    window.updateMIDIOutput();
  }

  window.updateMIDIOutput = function() {
    var sel = document.getElementById('ctrl-midi-out');
    var id = sel ? sel.value : '';
    _midiOutput = null;
    if (id && _midiAccess2) {
      _midiAccess2.outputs.forEach(function(out) { if (out.id === id) _midiOutput = out; });
    }
    var badge = document.getElementById('midi-out-badge');
    if (badge) badge.textContent = _midiOutput ? '● LIVE' : '';
    // If accompaniment is running, restart the clock on the new port
    if (_accompRunning) {
      if (_midiOutput) _startMidiClock();
      else _stopMidiClock();
    }
  };

  window.refreshMIDIOutputs = function() {
    if (_midiAccess2) _populateMIDIOutputs(_midiAccess2);
    else _initMIDIOut();
  };

  // -- Listen In ---------------------------------------------------------------
  var _listenMode     = 'off';
  var _midiAccess     = null;
  var _midiNotes      = new Set();   // active pitch-classes from MIDI
  var _midiOnsets     = [];          // recent note-on timestamps for tempo
  var _chordHoldTimer = null;
  var _listenStream   = null;
  var _pitchAnalyser  = null;
  var _pitchTimer     = null;

  function _listenStatus(msg) {
    var el = document.getElementById('listen-status'); if (el) el.textContent = msg;
  }
  function _listenBtnHighlight(mode) {
    ['off','midi','audio'].forEach(function(m) {
      var el = document.getElementById('listen-btn-'+m);
      if (el) el.style.opacity = (m === mode) ? '1' : '0.42';
    });
  }

  // Match active pitch-classes to the best palette chord index (-1 = no match)
  function _matchChord(pcs) {
    var chords = _accompState.chords; if (!chords || !chords.length) return -1;
    var best = -1, bestScore = 0;
    chords.forEach(function(c, i) {
      var cPCs = c.notes.map(function(n) { return _NOTE_PC[n] || 0; });
      var overlap = pcs.filter(function(pc) { return cPCs.indexOf(pc) >= 0; }).length;
      if (!overlap) return;
      var score = overlap / Math.max(pcs.length, cPCs.length);
      if (score > bestScore) { bestScore = score; best = i; }
    });
    return bestScore >= 0.34 ? best : -1;
  }

  function _detectTempo(onsets) {
    if (onsets.length < 4) return null;
    var ivs = [];
    for (var i = 1; i < onsets.length; i++) {
      var d = onsets[i] - onsets[i-1]; if (d > 0.18 && d < 2.5) ivs.push(d);
    }
    if (!ivs.length) return null;
    ivs.sort(function(a,b){return a-b;});
    return 60 / ivs[Math.floor(ivs.length/2)];
  }

  function _applyListenResponse(pcs) {
    var resp = (document.querySelector('input[name="listen-resp"]:checked') || {}).value || 'chord';
    var matched = -1;
    if ((resp==='chord'||resp==='both') && pcs.length) {
      matched = _matchChord(pcs);
      if (matched >= 0 && _accompRunning) { _accompState.idx = matched; _accompBarsLeft = 1; }
    }
    if ((resp==='beat'||resp==='both') && _midiOnsets.length >= 4) {
      var bpm = _detectTempo(_midiOnsets);
      if (bpm && bpm > 30 && bpm < 280) {
        var sl = document.getElementById('ctrl-bpm'), vl = document.getElementById('ctrl-bpm-val');
        if (sl) sl.value = Math.round(bpm); if (vl) vl.textContent = Math.round(bpm);
      }
    }
    var names = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'];
    var noteStr = pcs.map(function(p){return names[p];}).join('+');
    var chordStr = (matched >= 0 && _accompState.chords[matched]) ? ' → '+_accompState.chords[matched].symbol : '';
    _listenStatus(noteStr + chordStr);
  }

  // -- MIDI --------------------------------------------------------------------
  function _onMIDIMsg(ev) {
    if (_listenMode !== 'midi') return;
    var type = ev.data[0] & 0xF0, note = ev.data[1], vel = ev.data[2];
    if (type === 0x90 && vel > 0) {
      _midiNotes.add(note % 12);
      if (_audioCtx) { _midiOnsets.push(_audioCtx.currentTime); if (_midiOnsets.length > 10) _midiOnsets.shift(); }
      clearTimeout(_chordHoldTimer);
      _chordHoldTimer = setTimeout(function() { _applyListenResponse(Array.from(_midiNotes)); }, 140);
    } else if (type === 0x80 || (type === 0x90 && vel === 0)) {
      _midiNotes.delete(note % 12);
    }
  }

  function _initMIDI() {
    if (!navigator.requestMIDIAccess) { _listenStatus('Web MIDI not supported in this browser'); return; }
    navigator.requestMIDIAccess().then(function(access) {
      _midiAccess = access;
      var count = 0;
      access.inputs.forEach(function(inp) { inp.onmidimessage = _onMIDIMsg; count++; });
      access.onstatechange = function() { access.inputs.forEach(function(inp){ inp.onmidimessage = _onMIDIMsg; }); };
      _listenStatus(count ? 'MIDI ready ('+count+' device'+(count>1?'s':'')+') -- play to trigger' : 'No MIDI devices found');
    }, function() { _listenStatus('MIDI access denied'); });
  }

  // -- Audio / Mic -------------------------------------------------------------
  function _autocorrelate(buf, sr) {
    var N = buf.length, H = Math.floor(N/2), rms = 0;
    for (var i=0; i<N; i++) rms += buf[i]*buf[i];
    if (Math.sqrt(rms/N) < 0.008) return -1;
    var last=1, best=-1, bestC=0, go=false, c;
    for (var off=0; off<H; off++) {
      c = 0; for (var i=0; i<H; i++) c += Math.abs(buf[i]-buf[i+off]);
      c = 1 - c/H;
      if (c>0.9&&c>last) go=true;
      if (go&&c<last) { if(c>bestC){bestC=c;best=off;} }
      last=c;
    }
    return best<1 ? -1 : sr/best;
  }

  function _pollPitch() {
    if (_listenMode !== 'audio' || !_pitchAnalyser) return;
    var buf = new Float32Array(_pitchAnalyser.fftSize);
    _pitchAnalyser.getFloatTimeDomainData(buf);
    var freq = _autocorrelate(buf, _audioCtx.sampleRate);
    if (freq > 60 && freq < 2000) {
      var midiRaw = Math.round(12 * Math.log2(freq / 440) + 69);
      // 3-frame median smoothing — reduces pitch wobble / vibrato false-triggers
      _micPitchHist.push(midiRaw); if (_micPitchHist.length > 3) _micPitchHist.shift();
      var h = _micPitchHist.slice().sort(function(a,b){return a-b;});
      var midiF = h[Math.floor(h.length/2)];
      var pc = ((midiF % 12) + 12) % 12;
      _applyListenResponse([pc]);
      // Phrase capture: new note when smoothed pitch shifts by ≥ 1 semitone
      if (_phraseArmed || _phraseCapturing) {
        if (midiF !== _micLastMidi) {
          if (_micLastMidi >= 0 && _phraseCapturing) {
            var d = now - _micLastT;
            if (d >= 0.1) _capturedPhrase.push({midi:_micLastMidi, t:_micLastT, dur:d});
          }
          _micLastMidi = midiF; _micLastT = now;
          if (_phraseArmed) { _phraseCapturing = true; _phraseArmed = false; _updateArmBtn(false); }
          clearTimeout(_phraseGapTimer);
        }
      }
    } else if (_phraseCapturing && _micLastMidi >= 0) {
      // Silence detected — close the last note and start gap timer
      var d = now - _micLastT;
      if (d >= 0.1) _capturedPhrase.push({midi:_micLastMidi, t:_micLastT, dur:d});
      _micLastMidi = -1;
      _phraseGapTimer = setTimeout(_phraseComplete, PHRASE_GAP_SEC * 1000);
    }
    _pitchTimer = setTimeout(_pollPitch, 80);
  }

  function _initAudioListen() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      _listenStatus('Microphone not available'); return;
    }
    navigator.mediaDevices.getUserMedia({audio:true,video:false}).then(function(stream) {
      _listenStream = stream;
      if (!_audioCtx) { var AC=window.AudioContext||window.webkitAudioContext; if(AC) _audioCtx=new AC(); }
      var src = _audioCtx.createMediaStreamSource(stream);
      _pitchAnalyser = _audioCtx.createAnalyser(); _pitchAnalyser.fftSize = 2048;
      src.connect(_pitchAnalyser);
      _listenStatus('Mic active -- sing or play a note');
      _pollPitch();
    }).catch(function(e) { _listenStatus('Mic denied: '+e.message); });
  }

  window.startListening = function(mode) {
    // Stop previous
    _listenMode = 'off';
    if (_chordHoldTimer) clearTimeout(_chordHoldTimer);
    if (_pitchTimer)     clearTimeout(_pitchTimer);
    if (_listenStream)   { _listenStream.getTracks().forEach(function(t){t.stop();}); _listenStream=null; _pitchAnalyser=null; }
    _midiNotes.clear(); _midiOnsets = [];
    _listenBtnHighlight(mode);
    if (mode === 'off') { _listenStatus('--'); return; }
    // Ensure audio context
    if (!_audioCtx) { var AC=window.AudioContext||window.webkitAudioContext; if(AC) _audioCtx=new AC(); }
    if (_audioCtx && _audioCtx.state==='suspended') _audioCtx.resume();
    _ensureGainNodes();
    _listenMode = mode;
    if (mode === 'midi')  _initMIDI();
    if (mode === 'audio') _initAudioListen();
  };

  window.armPhrase = function() {
    _capturedPhrase = []; _phraseArmed = true; _phraseCapturing = false;
    clearTimeout(_phraseGapTimer); _micLastMidi = -1;
    _updateArmBtn(true);
    _melStatus('Armed — play a phrase');
  };

  window.clearPhrase = function() {
    _capturedPhrase = []; _phraseArmed = false; _phraseCapturing = false;
    clearTimeout(_phraseGapTimer); _micLastMidi = -1;
    _updateArmBtn(false); _melStatus('—');
  };

  function _restoreMelodyGain() {
    if (!_melodyGain || !_audioCtx) return;
    var el = document.getElementById('mix-melody');
    var v = el ? parseFloat(el.value) : 0.6;
    _melodyGain.gain.cancelScheduledValues(_audioCtx.currentTime);
    _melodyGain.gain.setTargetAtTime(v, _audioCtx.currentTime, 0.01);
  }

  window.playCaptured = function() {
    if (!_capturedPhrase.length) { _melStatus('Nothing captured — Arm first'); return; }
    if (!_audioCtx) { var AC=window.AudioContext||window.webkitAudioContext; if(AC) _audioCtx=new AC(); }
    if (_audioCtx.state==='suspended') _audioCtx.resume();
    _ensureGainNodes(); _restoreMelodyGain();
    var lp = _readLiveParams();
    var q = _quantizePhrase(_capturedPhrase, lp.beatDur);
    var now = _audioCtx.currentTime + 0.12;
    q.forEach(function(n) { _playMelodyNote(n.midi, now + n.t, n.dur, 0.72); });
    _melStatus('▶ Captured: ' + q.length + ' notes (quantized)');
  };

  window.generateMelodicResponse = function() {
    var scalePCs = _getScalePCs();
    if (!scalePCs.length) { _melStatus('Generate the palette first'); return; }
    var style    = (document.getElementById('mel-style')||{}).value || 'answer';
    var bars     = parseInt((document.getElementById('mel-bars')||{}).value || 2);
    var lp       = _readLiveParams();
    var notes    = _generateMelody(_capturedPhrase, scalePCs, bars * lp.bpb, style);
    if (!notes.length) { _melStatus('Nothing generated'); return; }
    if (!_audioCtx) { var AC=window.AudioContext||window.webkitAudioContext; if(AC) _audioCtx=new AC(); }
    if (_audioCtx.state==='suspended') _audioCtx.resume();
    _ensureGainNodes(); _restoreMelodyGain();
    var now = _audioCtx.currentTime + 0.12;
    notes.forEach(function(n) { _playMelodyNote(n.midi, now + n.beat, n.dur, 0.72); });
    _melStatus('▶ ' + notes.length + ' notes · ' + bars + ' bar' + (bars>1?'s':''));
  };

  window.setRangeNote = function(track, which, midi) {
    midi = parseInt(midi);
    if (track === 'bass')  { if (which === 'lo') _bassRangeLo  = midi; else _bassRangeHi  = midi; }
    else                   { if (which === 'lo') _chordRangeLo = midi; else _chordRangeHi = midi; }
  };

  function _buildNoteOptions(id, selected) {
    var names = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'];
    var sel = document.getElementById(id); if (!sel) return;
    sel.innerHTML = '';
    for (var m = 24; m <= 96; m++) {
      var opt = document.createElement('option');
      opt.value = m;
      opt.textContent = names[m % 12] + (Math.floor(m / 12) - 1);
      if (m === selected) opt.selected = true;
      sel.appendChild(opt);
    }
  }

  function _tryInitRangeSelectors(attempt) {
    if (!document.getElementById('rng-bass-lo')) {
      if (attempt < 25) setTimeout(function(){ _tryInitRangeSelectors(attempt + 1); }, 300);
      return;
    }
    _buildNoteOptions('rng-bass-lo',  _bassRangeLo);
    _buildNoteOptions('rng-bass-hi',  _bassRangeHi);
    _buildNoteOptions('rng-chord-lo', _chordRangeLo);
    _buildNoteOptions('rng-chord-hi', _chordRangeHi);
  }
  _tryInitRangeSelectors(0);

  initDelegation();
  _initMIDIOut();
})();

/* -- Loop Pedal -- */
(function(){
  var _slots = {
    A: {buf:null, src:null, muted:false, recorder:null, chunks:[], startTime:0, loopLen:0},
    B: {buf:null, src:null, muted:false, recorder:null, chunks:[], startTime:0, loopLen:0},
    C: {buf:null, src:null, muted:false, recorder:null, chunks:[], startTime:0, loopLen:0},
  };
  var _micStream = null;
  var _loopDestNode = null; // capture band audio + mic into recorder
  var _status = function(msg){ var el=document.getElementById('loop-status'); if(el) el.textContent=msg; };

  function _getCtx() { return window._audioCtx || null; }

  // Draw waveform on slot canvas
  function _drawWave(slot, buf) {
    var canvas = document.querySelector('#loop-slot-'+slot+' .loop-wave');
    if (!canvas || !buf) return;
    var ctx2 = canvas.getContext('2d');
    var W=canvas.width, H=canvas.height;
    ctx2.clearRect(0,0,W,H);
    ctx2.strokeStyle='#5b9'; ctx2.lineWidth=1;
    var data = buf.getChannelData(0);
    var step = Math.max(1, Math.floor(data.length/W));
    ctx2.beginPath();
    for (var x=0;x<W;x++) {
      var i=x*step, sum=0;
      for(var j=0;j<step&&i+j<data.length;j++) sum+=Math.abs(data[i+j]);
      var amp = (sum/step)*H*1.6;
      ctx2.moveTo(x, H/2-amp/2);
      ctx2.lineTo(x, H/2+amp/2);
    }
    ctx2.stroke();
  }

  function _slotClass(slot, cls) {
    var el = document.getElementById('loop-slot-'+slot);
    if (!el) return;
    el.classList.remove('recording','looping','overdub','muted');
    if (cls) el.classList.add(cls);
  }

  function _stopLoop(slot) {
    var s = _slots[slot];
    if (s.src) { try{s.src.stop();}catch(e){} s.src=null; }
  }

  function _startLoop(slot) {
    var s = _slots[slot]; if (!s.buf) return;
    var ctx = _getCtx(); if (!ctx) return;
    _stopLoop(slot);
    if (s.muted) return;
    s.src = ctx.createBufferSource();
    s.src.buffer = s.buf;
    s.src.loop = true;
    s.src.loopEnd = s.buf.duration;
    // Route to master gain
    if (window._masterGain) s.src.connect(window._masterGain);
    else s.src.connect(ctx.destination);
    s.src.start(0);
    _slotClass(slot, 'looping');
  }

  function _mixBuffers(ctx, buf1, buf2) {
    var len = Math.max(buf1.length, buf2.length);
    var out = ctx.createBuffer(2, len, ctx.sampleRate);
    for (var ch=0; ch<2; ch++) {
      var d = out.getChannelData(ch);
      var d1 = buf1.numberOfChannels > ch ? buf1.getChannelData(ch) : buf1.getChannelData(0);
      var d2 = buf2.numberOfChannels > ch ? buf2.getChannelData(ch) : buf2.getChannelData(0);
      for (var i=0;i<len;i++) d[i] = ((d1[i]||0)*0.72 + (d2[i]||0)*0.72);
    }
    return out;
  }

  function _getMicStream() {
    return navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false, noiseSuppression:false, autoGainControl:false}, video:false});
  }

  // Snap loop length to nearest bar if Sync to BPM is on
  function _syncedLen(rawLen) {
    var sync = document.getElementById('loop-sync-bpm');
    if (!sync || !sync.checked) return rawLen;
    var bpmEl = document.getElementById('ctrl-bpm');
    var bpm = bpmEl ? parseFloat(bpmEl.value)||120 : 120;
    var tsEl = document.getElementById('ctrl-timesig');
    var bpb = tsEl ? parseInt((tsEl.value||'4/4').split('/')[0])||4 : 4;
    var barLen = bpb * 60 / bpm;
    var bars = Math.max(1, Math.round(rawLen / barLen));
    return bars * barLen;
  }

  function _blobToAudioBuf(ctx, blob, cb) {
    var reader = new FileReader();
    reader.onload = function(e) {
      ctx.decodeAudioData(e.target.result, function(decoded) {
        cb(null, decoded);
      }, function(err){ cb(err); });
    };
    reader.readAsArrayBuffer(blob);
  }

  // Create a capture node: mic + band output → recorder input
  // Returns {dest, cleanup} so caller can disconnect band after recording stops
  function _buildCaptureStream(micStream, ctx) {
    var dest = ctx.createMediaStreamDestination();
    var inputMode = (document.getElementById('loop-input-sel')||{}).value || 'mix';
    var bandConnected = false;
    if (inputMode === 'mix' && window._masterGain) {
      window._masterGain.connect(dest);
      bandConnected = true;
    }
    if (micStream) {
      var micSrc = ctx.createMediaStreamSource(micStream);
      micSrc.connect(dest);
    }
    return {
      dest: dest,
      cleanup: function() {
        if (bandConnected && window._masterGain) {
          try { window._masterGain.disconnect(dest); } catch(e){}
        }
      }
    };
  }

  function _finishRecording(slot, blob, isDub) {
    var s = _slots[slot];
    var ctx = _getCtx(); if (!ctx) return;
    _blobToAudioBuf(ctx, blob, function(err, decoded) {
      if (err) { _status('Decode error: '+err); return; }

      // Trim/snap to sync
      var targetLen = _syncedLen(decoded.duration);
      var targetSamples = Math.round(targetLen * ctx.sampleRate);
      var trimmed = ctx.createBuffer(decoded.numberOfChannels,
        Math.min(targetSamples, decoded.length), ctx.sampleRate);
      for (var ch=0;ch<decoded.numberOfChannels;ch++) {
        trimmed.getChannelData(ch).set(decoded.getChannelData(ch).slice(0, trimmed.length));
      }

      if (isDub && s.buf) {
        // Overdub: mix into existing loop (align to loop length)
        var mixed = _mixBuffers(ctx, s.buf, trimmed);
        s.buf = mixed;
      } else {
        s.buf = trimmed;
      }
      s.loopLen = s.buf.duration;

      // Update time display
      var timeEl = document.querySelector('#loop-slot-'+slot+' .loop-slot-time');
      if (timeEl) timeEl.textContent = s.loopLen.toFixed(2)+'s';
      _drawWave(slot, s.buf);
      _startLoop(slot);
      _status('Loop '+slot+' playing ('+s.loopLen.toFixed(2)+'s)');
    });
  }

  function _startRecording(slot, isDub) {
    var s = _slots[slot];
    var ctx = _getCtx();
    if (!ctx) {
      var AC = window.AudioContext||window.webkitAudioContext;
      if (AC) { window._audioCtx = new AC(); ctx = window._audioCtx; }
      if (!ctx) { _status('No AudioContext'); return; }
    }
    if (ctx.state==='suspended') ctx.resume();
    if (window._ensureGainNodes) window._ensureGainNodes();

    _getMicStream().then(function(micStream) {
      _micStream = micStream;
      var capture = _buildCaptureStream(micStream, ctx);

      var mimeType = ['audio/webm;codecs=opus','audio/webm','audio/ogg'].find(function(m){
        return MediaRecorder.isTypeSupported(m);
      }) || '';
      var rec = new MediaRecorder(capture.dest.stream, mimeType ? {mimeType:mimeType} : {});
      s.chunks = [];
      s.recorder = rec;

      rec.ondataavailable = function(e){ if(e.data.size>0) s.chunks.push(e.data); };
      rec.onstop = function() {
        capture.cleanup(); // disconnect band from capture node
        var blob = new Blob(s.chunks, {type: mimeType || 'audio/webm'});
        _slotClass(slot, '');
        _finishRecording(slot, blob, isDub);
        if (_micStream) { _micStream.getTracks().forEach(function(t){t.stop()}); _micStream=null; }
        var btn = document.querySelector('#loop-slot-'+slot+' .loop-rec-btn');
        if (btn) btn.classList.remove('active');
      };

      if (isDub) {
        _stopLoop(slot); // briefly stop while overdubbing start, re-starts on finish
        _slotClass(slot, 'overdub');
        _status('Overdubbing '+slot+' -- press ⊕ again to commit');
      } else {
        _slotClass(slot, 'recording');
        _status('Recording '+slot+' -- press ⏺ again to finish');
      }

      rec.start(100);

      // Store stop callback on slot so second press can stop it
      s._stop = function() { if(rec.state==='recording') rec.stop(); };
    }).catch(function(e){ _status('Mic denied: '+e.message); });
  }

  // Toggle record on slot
  window._loopRec = function(slot, btn) {
    var s = _slots[slot];
    if (s.recorder && s.recorder.state === 'recording') {
      // Second press: stop recording
      btn.classList.remove('active');
      if (s._stop) s._stop();
    } else {
      // First press: stop any existing loop, start recording
      _stopLoop(slot);
      btn.classList.add('active');
      _startRecording(slot, false);
    }
  };

  window._loopDub = function(slot, btn) {
    var s = _slots[slot];
    if (!s.buf) { _status('Nothing to overdub on slot '+slot+' -- record first'); return; }
    if (s.recorder && s.recorder.state === 'recording') {
      btn.classList.remove('active');
      if (s._stop) s._stop();
    } else {
      btn.classList.add('active');
      _startRecording(slot, true);
    }
  };

  window._loopMute = function(slot, btn) {
    var s = _slots[slot];
    s.muted = !s.muted;
    btn.classList.toggle('active', s.muted);
    var el = document.getElementById('loop-slot-'+slot);
    if (el) el.classList.toggle('muted', s.muted);
    if (s.muted) { _stopLoop(slot); }
    else if (s.buf) { _startLoop(slot); }
  };

  window._loopClear = function(slot) {
    var s = _slots[slot];
    _stopLoop(slot);
    if (s.recorder && s.recorder.state==='recording') { try{s.recorder.stop();}catch(e){} }
    s.buf = null; s.recorder = null; s.chunks = []; s.muted = false;
    _slotClass(slot, '');
    var timeEl = document.querySelector('#loop-slot-'+slot+' .loop-slot-time');
    if (timeEl) timeEl.textContent = '--';
    var canvas = document.querySelector('#loop-slot-'+slot+' .loop-wave');
    if (canvas) { var c=canvas.getContext('2d'); c.clearRect(0,0,canvas.width,canvas.height); }
    var btn = document.querySelector('#loop-slot-'+slot+' .loop-rec-btn');
    if (btn) btn.classList.remove('active');
    _status('Slot '+slot+' cleared');
  };

  window._loopStopAll = function() {
    ['A','B','C'].forEach(function(s){ _stopLoop(s); _slotClass(s,''); });
    _status('All loops stopped');
  };

  window._loopClearAll = function() {
    ['A','B','C'].forEach(function(s){ window._loopClear(s); });
    _status('All loops cleared');
  };

  // Expose restart for external sync
  window._loopRestartAll = function() {
    ['A','B','C'].forEach(function(s){ if(_slots[s].buf && !_slots[s].muted) _startLoop(s); });
  };
})();

/* -- Crystallize & Describe -- pure JS, zero server roundtrip -- */
(function() {

  // Map Live Recorder mode names → Composer key string suffix
  var _MODE_STR = {
    'Major (Ionian)':'major','Natural Minor (Aeolian)':'minor',
    'Dorian':'dorian','Phrygian':'phrygian','Lydian':'lydian',
    'Mixolydian':'mixolydian','Locrian':'locrian',
    'Harmonic Minor':'harmonic minor','Melodic Minor':'melodic minor',
    'Major Pentatonic':'major pentatonic','Minor Pentatonic':'minor pentatonic',
    'Blues Scale':'blues','Whole Tone':'whole tone',
    'Diminished (HW)':'diminished','Hungarian Minor':'hungarian minor',
    'Phrygian Dominant':'phrygian dominant',
  };

  function _liveChords() {
    var seq = window._seqOrder || [];
    if (seq.length) return seq.join('  ');
    var sel = window._selectedChords || {};
    var keys = Object.keys(sel).filter(function(k){ return sel[k]; });
    if (keys.length) return keys.join('  ');
    var root = window._selRoot || 'C';
    return root + 'maj7  ' + root + 'm7  ' + root + '7  ' + root + 'maj7';
  }

  function _liveBpm() {
    var el = document.getElementById('ctrl-bpm');
    return el ? parseFloat(el.value) || 120 : 120;
  }

  // -- CRYSTALLIZE ------------------------------------------------------------
  // Stores session data in localStorage, then navigates to Composer.
  // ocNav reads localStorage on arrival and applies values via _applyComposerData.

  window._crystallize = function(srcBtn) {
    var root   = window._selRoot || 'C';
    var mode   = window._selMode || 'Major (Ionian)';
    var chords = _liveChords();
    var key    = root + ' ' + (_MODE_STR[mode] || 'major');
    var bpm    = _liveBpm();
    try {
      localStorage.setItem('oc_crystallize', JSON.stringify({chords:chords, key:key, bpm:bpm}));
    } catch(e) {}
    if (window.ocNav) window.ocNav('composer');
    if (srcBtn) { srcBtn.disabled = false; srcBtn.classList.remove('loading'); }
  };

  // -- DESCRIBE ---------------------------------------------------------------
  // Translates the live session into a natural-language Composer brief.
  // Unlike Crystallize (raw data), Describe builds a creative prompt the user
  // can read, edit, and run -- bridging improvisation and composition intent.

  var _STYLE_MOOD = {
    'Jazz':'swinging jazz','Gospel':'soulful gospel','Soul':'deep soul',
    'R&B':'smooth R&B','Funk':'funky groove','Motown':'classic Motown',
    'Ballad':'heartfelt ballad','Cinematic':'cinematic orchestral','Pop':'bright pop',
    'Acoustic':'intimate acoustic','Bossa Nova':'bossa nova','Latin':'Afro-Latin',
    'Reggae':'laid-back reggae','Blues':'slow blues','Lo-Fi':'lo-fi hip-hop',
    'Country':'country folk','Disco Pop':'disco pop','Indie Pop':'indie pop',
    'Neo Soul':'neo-soul','Worship':'worship anthemic','Pad':'atmospheric pad',
    'Samba':'uptempo samba','Swing':'classic swing','Smooth Jazz':'smooth jazz',
    'Tropical':'tropical groove','Funk Chop':'funk chop',
    'Singer-Songwriter':'singer-songwriter','Honky Tonk':'honky-tonk',
  };
  var _TEMPO_WORD = [[50,'languid'],[68,'slow'],[84,'relaxed'],[100,'moderate'],
                     [116,'mid-tempo'],[132,'brisk'],[148,'uptempo'],[999,'driving']];

  window._describeSession = function(srcBtn) {
    var root   = window._selRoot || 'C';
    var mode   = window._selMode || 'Major (Ionian)';
    var chords = _liveChords();
    var bpm    = _liveBpm();
    var style  = (document.getElementById('ctrl-style') && document.getElementById('ctrl-style').value) || 'Ballad';
    var key    = root + ' ' + (_MODE_STR[mode] || 'major');

    var tempoWord = 'moderate';
    for (var i = 0; i < _TEMPO_WORD.length; i++) { if (bpm < _TEMPO_WORD[i][0]) { tempoWord = _TEMPO_WORD[i][1]; break; } }

    var moodStr   = _STYLE_MOOD[style] || style.toLowerCase();
    var modeShort = mode.replace(' (Ionian)','').replace(' (Aeolian)','');

    var prompt = tempoWord + ' ' + moodStr + ' in ' + key + ' at ' + Math.round(bpm) + ' BPM. ' +
      'Chord progression: ' + chords + '. ' +
      'Orchestrate with rich ' + moodStr + ' textures -- full arrangement.';

    try {
      localStorage.setItem('oc_describe', JSON.stringify({prompt:prompt, bpm:bpm, key:key, chords:chords}));
    } catch(e) {}
    if (window.ocNav) window.ocNav('composer');
    if (srcBtn) { srcBtn.disabled = false; }
  };

})();

/* -- Custom palette picker -- */
(function(){
  var NOTES = [
    {label:'C',acc:false},{label:'C#',acc:true},{label:'D',acc:false},
    {label:'D#',acc:true},{label:'E',acc:false},{label:'F',acc:false},
    {label:'F#',acc:true},{label:'G',acc:false},{label:'G#',acc:true},
    {label:'A',acc:false},{label:'A#',acc:true},{label:'B',acc:false}
  ];
  var MODES_COMMON = [
    'Major (Ionian)','Natural Minor (Aeolian)','Dorian','Phrygian','Lydian',
    'Mixolydian','Locrian','Harmonic Minor','Melodic Minor',
    'Major Pentatonic','Minor Pentatonic','Blues Scale','Whole Tone',
    'Diminished (HW)','Hungarian Minor','Phrygian Dominant'
  ];
  // Mode intervals (semitones from root) -- computed client-side, no Gradio roundtrip needed
  var MODE_INTERVALS = {
    'Major (Ionian)':         [0,2,4,5,7,9,11],
    'Natural Minor (Aeolian)':[0,2,3,5,7,8,10],
    'Dorian':                 [0,2,3,5,7,9,10],
    'Phrygian':               [0,1,3,5,7,8,10],
    'Lydian':                 [0,2,4,6,7,9,11],
    'Mixolydian':             [0,2,4,5,7,9,10],
    'Locrian':                [0,1,3,5,6,8,10],
    'Harmonic Minor':         [0,2,3,5,7,8,11],
    'Melodic Minor':          [0,2,3,5,7,9,11],
    'Major Pentatonic':       [0,2,4,7,9],
    'Minor Pentatonic':       [0,3,5,7,10],
    'Blues Scale':            [0,3,5,6,7,10],
    'Whole Tone':             [0,2,4,6,8,10],
    'Diminished (HW)':        [0,2,3,5,6,8,9,11],
    'Hungarian Minor':        [0,2,3,6,7,8,11],
    'Phrygian Dominant':      [0,1,4,5,7,8,10],
  };
  var _CHROM = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'];
  var _NOTE_PC2 = {C:0,'C#':1,Db:1,D:2,'D#':3,Eb:3,E:4,F:5,'F#':6,Gb:6,G:7,'G#':8,Ab:8,A:9,'A#':10,Bb:10,B:11};
  var _DEG_LABELS = ['I','II','III','IV','V','VI','VII','VIII'];

  // Build all diatonic chords for a root + intervals array
  function _buildScaleChords(rootNote, intervals) {
    var rootPc = _NOTE_PC2[rootNote] || 0;
    var chords = [];
    intervals.forEach(function(semitone, degIdx) {
      var chordRootPc = (rootPc + semitone) % 12;
      var chordRootName = _CHROM[chordRootPc];
      // Intervals available from this degree
      var avail = {};
      intervals.forEach(function(s) { avail[(s - semitone + 12) % 12] = true; });
      var has = function(n){ return !!avail[n]; };
      var m3=has(3),M3=has(4),p4=has(5),d5=has(6),p5=has(7),A5=has(8);
      var M2=has(2),d7=has(9),m7=has(10),M7=has(11),A4=has(6);
      var degBase = _DEG_LABELS[degIdx] || String(degIdx+1);
      function addChord(sym, degSuffix, quality) {
        chords.push({sym:chordRootName+sym, deg:degBase+degSuffix, quality:quality});
      }
      // Triads
      if (M3&&p5)      addChord('',     '', 'major');
      if (m3&&p5)      addChord('m',    '', 'minor');
      if (m3&&d5&&!m7&&!d7) addChord('dim','°','dim');
      if (M3&&A5)      addChord('aug',  '+','aug');
      // 7ths
      if (M3&&p5&&M7)  addChord('maj7','Δ','maj7');
      if (M3&&p5&&m7)  addChord('7',   '⁷','dom7');
      if (m3&&p5&&m7)  addChord('m7',  '⁷','min7');
      if (m3&&d5&&m7)  addChord('m7b5','ø','hdim');
      if (m3&&d5&&d7)  addChord('dim7','°⁷','dim7');
      if (M3&&p5&&M7&&M2) addChord('maj9','⁹','maj9');
      if (m3&&p5&&m7&&M2) addChord('m9', '⁹','min9');
      if (M3&&p5&&m7&&M2) addChord('9',  '⁹','dom9');
      if (M3&&p5&&m7&&A4) addChord('7#11','♯¹¹','lyd7');
      if (p4&&p5&&!m3&&!M3) addChord('sus4','sus','sus');
      if (M2&&p5&&!m3&&!M3) addChord('sus2','sus²','sus');
    });
    return chords;
  }

  var _selRoot = window._selRoot = 'C';
  var _selMode = window._selMode = 'Major (Ionian)';
  var _customIntervals = null; // set when user types custom intervals; overrides MODE_INTERVALS
  var _seqOrder = window._seqOrder = []; // ordered list of selected chord symbols
  var _selectedChords = window._selectedChords = {}; // sym → true
  var _extraChords = {}; // sym → notes[] for chords added via the input box (off-palette)

  function _setGradioDropdown(elemId, value) {
    var wrap = document.getElementById(elemId) || document.querySelector('[id="'+elemId+'"]');
    if (!wrap) { wrap = document.querySelector('#pal-gradio-controls [data-testid]'); }
    // Gradio Svelte dropdown: find the inner <input> or <select>
    if (!wrap) return;
    var input = wrap.querySelector('input:not([type=checkbox])');
    if (input) {
      var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
      nativeSetter.call(input, value);
      input.dispatchEvent(new Event('input', {bubbles:true}));
    }
  }

  function _triggerGenerate() {
    var btn = document.getElementById('palette-gen-btn');
    if (!btn) btn = document.querySelector('#pal-gradio-controls button');
    if (btn) btn.click();
  }

  function _buildNoteRow() {
    var row = document.getElementById('pal-note-row');
    if (!row) return;
    row.innerHTML = '';
    NOTES.forEach(function(n) {
      var b = document.createElement('button');
      b.className = 'pal-note' + (n.acc ? ' acc' : '') + (n.label === _selRoot ? ' on' : '');
      b.textContent = n.label;
      b.onclick = function() {
        _selRoot = window._selRoot = n.label;
        row.querySelectorAll('.pal-note').forEach(function(x){x.classList.remove('on');});
        b.classList.add('on');
        _selectedChords = window._selectedChords = {}; _seqOrder = window._seqOrder = [];
        _rebuildTiles();
      };
      row.appendChild(b);
    });
  }

  function _buildModeRow() {
    var row = document.getElementById('pal-mode-row');
    if (!row) return;
    row.innerHTML = '';
    MODES_COMMON.forEach(function(m) {
      var c = document.createElement('div');
      c.className = 'pal-mode' + (m === _selMode ? ' on' : '');
      c.textContent = m.replace(' (Ionian)','').replace(' (Aeolian)','');
      c.title = m;
      c.onclick = function() {
        _selMode = window._selMode = m;
        _customIntervals = null;
        var ci = document.getElementById('pal-custom-input'); if (ci) ci.value = '';
        row.querySelectorAll('.pal-mode').forEach(function(x){x.classList.remove('on');});
        c.classList.add('on');
        _selectedChords = window._selectedChords = {}; _seqOrder = window._seqOrder = [];
        _rebuildTiles();
      };
      row.appendChild(c);
    });
  }

  // Parse a custom intervals string (same rules as Python's parse_mode_input):
  //   space/comma separated numbers; if min>=1 treat as 1-based swarasthanas, else 0-based semitones
  function _parseCustomIntervals(text) {
    text = (text || '').trim();
    if (!text) return null;
    var nums = text.replace(/,/g, ' ').split(/\s+/).map(Number).filter(function(n){ return !isNaN(n) && isFinite(n); });
    if (!nums.length) return null;
    if (Math.min.apply(null, nums) >= 1) {
      // 1-based swarasthanas
      var s = {}; nums.forEach(function(n){ s[(n-1)%12]=true; });
      return Object.keys(s).map(Number).sort(function(a,b){return a-b;});
    }
    var s = {}; nums.forEach(function(n){ s[((n%12)+12)%12]=true; });
    return Object.keys(s).map(Number).sort(function(a,b){return a-b;});
  }

  function _applyCustomIntervals() {
    var ci = document.getElementById('pal-custom-input');
    if (!ci) return;
    var parsed = _parseCustomIntervals(ci.value);
    _customIntervals = parsed;
    _selectedChords = window._selectedChords = {};
    _seqOrder = window._seqOrder = [];
    _rebuildTiles();
  }

  // Custom intervals box → parse in JS and rebuild tiles directly
  function _wireCustomIntervals() {
    var ci = document.getElementById('pal-custom-input');
    if (!ci) { setTimeout(_wireCustomIntervals, 400); return; }
    ci.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') { _applyCustomIntervals(); }
    });
    ci.addEventListener('input', function() {
      clearTimeout(ci._t);
      // Clear immediately if box is emptied so mode row takes over
      if (!ci.value.trim()) { _customIntervals = null; _rebuildTiles(); return; }
      ci._t = setTimeout(_applyCustomIntervals, 700);
    });
  }

  // -- Tile grid: JS-native, no Gradio roundtrip ---------------------
  var _QUALITY_COLOR = {
    major:'T', minor:'P', maj7:'T', min7:'P', dom7:'D', dom9:'D',
    hdim:'B', dim:'B', dim7:'B', aug:'B', maj9:'T', min9:'P',
    lyd7:'D', sus:'T'
  };
  function _qualLabel(q) {
    return {major:'major',minor:'minor',maj7:'maj7',min7:'min7',dom7:'dom7',dom9:'9th',
            hdim:'ø',dim:'dim',dim7:'dim7',aug:'aug',maj9:'maj9',min9:'min9',
            lyd7:'lyd♭7',sus:'sus'}[q] || q;
  }

  function _rebuildTiles() {
    var grid = document.getElementById('pal-tile-grid');
    if (!grid) return;
    var intervals = _customIntervals || MODE_INTERVALS[_selMode] || MODE_INTERVALS['Major (Ionian)'];
    var chords = _buildScaleChords(_selRoot, intervals);
    grid.innerHTML = '';
    // Remove chords from seqOrder that are no longer in scale, but preserve manually-added extras
    var syms = chords.map(function(c){return c.sym;});
    _seqOrder = window._seqOrder = _seqOrder.filter(function(s){ return syms.indexOf(s) >= 0 || !!_extraChords[s]; });
    chords.forEach(function(c) {
      var isOn = !!_selectedChords[c.sym];
      var tile = document.createElement('div');
      tile.className = 'pal-tile' + (isOn ? ' on' : '');
      tile.dataset.fn = _QUALITY_COLOR[c.quality] || 'T';
      tile.dataset.sym = c.sym;
      tile.innerHTML =
        '<span class="pt-deg">'+c.deg+'</span>' +
        '<span class="pt-sym">'+c.sym+'</span>' +
        '<span class="pt-qual">'+_qualLabel(c.quality)+'</span>';
      tile.onclick = function() {
        var on = !_selectedChords[c.sym];
        _selectedChords[c.sym] = on ? true : undefined;
        tile.classList.toggle('on', on);
        if (on) { if (_seqOrder.indexOf(c.sym)<0) _seqOrder.push(c.sym); }
        else { _seqOrder = window._seqOrder = _seqOrder.filter(function(x){return x!==c.sym;}); }
        _rebuildSeq();
        _syncGradioCheckboxes();
      };
      grid.appendChild(tile);
    });
    // Expose all current palette chords so accompaniment can use them as fallback
    window._currentPaletteTiles = chords.map(function(c) {
      return {symbol: c.sym, label: c.sym,
              notes: window._chordNotesFromSym ? window._chordNotesFromSym(c.sym) : [c.sym]};
    });
    _rebuildSeq();
  }

  // Push current selection back to hidden Gradio CheckboxGroup so backend can use it
  function _syncGradioCheckboxes() {
    var picker = document.querySelector('#chord_picker');
    if (!picker) return;
    var selected = Object.keys(_selectedChords).filter(function(k){ return _selectedChords[k]; });
    // Set checkboxes to match _selectedChords
    picker.querySelectorAll('label').forEach(function(lbl) {
      var inp = lbl.querySelector('input[type=checkbox]');
      var span = lbl.querySelector('span');
      if (!inp || !span) return;
      var sym = span.textContent.trim().split('·')[0].trim();
      var shouldBeChecked = !!_selectedChords[sym];
      if (inp.checked !== shouldBeChecked) inp.click();
    });
  }

  function _rebuildSeq() {
    var row = document.getElementById('pal-seq-row');
    if (!row) return;
    row.innerHTML = '';
    _seqOrder.forEach(function(sym, i) {
      var pill = document.createElement('div');
      pill.className = 'pal-seq-pill' + (_extraChords[sym] ? ' custom' : '');
      pill.draggable = true;
      pill.dataset.i = i;
      var x = document.createElement('span');
      x.className = 'pal-seq-x'; x.textContent = '×';
      x.onclick = function(e) {
        e.stopPropagation();
        _seqOrder.splice(i,1);
        window._seqOrder = _seqOrder;
        _selectedChords[sym] = undefined;
        delete _extraChords[sym];
        _rebuildTiles();
      };
      pill.innerHTML = '<span>'+sym+'</span>';
      pill.appendChild(x);
      // drag-to-reorder
      pill.ondragstart = function(e){ e.dataTransfer.setData('text/plain', i); };
      pill.ondragover = function(e){ e.preventDefault(); };
      pill.ondrop = function(e){
        e.preventDefault();
        var from = +e.dataTransfer.getData('text/plain');
        var to = i;
        if (from===to) return;
        var item = _seqOrder.splice(from,1)[0];
        _seqOrder.splice(to,0,item);
        _rebuildSeq();
      };
      row.appendChild(pill);
    });
  }

  function _findLabel(sym) {
    var picker = document.querySelector('#chord_picker');
    if (!picker) return null;
    var found = null;
    picker.querySelectorAll('label').forEach(function(l){
      var s = l.querySelector('span');
      if (s && s.textContent.trim().split('·')[0].trim() === sym) found = l;
    });
    return found;
  }

  // ── Add-chord input: parse symbol, validate, inject into sequence ─────────
  var _VALID_ROOTS = {C:1,'C#':1,Db:1,D:1,'D#':1,Eb:1,E:1,F:1,'F#':1,Gb:1,G:1,'G#':1,Ab:1,A:1,'A#':1,Bb:1,B:1,Cb:1,'B#':1};
  var _QUAL_ALIASES = {
    'M7':'maj7','Δ7':'maj7','△7':'maj7','^7':'maj7','MA7':'maj7','ma7':'maj7',
    'M9':'maj9','Δ9':'maj9','△9':'maj9',
    'min':'m','mi':'m','-':'m',
    'min7':'m7','mi7':'m7','-7':'m7',
    'min9':'m9','mi9':'m9','-9':'m9',
    '°':'dim','o':'dim','0':'dim',
    '°7':'dim7','o7':'dim7',
    'ø':'m7b5','ø7':'m7b5','m7♭5':'m7b5','m7b5':'m7b5',
    '+':'aug',
    'dom7':'7','dom':'7','dom9':'9',
    'sus':'sus4'
  };
  var _VALID_QUALS = {'':1,'m':1,'dim':1,'aug':1,'maj7':1,'7':1,'m7':1,'m7b5':1,'dim7':1,
                      'maj9':1,'m9':1,'9':1,'7#11':1,'sus4':1,'sus2':1};

  function _parseChordSym(raw) {
    raw = (raw || '').trim();
    if (!raw) return {error: 'type a chord first'};
    var rm = raw.match(/^([A-Ga-g])(##|bb|[#b])?/);
    if (!rm) return {error: 'must start with A – G'};
    var root = rm[1].toUpperCase() + (rm[2] || '').replace('##','#').replace('bb','b');
    if (!_VALID_ROOTS[root]) return {error: root + ' not a valid root'};
    var qual = raw.slice(rm[0].length).replace(/\s/g,'');
    // Try alias table, then exact
    if (_QUAL_ALIASES[qual] !== undefined) qual = _QUAL_ALIASES[qual];
    if (!_VALID_QUALS[qual]) return {error: '"' + qual + '" not a known quality'};
    var sym = root + qual;
    // Use global _chordNotesFromSym exposed from IIFE 1
    var notes = window._chordNotesFromSym ? window._chordNotesFromSym(sym) : null;
    if (!notes || !notes.length || (notes.length===1 && notes[0]===sym)) {
      return {error: 'could not compute notes for ' + sym};
    }
    return {sym: sym, notes: notes};
  }

  function _wireAddChord() {
    var inp = document.getElementById('pal-add-input');
    var btn = document.getElementById('pal-add-btn');
    var err = document.getElementById('pal-add-err');
    if (!inp || !btn) { setTimeout(_wireAddChord, 400); return; }

    function doAdd() {
      var parsed = _parseChordSym(inp.value);
      if (parsed.error) {
        err.textContent = parsed.error;
        setTimeout(function(){ err.textContent=''; }, 2500);
        return;
      }
      err.textContent = '';
      var sym = parsed.sym;
      _extraChords[sym] = parsed.notes;
      if (_seqOrder.indexOf(sym) < 0) {
        _seqOrder.push(sym);
        window._seqOrder = _seqOrder;
      }
      _selectedChords[sym] = true;
      inp.value = '';
      _rebuildSeq();
    }

    btn.onclick = doAdd;
    inp.addEventListener('keydown', function(e){ if (e.key==='Enter') doAdd(); });
  }

  // Initial tile build — triggered once DOM is ready
  function _watchPicker() {
    var grid = document.getElementById('pal-tile-grid');
    if (!grid) { setTimeout(_watchPicker, 400); return; }
    _rebuildTiles();
  }

  function _init() {
    if (!document.getElementById('pal-note-row')) { setTimeout(_init, 200); return; }
    _buildNoteRow();
    _buildModeRow();
    _wireCustomIntervals();
    _wireAddChord();
    _watchPicker();
    // Auto-generate on load
    setTimeout(_triggerGenerate, 800);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', _init);
  else _init();
})();

/* -- Studio tab switching -- */
function studioShowTab(tab) {
  document.querySelectorAll('.studio-panel').forEach(function(p){ p.classList.remove('visible'); });
  document.querySelectorAll('.studio-tab-btn').forEach(function(b){ b.classList.remove('active'); });
  var panel = document.getElementById('panel-' + tab);
  var btn   = document.getElementById('tab-btn-' + tab);
  if (panel) panel.classList.add('visible');
  if (btn)   btn.classList.add('active');
}
(function(){
  var _ocStatusObs = new MutationObserver(function(){
    var gr = document.querySelector('.gr-markdown');
    var txt = gr ? (gr.textContent || 'Checking…') : '';
    ['oc-ollama-status','oc-ollama-status-comp'].forEach(function(id){
      var el = document.getElementById(id); if(el) el.textContent = txt;
    });
  });
  document.addEventListener('DOMContentLoaded', function(){
    var target = document.querySelector('[data-testid="markdown"]');
    if (target) _ocStatusObs.observe(target, {childList:true, subtree:true});
  });
})();

/* -- Home/section navigation -- */
(function(){
  function _show(id){ var el=document.getElementById(id); if(el) el.style.removeProperty('display'); }
  function _hide(id){ var el=document.getElementById(id); if(el) el.style.display='none'; }

  // Gradio Svelte dispatch -- the only reliable way to set component values
  function _gradioSet(el, value) {
    if (!el) return;
    // Try Svelte internal dispatch first
    if (el.__svelte_meta || el._svelte_component) {
      try { el.dispatchEvent(new CustomEvent('change', {detail: value, bubbles: true})); } catch(e){}
    }
    // Then native setter + synthetic events
    var input = el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' ? el
      : el.querySelector('input, textarea');
    if (!input) return;
    var proto = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    var setter = Object.getOwnPropertyDescriptor(proto, 'value');
    if (setter && setter.set) setter.set.call(input, value);
    ['input','change','blur'].forEach(function(ev) {
      input.dispatchEvent(new Event(ev, {bubbles:true}));
    });
  }

  function _applyComposerData(data) {
    if (!data) return;
    // Chord input: find by placeholder
    var chordEl = document.querySelector(
      'textarea[placeholder*="Dm7"], input[placeholder*="Dm7"]');
    if (chordEl) _gradioSet(chordEl, data.chords);

    // Key textbox: find by placeholder
    var keyEl = document.querySelector(
      'input[placeholder*="C major"], textarea[placeholder*="C major"]');
    if (keyEl) _gradioSet(keyEl, data.key);

    // BPM: find slider with min=40 max=240 inside section-composer
    var bpmEl = document.querySelector(
      '#section-composer input[type=range][min="40"], #section-composer input[type=range][min="40.0"]');
    if (!bpmEl) bpmEl = document.querySelector('input[type=range][min="40"][max="240"]');
    if (bpmEl) {
      var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
      if (setter && setter.set) setter.set.call(bpmEl, data.bpm);
      ['input','change'].forEach(function(ev){ bpmEl.dispatchEvent(new Event(ev,{bubbles:true})); });
    }

    // Key mode radio: click "Manual"
    document.querySelectorAll('#section-composer input[type=radio]').forEach(function(r){
      var lbl = r.closest('label');
      if (lbl && lbl.textContent.trim() === 'Manual' && !r.checked) r.click();
    });

    // Melody source: click "Harmony only"
    document.querySelectorAll('#section-composer input[type=radio]').forEach(function(r){
      var lbl = r.closest('label');
      if (lbl && lbl.textContent.trim() === 'Harmony only' && !r.checked) r.click();
    });
  }

  var _ALL_SECTIONS = ['section-live', 'section-composer', 'section-garcia'];

  window.ocNav = function(section) {
    _hide('home-screen');
    _ALL_SECTIONS.forEach(function(id){ _hide(id); });
    _show('section-' + section);
    if (section === 'composer') {
      try {
        var raw = localStorage.getItem('oc_crystallize');
        if (raw) {
          var data = JSON.parse(raw);
          localStorage.removeItem('oc_crystallize');
          setTimeout(function(){
            _applyComposerData(data);
            setTimeout(function(){
              var genBtn = document.querySelector('#generate-btn-wrap button.primary');
              if (!genBtn) genBtn = document.querySelector('#generate-btn-wrap button');
              if (genBtn) genBtn.click();
            }, 300);
          }, 120);
        }
      } catch(e){}
    }
    window._ocActiveSection = section;
  };

  window.ocGoHome = function() {
    _show('home-screen');
    _ALL_SECTIONS.forEach(function(id){ _hide(id); });
  };

  function _hideSections() {
    _ALL_SECTIONS.forEach(function(id){
      var el = document.getElementById(id);
      if (el && el.style.display !== 'none') el.style.display = 'none';
    });
  }

  _hideSections();
  var _sectionObserver = new MutationObserver(function() { _hideSections(); });
  _sectionObserver.observe(document.body, { childList: true, subtree: true });

  function _waitSections() {
    var found = _ALL_SECTIONS.filter(function(id){ return !!document.getElementById(id); });
    if (found.length === _ALL_SECTIONS.length) {
      _ALL_SECTIONS.forEach(function(id){ document.getElementById(id).style.display = 'none'; });
      _sectionObserver.disconnect();
    } else {
      setTimeout(_waitSections, 80);
    }
  }
  _waitSections();
})();
"""

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tahoma:wght@400;700&display=swap');

/* ═══════════════════════════════════════════════
   DESIGN TOKENS -- DAW / Pro Audio (2000s)
   ═══════════════════════════════════════════════ */
:root {
  --bg:          #1e1e1e;
  --panel:       #2c2c2c;
  --panel-hi:    #383838;
  --panel-lo:    #222222;
  --raised:      #3a3a3a;
  --sunken:      #161616;
  --border-hi:   #585858;
  --border-lo:   #0e0e0e;
  --border-mid:  #404040;
  --text:        #d4d4d4;
  --text-dim:    #888888;
  --text-bright: #f0f0f0;
  --text-label:  #aaaaaa;
  --accent:      #3a7bd5;
  --accent-hi:   #5599ff;
  --accent-lo:   #1e4a99;
  --green-led:   #44cc55;
  --green-dim:   #1a4422;
  --red-led:     #dd3322;
  --red-dim:     #441111;
  --orange-led:  #dd8822;
  --lcd-bg:      #0d1a0d;
  --lcd-text:    #44dd44;
  --tape-green:  #2a5c3a;
  --font-ui:     Tahoma, 'Segoe UI', Arial, sans-serif;
  --font-mono:   'Courier New', Consolas, monospace;
  --font-serif:  Tahoma, Arial, sans-serif;
  --radius-sm:   2px;
  --radius-md:   3px;
  --radius-lg:   4px;
  --bevel-out:   1px solid var(--border-hi);
  --bevel-in:    1px solid var(--border-lo);
  --btn-face:    linear-gradient(to bottom, #4a4a4a 0%, #323232 50%, #2a2a2a 51%, #383838 100%);
  --btn-pressed: linear-gradient(to bottom, #222222 0%, #2a2a2a 50%, #323232 51%, #3a3a3a 100%);
  --shadow-card: inset 0 1px 0 #505050, inset 0 -1px 0 #111111, 1px 1px 3px rgba(0,0,0,0.6);
  --shadow-inset: inset 1px 1px 3px rgba(0,0,0,0.8), inset -1px -1px 0 rgba(80,80,80,0.15);
}

/* ═══════════════════════════════════════════════
   ROOT / SHELL
   ═══════════════════════════════════════════════ */
gradio-app {
  background: var(--bg) !important;
  font-family: var(--font-ui) !important;
}
.gradio-container {
  background: var(--bg) !important;
  max-width: 1340px !important;
  padding: 0 12px !important;
}

/* ═══════════════════════════════════════════════
   PANELS / BLOCKS -- beveled raised surface
   ═══════════════════════════════════════════════ */
.block, .form, .gr-group, .gr-box {
  background: var(--panel) !important;
  border-top: 1px solid var(--border-hi) !important;
  border-left: 1px solid var(--border-hi) !important;
  border-right: 1px solid var(--border-lo) !important;
  border-bottom: 1px solid var(--border-lo) !important;
  border-radius: var(--radius-sm) !important;
  box-shadow: none !important;
}
.gap, .gr-gap { gap: 6px !important; }

/* ═══════════════════════════════════════════════
   TYPOGRAPHY
   ═══════════════════════════════════════════════ */
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3,
.markdown h1, .markdown h2, .markdown h3 {
  font-family: var(--font-ui) !important;
  font-weight: 700 !important;
  color: var(--text-bright) !important;
  letter-spacing: 0;
  font-size: 13px !important;
}
.gr-markdown p, .markdown p, .gr-markdown li {
  color: var(--text-dim) !important;
  font-size: 11px !important;
  font-family: var(--font-ui) !important;
}
strong { color: var(--text) !important; }

/* -- Labels -- */
label, label span {
  color: var(--text-label) !important;
  font-size: 10px !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  font-family: var(--font-ui) !important;
}

/* ═══════════════════════════════════════════════
   INPUTS -- sunken inset look
   ═══════════════════════════════════════════════ */
textarea, input[type="text"], input[type="number"],
.gr-textbox textarea, .gr-input input {
  background: var(--sunken) !important;
  border-top: 1px solid var(--border-lo) !important;
  border-left: 1px solid var(--border-lo) !important;
  border-right: 1px solid var(--border-hi) !important;
  border-bottom: 1px solid var(--border-hi) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  box-shadow: var(--shadow-inset) !important;
  transition: none !important;
}
textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
  border-color: var(--accent) !important;
  outline: 1px solid var(--accent) !important;
  outline-offset: -1px !important;
  box-shadow: var(--shadow-inset) !important;
}

/* ═══════════════════════════════════════════════
   DROPDOWNS -- Windows-style select
   ═══════════════════════════════════════════════ */
select, .gr-dropdown select {
  background: var(--sunken) !important;
  border-top: 1px solid var(--border-lo) !important;
  border-left: 1px solid var(--border-lo) !important;
  border-right: 1px solid var(--border-hi) !important;
  border-bottom: 1px solid var(--border-hi) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-size: 11px !important;
  font-family: var(--font-ui) !important;
}

/* ═══════════════════════════════════════════════
   SLIDERS -- accent blue track
   ═══════════════════════════════════════════════ */
input[type="range"] { accent-color: var(--accent) !important; cursor: pointer; }

/* ═══════════════════════════════════════════════
   BUTTONS -- beveled pill, Windows XP era
   ═══════════════════════════════════════════════ */
.gr-button, button.gr-button {
  font-family: var(--font-ui) !important;
  font-weight: 700 !important;
  font-size: 11px !important;
  letter-spacing: 0.02em !important;
  border-radius: var(--radius-sm) !important;
  cursor: pointer !important;
  transition: none !important;
  text-transform: uppercase !important;
  padding: 5px 14px !important;
  background: var(--btn-face) !important;
  border-top: 1px solid var(--border-hi) !important;
  border-left: 1px solid var(--border-hi) !important;
  border-right: 1px solid var(--border-lo) !important;
  border-bottom: 1px solid var(--border-lo) !important;
  color: var(--text-bright) !important;
  box-shadow: inset 0 1px 0 rgba(100,100,100,0.3) !important;
}
.gr-button:hover, button.gr-button:hover {
  filter: brightness(1.15) !important;
  transform: none !important;
}
.gr-button:active, button.gr-button:active {
  background: var(--btn-pressed) !important;
  border-top: 1px solid var(--border-lo) !important;
  border-left: 1px solid var(--border-lo) !important;
  border-right: 1px solid var(--border-hi) !important;
  border-bottom: 1px solid var(--border-hi) !important;
  box-shadow: inset 1px 1px 3px rgba(0,0,0,0.6) !important;
}
.gr-button.primary, button.primary {
  background: linear-gradient(to bottom, #4a7fe8 0%, #2a5cc8 50%, #1e4db0 51%, #3366dd 100%) !important;
  border-top: 1px solid #6699ff !important;
  border-left: 1px solid #5588ee !important;
  border-right: 1px solid #1133aa !important;
  border-bottom: 1px solid #1133aa !important;
  color: #ffffff !important;
  box-shadow: inset 0 1px 0 rgba(120,160,255,0.4) !important;
}
.gr-button.primary:active {
  background: linear-gradient(to bottom, #1e4db0 0%, #2a5cc8 50%, #4a7fe8 100%) !important;
}
.gr-button.secondary, button.secondary {
  background: var(--btn-face) !important;
  color: var(--text-bright) !important;
}
.gr-button.stop, button.stop {
  background: linear-gradient(to bottom, #dd5544 0%, #bb2211 50%, #991100 51%, #cc3322 100%) !important;
  border-top: 1px solid #ff7766 !important;
  border-left: 1px solid #ee6655 !important;
  border-right: 1px solid #880000 !important;
  border-bottom: 1px solid #880000 !important;
  color: #ffffff !important;
}

/* ═══════════════════════════════════════════════
   CHECKBOXES & RADIOS
   ═══════════════════════════════════════════════ */
input[type="checkbox"] { accent-color: var(--accent) !important; }
input[type="radio"]    { accent-color: var(--accent) !important; }
.gr-check-radio { color: var(--text-dim) !important; font-size: 11px !important; }

/* ═══════════════════════════════════════════════
   ACCORDIONS -- title bar style
   ═══════════════════════════════════════════════ */
.gr-accordion {
  border-top: 1px solid var(--border-hi) !important;
  border-left: 1px solid var(--border-hi) !important;
  border-right: 1px solid var(--border-lo) !important;
  border-bottom: 1px solid var(--border-lo) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--panel) !important;
  box-shadow: none !important;
}
.gr-accordion > .label-wrap {
  background: linear-gradient(to bottom, #3e3e3e 0%, #2e2e2e 100%) !important;
  border-bottom: 1px solid var(--border-lo) !important;
  border-radius: 0 !important;
  padding: 6px 12px !important;
}
.gr-accordion > .label-wrap span {
  color: var(--text-bright) !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  font-family: var(--font-ui) !important;
}

/* ═══════════════════════════════════════════════
   DATAFRAME -- grid / table
   ═══════════════════════════════════════════════ */
.gr-dataframe table {
  background: var(--sunken) !important;
  color: var(--text) !important;
  font-size: 11px !important;
  font-family: var(--font-mono) !important;
  border-collapse: collapse !important;
}
.gr-dataframe th {
  background: linear-gradient(to bottom, #404040, #303030) !important;
  color: var(--text-bright) !important;
  font-weight: 700 !important;
  font-size: 10px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
  border: 1px solid var(--border-lo) !important;
  padding: 4px 8px !important;
}
.gr-dataframe td {
  border: 1px solid var(--border-mid) !important;
  padding: 3px 6px !important;
}

/* ═══════════════════════════════════════════════
   FILE UPLOAD + AUDIO
   ═══════════════════════════════════════════════ */
.gr-file {
  background: var(--sunken) !important;
  border: 1px dashed var(--border-mid) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-dim) !important;
}
.gr-audio {
  background: var(--panel-lo) !important;
  border-radius: var(--radius-sm) !important;
}

/* ═══════════════════════════════════════════════
   HOME SCREEN
   ═══════════════════════════════════════════════ */
#home-screen {
  background: #161616;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0;
  min-height: 92vh;
  padding: 40px 20px;
}
#home-wordmark {
  font-family: var(--font-ui);
  font-size: 26px;
  font-weight: 700;
  letter-spacing: 0.22em;
  color: #e8e8e8;
  text-transform: uppercase;
  margin-bottom: 8px;
}
#home-wordmark span { color: #6eaaff; margin-left: 4px; }
#home-sub {
  font-family: var(--font-ui);
  font-size: 10px;
  letter-spacing: 0.18em;
  color: #555;
  text-transform: uppercase;
  margin-bottom: 64px;
}
#home-cards {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  justify-content: center;
  max-width: 1120px;
}

/* -- Live Recorder card -- RED identity -- */
#card-live {
  width: 340px;
  height: 300px;
  background: linear-gradient(160deg, #2a1010 0%, #1a0a0a 60%, #120000 100%);
  border-top: 3px solid #cc3333;
  border-left: 3px solid #cc3333;
  border-right: 3px solid #550000;
  border-bottom: 3px solid #550000;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 36px 28px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 6px 24px rgba(180,0,0,0.25), 4px 4px 12px rgba(0,0,0,0.8);
  transition: box-shadow 0.15s, filter 0.15s;
}
#card-live::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(to right, #ff4444, #ff8888, #ff4444);
}
#card-live:hover {
  filter: brightness(1.2);
  box-shadow: 0 6px 32px rgba(220,40,40,0.45), 4px 4px 12px rgba(0,0,0,0.8);
}
#card-live:active {
  filter: brightness(0.85);
  border-top-color: #550000;
  border-left-color: #550000;
  border-right-color: #cc3333;
  border-bottom-color: #cc3333;
  box-shadow: inset 3px 3px 10px rgba(0,0,0,0.6);
}

/* -- Orchestral Composer card -- BLUE identity -- */
#card-composer {
  width: 340px;
  height: 300px;
  background: linear-gradient(160deg, #0d1826 0%, #080f1a 60%, #040a12 100%);
  border-top: 3px solid #2a6bc8;
  border-left: 3px solid #2a6bc8;
  border-right: 3px solid #0a2040;
  border-bottom: 3px solid #0a2040;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 36px 28px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 6px 24px rgba(0,80,200,0.2), 4px 4px 12px rgba(0,0,0,0.8);
  transition: box-shadow 0.15s, filter 0.15s;
}
#card-composer::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(to right, #4488ff, #88bbff, #4488ff);
}
#card-composer:hover {
  filter: brightness(1.2);
  box-shadow: 0 6px 32px rgba(40,100,220,0.4), 4px 4px 12px rgba(0,0,0,0.8);
}
#card-composer:active {
  filter: brightness(0.85);
  border-top-color: #0a2040;
  border-left-color: #0a2040;
  border-right-color: #2a6bc8;
  border-bottom-color: #2a6bc8;
  box-shadow: inset 3px 3px 10px rgba(0,0,0,0.6);
}

/* -- Card internals -- */
.home-card-icon {
  font-size: 52px;
  line-height: 1;
}
#card-live .home-card-icon { color: #ff5555; filter: drop-shadow(0 0 8px rgba(255,60,60,0.6)); }
#card-composer .home-card-icon { color: #5599ff; filter: drop-shadow(0 0 8px rgba(60,120,255,0.5)); }

.home-card-title {
  font-family: var(--font-ui);
  font-size: 15px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
#card-live .home-card-title { color: #ff8888; }
#card-composer .home-card-title { color: #88bbff; }

.home-card-desc {
  font-family: var(--font-ui);
  font-size: 11px;
  color: #666;
  text-align: center;
  line-height: 1.7;
}
#card-live .home-card-desc { color: #8a5555; }
#card-composer .home-card-desc { color: #3a5580; }

/* -- GarSIa card -- GREEN identity -- */
#card-garcia {
  background: linear-gradient(160deg, #0f2a1a 0%, #1a3a25 60%, #0a1f12 100%);
  border: 1px solid #1a5c30;
  box-shadow: var(--shadow-card), 0 0 18px rgba(30,180,80,0.08);
}
#card-garcia::after {
  background: radial-gradient(ellipse at 60% 30%, rgba(30,200,80,0.18) 0%, transparent 70%);
}
#card-garcia:hover { border-color: #2aaa55; box-shadow: var(--shadow-card), 0 0 28px rgba(30,200,80,0.22); }
#card-garcia:active { transform: translateY(1px); }
#card-garcia .home-card-icon { color: #33cc66; filter: drop-shadow(0 0 8px rgba(30,200,80,0.6)); }
#card-garcia .home-card-title { color: #66dd99; }
#card-garcia .home-card-desc { color: #2d6644; }

/* -- GarSIa section topbar -- */
#section-garcia .section-topbar {
  background: linear-gradient(to bottom, #0f2a1a, #0a1f12);
  border-bottom: 1px solid #1a5c30;
}
#section-garcia .section-back-btn {
  background: linear-gradient(to bottom, #1a3a25, #0f2a1a);
  color: #66dd99; border: 1px solid #1a5c30;
}
#section-garcia .section-back-btn:hover { background: linear-gradient(to bottom, #1e4a2e, #0f2a1a); color: #99ffbb; }
#section-garcia .section-topbar-title { color: #33cc66; }

/* ── GarSIa main layout ── */
#garcia-layout {
  display: flex;
  height: calc(100vh - 48px);
  background: #080f0a;
  overflow: hidden;
}

/* Left preset panel */
#garcia-preset {
  width: 210px;
  min-width: 210px;
  background: rgba(8,20,12,0.98);
  border-right: 1px solid #142a1a;
  padding: 14px 12px;
  display: flex;
  flex-direction: column;
  gap: 5px;
  overflow-y: auto;
}
.garcia-label {
  font-size: 0.58rem;
  letter-spacing: 0.16em;
  color: #1a6633;
  font-weight: 700;
  text-transform: uppercase;
  margin-top: 10px;
}
.garcia-label:first-child { margin-top: 0; }
.garcia-big {
  font-size: 1.7rem;
  font-weight: 700;
  font-family: monospace;
  line-height: 1.1;
}
.garcia-big.green  { color: #55dd88; }
.garcia-big.bright { color: #33cc66; }
.garcia-hint {
  font-size: 0.65rem;
  color: #1a4a28;
  font-family: monospace;
  line-height: 1.4;
}
.garcia-guide {
  font-size: 0.72rem;
  color: #2a6640;
  font-family: monospace;
  line-height: 1.9;
}
.gg-icon { display: inline-block; width: 20px; }
#garcia-preset select, #garcia-preset input[type=text] {
  width: 100%;
  background: #0a1f12;
  border: 1px solid #1a5c30;
  color: #66dd99;
  padding: 6px 8px;
  font-family: monospace;
  font-size: 0.78rem;
  border-radius: 2px;
  box-sizing: border-box;
}
#garcia-preset input[type=range] {
  accent-color: #33cc66;
}

/* Right: camera + layers */
#garcia-right {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
#garcia-viewport {
  flex: 1;
  position: relative;
  background: #000;
  overflow: hidden;
}
#garcia-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
#garcia-start-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
  background: rgba(0,0,0,0.7);
}
.garcia-start-glyph {
  font-size: 4rem;
  filter: drop-shadow(0 0 20px rgba(30,200,80,0.5));
  animation: garcia-pulse 2s ease-in-out infinite;
}
@keyframes garcia-pulse {
  0%,100%{ filter:drop-shadow(0 0 10px rgba(30,200,80,0.3)); }
  50%    { filter:drop-shadow(0 0 28px rgba(30,200,80,0.7)); }
}
#garcia-start-btn {
  background: #122a18;
  border: 1px solid #33cc66;
  color: #33cc66;
  font-size: 1.05rem;
  font-family: monospace;
  letter-spacing: 0.1em;
  padding: 14px 40px;
  border-radius: 3px;
  cursor: pointer;
  transition: box-shadow 0.15s, background 0.15s;
}
#garcia-start-btn:hover {
  background: #1a3a22;
  box-shadow: 0 0 20px rgba(30,200,80,0.35);
}
#garcia-start-btn:disabled { opacity: 0.5; cursor: default; }

/* Layer bar */
#garcia-layers {
  display: flex;
  height: 78px;
  border-top: 1px solid #142a1a;
  flex-shrink: 0;
}
.garcia-layer {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  border-right: 1px solid #0a180e;
  background: #080f0a;
  transition: background 0.25s, box-shadow 0.25s;
}
.garcia-layer:last-child { border-right: none; }
.garcia-layer.active {
  background: color-mix(in srgb, var(--lc) 12%, #080f0a);
  box-shadow: inset 0 0 24px color-mix(in srgb, var(--lc) 18%, transparent);
}
.garcia-layer.active .gl-name { color: var(--lc); }
.garcia-layer.active .gl-icon {
  filter: drop-shadow(0 0 6px var(--lc));
}
.gl-icon  { font-size: 1.3rem; }
.gl-name  { font-size: 0.55rem; letter-spacing: 0.12em; color: #1a4a22; font-weight: 700; }
.gl-hint  { font-size: 0.75rem; color: #0f2a18; }
.garcia-style-name { font-size: 0.48rem; letter-spacing: 0.08em; color: #1e5a2a; text-align: center; min-height: 0.7rem; }
.garcia-active-target { outline: 1px solid color-mix(in srgb, var(--lc) 40%, transparent); }
.garcia-active-target .garcia-style-name { color: var(--lc); }

/* -- click-hint label under each card -- */
.home-card-hint {
  font-family: var(--font-ui);
  font-size: 9px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-top: -4px;
  padding: 5px 18px;
  border-radius: 2px;
}
#card-live .home-card-hint { color: #993333; border: 1px solid #441111; background: #1a0808; }
#card-composer .home-card-hint { color: #224488; border: 1px solid #112233; background: #060e18; }

#home-footer {
  margin-top: 52px;
  font-family: var(--font-ui);
  font-size: 9px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #333;
}

/* ═══════════════════════════════════════════════
   SECTION TOPBAR
   Live Recorder = red-tinted  |  Composer = blue-tinted
   ═══════════════════════════════════════════════ */
.section-topbar {
  display: flex;
  align-items: center;
  gap: 0;
  height: 36px;
  margin-bottom: 10px;
  border-radius: var(--radius-sm);
  overflow: hidden;
  border-bottom: 2px solid #0a0a0a;
}
#section-live .section-topbar {
  background: linear-gradient(to bottom, #2a1010 0%, #180808 100%);
  border-top: 1px solid #993333;
  border-left: 1px solid #993333;
  border-right: 1px solid #330000;
}
#section-composer .section-topbar {
  background: linear-gradient(to bottom, #0d1826 0%, #060f1a 100%);
  border-top: 1px solid #2255aa;
  border-left: 1px solid #2255aa;
  border-right: 1px solid #061020;
}
.section-back-btn {
  height: 100%;
  padding: 0 16px;
  border: none;
  font-family: var(--font-ui);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
  white-space: nowrap;
}
#section-live .section-back-btn {
  background: linear-gradient(to bottom, #3a1515 0%, #220a0a 100%);
  color: #cc4444;
  border-right: 1px solid #551111;
}
#section-live .section-back-btn:hover { background: linear-gradient(to bottom, #551818, #330d0d); color: #ff6666; }
#section-composer .section-back-btn {
  background: linear-gradient(to bottom, #0e1c34 0%, #070e1c 100%);
  color: #4488cc;
  border-right: 1px solid #0a1830;
}
#section-composer .section-back-btn:hover { background: linear-gradient(to bottom, #122040, #0a1428); color: #66aaff; }

.section-topbar-title {
  flex: 1;
  padding: 0 16px;
  font-family: var(--font-ui);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
#section-live .section-topbar-title { color: #dd5555; }
#section-composer .section-topbar-title { color: #5599ee; }

.section-topbar-status {
  padding: 0 12px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--green-led);
  background: var(--lcd-bg);
  border-left: 1px solid var(--border-lo);
  height: 100%;
  display: flex;
  align-items: center;
  letter-spacing: 0.04em;
}

/* Strip Group chrome from section wrappers and home screen container */
#section-live, #section-composer, #section-garcia {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  border-radius: 0 !important;
}
/* Home screen: strip Gradio component wrapper padding */
#home-screen {
  width: 100% !important;
}
.gradio-html:has(> #home-screen) {
  padding: 0 !important;
  margin: 0 !important;
}

/* ═══════════════════════════════════════════════
   STUDIO HERO -- split panel
   ═══════════════════════════════════════════════ */
#studio-hero {
  background: var(--panel-lo);
  border-top: 1px solid var(--border-hi);
  border-left: 1px solid var(--border-hi);
  border-right: 1px solid var(--border-lo);
  border-bottom: 1px solid var(--border-lo);
  border-radius: var(--radius-sm);
  overflow: hidden;
  margin-bottom: 8px;
}

#studio-tabs {
  display: flex;
  border-bottom: 1px solid var(--border-lo);
  background: linear-gradient(to bottom, #333 0%, #222 100%);
}
.studio-tab-btn {
  flex: 1;
  padding: 7px 16px;
  background: transparent;
  border: none;
  border-right: 1px solid var(--border-lo);
  border-bottom: 2px solid transparent;
  color: var(--text-dim);
  font-family: var(--font-ui);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  cursor: pointer;
}
.studio-tab-btn.active {
  color: var(--text-bright);
  border-bottom-color: var(--accent-hi);
  background: rgba(58,123,213,0.12);
}
.studio-tab-btn:hover:not(.active) { color: var(--text); background: rgba(255,255,255,0.04); }
.studio-tab-btn .tab-icon { font-size: 11px; margin-right: 5px; }

.studio-panel { display: none; padding: 14px 16px 16px; }
.studio-panel.visible { display: block; }

@media (min-width: 900px) {
  #studio-panels { display: flex; }
  .studio-panel { flex: 1; display: block !important; border-left: 1px solid var(--border-lo); }
  .studio-panel:first-child { border-left: none; }
  #studio-tabs { display: none; }
  #studio-sidebyside-label { display: flex; border-bottom: 1px solid var(--border-lo); background: linear-gradient(to bottom, #333, #252525); }
}
@media (max-width: 899px) { #studio-sidebyside-label { display: none; } }

.studio-col-label {
  flex: 1;
  padding: 6px 16px;
  font-family: var(--font-ui);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent-hi);
  border-left: 1px solid var(--border-lo);
}
.studio-col-label:first-child { border-left: none; }

#panel-palette { background: transparent; }
#palette-hero { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
#palette-hero .form { background: transparent !important; border: none !important; }
#palette-hero label span, #palette-hero .block label, #palette-hero .gr-markdown p { color: var(--text-dim) !important; }
#palette-hero input, #palette-hero select { background: var(--sunken) !important; color: var(--text) !important; border-color: var(--border-lo) !important; }

/* Hide native Gradio palette controls -- replaced by custom picker */
#pal-gradio-controls, #pal-custom-gradio { display: none !important; }
#chord_picker { display: none !important; }

/* -- Custom palette picker ------------------------------------------- */
#pal-picker { padding: 4px 0 12px; }
.pal-section { margin-bottom: 14px; }
.pal-label { font-size: 11px; font-weight: 500; letter-spacing: .07em; text-transform: uppercase;
  color: var(--text-dim, #888); margin-bottom: 8px; }
.pal-notes { display: flex; flex-wrap: wrap; gap: 5px; }
.pal-note { width: 42px; height: 40px; border-radius: 7px; border: 1px solid var(--border-lo, #444);
  background: var(--sunken, #222); color: var(--text, #ddd); font-size: 12px; font-weight: 500;
  cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all .1s; }
.pal-note:hover { border-color: var(--border-hi, #888); background: var(--surface, #2a2a2a); }
.pal-note.acc { opacity: .75; font-size: 11px; }
.pal-note.on { background: #185FA5; border-color: #378ADD; color: #fff; opacity: 1; }
.pal-modes { display: flex; flex-wrap: wrap; gap: 6px; }
.pal-mode { padding: 5px 13px; border-radius: 20px; font-size: 12px; font-weight: 500;
  border: 1px solid var(--border-lo, #444); background: var(--sunken, #222);
  color: var(--text-dim, #888); cursor: pointer; white-space: nowrap; transition: all .1s; }
.pal-mode:hover { color: var(--text, #ddd); border-color: var(--border-hi, #888); }
.pal-mode.on { background: #185FA5; border-color: #378ADD; color: #fff; }

/* -- Custom chord tile grid ------------------------------------------- */
#pal-tile-grid { display: flex; flex-wrap: wrap; gap: 6px; padding: 4px 0 10px; min-height: 30px; }
.pal-tile { display: flex; flex-direction: column; align-items: center; justify-content: center;
  width: 76px; height: 62px; border-radius: 8px; cursor: pointer; transition: all .1s;
  border: 1.5px solid transparent; background: var(--sunken, #222); }
.pal-tile:hover { border-color: var(--border-hi, #666); }
.pal-tile.on { border-color: #378ADD; background: rgba(24,95,165,.18); }
.pal-tile .pt-deg { font-size: 10px; font-weight: 500; color: var(--text-dim, #888); margin-bottom: 2px; }
.pal-tile .pt-sym { font-size: 13px; font-weight: 600; color: var(--text, #ddd); }
.pal-tile .pt-qual { font-size: 10px; color: var(--text-dim, #888); margin-top: 1px; }
.pal-tile.on .pt-deg, .pal-tile.on .pt-sym { color: #7eb8f7; }
.pal-tile[data-fn="T"].on { border-color: #639922; background: rgba(99,153,34,.15); }
.pal-tile[data-fn="T"].on .pt-sym { color: #a3d45a; }
.pal-tile[data-fn="P"].on { border-color: #7F77DD; background: rgba(127,119,221,.15); }
.pal-tile[data-fn="P"].on .pt-sym { color: #b8b4f0; }
.pal-tile[data-fn="D"].on { border-color: #378ADD; background: rgba(55,138,221,.15); }
.pal-tile[data-fn="D"].on .pt-sym { color: #7eb8f7; }
.pal-tile[data-fn="B"].on { border-color: #D85A30; background: rgba(216,90,48,.15); }
.pal-tile[data-fn="B"].on .pt-sym { color: #f09970; }

/* Sequence pill row */
#pal-seq-row { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; min-height: 34px; margin-bottom: 10px; }
.pal-seq-pill { display: flex; align-items: center; gap: 5px; padding: 4px 8px 4px 12px;
  border-radius: 20px; background: var(--sunken, #222); border: 1px solid var(--border-lo, #444);
  font-size: 13px; font-weight: 500; color: var(--text, #ddd); cursor: grab; user-select: none; }
.pal-seq-x { font-size: 15px; color: var(--text-dim, #888); cursor: pointer; line-height: 1; }
.pal-seq-x:hover { color: #e2534a; }
.pal-seq-pill.custom { border-style: dashed; border-color: #5a8abf; color: #9ecbf7; }
.pal-seq-label { font-size: 11px; font-weight: 500; letter-spacing: .07em; text-transform: uppercase;
  color: var(--text-dim, #888); margin-bottom: 6px; margin-top: 4px; }
#pal-add-chord-row { display:flex; align-items:center; gap:6px; margin-top:8px; }
#pal-add-input { flex:1; padding:5px 10px; border-radius:6px; border:.5px solid #555;
  background:#1a1a2e; color:inherit; font-family:inherit; font-size:13px; }
#pal-add-btn { padding:5px 13px; border-radius:6px; background:#1a3d6e; color:#9ecbf7;
  border:1px solid #5a8abf; cursor:pointer; font-size:13px; white-space:nowrap; }
#pal-add-btn:hover { background:#1e5099; }
#pal-add-err { font-size:11px; color:#e2534a; white-space:nowrap; }

/* ═══════════════════════════════════════════════
   LIVE ACCOMPANIMENT -- hardware console
   ═══════════════════════════════════════════════ */
#live-console {
  background: var(--panel-lo);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
}
.lc-section-title {
  font-family: var(--font-ui) !important;
  font-size: 9px !important;
  font-weight: 700 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--accent-hi) !important;
  margin-bottom: 8px;
}
.ac-lbl {
  font-family: var(--font-ui) !important;
  color: var(--text-label) !important;
  font-size: 9px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  margin-bottom: 3px;
}
.ac-val {
  font-family: var(--font-mono) !important;
  color: var(--green-led) !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  margin-top: 2px;
}
.ac-sel {
  width: 100%;
  background: var(--sunken) !important;
  color: var(--text) !important;
  border-top: 1px solid var(--border-lo) !important;
  border-left: 1px solid var(--border-lo) !important;
  border-right: 1px solid var(--border-hi) !important;
  border-bottom: 1px solid var(--border-hi) !important;
  border-radius: var(--radius-sm) !important;
  padding: 3px 6px;
  font-size: 11px;
  font-family: var(--font-ui);
  cursor: pointer;
}
.ac-sel option { background: #1e1e1e !important; color: var(--text) !important; }
.dens-row {
  display: flex;
  gap: 2px;
  margin-top: 4px;
  width: 100%;
}
.dens-btn {
  flex: 1;
  background: var(--sunken);
  color: var(--text-dim);
  border: 1px solid var(--border-mid);
  border-radius: 3px;
  font-size: 8px;
  font-family: var(--font-ui);
  font-weight: 600;
  padding: 3px 0;
  cursor: pointer;
  letter-spacing: 0.03em;
  transition: background 0.1s, color 0.1s;
}
.dens-btn:hover { background: var(--panel); color: var(--text); }
.dens-btn.active { background: #2a4a6b; color: #6db3f2; border-color: #4a7fb5; }

/* Transport buttons -- hardware style */
.ac-btn {
  border-radius: var(--radius-sm);
  padding: 7px 16px;
  font-size: 10px;
  font-weight: 700;
  font-family: var(--font-ui);
  letter-spacing: 0.06em;
  cursor: pointer;
  text-transform: uppercase;
  background: var(--btn-face);
  border-top: 1px solid var(--border-hi);
  border-left: 1px solid var(--border-hi);
  border-right: 1px solid var(--border-lo);
  border-bottom: 1px solid var(--border-lo);
  color: var(--text-bright);
}
.ac-btn:active {
  background: var(--btn-pressed);
  border-top: 1px solid var(--border-lo);
  border-left: 1px solid var(--border-lo);
  border-right: 1px solid var(--border-hi);
  border-bottom: 1px solid var(--border-hi);
}
.btn-play {
  background: linear-gradient(to bottom, #2a6e38 0%, #1a4a24 50%, #133d1c 51%, #226030 100%);
  border-top: 1px solid #449955;
  border-left: 1px solid #338844;
  border-right: 1px solid #0a2210;
  border-bottom: 1px solid #0a2210;
  color: #88ee99;
}
.btn-stop {
  background: linear-gradient(to bottom, #6e2a22 0%, #4a1a14 50%, #3d1310 51%, #602218 100%);
  border-top: 1px solid #995544;
  border-left: 1px solid #884433;
  border-right: 1px solid #220a08;
  border-bottom: 1px solid #220a08;
  color: #ffaa99;
}
.btn-rec {
  background: var(--btn-face);
  color: var(--red-led);
  border-top: 1px solid var(--border-hi);
  border-left: 1px solid var(--border-hi);
  border-right: 1px solid var(--border-lo);
  border-bottom: 1px solid var(--border-lo);
  outline: 1px solid var(--red-led);
  outline-offset: -3px;
}
.btn-rec.recording { animation: rec-pulse 1.2s ease-in-out infinite; }
.btn-tap {
  background: var(--btn-face);
  color: var(--accent-hi);
  border-top: 1px solid var(--border-hi);
  border-left: 1px solid var(--border-hi);
  border-right: 1px solid var(--border-lo);
  border-bottom: 1px solid var(--border-lo);
}

@keyframes rec-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(221,51,34,0.5); }
  50%       { box-shadow: 0 0 0 5px rgba(221,51,34,0); }
}

#now-playing-strip {
  flex: 1;
  background: var(--lcd-bg);
  border-radius: var(--radius-sm);
  padding: 5px 10px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: #336633;
  border-top: 1px solid var(--border-lo);
  border-left: 1px solid var(--border-lo);
  border-right: 1px solid var(--border-hi);
  border-bottom: 1px solid var(--border-hi);
  box-shadow: inset 1px 1px 4px rgba(0,0,0,0.8);
}
#now-playing-strip span { color: var(--lcd-text); font-weight: 700; }

.mixer-strip {
  flex: 1;
  text-align: center;
  min-width: 60px;
  padding: 6px 4px;
  background: var(--panel-lo);
  border-radius: var(--radius-sm);
  border-top: 1px solid var(--border-hi);
  border-left: 1px solid var(--border-hi);
  border-right: 1px solid var(--border-lo);
  border-bottom: 1px solid var(--border-lo);
}
.mixer-strip .ac-lbl { font-size: 8px !important; margin-bottom: 5px; }

/* Listen buttons */
/* -- Loop Pedal -- */
.loop-slot {
  background: var(--sunken);
  border: 1px solid var(--border-mid);
  border-radius: 8px;
  padding: 8px 10px;
  min-width: 110px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  transition: border-color 0.15s;
}
.loop-slot.recording { border-color: #e55; box-shadow: 0 0 8px rgba(220,60,60,0.4); }
.loop-slot.looping   { border-color: #5b3; box-shadow: 0 0 8px rgba(80,200,60,0.3); }
.loop-slot.overdub   { border-color: #f90; box-shadow: 0 0 8px rgba(255,150,0,0.4); }
.loop-slot.muted     { opacity: 0.45; }
.loop-slot-label { font-size: 18px; font-weight: 800; color: var(--text-bright); font-family: var(--font-ui); line-height:1; }
.loop-slot-time  { font-size: 10px; font-family:'JetBrains Mono',monospace; color: var(--text-dim); }
.loop-wave { display:block; background:rgba(255,255,255,0.03); border-radius:3px; }
.loop-rec-btn, .loop-dub-btn, .loop-mute-btn, .loop-clear-btn {
  flex:1; padding:3px 0; font-size:11px; border-radius:4px; cursor:pointer;
  border:1px solid var(--border-mid); background:var(--panel); color:var(--text-dim);
  font-family:var(--font-ui); transition: background 0.1s, color 0.1s;
}
.loop-rec-btn:hover   { background:#7b1515; color:#ff8080; border-color:#c33; }
.loop-rec-btn.active  { background:#c0392b; color:#fff; border-color:#e55; animation: pulse-rec 0.8s infinite; }
.loop-dub-btn.active  { background:#b07300; color:#ffe; border-color:#f90; }
.loop-mute-btn.active { background:#445; color:#aac; }
.loop-clear-btn:hover { background:#3a1515; color:#f88; }
@keyframes pulse-rec { 0%,100%{opacity:1} 50%{opacity:0.65} }
.loop-global-btn {
  padding:4px 10px; font-size:9px; font-weight:700; letter-spacing:.05em;
  font-family:var(--font-ui); text-transform:uppercase;
  background:var(--sunken); color:var(--text-dim); border:1px solid var(--border-mid);
  border-radius:4px; cursor:pointer; white-space:nowrap;
}
.loop-global-btn:hover { color:var(--text); border-color:var(--border-hi); }

.lst-btn {
  background: var(--btn-face);
  color: var(--text-dim);
  border-top: 1px solid var(--border-hi);
  border-left: 1px solid var(--border-hi);
  border-right: 1px solid var(--border-lo);
  border-bottom: 1px solid var(--border-lo);
  border-radius: var(--radius-sm);
  padding: 4px 12px;
  font-size: 10px;
  font-weight: 700;
  font-family: var(--font-ui);
  cursor: pointer;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.lst-btn.active, .lst-btn:focus {
  background: linear-gradient(to bottom, #2a5cc8, #1e4db0) !important;
  color: #fff !important;
  border-top: 1px solid #6699ff !important;
  border-left: 1px solid #5588ee !important;
  border-right: 1px solid #1133aa !important;
  border-bottom: 1px solid #1133aa !important;
}
.lst-btn:hover:not(.active) { color: var(--text-bright); filter: brightness(1.15); }

/* -- Bridge buttons: Crystallize & Ghost Orchestra -- */
.bridge-btn {
  display: flex; flex-direction: column; align-items: flex-start;
  gap: 1px; padding: 10px 14px; border-radius: 8px; cursor: pointer;
  border: 1px solid; font-family: var(--font-ui); transition: all 0.18s ease;
  min-width: 160px; flex: 1;
}
.bridge-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.bridge-btn.loading { animation: bridge-pulse 1.2s ease-in-out infinite; }
@keyframes bridge-pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
.bridge-btn-icon { font-size: 18px; line-height: 1; margin-bottom: 2px; }
.bridge-btn-label { font-size: 12px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
.bridge-btn-sub { font-size: 9px; font-weight: 400; opacity: 0.65; letter-spacing: .03em; }

/* Crystallize -- warm gold/amber */
.crystallize-btn {
  background: linear-gradient(135deg, #1a1200 0%, #2a1e00 100%);
  border-color: #6b4f00; color: #f0c040;
}
.crystallize-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, #2a1e00 0%, #3a2c00 100%);
  border-color: #c89a20; box-shadow: 0 0 14px rgba(200,150,0,0.25);
}


/* ═══════════════════════════════════════════════
   COMPOSER STUDIO ZONE
   ═══════════════════════════════════════════════ */
#composer-studio {
  border-top: 2px solid var(--border-lo);
  margin-top: 4px;
  padding-top: 12px;
}
#composer-studio-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
#composer-studio-wordmark {
  font-family: var(--font-ui);
  font-size: 10px;
  font-weight: 700;
  color: var(--text-dim);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
#composer-studio-rule { flex: 1; height: 1px; background: var(--border-mid); }
#composer-studio-badge {
  font-family: var(--font-ui);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent-hi);
  background: rgba(58,123,213,0.15);
  border: 1px solid rgba(58,123,213,0.4);
  border-radius: var(--radius-sm);
  padding: 2px 8px;
}

/* Generate button -- prominent DAW-style */
#generate-btn-wrap .gr-button.primary {
  font-family: var(--font-ui) !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 12px 28px !important;
  background: linear-gradient(to bottom, #4a7fe8 0%, #2a5cc8 50%, #1e4db0 51%, #3366dd 100%) !important;
  border-top: 1px solid #6699ff !important;
  border-left: 1px solid #5588ee !important;
  border-right: 1px solid #1133aa !important;
  border-bottom: 1px solid #1133aa !important;
  border-radius: var(--radius-sm) !important;
  box-shadow: inset 0 1px 0 rgba(120,160,255,0.4) !important;
  color: #fff !important;
}
#generate-btn-wrap .gr-button.primary:active {
  background: linear-gradient(to bottom, #1e4db0 0%, #2a5cc8 51%, #4a7fe8 100%) !important;
}

/* Arrange bridge button -- teal accent to signal cross-module action */
#arrange-btn {
  background: linear-gradient(to bottom, #1a5a4a 0%, #0f3a30 50%, #0a2a22 51%, #153d30 100%) !important;
  border-top: 1px solid #33aa88 !important;
  border-left: 1px solid #22997a !important;
  border-right: 1px solid #0a3025 !important;
  border-bottom: 1px solid #0a3025 !important;
  color: #55ddaa !important;
  font-family: var(--font-ui) !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border-radius: var(--radius-sm) !important;
}
#arrange-btn:hover {
  background: linear-gradient(to bottom, #226655 0%, #144433 50%, #0f3028 51%, #1c4a38 100%) !important;
  color: #77ffcc !important;
}
#arrange-btn:active {
  background: linear-gradient(to bottom, #0a2a22 0%, #0f3a30 51%, #1a5a4a 100%) !important;
  border-top-color: #0a3025 !important;
  border-left-color: #0a3025 !important;
  border-right-color: #33aa88 !important;
  border-bottom-color: #33aa88 !important;
}

/* Output status -- LCD readout */
#out-status textarea {
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  color: var(--lcd-text) !important;
  background: var(--lcd-bg) !important;
  border-top: 1px solid var(--border-lo) !important;
  border-left: 1px solid var(--border-lo) !important;
  border-right: 1px solid var(--border-hi) !important;
  border-bottom: 1px solid var(--border-hi) !important;
  box-shadow: inset 1px 1px 6px rgba(0,0,0,0.8) !important;
}

/* ═══════════════════════════════════════════════
   SCROLLBARS -- thin dark
   ═══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--sunken); border: 1px solid var(--border-lo); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(to bottom, #505050, #383838);
  border-top: 1px solid var(--border-hi);
  border-left: 1px solid var(--border-hi);
  border-right: 1px solid var(--border-lo);
  border-bottom: 1px solid var(--border-lo);
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(to bottom, #606060, #484848); }

/* ═══════════════════════════════════════════════
   WORKFLOW SECTION DIVIDERS
   ═══════════════════════════════════════════════ */
.oc-section-div {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 12px 0 6px;
}
.oc-section-div::before, .oc-section-div::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border-mid);
}
.oc-section-div span {
  font-family: var(--font-ui) !important;
  font-size: 9px !important;
  font-weight: 700 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--accent-hi) !important;
  white-space: nowrap;
  background: var(--panel);
  padding: 0 6px;
}

/* -- Key + Tempo always-visible group -- */
#key-tempo-group {
  background: var(--panel-lo) !important;
  border-top: 1px solid var(--border-lo) !important;
  border-left: 1px solid var(--border-lo) !important;
  border-right: 1px solid var(--border-hi) !important;
  border-bottom: 1px solid var(--border-hi) !important;
  border-radius: var(--radius-sm) !important;
  padding: 8px 10px !important;
  margin-bottom: 6px !important;
  box-shadow: var(--shadow-inset) !important;
}
#key-tempo-group label { color: var(--accent-hi) !important; }

/* -- Indian mode banner -- */
#indian-mode-banner .gr-markdown, #indian-mode-banner p {
  background: rgba(26, 68, 34, 0.5) !important;
  border-top: 1px solid var(--tape-green) !important;
  border-left: 1px solid var(--tape-green) !important;
  border-right: 1px solid #0a1a0e !important;
  border-bottom: 1px solid #0a1a0e !important;
  border-radius: var(--radius-sm) !important;
  padding: 8px 12px !important;
  color: var(--green-led) !important;
  font-size: 11px !important;
  font-family: var(--font-mono) !important;
  margin: 4px 0 8px !important;
}
"""

with gr.Blocks(title="Orchestral Composer", css=_CSS, js=_TOOLTIP_JS + "\n" + _GARCIA_JS) as demo:

    # -- Home screen overlay ---------------------------------------------------
    gr.HTML("""
<div id="home-screen">
  <div id="home-wordmark">ORCHESTRAL<span>COMPOSER</span></div>
  <div id="home-sub">Professional AI Music Production Suite</div>
  <div id="home-cards">
    <button id="card-live" onclick="window.ocNav('live')">
      <div class="home-card-icon">⏺</div>
      <div class="home-card-title">LIVE RECORDER</div>
      <div class="home-card-desc">Chord palette · real-time accompaniment<br>MIDI listen-in · mic input · live playback</div>
      <div class="home-card-hint">Click to open</div>
    </button>
    <button id="card-composer" onclick="window.ocNav('composer')">
      <div class="home-card-icon">♩</div>
      <div class="home-card-title">ORCHESTRAL COMPOSER</div>
      <div class="home-card-desc">AI melody generation · full arrangement<br>Multi-instrument MIDI · raga engine · export</div>
      <div class="home-card-hint">Click to open</div>
    </button>
    <button id="card-garcia" onclick="window.ocNav('garcia')">
      <div class="home-card-icon">🖐</div>
      <div class="home-card-title">GarSIa</div>
      <div class="home-card-desc">Gesture-driven live looper · hand tracking<br>Camera control · BPM sync · no keyboard</div>
      <div class="home-card-hint">Click to open</div>
    </button>
  </div>
  <div id="home-footer">Select a module to continue</div>
</div>
""")
    ollama_status = gr.Markdown("_Checking Ollama…_", visible=False)

    # -- LIVE RECORDER SECTION -------------------------------------------------
    with gr.Group(elem_id="section-live"):
        gr.HTML("""
    <div class="section-topbar">
      <button class="section-back-btn" onclick="window.ocGoHome()">&#9664; HOME</button>
      <div class="section-topbar-title">&#9679;&nbsp; LIVE RECORDER</div>
      <div id="oc-ollama-status" class="section-topbar-status">--</div>
    </div>
    """)
    
        # -- Studio Hero -- side-by-side tab panel ---------------------------------
        gr.HTML("""
    <div id="studio-hero">
    
      <!-- Mobile tab switcher (hidden on wide screens) -->
      <div id="studio-tabs">
        <button class="studio-tab-btn active" id="tab-btn-palette"
          onclick="studioShowTab('palette')">
          <span class="tab-icon">🎵</span> Harmonic Palette
        </button>
        <button class="studio-tab-btn" id="tab-btn-live"
          onclick="studioShowTab('live')">
          <span class="tab-icon">⬤</span> Live Studio
        </button>
      </div>
    
      <!-- Wide-screen column labels -->
      <div id="studio-sidebyside-label">
        <div class="studio-col-label"><span class="tab-icon">🎵</span>&nbsp; Harmonic Palette</div>
        <div class="studio-col-label"><span class="tab-icon">⬤</span>&nbsp; Live Studio</div>
      </div>
    
      <div id="studio-panels">
    """)
    
        with gr.Group(elem_id="palette-hero"):
            gr.HTML('<div class="studio-panel visible" id="panel-palette">')
            # Custom palette picker UI -- note buttons + mode chips replace dropdowns
            gr.HTML('''<div id="pal-picker">
  <div class="pal-section">
    <div class="pal-label">Root</div>
    <div class="pal-notes" id="pal-note-row"></div>
  </div>
  <div class="pal-section">
    <div class="pal-label">Scale / mode</div>
    <div class="pal-modes" id="pal-mode-row"></div>
  </div>
  <div class="pal-section" id="pal-custom-wrap">
    <div class="pal-label">Custom intervals <span style="font-weight:400;opacity:.6">(overrides mode -- semitones from root)</span></div>
    <input id="pal-custom-input" type="text" placeholder="e.g. 0 2 3 6 7 8 10" style="width:100%;padding:6px 10px;border-radius:6px;border:.5px solid var(--color-border-secondary,#555);background:var(--color-background-secondary,#222);color:inherit;font-family:inherit;font-size:13px"/>
  </div>
</div>''')
            # Hidden Gradio controls -- driven by JS above
            with gr.Row(elem_id="pal-gradio-controls"):
                palette_root = gr.Dropdown(
                    ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
                     "Db","Eb","Gb","Ab","Bb"],
                    value="C", label="Root", scale=1,
                )
                palette_mode = gr.Dropdown(
                    _MODE_NAMES, value="Major (Ionian)", label="Mode / Raga", scale=4,
                )
                palette_btn = gr.Button("Generate palette", variant="primary", scale=1, size="lg", elem_id="palette-gen-btn")
            custom_intervals = gr.Textbox(
                value="",
                label="Custom intervals (overrides mode above)",
                placeholder="e.g.  0 2 3 6 7 8 10  -- semitones from root, space-separated",
                lines=1, elem_id="pal-custom-gradio",
            )
            # Custom tile grid overlays the CheckboxGroup
            gr.HTML('<div id="pal-tile-grid"></div>')
            chord_picker = gr.CheckboxGroup(
                label="",
                choices=[], value=[], interactive=True,
                elem_id="chord_picker",
            )
            gr.HTML('<div class="pal-seq-label">Sequence</div><div id="pal-seq-row"></div>'
                    '<div id="pal-add-chord-row">'
                    '<input id="pal-add-input" type="text" placeholder="Add chord: Ebmaj7, F#dim7, Bb7…"/>'
                    '<button id="pal-add-btn">＋ Add</button>'
                    '<span id="pal-add-err"></span>'
                    '</div>')
            chord_data_store = gr.HTML(value="", elem_id="chord-data-wrapper")
            with gr.Row():
                use_selected_btn    = gr.Button("→ Copy to chord input",    size="sm", scale=2)
                fill_all_sections_btn = gr.Button("→ Fill all A / B / C sections", size="sm", scale=2)
                arrange_btn = gr.Button("⬡ Arrange checked chords as MIDI →", size="sm", scale=2, variant="primary", elem_id="arrange-btn")
                gr.HTML('<button id="hover-audio-btn" onclick="window.toggleHoverAudio&&window.toggleHoverAudio()" '
                        'style="background:transparent;border:1px solid #3a72b8;border-radius:6px;'
                        'color:#7eb8f7;padding:4px 12px;font-size:12px;cursor:pointer;white-space:nowrap">'
                        '🔊 Palette preview</button>')
                use_flat_prog = gr.Checkbox(
                    label="Use palette selection as flat progression",
                    value=False, scale=1,
                    info="Bypasses the chord text field -- pipeline uses your checked chords directly.",
                )
    
            with gr.Accordion("Edit / add mode or raga", open=False):
                gr.Markdown(
                    "_Correct a raga, create your own scale, or add a new mode. "
                    "Changes are saved to `data/custom_modes.json` and survive restarts. "
                    "Built-in modes can be overridden by saving under the same name._"
                )
                with gr.Row():
                    mode_name_edit = gr.Textbox(
                        label="Mode / raga name", scale=3,
                        placeholder="e.g. Desh  or  My Scale",
                    )
                    intervals_edit = gr.Textbox(
                        label="Intervals -- semitones from root (0–11, space-separated)",
                        scale=4,
                        placeholder="e.g.  0 2 4 5 7 9 10 11",
                    )
                mode_notes_preview = gr.Markdown("")
                with gr.Row():
                    save_mode_btn   = gr.Button("Save to database",       variant="primary", size="sm", scale=2)
                    delete_mode_btn = gr.Button("Delete from database",   variant="stop",    size="sm", scale=1)
                mode_edit_status = gr.Textbox(label="", interactive=False, lines=1, show_label=False)
    
        # -- Live Accompaniment ----------------------------------------------------
    
        with gr.Group():
            gr.HTML('</div>') # close panel-palette
            gr.HTML('<div class="studio-panel" id="panel-live">')
            gr.Markdown("")
            gr.HTML("""
    <div id="live-console">
    
      <!-- BPM + controls row -->
    
      <!-- Controls row -->
      <div style="display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;margin-bottom:13px">
        <div style="flex:2;min-width:130px">
          <div class="ac-lbl">BPM &nbsp;<span id="ctrl-bpm-val" style="color:#5ba3f5;font-size:13px">85</span></div>
          <input id="ctrl-bpm" type="range" min="40" max="240" value="85" step="1"
            style="width:100%;accent-color:#5ba3f5;cursor:pointer"
            oninput="document.getElementById('ctrl-bpm-val').textContent=this.value">
        </div>
        <div style="flex:1;min-width:72px">
          <div class="ac-lbl">Time sig</div>
          <select id="ctrl-timesig" class="ac-sel">
            <option>4/4</option><option>3/4</option><option>6/8</option><option>5/4</option><option>7/8</option>
          </select>
        </div>
        <div style="flex:2;min-width:110px">
          <div class="ac-lbl">Style</div>
          <select id="ctrl-style" class="ac-sel">
            <optgroup label="-- Piano / Keys --">
              <option>Pop</option><option>Ballad</option><option>Indie Pop</option>
              <option>Singer-Songwriter</option><option>Rhodes</option><option>Lo-Fi</option>
              <option>Honky Tonk</option>
            </optgroup>
            <optgroup label="-- Soul / R&amp;B --">
              <option>R&amp;B</option><option>Neo Soul</option><option>Soul</option>
              <option>Motown</option><option>Gospel</option><option>Worship</option>
              <option>New Soul</option>
            </optgroup>
            <optgroup label="-- Jazz --">
              <option>Jazz</option><option>Smooth Jazz</option><option>Jazz Shell</option>
              <option>Brushed Trio</option><option>Rhodes Jazz</option>
              <option>Vibraphone</option><option>Swing</option>
            </optgroup>
            <optgroup label="-- Funk / Groove --">
              <option>Funk</option><option>Funk Chop</option><option>Disco Pop</option>
              <option>Brass</option>
            </optgroup>
            <optgroup label="-- Guitar --">
              <option>Acoustic</option><option>Country</option>
            </optgroup>
            <optgroup label="-- World / Latin --">
              <option>Bossa Nova</option><option>Samba</option><option>Latin</option>
              <option>Reggae</option><option>Blues</option><option>Waltz</option>
              <option>Tropical</option>
            </optgroup>
            <optgroup label="-- Cinematic --">
              <option>Cinematic</option><option>Pad</option>
              <option>Arpeggio Up</option><option>Arpeggio Down</option>
            </optgroup>
          </select>
        </div>
        <div style="flex:1;min-width:100px">
          <div class="ac-lbl">Bars/chord &nbsp;<span id="ctrl-bpc-val" style="color:#5ba3f5;font-size:13px">2</span></div>
          <input id="ctrl-bpc" type="range" min="1" max="8" value="2" step="0.5"
            style="width:100%;accent-color:#5ba3f5;cursor:pointer"
            oninput="var v=parseFloat(this.value);document.getElementById('ctrl-bpc-val').textContent=v%1===0?v.toFixed(0):v.toFixed(1)">
        </div>
      </div>
    
      <!-- Feel + Passing chord + MIDI Out row -->
      <div style="display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;margin-bottom:13px">
        <div style="flex:2;min-width:130px">
          <div class="ac-lbl">Feel &nbsp;<span id="ctrl-feel-val" style="color:#5ba3f5;font-size:13px">10%</span>
            <span style="color:#778899;font-size:10px;margin-left:6px">0=robotic · 100=loose</span></div>
          <input id="ctrl-feel" type="range" min="0" max="1" value="0.10" step="0.01"
            style="width:100%;accent-color:#a78bfa;cursor:pointer"
            oninput="window.setHumanize&&window.setHumanize(this.value);document.getElementById('ctrl-feel-val').textContent=Math.round(this.value*100)+'%'">
        </div>
        <div style="flex:1;min-width:110px">
          <div class="ac-lbl">Passing chord &nbsp;<span id="ctrl-pass-val" style="color:#f9a84d;font-size:13px">Off</span></div>
          <input id="ctrl-pass" type="range" min="0" max="0.5" value="0" step="0.01"
            style="width:100%;accent-color:#f9a84d;cursor:pointer"
            oninput="var v=parseFloat(this.value);window.setPassProb&&window.setPassProb(v);document.getElementById('ctrl-pass-val').textContent=v===0?'Off':Math.round(v*100)+'%'">
          <select id="ctrl-pass-type" class="ac-sel" style="margin-top:4px;width:100%"
            onchange="window.setPassType&&window.setPassType(this.value)">
            <option value="sec_dom">V7 / next (jazz)</option>
            <option value="dim">dim7 approach</option>
            <option value="chromatic">chromatic ½↓</option>
          </select>
        </div>
        <div style="flex:2;min-width:150px">
          <div class="ac-lbl">MIDI Out → DAW
            <span id="midi-out-badge" style="color:#4ade80;font-size:10px;font-weight:700;margin-left:6px"></span>
          </div>
          <div style="display:flex;gap:5px;align-items:center">
            <select id="ctrl-midi-out" class="ac-sel" style="flex:1"
              onchange="window.updateMIDIOutput&&window.updateMIDIOutput()">
              <option value="">Off</option>
            </select>
            <button class="lst-btn" onclick="window.refreshMIDIOutputs&&window.refreshMIDIOutputs()" title="Refresh MIDI ports" style="padding:5px 9px;font-size:13px">↺</button>
          </div>
          <div style="color:#556677;font-size:10px;margin-top:3px">Enable IAC Driver in macOS Audio MIDI Setup</div>
        </div>
      </div>
    
      <!-- Transport row -->
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:14px;flex-wrap:wrap">
        <button class="ac-btn btn-play"
          onclick="window.startAccompaniment&&window.startAccompaniment()">▶ Play</button>
        <button class="ac-btn btn-stop"
          onclick="window.stopAccompaniment&&window.stopAccompaniment()">■ Stop</button>
        <button id="rec-btn" class="ac-btn btn-rec"
          onclick="window.toggleRecording&&window.toggleRecording()">⏺ Rec</button>
        <span id="rec-dur" style="font-family:'JetBrains Mono',monospace;color:#e04040;font-size:11px;font-weight:700;min-width:38px"></span>
        <button id="tap-btn" class="ac-btn btn-tap"
          onclick="window.tapTempo&&window.tapTempo()">Tap</button>
        <div id="now-playing-strip">
          Now playing: <span id="accomp-now-playing">--</span>
        </div>
      </div>
    
      <!-- Mixer -->
      <div style="margin-bottom:12px">
        <div class="lc-section-title">🎚&nbsp; Mixer</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <div class="mixer-strip">
            <div class="ac-lbl">Master</div>
            <input id="mix-master" type="range" min="0" max="1.5" step="0.01" value="0.45" style="width:100%;accent-color:#c8874a;cursor:pointer"
              oninput="window.setAccompVolume&&window.setAccompVolume('master',this.value);document.getElementById('mix-master-val').textContent=Math.round(this.value/1.5*100)+'%'">
            <div id="mix-master-val" class="ac-val">30%</div>
          </div>
          <div class="mixer-strip">
            <div class="ac-lbl">Bass</div>
            <input id="mix-bass" type="range" min="0" max="2" step="0.01" value="0.6" style="width:100%;accent-color:#c8874a;cursor:pointer"
              oninput="window.setAccompVolume&&window.setAccompVolume('bass',this.value);document.getElementById('mix-bass-val').textContent=Math.round(this.value/2*100)+'%'">
            <div id="mix-bass-val" class="ac-val">30%</div>
            <div class="dens-row" id="dens-bass">
              <button class="dens-btn" data-v="0" onclick="window._setDens('bass',0,this)">Off</button>
              <button class="dens-btn" data-v="1" onclick="window._setDens('bass',1,this)">Sparse</button>
              <button class="dens-btn active" data-v="2" onclick="window._setDens('bass',2,this)">Mod</button>
              <button class="dens-btn" data-v="3" onclick="window._setDens('bass',3,this)">Dense</button>
            </div>
          </div>
          <div class="mixer-strip">
            <div class="ac-lbl" id="mix-chord-label">Chords</div>
            <input id="mix-chord" type="range" min="0" max="2" step="0.01" value="0.6" style="width:100%;accent-color:#c8874a;cursor:pointer"
              oninput="window.setAccompVolume&&window.setAccompVolume('chord',this.value);document.getElementById('mix-chord-val').textContent=Math.round(this.value/2*100)+'%'">
            <div id="mix-chord-val" class="ac-val">30%</div>
            <div class="dens-row" id="dens-chord">
              <button class="dens-btn" data-v="0" onclick="window._setDens('chord',0,this)">Off</button>
              <button class="dens-btn" data-v="1" onclick="window._setDens('chord',1,this)">Sparse</button>
              <button class="dens-btn active" data-v="2" onclick="window._setDens('chord',2,this)">Mod</button>
              <button class="dens-btn" data-v="3" onclick="window._setDens('chord',3,this)">Dense</button>
            </div>
            <select class="mix-density" style="margin-top:4px" id="chord-inst-sel"
              onchange="window.setChordInstrument&&window.setChordInstrument(this.value)">
              <optgroup label="Keys">
                <option value="">Auto (from Style)</option>
                <option value="piano2">Acoustic Piano</option>
                <option value="rhodes">Rhodes</option>
                <option value="organ">Hammond Organ</option>
                <option value="honkyton">Honky Tonk Piano</option>
              </optgroup>
              <optgroup label="Guitar &amp; Strings">
                <option value="guitar">Jazz Guitar</option>
                <option value="guitar_clean">Clean Electric</option>
                <option value="guitar_bossa">Nylon Guitar</option>
                <option value="acoustic">Acoustic Guitar</option>
                <option value="strings">Strings</option>
                <option value="strings_ens">String Ensemble</option>
              </optgroup>
              <optgroup label="Synth &amp; Misc">
                <option value="vibes">Vibraphone</option>
                <option value="clav">Clavinet</option>
                <option value="pad">Synth Pad</option>
                <option value="disco">Disco Synth</option>
                <option value="brass">Brass</option>
                <option value="accordion">Accordion</option>
              </optgroup>
            </select>
          </div>
          <div class="mixer-strip">
            <div class="ac-lbl">Melody</div>
            <input id="mix-melody" type="range" min="0" max="2" step="0.01" value="0.0" style="width:100%;accent-color:#a78bfa;cursor:pointer"
              oninput="window.setMelodyVolume&&window.setMelodyVolume(this.value);document.getElementById('mix-melody-val').textContent=Math.round(this.value/2*100)+'%'">
            <div id="mix-melody-val" class="ac-val">0%</div>
            <select class="mix-density" style="margin-top:4px" id="melody-inst-sel"
              onchange="window.setMelodyInstrument&&window.setMelodyInstrument(this.value)">
              <option value="piano2">Piano Lead</option>
              <option value="rhodes">Rhodes Lead</option>
              <option value="vibes">Vibraphone</option>
              <option value="guitar_clean">Clean Guitar</option>
              <option value="strings_ens">Strings</option>
              <option value="organ">Organ Lead</option>
            </select>
          </div>
          <div class="mixer-strip">
            <div class="ac-lbl">Reverb</div>
            <input id="mix-reverb" type="range" min="0" max="1" step="0.01" value="0.18" style="width:100%;accent-color:#60a5fa;cursor:pointer"
              oninput="window.setReverbLevel&&window.setReverbLevel(this.value);document.getElementById('mix-reverb-val').textContent=Math.round(this.value*100)+'%'">
            <div id="mix-reverb-val" class="ac-val">18%</div>
            <select class="mix-density" style="margin-top:4px"
              onchange="window.setReverbRoom&&window.setReverbRoom(this.value)">
              <option value="sm">Small Room</option>
              <option value="md" selected>Medium Hall</option>
              <option value="lg">Large Hall</option>
              <option value="pl">Plate</option>
            </select>
          </div>
          <div class="mixer-strip">
            <div class="ac-lbl">Drums</div>
            <input id="mix-drum" type="range" min="0" max="2" step="0.01" value="0.6" style="width:100%;accent-color:#c8874a;cursor:pointer"
              oninput="window.setAccompVolume&&window.setAccompVolume('drum',this.value);document.getElementById('mix-drum-val').textContent=Math.round(this.value/2*100)+'%'">
            <div id="mix-drum-val" class="ac-val">30%</div>
            <div class="dens-row" id="dens-drum">
              <button class="dens-btn" data-v="0" onclick="window._setDens('drum',0,this)">Off</button>
              <button class="dens-btn active" data-v="1" onclick="window._setDens('drum',1,this)">Sparse</button>
              <button class="dens-btn" data-v="2" onclick="window._setDens('drum',2,this)">Mod</button>
              <button class="dens-btn" data-v="3" onclick="window._setDens('drum',3,this)">Dense</button>
            </div>
            <select id="drum-pattern-sel" class="mix-density" style="margin-top:4px"
              onchange="window._setDrumUserOverride&&window._setDrumUserOverride(true);window.setDrumPattern&&window.setDrumPattern(this.value)">
              <optgroup label="Soft"><option>Ballad</option></optgroup>
              <optgroup label="Rock">
                <option selected>Rock Basic</option><option>Rock Groove</option><option>Rock Heavy</option>
              </optgroup>
              <optgroup label="Half-Time">
                <option>Half-Time</option><option>Half-Time Heavy</option>
              </optgroup>
              <optgroup label="Funk">
                <option>Funk Light</option><option>Funk Heavy</option>
              </optgroup>
              <optgroup label="Jazz / Latin">
                <option>Jazz Swing</option><option>Brushed Trio</option>
                <option>Jazz Shell</option><option>Bossa Nova</option>
              </optgroup>
              <optgroup label="Pop / World">
                <option>Disco Pop</option><option>Acoustic</option>
                <option>Double Time</option><option>Reggae</option>
              </optgroup>
            </select>
          </div>
        </div>
      </div>
    
      <!-- Listen In -->
      <div style="background:var(--ink);border:1px solid var(--divider);border-radius:var(--radius-sm);padding:11px 14px">
        <!-- -- Loop Pedal -- -->
        <div class="lc-section-title" style="margin-top:14px">🔴&nbsp; Loop Pedal</div>
        <div id="looper-wrap">
          <div style="display:flex;gap:8px;align-items:stretch;flex-wrap:wrap;margin-bottom:8px">
            <!-- Slots A B C -->
            <div id="loop-slot-A" class="loop-slot" data-slot="A">
              <div class="loop-slot-label">A</div>
              <canvas class="loop-wave" width="90" height="28"></canvas>
              <div class="loop-slot-time">--</div>
              <div style="display:flex;gap:3px;margin-top:4px">
                <button class="loop-rec-btn" onclick="window._loopRec('A',this)">⏺</button>
                <button class="loop-dub-btn" onclick="window._loopDub('A',this)">⊕</button>
                <button class="loop-mute-btn" onclick="window._loopMute('A',this)">M</button>
                <button class="loop-clear-btn" onclick="window._loopClear('A')">✕</button>
              </div>
            </div>
            <div id="loop-slot-B" class="loop-slot" data-slot="B">
              <div class="loop-slot-label">B</div>
              <canvas class="loop-wave" width="90" height="28"></canvas>
              <div class="loop-slot-time">--</div>
              <div style="display:flex;gap:3px;margin-top:4px">
                <button class="loop-rec-btn" onclick="window._loopRec('B',this)">⏺</button>
                <button class="loop-dub-btn" onclick="window._loopDub('B',this)">⊕</button>
                <button class="loop-mute-btn" onclick="window._loopMute('B',this)">M</button>
                <button class="loop-clear-btn" onclick="window._loopClear('B')">✕</button>
              </div>
            </div>
            <div id="loop-slot-C" class="loop-slot" data-slot="C">
              <div class="loop-slot-label">C</div>
              <canvas class="loop-wave" width="90" height="28"></canvas>
              <div class="loop-slot-time">--</div>
              <div style="display:flex;gap:3px;margin-top:4px">
                <button class="loop-rec-btn" onclick="window._loopRec('C',this)">⏺</button>
                <button class="loop-dub-btn" onclick="window._loopDub('C',this)">⊕</button>
                <button class="loop-mute-btn" onclick="window._loopMute('C',this)">M</button>
                <button class="loop-clear-btn" onclick="window._loopClear('C')">✕</button>
              </div>
            </div>
            <!-- Global controls -->
            <div style="display:flex;flex-direction:column;gap:6px;justify-content:center;padding-left:6px;border-left:1px solid var(--border-mid)">
              <button id="loop-stop-all" class="loop-global-btn" onclick="window._loopStopAll()">⏹ Stop all</button>
              <button id="loop-clear-all" class="loop-global-btn" onclick="window._loopClearAll()">✕ Clear all</button>
              <label style="display:flex;align-items:center;gap:5px;font-size:9px;color:var(--text-dim);cursor:pointer">
                <input type="checkbox" id="loop-sync-bpm" style="accent-color:#5ba3f5"> Sync to BPM
              </label>
              <div style="font-size:9px;color:var(--text-dim)">Input:
                <select id="loop-input-sel" class="mix-density" style="margin-top:2px;width:100%">
                  <option value="mic">Mic</option>
                  <option value="mix" selected>Mic + Band</option>
                </select>
              </div>
            </div>
          </div>
          <div id="loop-status" style="font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--amber);min-height:16px">Ready -- press ⏺ on any slot to record</div>
        </div>

        <!-- -- Bridge buttons -- -->
        <div id="bridge-btns" style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap">
          <button id="crystallize-btn" class="bridge-btn crystallize-btn"
            onclick="window._crystallize && window._crystallize(this)">
            <span class="bridge-btn-icon">✦</span>
            <span class="bridge-btn-label">Crystallize</span>
            <span class="bridge-btn-sub">Send session → Composer &amp; generate</span>
          </button>
        </div>

        <div class="lc-section-title" style="margin-top:14px">🎧&nbsp; Listen In</div>
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
          <div style="display:flex;gap:6px">
            <button id="listen-btn-off"   class="lst-btn active" onclick="window.startListening&&window.startListening('off')">Off</button>
            <button id="listen-btn-midi"  class="lst-btn"        onclick="window.startListening&&window.startListening('midi')">MIDI</button>
            <button id="listen-btn-audio" class="lst-btn"        onclick="window.startListening&&window.startListening('audio')">Mic</button>
          </div>
          <div style="display:flex;gap:12px;align-items:center;font-family:'Inter',sans-serif;font-size:11px;color:var(--mist)">
            <span style="color:var(--dim);font-size:10px;letter-spacing:.06em;text-transform:uppercase">Respond to:</span>
            <label style="cursor:pointer;color:var(--mist)"><input type="radio" name="listen-resp" value="chord" checked style="accent-color:#c8874a"> Chord</label>
            <label style="cursor:pointer;color:var(--mist)"><input type="radio" name="listen-resp" value="beat"  style="accent-color:#c8874a"> Beat</label>
            <label style="cursor:pointer;color:var(--mist)"><input type="radio" name="listen-resp" value="both"  style="accent-color:#c8874a"> Both</label>
          </div>
          <div id="listen-status" style="flex:1;font-family:'JetBrains Mono',monospace;color:var(--amber);font-size:11px;text-align:right;min-width:80px">--</div>
        </div>
      </div>

      <!-- Range -->
      <div style="margin-top:11px;padding:11px 13px;background:#060c16;border:1px solid #122035;border-radius:8px">
        <div style="color:#c8874a;font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:9px">&#127929; Range</div>
        <div style="display:flex;flex-direction:column;gap:7px">
          <div style="display:flex;align-items:center;gap:8px">
            <div style="width:44px;flex-shrink:0;font-size:11px;color:var(--mist)">Bass</div>
            <select id="rng-bass-lo" onchange="window.setRangeNote&&window.setRangeNote('bass','lo',this.value)" style="background:#0a1628;border:1px solid #2a4a6a;border-radius:4px;color:#c8c8d0;font-size:11px;padding:2px 4px"></select>
            <span style="color:#c8874a;font-size:11px;flex-shrink:0">&#8211;</span>
            <select id="rng-bass-hi" onchange="window.setRangeNote&&window.setRangeNote('bass','hi',this.value)" style="background:#0a1628;border:1px solid #2a4a6a;border-radius:4px;color:#c8c8d0;font-size:11px;padding:2px 4px"></select>
          </div>
          <div style="display:flex;align-items:center;gap:8px">
            <div style="width:44px;flex-shrink:0;font-size:11px;color:var(--mist)">Chord</div>
            <select id="rng-chord-lo" onchange="window.setRangeNote&&window.setRangeNote('chord','lo',this.value)" style="background:#0a1628;border:1px solid #2a4a6a;border-radius:4px;color:#c8c8d0;font-size:11px;padding:2px 4px"></select>
            <span style="color:#c8874a;font-size:11px;flex-shrink:0">&#8211;</span>
            <select id="rng-chord-hi" onchange="window.setRangeNote&&window.setRangeNote('chord','hi',this.value)" style="background:#0a1628;border:1px solid #2a4a6a;border-radius:4px;color:#c8c8d0;font-size:11px;padding:2px 4px"></select>
          </div>
        </div>
      </div>

      <!-- Melodic Response -->
      <div style="margin-top:11px;padding:11px 13px;background:#060c16;border:1px solid #122035;border-radius:8px">
        <div style="color:#c8874a;font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:9px">&#127925; Melodic Response</div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
          <button id="mel-arm-btn" onclick="window.armPhrase&&window.armPhrase()"
            style="background:#1a3050;color:#8af7b8;border:1px solid #2a7040;border-radius:5px;padding:4px 10px;font-size:12px;cursor:pointer">&#128225; Arm</button>
          <button onclick="window.playCaptured&&window.playCaptured()"
            style="background:#0a1628;border:1px solid #2a4a6a;border-radius:5px;color:#c8c8d0;padding:4px 10px;font-size:12px;cursor:pointer">&#9654; Play</button>
          <button onclick="window.generateMelodicResponse&&window.generateMelodicResponse()"
            style="background:#0a1628;border:1px solid #2a4a6a;border-radius:5px;color:#c8c8d0;padding:4px 10px;font-size:12px;cursor:pointer">&#10022; Generate</button>
          <button onclick="window.clearPhrase&&window.clearPhrase()"
            style="background:#0a1628;border:1px solid #2a4a6a;border-radius:5px;color:#c8c8d0;padding:4px 10px;font-size:12px;cursor:pointer;opacity:.6">&#10005; Clear</button>
          <div id="mel-status" style="flex:1;color:var(--amber);font-size:11px;font-style:italic">&#8212;</div>
        </div>
        <div style="display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap">
          <div>
            <div style="font-size:10px;color:var(--mist);margin-bottom:3px">Style</div>
            <select id="mel-style" style="background:#0a1628;border:1px solid #2a4a6a;border-radius:4px;color:#c8c8d0;font-size:11px;padding:2px 4px;width:88px">
              <option value="answer" selected>Answer</option>
              <option value="imitate">Imitate</option>
              <option value="free">Free</option>
            </select>
          </div>
          <div>
            <div style="font-size:10px;color:var(--mist);margin-bottom:3px">Length</div>
            <select id="mel-bars" style="background:#0a1628;border:1px solid #2a4a6a;border-radius:4px;color:#c8c8d0;font-size:11px;padding:2px 4px;width:72px">
              <option value="1">1 bar</option>
              <option value="2" selected>2 bars</option>
              <option value="4">4 bars</option>
              <option value="8">8 bars</option>
            </select>
          </div>
          <label style="display:flex;align-items:center;gap:5px;cursor:pointer;padding-bottom:2px;font-size:11px;color:var(--mist)">
            <input type="checkbox" id="mel-auto-respond" style="cursor:pointer;accent-color:#c8874a">
            Auto after phrase
          </label>
        </div>
      </div>

    </div>
    </div></div>
    """)
    


    # -- Bridge: Live Recorder → Composer -------------------------------------
    # Hidden textboxes receive JSON payloads from JS; hidden buttons trigger Python bridges
    with gr.Group(visible=False, elem_id="bridge-hidden"):
        _crystallize_payload = gr.Textbox(value="", interactive=True, visible=False, elem_id="crystallize-payload")
        _crystallize_trigger = gr.Button("crystallize-trigger", visible=False, elem_id="crystallize-trigger")

    # -- ORCHESTRAL COMPOSER SECTION -------------------------------------------
    with gr.Group(elem_id="section-composer"):
        gr.HTML("""
    <div class="section-topbar">
      <button class="section-back-btn" onclick="window.ocGoHome()">&#9664; HOME</button>
      <div class="section-topbar-title">&#9654;&nbsp; ORCHESTRAL COMPOSER</div>
      <div id="oc-ollama-status-comp" class="section-topbar-status">--</div>
    </div>
    <div id="composer-studio">
      <div id="composer-studio-header">
        <div id="composer-studio-wordmark">Composer Studio</div>
        <div id="composer-studio-rule"></div>
        <div id="composer-studio-badge">MIDI Export</div>
      </div>
    </div>
    """)
    
        # -- Single-column layout -------------------------------------------------
        gr.HTML('<div class="oc-section-div"><span>1 · Melody</span></div>')
    
        melody_source = gr.Radio(
            ["Upload / Record", "AI-generated", "Harmony only"],
            value="Harmony only", label="Melody source",
            info="Upload/Record: transcribe your voice. AI-generated: Ollama writes the melody. Harmony only: algorithmic, no AI.",
        )
        with gr.Row(visible=True, elem_id="synfire-row"):
            melody_contour = gr.Dropdown(
                ["None", "Arch", "Rise", "Fall", "Valley", "Wave", "Step"],
                value="Arch", label="Melody contour",
                info="Synfire-style: shape is defined independently of harmony, then fitted to chord tones.",
                scale=2,
            )
            motif_input = gr.Textbox(
                value="", label="Motif (overrides contour if set)",
                placeholder="C4 E4 G4 E4   — space-separated notes, repeats across chords",
                lines=1, scale=3,
            )
        audio_input = gr.Audio(
            label="Melody audio (microphone or file)",
            sources=["microphone", "upload"], type="filepath", visible=False,
        )
        pitch_tracker = gr.Radio(
            ["CREPE (neural)", "pyin (fast)"], value="CREPE (neural)",
            label="Pitch tracker",
            info="CREPE is more accurate on expressive/ornamental singing; pyin is faster.",
            visible=False,
        )
        with gr.Accordion("Vocal enhancement", open=False):
            gr.Markdown(
                "_Six-stage DSP chain applied automatically before transcription: "
                "noise gate → noise reduction → EQ → compression → tuning → pitch correction._"
            )
            with gr.Row():
                reverb_preset = gr.Dropdown(
                    ["studio", "small_room", "hall", "cathedral", "none"],
                    value="studio", label="Reverb preset", scale=2,
                )
                reverb_mix = gr.Slider(
                    0.0, 1.0, value=0.18, step=0.01, label="Reverb amount", scale=3,
                )
            with gr.Row():
                pitch_strength = gr.Slider(
                    0.0, 1.0, value=0.8, step=0.05,
                    label="Pitch correction strength", scale=3,
                    info="1.0 = hard quantise to scale · 0.0 = off",
                )
                noise_strength = gr.Slider(
                    0.0, 1.0, value=0.7, step=0.05,
                    label="Noise reduction strength", scale=3,
                    info="Higher values remove more background noise before transcription",
                )
    
        with gr.Row():
            duration_bars = gr.Slider(
                4, 64, value=16, step=4, label="Duration (bars)", scale=3,
            )
            num_variations = gr.Slider(
                1, 4, value=1, step=1, label="Variations", scale=1,
            )
        duration_display = gr.Markdown(value=calc_duration(16, DEFAULT_TEMPO_BPM))
    
        gr.HTML('<div class="oc-section-div"><span>2 · Composition</span></div>')
    
        # Preset + Key + Tempo (always visible -- the three most essential parameters)
        with gr.Row():
            preset_name = gr.Dropdown(
                list(INSTRUMENT_PRESETS.keys()), value="Jazz Quartet",
                label="Orchestration preset", scale=2,
            )
            harmony_style = gr.Dropdown(
                list(HARMONY_STYLES.keys()), value="Jazz comping",
                label="Chord voicing style", scale=2,
                info="Voicing style for A/B/C sections. Korvai bridges (K1, K2) always use their own solkattu engine.",
            )
    
        with gr.Group(elem_id="key-tempo-group"):
            with gr.Row():
                key_mode = gr.Radio(
                    ["Auto-detect", "Manual"], value="Manual",
                    label="Key", interactive=True, scale=1,
                )
                manual_key = gr.Textbox(
                    value="C major", label="",
                    placeholder="C major · A minor · D Shankarabharanam…", scale=3,
                )
                tempo_mode = gr.Radio(
                    ["Auto-detect", "Auto-detect (÷2)", "Manual"],
                    value="Manual", label="Tempo", scale=1,
                )
                manual_tempo = gr.Slider(
                    40, 240, value=DEFAULT_TEMPO_BPM, step=1, label="BPM", scale=2,
                )
                time_sig = gr.Dropdown(
                    list(TIME_SIGNATURES.keys()), value="4/4", label="Time sig", scale=1,
                )
    
        # Indian mode banner -- shown when an Indian/raga preset is selected
        indian_banner = gr.Markdown("", visible=False, elem_id="indian-mode-banner")
    
        # Chord input (hidden for Indian presets -- replaced by raga drone)
        with gr.Group(elem_id="chord-group") as chord_group:
            gr.Markdown("**Chord progression**  _(from palette above, or type your own)_")
            chord_input = gr.Textbox(
                value="Dm7 G7 Cmaj7 Am7",
                label="Chord symbols  (or Roman numerals: I7-iii-V)",
                placeholder="Dm7  G7  Cmaj7  Am7  --  or  I7-iii-V  (resolved to current key)", lines=1,
            )
            beats_per_chord = gr.Slider(
                0.5, 8.0, value=4.0, step=0.5, label="Beats per chord",
            )
    
        # Song structure -- top-level so it's always findable
        with gr.Accordion("Song structure  -- form builder", open=False):
            gr.Markdown(
                "Add rows to define sections (any label). Korvai bridges are configured in the Korvai slots below. "
                "Form examples: `A B A` · `A K B L A` · `K` (korvai only) · `A B K1 A B K2 A`."
            )
            form_input = gr.Textbox(
                value="", label="Form", placeholder="A B K1 A B K2 A", lines=1,
            )
            form_status = gr.Markdown("")
    
            gr.Markdown("**Quick section builder** -- pick a named progression and apply it to a section:")
            with gr.Row():
                sec_quick_label = gr.Textbox(label="Label", placeholder="A", scale=1, lines=1)
                sec_quick_bars  = gr.Number(label="Bars", value=8, scale=1)
                sec_quick_prog  = gr.Dropdown(
                    ["-- pick --"] + PROGRESSION_NAMES,
                    value="-- pick --", label="Progression", scale=3,
                )
                sec_quick_btn   = gr.Button("→ Add / update", size="sm", scale=1)
    
            section_table = gr.Dataframe(
                headers=["Label", "Bars", "Chords"],
                datatype=["str", "number", "str"],
                value=[["A", 8, ""], ["B", 8, ""]],
                row_count=(2, "dynamic"),
                label="Sections -- edit rows or add new ones; labels appear in the Form above",
                interactive=True,
            )
    
            gr.Markdown(
                "---\n**Korvai definitions** -- one row per korvai bridge; add as many as you need. "
                "Label must match the Form. "
                "Phrase / connector: syllables (`ta ka dhi mi`) **or numbers** "
                "(`8` · `5, 4, 3` gopuccha · `1, 1, 1` spaced). "
                "Gati: `4` chatusram · `3` tisram · `5` khandam. "
                "Kalam: `n` normal ♬ · `k` keezh ♪ · `v` vilambita ♩."
            )
            with gr.Row():
                korvai_rand_btn = gr.Button(
                    "🎲 Randomize empty rows", size="sm", variant="secondary", scale=1,
                )
            korvai_table = gr.Dataframe(
                headers=["Label", "A phrase", "Connector", "Gati (4/3/5)", "Kalam (n/k/v)", "Target", "Chord (opt)"],
                datatype=["str", "str", "str", "str", "str", "number", "str"],
                value=[
                    ["K1", "ta ka dhi mi ta ka dhi mi",                   "ta ka",         "4", "n", 32, ""],
                    ["K2", "ta ka dhi mi ta ka dhi mi ta ka dhi mi", "ta ka dhi mi", "4", "n", 32, ""],
                ],
                row_count=(2, "dynamic"),
                label="Korvai definitions -- add rows freely",
                interactive=True,
            )
            korvai_status = gr.Markdown("_Add korvai rows above to see matra counts._")
    
        # Advanced options
        with gr.Accordion("Advanced options", open=False):
    
            with gr.Row():
                user_style = gr.Textbox(
                    value="",
                    label="Style direction  (AI-generated melody only)",
                    placeholder="sparse · melancholic · bebop · late-night jazz…",
                    lines=2, scale=3,
                )
                with gr.Column(scale=1):
                    rhythm_style = gr.Dropdown(
                        list(RHYTHM_STYLES.keys()), value="Free", label="Rhythm feel",
                    )
                    chord_mode = gr.Radio(
                        ["Strict", "Loose"], value="Strict", label="Chord compliance",
                    )
    
            with gr.Row():
                prog_dropdown = gr.Dropdown(
                    ["-- pick a progression --"] + PROGRESSION_NAMES,
                    value="-- pick a progression --",
                    label="Common progressions", scale=3,
                )
                prog_apply_btn = gr.Button("→ Use", size="sm", scale=1)
    
            with gr.Group(elem_id="western-harmony-group"):
                gr.Markdown("**Harmonic enrichment**  _(Western presets only)_")
                with gr.Row():
                    use_sec_dom = gr.Checkbox(
                        label="Secondary dominants", value=False,
                        info="V7/X before non-tonic chords -- forward motion.",
                    )
                    use_auto_seventh = gr.Checkbox(
                        label="Auto 7ths", value=True,
                        info="Upgrades plain triads to 7ths (C→Cmaj7, Am→Am7…).",
                    )
                    use_back_cycle = gr.Checkbox(
                        label="Back-cycling (ii–V)", value=True,
                        info="Inserts ii7 before each dominant (G7 → Dm7 G7).",
                    )
                with gr.Row():
                    use_tritone_sub = gr.Checkbox(
                        label="Tritone subs", value=False,
                        info="Replaces dominant chords with tritone substitution.",
                    )
                    use_passing_chords = gr.Checkbox(
                        label="Passing chords", value=False,
                        info="Chromatic passing diminished between whole-step roots.",
                    )
                forbidden_input = gr.Textbox(
                    value="", label="Forbidden chords",
                    placeholder="e.g.  Dm  Bb7", lines=1,
                )
    
            with gr.Group():
                gr.Markdown("**LLM / generation settings**  _(AI-generated melody only)_")
                notes_per_beat = gr.Slider(
                    0.5, 4.0, value=DEFAULT_NOTES_PER_BEAT, step=0.25,
                    label="Note density (notes per beat per part)",
                )
                with gr.Row():
                    max_tokens = gr.Slider(
                        4096, 32768, value=DEFAULT_MAX_TOKENS, step=4096, label="Max tokens",
                    )
                    temperature = gr.Slider(
                        0.1, 1.5, value=DEFAULT_TEMPERATURE, step=0.1, label="Temperature",
                    )
                humanize = gr.Slider(
                    0.0, 1.0, value=0.35, step=0.05, label="Humanize",
                    info="0 = mechanical · 0.35 = natural (default) · 0.7+ = loose",
                )
    
        # -- Generate + output ----------------------------------------------------
        gr.HTML('<div class="oc-section-div"><span>3 · Generate</span></div>')
        with gr.Row(elem_id="generate-btn-wrap"):
            generate_btn = gr.Button("Generate Arrangement", variant="primary", size="lg", scale=4)
            stop_btn     = gr.Button("⏹ Stop",               variant="stop",    size="lg", scale=1, visible=False)
    
        out_player = gr.HTML(label="Preview & playback")
        out_status = gr.Textbox(
            label="Status", lines=6, interactive=False,
            placeholder="Hit Generate -- output appears here.",
            elem_id="out-status",
        )
        out_file = gr.File(label="Download MIDI", file_types=[".mid"])
    
        with gr.Accordion("Diagnose & re-arrange", open=False):
            diag_btn = gr.Button("Analyse last MIDI", size="sm")
            diag_out = gr.Textbox(label="Diagnostic report", lines=8, interactive=False)
    
            gr.Markdown("---\n**Re-arrange with reharmonization**")
            gr.Markdown("_Check any enrichments to apply on top of the last generation, then click Re-arrange._")
            with gr.Row():
                reharm_auto_seventh  = gr.Checkbox(label="Auto 7ths",             value=False)
                reharm_sec_dom       = gr.Checkbox(label="Secondary dominants",    value=False)
                reharm_back_cycle    = gr.Checkbox(label="Back-cycling (ii–V)",    value=False)
            with gr.Row():
                reharm_tritone       = gr.Checkbox(label="Tritone subs",           value=False)
                reharm_passing       = gr.Checkbox(label="Passing chords",         value=False)
            reharm_btn = gr.Button("Re-arrange", variant="secondary", size="sm")
    
            gr.Markdown("---\n**Swap chord progression (reharmonise)**")
            gr.Markdown("_Keeps the last melody contour, replaces the chords underneath, rebuilds harmony/bass._")
            reharm_new_chords = gr.Textbox(
                value="", label="New chord progression",
                placeholder="e.g.  Fmaj7  Dm7  Gm7  C7  -- leave blank to keep current",
                lines=1,
            )
            reharm_chords_btn = gr.Button("Reharmonise", variant="secondary", size="sm")

            gr.Markdown("---\n**Section variation**")
            gr.Markdown("_Applies a Synfire-style transformation to the melody of the last generation._")
            with gr.Row():
                vary_technique = gr.Dropdown(
                    ["invert", "retrograde", "augment", "diminish"],
                    value="invert", label="Technique", scale=2,
                )
                vary_btn = gr.Button("Create Variation", variant="secondary", size="sm", scale=1)
            vary_out = gr.Textbox(label="Variation status", lines=2, interactive=False)

            gr.Markdown("---\n**Feedback**")
            feedback_input = gr.Textbox(
                value="", label="What's wrong?",
                placeholder="too mechanical · too sparse · wrong key…", lines=2,
            )
            feedback_out = gr.Textbox(label="Suggestions", lines=3, interactive=False)
            feedback_btn = gr.Button("Parse feedback", size="sm")
    


    # -- Event wiring --------------------------------------------------------

    # Korvai table -- live matra/beat status for every defined row
    def _korvai_table_status(kv_data, tempo_bpm):
        rows = _table_to_rows_korvai(kv_data)
        bpm  = max(20.0, float(tempo_bpm or 90))
        lines = []
        for row in rows:
            label, phrase, connector, gati_s, kalam_s, target = row[:6]
            if not label or not phrase:
                continue
            gati_ratio = _parse_gati(gati_s)
            bpm_factor = _parse_kalam(kalam_s)
            try:
                info = _korvai_info(phrase, connector, gati_ratio, int(target or 32))
                rem  = info["remainder"]
                icon = "✓" if info["fits"] else ("▲" if abs(rem) < 4 else "✗")
                beats = info["total"] * bpm_factor
                secs  = beats * 60.0 / bpm
                fit_s = f"  rem {rem:+.1f}" if not info["fits"] else "  perfect"
                lines.append(
                    f"**{label}**: {icon}  "
                    f"3×{info['phrase_matras']:.0f} + 2×{info['connector_matras']:.0f} "
                    f"= {info['total']:.0f}/{int(target or 32)} matras"
                    f"  ·  {beats:.1f} beats  ≈ {secs:.1f}s @ {int(bpm)} BPM{fit_s}"
                )
            except Exception:
                lines.append(f"**{label}**: parse error")
        return "\n\n".join(lines) if lines else "_Add korvai rows above to see matra counts._"

    def _randomize_korvai_table(kv_data, tempo_bpm):
        """Fill every empty-phrase korvai row with a random phrase+connector."""
        rows = _table_to_rows_korvai(kv_data) or [["K1", "", "", "4", "n", 32, ""]]
        new_rows = []
        for row in rows:
            label, phrase, connector, gati_s, kalam_s, target = row[:6]
            chord_override = str(row[6]).strip() if len(row) > 6 else ""
            if not phrase.strip():
                try:
                    phrase, connector = _random_korvai(
                        target_matras=int(target or 32),
                        gati_ratio=_parse_gati(gati_s),
                    )
                except Exception:
                    pass
            new_rows.append([label, phrase, connector, gati_s or "4", kalam_s or "n", target or 32, chord_override])
        return new_rows

    korvai_table.change(_korvai_table_status, inputs=[korvai_table, manual_tempo], outputs=[korvai_status])
    manual_tempo.change(_korvai_table_status, inputs=[korvai_table, manual_tempo], outputs=[korvai_status])
    korvai_rand_btn.click(_randomize_korvai_table, inputs=[korvai_table, manual_tempo], outputs=[korvai_table])

    # Section quick-builder -- resolves a named progression and upserts a row
    def _sec_quick_add(label, bars, prog_name, key, current_table):
        label = (label or "").strip().upper()
        if not label or str(prog_name or "").startswith("--"):
            return gr.update()
        chords = progression_to_chords(prog_name, key or "C major")
        if not chords:
            return gr.update()
        chord_str = " ".join(chords)
        rows = _table_to_rows(current_table) or []
        new_rows, updated = [], False
        for row in rows:
            if str(row[0]).strip().upper() == label:
                new_rows.append([label, int(bars or 8), chord_str])
                updated = True
            else:
                new_rows.append(row)
        if not updated:
            new_rows.append([label, int(bars or 8), chord_str])
        return new_rows

    sec_quick_btn.click(
        _sec_quick_add,
        inputs=[sec_quick_label, sec_quick_bars, sec_quick_prog, manual_key, section_table],
        outputs=[section_table],
    )

    # Show/hide audio + duration based on melody source
    def _source_changed(src):
        show_audio      = src == "Upload / Record"
        show_duration   = src != "Upload / Record"
        show_tempo_auto = src == "Upload / Record"
        return (
            gr.update(visible=show_audio),     # audio_input
            gr.update(visible=show_audio),     # pitch_tracker
            gr.update(visible=show_duration),  # duration_bars
            gr.update(visible=show_duration),  # num_variations
            gr.update(visible=show_duration),  # duration_display
            gr.update(value="Auto-detect" if show_tempo_auto else "Manual"),  # tempo_mode
        )

    melody_source.change(
        _source_changed, inputs=melody_source,
        outputs=[audio_input, pitch_tracker, duration_bars, num_variations, duration_display, tempo_mode],
    )

    # Duration display
    duration_bars.change(calc_duration, inputs=[duration_bars, manual_tempo, time_sig], outputs=duration_display)
    manual_tempo.change(calc_duration, inputs=[duration_bars, manual_tempo, time_sig], outputs=duration_display)
    time_sig.change(calc_duration, inputs=[duration_bars, manual_tempo, time_sig], outputs=duration_display)

    # Key auto-detect visibility
    key_mode.change(
        lambda m: gr.update(visible=(m == "Manual")), inputs=key_mode, outputs=manual_key,
    )

    # Indian preset -- show banner, hide chord/western-harmony groups
    _INDIAN_PRESETS_UI = frozenset({
        "Carnatic Ensemble", "Bollywood Golden Era", "Bollywood Modern",
        "Sufi / Qawwali", "Koothu / Folk", "Dholak Party",
        "Hindustani Classical", "Sitar & Strings",
    })

    def _on_preset_change(preset):
        is_indian = preset in _INDIAN_PRESETS_UI
        banner_md = (
            "**Raga mode active** -- chord progression replaced by Sa–Pa drone. "
            "Set Key as `Note Raga` e.g. `D Shankarabharanam` or `G Yaman`. "
            "Harmonic enrichment has no effect in this mode."
            if is_indian else ""
        )
        return (
            gr.update(visible=is_indian, value=banner_md),  # indian_banner
            gr.update(visible=not is_indian),               # chord-group
            gr.update(visible=not is_indian),               # harmony_style
        )

    preset_name.change(
        _on_preset_change, inputs=[preset_name],
        outputs=[indian_banner, chord_group, harmony_style],
    )

    # Progression dropdown → chord_input
    def _apply_prog(prog_name, key):
        if prog_name.startswith("--"):
            return gr.update()
        return gr.update(value=" ".join(progression_to_chords(prog_name, key or "C major")))

    prog_apply_btn.click(_apply_prog, inputs=[prog_dropdown, manual_key], outputs=[chord_input])

    # Form validation -- live feedback when form, sections, or korvai table changes
    def _validate_form_fn(form_str, sec_data, kv_data):
        form = parse_form(form_str or "")
        if not form:
            return ""
        defined = {row[0] for row in _table_to_rows(sec_data) if row[0]}
        defined |= {row[0] for row in _table_to_rows_korvai(kv_data) if row[0]}
        undefined = [l for l in form if l not in defined]
        if undefined:
            return f"⚠ Not defined: **{', '.join(undefined)}** -- add rows to the sections or korvai table"
        return f"✓  {' → '.join(form)}"

    _form_val_inputs = [form_input, section_table, korvai_table]
    for _comp in _form_val_inputs:
        _comp.change(_validate_form_fn, inputs=_form_val_inputs, outputs=[form_status])

    # Palette
    palette_btn.click(
        generate_palette,
        inputs=[palette_root, palette_mode, custom_intervals, beats_per_chord],
        outputs=[chord_picker, key_mode, manual_key, chord_data_store],
    ).then(fn=None, js="() => window.attachChordTooltips && window.attachChordTooltips()")
    use_selected_btn.click(use_selected_chords, inputs=[chord_picker], outputs=[chord_input])
    arrange_btn.click(
        arrange_in_composer,
        inputs=[chord_picker, palette_root, palette_mode, manual_tempo],
        outputs=[chord_input, key_mode, manual_key, manual_tempo, use_flat_prog],
    ).then(fn=None, js="() => window.ocNav && window.ocNav('composer')")
    fill_all_sections_btn.click(
        fill_all_sections_from_palette,
        inputs=[chord_picker, section_table],
        outputs=[section_table],
    )

    # Mode editor
    palette_mode.change(
        fill_mode_editor, inputs=[palette_mode],
        outputs=[mode_name_edit, intervals_edit],
    )
    for _trigger in [intervals_edit, palette_root]:
        _trigger.change(
            preview_mode_notes, inputs=[palette_root, intervals_edit],
            outputs=[mode_notes_preview],
        )
    save_mode_btn.click(
        save_mode, inputs=[mode_name_edit, intervals_edit],
        outputs=[palette_mode, mode_edit_status],
    )
    delete_mode_btn.click(
        delete_mode, inputs=[mode_name_edit],
        outputs=[palette_mode, mode_edit_status],
    )

    # Main generate button -- show Stop while running, restore when done
    _pipeline_inputs = [
        melody_source, audio_input,
        pitch_tracker,
        tempo_mode, manual_tempo,
        duration_bars, key_mode, manual_key,
        preset_name, user_style,
        chord_input, beats_per_chord, chord_mode,
        rhythm_style, harmony_style,
        forbidden_input,
        notes_per_beat, max_tokens, temperature,
        time_sig, use_sec_dom, use_tritone_sub,
        humanize, num_variations,
        form_input,
        section_table,
        chord_picker, use_flat_prog,
        use_auto_seventh, use_back_cycle, use_passing_chords,
        korvai_table,
        reverb_preset, reverb_mix, pitch_strength, noise_strength,
        melody_contour, motif_input,
    ]
    _btn_running  = lambda: (gr.update(visible=False), gr.update(visible=True))
    _btn_restored = lambda: (gr.update(visible=True),  gr.update(visible=False))

    _show_stop = generate_btn.click(
        fn=_btn_running, outputs=[generate_btn, stop_btn], queue=False,
    )
    _gen_event = _show_stop.then(
        run_pipeline, inputs=_pipeline_inputs, outputs=[out_player, out_file, out_status],
    )
    _gen_event.then(
        fn=_btn_restored, outputs=[generate_btn, stop_btn], queue=False,
    )
    stop_btn.click(
        fn=_btn_restored, outputs=[generate_btn, stop_btn],
        cancels=[_gen_event], queue=False,
    )

    # Diagnostics
    diag_btn.click(run_diagnostics, inputs=[], outputs=[diag_out])

    # Feedback
    feedback_btn.click(parse_feedback, inputs=[feedback_input], outputs=[feedback_out])

    # Re-arrange with reharmonization
    reharm_btn.click(
        run_reharmonize,
        inputs=[reharm_auto_seventh, reharm_sec_dom, reharm_back_cycle, reharm_tritone, reharm_passing],
        outputs=[out_player, out_file, out_status],
    )

    # Reharmonise with new chord progression
    def run_reharm_chords(new_chords_str, harmony_sty, key_str, time_sig_str, humanize_amt):
        from src.algo_arranger import inject_algo_parts
        ctx = _last_context
        if not ctx.get("orchestration"):
            return gr.update(), gr.update(), "No previous generation to reharmonise."
        import re as _re
        from datetime import datetime as _dt
        chords_str = (new_chords_str or "").strip() or ctx.get("chord_input", "")
        if not chords_str:
            return gr.update(), gr.update(), "No chord progression supplied."
        new_ct, total_b, _, chord_sum, sec_map = _build_chord_timeline(
            chords_str, ctx.get("beats_per_chord", 4.0),
            ctx.get("total_beats", 32), ctx.get("beats_per_bar", 4.0),
            key_str, False, False, "", None,
        )
        if not new_ct:
            return gr.update(), gr.update(), "Could not parse new chord progression."
        ts = TIME_SIGNATURES.get(time_sig_str, TIME_SIGNATURES["4/4"])
        bpb = ts["beats_per_bar"]
        orch = dict(ctx["orchestration"])
        orch = inject_algo_parts(
            orch, new_ct, total_b,
            CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
            dict(INSTRUMENT_RANGES), harmony_sty, bpb, key_str,
        )
        if humanize_amt > 0:
            orch = humanize_orchestration(orch, amount=humanize_amt, melody_instruments=_MELODY_INSTRUMENTS)
        ts_inf = ts
        stamp = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = str(OUTPUTS_DIR / f"{stamp}_reharm.mid")
        try:
            build_midi(orch, out_path,
                       time_sig_num=ts_inf["numerator"], time_sig_den=ts_inf["denominator"],
                       chord_timeline=new_ct, key=key_str)
            player_html = _make_simple_player(out_path)
            _last_context.update({"orchestration": orch, "midi_path": out_path,
                                   "chord_timeline": new_ct, "total_beats": total_b})
            return player_html, out_path, f"Reharmonised.\nNew chords: {chord_sum}"
        except Exception as e:
            return gr.update(), gr.update(), f"Reharmonise failed: {e}"

    reharm_chords_btn.click(
        run_reharm_chords,
        inputs=[reharm_new_chords, harmony_style, manual_key, time_sig, humanize],
        outputs=[out_player, out_file, out_status],
    )

    # Section variation
    def run_vary(technique):
        from src.algo_arranger import vary_orchestration
        from datetime import datetime as _dt
        ctx = _last_context
        if not ctx.get("orchestration"):
            return "No previous generation to vary."
        varied = vary_orchestration(ctx["orchestration"], technique=technique)
        stamp = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = str(OUTPUTS_DIR / f"{stamp}_variation_{technique}.mid")
        ts_inf = TIME_SIGNATURES.get(ctx.get("time_sig", "4/4"), TIME_SIGNATURES["4/4"])
        try:
            build_midi(varied, out_path,
                       time_sig_num=ts_inf["numerator"], time_sig_den=ts_inf["denominator"],
                       chord_timeline=ctx.get("chord_timeline"), key=ctx.get("key", "C major"))
            _last_context["orchestration"] = varied
            _last_context["midi_path"] = out_path
            return f"Variation ({technique}) saved → {out_path}"
        except Exception as e:
            return f"Variation failed: {e}"

    vary_btn.click(run_vary, inputs=[vary_technique], outputs=[vary_out])

    # -- GarSIa SECTION -------------------------------------------------------
    with gr.Group(elem_id="section-garcia"):
        gr.HTML("""
<div class="section-topbar">
  <button class="section-back-btn" onclick="window.ocGoHome()">&#9664; HOME</button>
  <div class="section-topbar-title">🖐&nbsp; GarSIa</div>
  <div id="garcia-status" class="section-topbar-status">Ready</div>
</div>

<div id="garcia-layout">

  <!-- ── Left preset panel ── -->
  <div id="garcia-preset">

    <div class="garcia-label">SCALE</div>
    <select id="garcia-scale-sel" onchange="window.garciaChordSet&&window.garciaChordSet(document.getElementById('garcia-chord-input').value)">
      <option>C major</option><option>D minor</option><option>G major</option>
      <option>A minor</option><option>E minor</option><option>F major</option>
      <option>D Dorian</option><option>A Mixolydian</option>
    </select>

    <div class="garcia-label">CHORD SEQUENCE</div>
    <input id="garcia-chord-input" type="text" value="Cmaj7, Am7, Dm7, G7"
           placeholder="e.g. Dm7, G7, Cmaj7"
           onchange="window.garciaChordSet&&window.garciaChordSet(this.value)" />
    <div class="garcia-hint">comma-separated · pinky gesture advances</div>

    <div class="garcia-label">NOW PLAYING</div>
    <div id="garcia-chord-now" class="garcia-big green">Cmaj7</div>

    <div class="garcia-label">BPM</div>
    <div id="garcia-bpm-val" class="garcia-big bright">120</div>
    <input type="range" min="60" max="180" value="120" step="5"
           oninput="window.garciaBPMSet&&window.garciaBPMSet(this.value)"
           style="width:100%;margin:4px 0 0;" />
    <div class="garcia-hint">or tap wrist to set live</div>

    <div class="garcia-label" style="margin-top:16px;">GESTURE GUIDE</div>
    <div class="garcia-guide">
      <div><span class="gg-icon">✊</span> Fist &nbsp;→ Drums</div>
      <div><span class="gg-icon">☝</span> Point → Bass</div>
      <div><span class="gg-icon">✌</span> Peace → Keys</div>
      <div><span class="gg-icon">🖖</span> Three → Lead</div>
      <div><span class="gg-icon">🖐</span> Four &nbsp;→ Pad</div>
      <div><span class="gg-icon">🖐</span> Palm &nbsp;→ All</div>
      <div><span class="gg-icon">👍</span> Thumb → BPM ▲</div>
      <div><span class="gg-icon">🤙</span> Shaka → BPM ▼</div>
    </div>
    <div class="garcia-hint" style="margin-top:6px;">Hold gesture 0.6 s to activate</div>

    <button onclick="window.garciaStop()"
            style="margin-top:auto;background:#1a2a1a;border:1px solid #1a5c30;color:#66dd99;
                   padding:8px;cursor:pointer;font-family:monospace;letter-spacing:0.08em;width:100%;">
      ■ STOP
    </button>
  </div>

  <!-- ── Camera / canvas area ── -->
  <div id="garcia-right">
    <div id="garcia-viewport">
      <video id="garcia-video" playsinline muted style="display:none"></video>
      <canvas id="garcia-canvas"></canvas>
      <div id="garcia-start-overlay">
        <div class="garcia-start-glyph">🖐</div>
        <button id="garcia-start-btn" onclick="window.garciaStart()">▶ Start Camera</button>
        <div class="garcia-hint" style="color:#2d6644;">Allow camera access when prompted</div>
      </div>
    </div>

    <!-- ── Layer indicators ── -->
    <div id="garcia-layers">
      <div id="garcia-layer-drums" class="garcia-layer" style="--lc:#ff5555">
        <div class="gl-icon">🥁</div>
        <div class="gl-name">DRUMS</div>
        <div class="garcia-style-name">Basic 4/4</div>
        <div class="gl-hint">✊</div>
      </div>
      <div id="garcia-layer-bass" class="garcia-layer" style="--lc:#ff9933">
        <div class="gl-icon">🎸</div>
        <div class="gl-name">BASS</div>
        <div class="garcia-style-name">Root</div>
        <div class="gl-hint">☝</div>
      </div>
      <div id="garcia-layer-keys" class="garcia-layer" style="--lc:#ffee44">
        <div class="gl-icon">🎹</div>
        <div class="gl-name">KEYS</div>
        <div class="garcia-style-name">Block</div>
        <div class="gl-hint">✌</div>
      </div>
      <div id="garcia-layer-lead" class="garcia-layer" style="--lc:#44ff88">
        <div class="gl-icon">🎺</div>
        <div class="gl-name">LEAD</div>
        <div class="garcia-style-name"></div>
        <div class="gl-hint">🖖</div>
      </div>
      <div id="garcia-layer-pad" class="garcia-layer" style="--lc:#44aaff">
        <div class="gl-icon">🌊</div>
        <div class="gl-name">PAD</div>
        <div class="garcia-style-name">Whole</div>
        <div class="gl-hint">🖐</div>
      </div>
    </div>
  </div>

</div>
""")

    # Startup check -- feeds the hidden markdown (for wiring) + status box
    def _startup_check():
        ok, msg = check_ollama()
        badge = f"**{'✓' if ok else '⚠'} {msg}**" if ok else f"⚠ {msg} -- AI melody unavailable"
        status_txt = f"{'✓' if ok else '✗'} {msg}\n\nReady."
        return badge, status_txt

    demo.load(_startup_check, outputs=[ollama_status, out_status])
    # Push Ollama status to the header pill after page load
    demo.load(
        fn=None,
        js="""() => {
          setTimeout(function() {
            var md = document.querySelector('#component-4 p, .gr-markdown p');
            var pill = document.getElementById('oc-ollama-status');
            if (pill && md) pill.textContent = md.textContent.replace(/^[✓⚠✗]\\s*/, '').split(' -- ')[0] || 'Ollama';
          }, 2000);
        }"""
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_port=SERVER_PORT, show_error=True, server_name="127.0.0.1")