"""
Orchestration via a local Ollama LLM.

The LLM receives a compact melody description and returns a multi-part
orchestration as JSON. We use Ollama's native JSON-mode (`"format": "json"`)
to dramatically reduce parse failures. On failure we retry up to max_retries
times with a slightly simplified prompt.
"""

import json
import re
import requests
from typing import Generator

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.prompts import (
    SYSTEM_PROMPT, build_orchestration_prompt, build_composition_prompt, INSTRUMENT_PROGRAMS
)
from src.harmony import (
    enforce_chord_compliance, fill_chord_voicings,
    snap_all_parts_to_scale, parse_key_string, scale_note_names, _NOTE_PC,
)
from src.prompts import INSTRUMENT_RANGES, CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS
from src.algo_arranger import inject_algo_parts

_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
_TAGS_URL     = f"{OLLAMA_BASE_URL}/api/tags"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_ollama() -> tuple[bool, str]:
    """
    Check that Ollama is running and the configured model is available.

    Returns:
        (ok: bool, message: str)
    """
    try:
        resp = requests.get(_TAGS_URL, timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return False, (
            f"Ollama is not running.\n"
            f"Start it with:  ollama serve\n"
            f"Then retry."
        )
    except Exception as e:
        return False, f"Ollama health check failed: {e}"

    models = [m["name"] for m in resp.json().get("models", [])]
    # Accept partial matches so "llama3.1:8b" matches "llama3.1:8b-instruct-q4_K_M" etc.
    if not any(OLLAMA_MODEL.split(":")[0] in m for m in models):
        return False, (
            f"Model '{OLLAMA_MODEL}' not found in Ollama.\n"
            f"Pull it with:  ollama pull {OLLAMA_MODEL}\n"
            f"Available: {', '.join(models) or 'none'}"
        )

    return True, f"Ollama ready — using {OLLAMA_MODEL}"


def orchestrate(
    notes: list[dict],
    key: str,
    tempo_bpm: float,
    total_beats: float,
    instruments: list[str] | None = None,
    style_prompt: str = "",
    max_retries: int = 2,
) -> dict:
    """
    Ask the local LLM to orchestrate a melody into multiple parts.

    Args:
        notes:        Quantised notes from transcribe.quantize_to_grid()
        key:          Key string, e.g. 'D major'
        tempo_bpm:    Tempo in BPM
        total_beats:  Length of piece in beats
        instruments:  Part names (defaults to String Trio)
        style_prompt: Combined preset hint + user free-text direction
        max_retries:  Number of extra attempts on JSON parse failure

    Returns:
        Orchestration dict: {"key": str, "tempo": float, "parts": {name: [notes]}}

    Raises:
        RuntimeError: If Ollama is unreachable or all retries fail.
    """
    if instruments is None:
        instruments = ["strings_melody", "strings_harmony", "bass"]

    ok, msg = check_ollama()
    if not ok:
        raise RuntimeError(msg)

    last_error = None
    for attempt in range(1 + max_retries):
        prompt = build_orchestration_prompt(
            notes, key, tempo_bpm, total_beats, instruments, style_prompt
        )

        try:
            raw = _call_ollama(prompt)
            result = _parse_orchestration(raw, instruments, total_beats)
            return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            # On retry, simplify the melody (every other note) to reduce token load
            if attempt < max_retries:
                notes = notes[::2] if len(notes) > 8 else notes

    raise RuntimeError(
        f"LLM failed to produce valid orchestration after {1 + max_retries} attempts.\n"
        f"Last error: {last_error}\n"
        f"Try a different model or check that ollama is healthy."
    )


def orchestrate_streaming(
    notes: list[dict],
    key: str,
    tempo_bpm: float,
    total_beats: float,
    melody_instrument: str,
    harmony_instruments: list[str],
    style_prompt: str = "",
    notes_per_beat: float = 1.0,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    chord_chart: str = "",
    chord_mode: str = "Strict",
    rhythm_style: str = "Free",
    harmony_style: str = "Block chords",
    beats_per_chord: float = 4.0,
    forbidden_chords: list[str] | None = None,
    chord_timeline: list[tuple[float, str]] | None = None,
    time_sig: str = "4/4",
    max_retries: int = 2,
) -> Generator[tuple[str, dict | None], None, None]:
    """
    Generator that streams LLM harmony generation. Yields (progress, result_or_None).

    The melody track is NOT generated here — it is built directly from the
    transcription in app.py and injected into the orchestration dict afterwards.
    Only harmony_instruments are requested from the LLM.
    """
    ok, msg = check_ollama()
    if not ok:
        raise RuntimeError(msg)

    melody_notes = notes  # kept for retry simplification reference
    last_error = None

    for attempt in range(1 + max_retries):
        prompt = build_orchestration_prompt(
            melody_notes, melody_instrument, key, tempo_bpm, total_beats,
            harmony_instruments, style_prompt, notes_per_beat, chord_chart,
            chord_mode, rhythm_style, forbidden_chords, time_sig,
            harmony_style, beats_per_chord,
        )

        full_text = ""
        char_count = 0
        for chunk in _stream_ollama(prompt, max_tokens, temperature):
            full_text += chunk
            char_count += len(chunk)
            yield f"{char_count:,} characters generated…", None

        try:
            # Harmony instruments are all replaced by inject_algo_parts — don't
            # retry just because the LLM generated sparse harmony that will be discarded.
            # Only validate non-algo parts (any harmony instrument that is NOT chordal/bass).
            non_algo_harmony = [
                i for i in harmony_instruments
                if i not in CHORDAL_INSTRUMENTS and i not in BASS_INSTRUMENTS
            ]
            # Empty list is valid: all harmony replaced by algo, nothing to validate
            validate_insts = non_algo_harmony
            result = _parse_orchestration(full_text, validate_insts, total_beats, notes_per_beat)
            ranges = dict(INSTRUMENT_RANGES)
            ts_info = {}
            try:
                from src.prompts import TIME_SIGNATURES
                ts_info = TIME_SIGNATURES.get(time_sig, TIME_SIGNATURES["4/4"])
            except Exception:
                pass
            beats_per_bar = ts_info.get("beats_per_bar", 4.0)

            # Strip any spurious parts the LLM invented beyond what was requested.
            # LLMs often add extra instruments; we must not let them contaminate the output.
            _allowed = set(harmony_instruments) | {melody_instrument}
            for _inst in list(result["parts"].keys()):
                if _inst not in _allowed:
                    del result["parts"][_inst]

            # Guarantee all requested harmony instruments are present — LLMs sometimes
            # omit a part or use a wrong name. Missing parts get filled by inject_algo_parts.
            for inst in harmony_instruments:
                result["parts"].setdefault(inst, [])

            # Snap LLM-generated harmony to the actual scale so chromatic drift can't
            # survive into the output. Algo-replaced parts are not snapped (their
            # voicings are derived directly from chord symbols and are already correct).
            parsed_key = parse_key_string(key)
            if parsed_key and non_algo_harmony:
                _root_k, _ivs_k = parsed_key
                _root_pc_k = _NOTE_PC.get(_root_k, 0)
                _scale_pcs = frozenset((_root_pc_k + i) % 12 for i in _ivs_k)
                for inst in non_algo_harmony:
                    if result["parts"].get(inst):
                        _tmp = {"parts": {inst: result["parts"][inst]}}
                        _tmp = snap_all_parts_to_scale(_tmp, _scale_pcs, ranges)
                        result["parts"][inst] = _tmp["parts"][inst]

            if chord_timeline:
                # Algorithmic replacement: guarantees coherent chords and bass
                result = inject_algo_parts(
                    result, chord_timeline, total_beats,
                    CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
                    ranges, harmony_style, beats_per_bar, key,
                )
            yield "complete", result
            return
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            if attempt < max_retries:
                melody_notes = melody_notes[::2] if len(melody_notes) > 8 else melody_notes
                yield f"Attempt {attempt + 1} failed ({e}) — retrying with simplified melody…", None

    raise RuntimeError(
        f"LLM failed to produce valid orchestration after {1 + max_retries} attempts.\n"
        f"Last error: {last_error}\n"
        f"Try a different model or check that Ollama is healthy."
    )


def compose_streaming(
    key: str,
    tempo_bpm: float,
    total_beats: float,
    all_instruments: list[str],
    style_prompt: str = "",
    notes_per_beat: float = 1.0,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    chord_chart: str = "",
    chord_mode: str = "Strict",
    rhythm_style: str = "Free",
    harmony_style: str = "Block chords",
    beats_per_chord: float = 4.0,
    forbidden_chords: list[str] | None = None,
    chord_timeline: list[tuple[float, str]] | None = None,
    time_sig: str = "4/4",
    max_retries: int = 2,
) -> Generator[tuple[str, dict | None], None, None]:
    """
    Generator that streams LLM composition (all parts, no fixed melody).
    Yields (progress, result_or_None). Same streaming protocol as orchestrate_streaming.
    """
    ok, msg = check_ollama()
    if not ok:
        raise RuntimeError(msg)

    last_error = None
    for attempt in range(1 + max_retries):
        prompt = build_composition_prompt(
            key, tempo_bpm, total_beats, all_instruments,
            style_prompt, notes_per_beat, chord_chart,
            chord_mode, rhythm_style, forbidden_chords, time_sig,
            harmony_style, beats_per_chord,
        )
        full_text, char_count = "", 0
        for chunk in _stream_ollama(prompt, max_tokens, temperature):
            full_text  += chunk
            char_count += len(chunk)
            yield f"{char_count:,} characters generated…", None

        try:
            # Only validate the melody instrument(s). Chordal and bass parts are
            # replaced by inject_algo_parts regardless of LLM quality — validating
            # them causes spurious retries when the LLM generates sparse harmony.
            melody_insts = [
                i for i in all_instruments
                if i not in CHORDAL_INSTRUMENTS and i not in BASS_INSTRUMENTS
            ]
            validate_insts = melody_insts if melody_insts else all_instruments
            result = _parse_orchestration(full_text, validate_insts, total_beats, notes_per_beat)
            ranges = dict(INSTRUMENT_RANGES)
            melody_inst = all_instruments[0]

            # Strip any spurious parts the LLM invented beyond what was requested.
            _allowed_all = set(all_instruments)
            for _inst in list(result["parts"].keys()):
                if _inst not in _allowed_all:
                    del result["parts"][_inst]

            # Guarantee every requested instrument is present (LLM may omit or misname parts)
            for inst in all_instruments:
                result["parts"].setdefault(inst, [])

            # Step 1: snap melody to scale (LLM melody is kept)
            parsed_key = parse_key_string(key)
            if parsed_key:
                _root, _intervals = parsed_key
                _root_pc = _NOTE_PC.get(_root, 0)
                scale_pcs = frozenset((_root_pc + i) % 12 for i in _intervals)
                melody_part = result["parts"].get(melody_inst, [])
                tmp = {"parts": {melody_inst: melody_part}}
                tmp = snap_all_parts_to_scale(tmp, scale_pcs, ranges)
                result["parts"][melody_inst] = tmp["parts"][melody_inst]

            # Step 2: replace harmony/bass with algorithmically generated versions
            if chord_timeline:
                try:
                    from src.prompts import TIME_SIGNATURES
                    ts_info = TIME_SIGNATURES.get(time_sig, TIME_SIGNATURES["4/4"])
                except Exception:
                    ts_info = {"beats_per_bar": 4.0}
                beats_per_bar = ts_info.get("beats_per_bar", 4.0)
                result = inject_algo_parts(
                    result, chord_timeline, total_beats,
                    CHORDAL_INSTRUMENTS, BASS_INSTRUMENTS,
                    ranges, harmony_style, beats_per_bar, key,
                )
            yield "complete", result
            return
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            if attempt < max_retries:
                yield f"Attempt {attempt + 1} failed ({e}) — retrying…", None

    raise RuntimeError(
        f"LLM failed to produce valid composition after {1 + max_retries} attempts.\n"
        f"Last error: {last_error}\n"
        f"Try a different model or check that Ollama is healthy."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stream_ollama(
    prompt: str, max_tokens: int = 16384, temperature: float = 0.7
) -> Generator[str, None, None]:
    """Yield raw text chunks from a streaming Ollama request."""
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "format": "json",
        "stream": True,
        "options": {"temperature": temperature, "top_p": 0.9, "num_predict": max_tokens},
    }
    with requests.post(_GENERATE_URL, json=payload, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done", False):
                    break


def _call_ollama(prompt: str, max_tokens: int = 16384, temperature: float = 0.7) -> str:
    """Accumulate a full streaming response and return it as a single string."""
    return "".join(_stream_ollama(prompt, max_tokens, temperature))


def _parse_orchestration(
    raw: str,
    expected_instruments: list[str],
    total_beats: float,
    notes_per_beat: float = 1.0,
) -> dict:
    """
    Extract and validate the orchestration JSON from LLM output.

    Checks density (minimum notes per part) and coverage (each part must reach
    at least 80% of total_beats) so stub or early-stopping responses are rejected.
    """
    text = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        text = fence_match.group(1)

    data = json.loads(text)

    if "parts" not in data:
        raise ValueError("LLM response missing 'parts' key.")

    # At least 1 note per 6 beats — catches near-empty stubs without being so
    # strict that models generating quarter-note density on long pieces always retry.
    min_notes    = max(4, int(total_beats / 6))
    min_coverage = total_beats * 0.60             # last note must reach 60% of duration

    for inst in expected_instruments:
        if inst not in data["parts"]:
            raise KeyError(f"Missing part '{inst}' in LLM response.")
        part = data["parts"][inst]
        if not isinstance(part, list) or len(part) == 0:
            raise ValueError(f"Part '{inst}' is empty.")
        non_dict = [n for n in part if not isinstance(n, dict)]
        if non_dict:
            raise ValueError(
                f"Part '{inst}' contains {len(non_dict)} non-object items "
                f"(LLM returned note names as strings instead of dicts)."
            )
        if len(part) < min_notes:
            raise ValueError(
                f"Part '{inst}' too sparse: {len(part)} notes for "
                f"{total_beats:.0f} beats (need ≥{min_notes})."
            )
        last_end = max(
            n.get("start_beat", 0) + n.get("duration_beats", 1)
            for n in part
            if isinstance(n, dict)
        )
        if last_end < min_coverage:
            raise ValueError(
                f"Part '{inst}' stops at beat {last_end:.1f} but arrangement "
                f"runs to {total_beats:.0f} (need ≥{min_coverage:.0f})."
            )

    # Normalise field name: LLMs occasionally drop the trailing 's'
    for part_notes in data["parts"].values():
        for note in part_notes:
            if "duration_beat" in note and "duration_beats" not in note:
                note["duration_beats"] = note.pop("duration_beat")

    # Clamp out-of-range notes to the nearest valid octave.
    # LLMs regularly ignore range instructions; this silently corrects them rather than
    # discarding notes, so we get a complete arrangement even if pitches needed adjustment.
    for inst, part_notes in data["parts"].items():
        lo, hi = INSTRUMENT_RANGES.get(inst, (36, 96))
        for note in part_notes:
            if not isinstance(note, dict) or "note" not in note:
                continue
            try:
                from src.harmony import note_name_to_midi, midi_to_note_name
                midi = note_name_to_midi(note["note"])
                if midi < lo:
                    # Shift up by octaves until in range
                    while midi < lo:
                        midi += 12
                    note["note"] = midi_to_note_name(min(midi, hi))
                elif midi > hi:
                    while midi > hi:
                        midi -= 12
                    note["note"] = midi_to_note_name(max(midi, lo))
            except Exception:
                pass

    return data
