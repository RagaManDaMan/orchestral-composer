"""
src/melody_rnn.py
=================
Melody generator for Orchestral Composer.

Primary: sander-wood/text-to-music (BART fine-tuned on 282,870 text-music pairs)
         Takes a text style description → generates ABC notation → converts to MIDI notes
         Styles: blues, classical, folk, jazz, pop, world music
         Install: py -3.11 -m pip install transformers torch samplings

Fallback: Markov chain generator — works with zero extra dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

# Raga engine — used for Indian preset melody generation
try:
    from src.raga_engine import generate_raga_melody, get_raga_for_key, is_indian_preset
    _RAGA_AVAILABLE = True
except ImportError:
    _RAGA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Music theory helpers
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

_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
_DORIAN_INTERVALS = [0, 2, 3, 5, 7, 9, 10]


def _key_to_scale(key: str) -> list[int]:
    parts   = key.strip().split()
    root    = parts[0] if parts else "C"
    quality = parts[1].lower() if len(parts) > 1 else "major"
    root_pc = _NOTE_TO_PC.get(root, 0)
    if "dor" in quality:
        intervals = _DORIAN_INTERVALS
    elif "min" in quality:
        intervals = _MINOR_INTERVALS
    else:
        intervals = _MAJOR_INTERVALS
    return [(root_pc + i) % 12 for i in intervals]


def _midi_to_note_name(midi: int) -> str:
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _build_scale_midi(key: str, low: int = 52, high: int = 81) -> list[int]:
    pcs = _key_to_scale(key)
    return [m for m in range(low, high + 1) if m % 12 in pcs]


# ---------------------------------------------------------------------------
# ABC notation parser
# ---------------------------------------------------------------------------

_ABC_NOTE_TO_SEMITONE = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
    "c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11,
}


def _parse_abc_to_notes(abc_text: str, key: str, tempo_bpm: float, snap_to_key: bool = True) -> list[dict]:
    scale_pcs = _key_to_scale(key)

    # Default note length
    l_match = re.search(r"L:\s*(\d+)/(\d+)", abc_text)
    default_beats = (int(l_match.group(1)) / int(l_match.group(2)) * 8) if l_match else 0.5

    # Key sig accidentals
    key_sig_sharps: set[str] = set()
    key_sig_flats:  set[str] = set()
    k_match = re.search(r"K:\s*([A-Ga-g][b#]?\s*\w*)", abc_text)
    if k_match:
        sharp_order = ["F","C","G","D","A","E","B"]
        flat_order  = ["B","E","A","D","G","C","F"]
        major_sharps = {"G":1,"D":2,"A":3,"E":4,"B":5,"F#":6,"C#":7}
        major_flats  = {"F":1,"Bb":2,"Eb":3,"Ab":4,"Db":5,"Gb":6,"Cb":7}
        root = k_match.group(1).strip().split()[0]
        if root in major_sharps:
            key_sig_sharps = set(sharp_order[:major_sharps[root]])
        elif root in major_flats:
            key_sig_flats = set(flat_order[:major_flats[root]])

    # Strip headers and chord symbols
    body = re.sub(r"^[A-Za-z]:.*$", "", abc_text, flags=re.MULTILINE)
    body = re.sub(r'"[^"]*"', "", body)
    body = re.sub(r"\[[^\]]*\]", "", body)

    token_pat = re.compile(
        r"(\^{1,2}|_{1,2}|=)?"
        r"([A-Ga-gz])"
        r"([,']*)?"
        r"(\d+)?(/\d+)?"
    )

    notes_out: list[dict] = []
    beat = 0.0
    pos  = 0

    while pos < len(body):
        ch = body[pos]
        if ch in "| \t\n\r-(){}":
            pos += 1
            continue

        m = token_pat.match(body, pos)
        if not m:
            pos += 1
            continue

        accidental = m.group(1) or ""
        letter     = m.group(2)
        octave_mod = m.group(3) or ""
        dur_num    = m.group(4)
        dur_frac   = m.group(5)
        pos        = m.end()

        if dur_num and dur_frac:
            duration_beats = default_beats * int(dur_num) / int(dur_frac[1:])
        elif dur_num:
            duration_beats = default_beats * int(dur_num)
        elif dur_frac:
            duration_beats = default_beats / int(dur_frac[1:])
        else:
            duration_beats = default_beats
        duration_beats = max(0.125, duration_beats)

        if letter == "z":
            beat += duration_beats
            continue

        base_octave = 5 if letter.islower() else 4
        base_octave += octave_mod.count("'") - octave_mod.count(",")
        semitone = _ABC_NOTE_TO_SEMITONE.get(letter, 0)

        if   accidental == "^":  semitone += 1
        elif accidental == "^^": semitone += 2
        elif accidental == "_":  semitone -= 1
        elif accidental == "__": semitone -= 2
        elif not accidental:
            upper = letter.upper()
            if upper in key_sig_sharps: semitone += 1
            elif upper in key_sig_flats: semitone -= 1

        midi = max(36, min(96, base_octave * 12 + semitone))

        if snap_to_key and midi % 12 not in scale_pcs:
            best = min(scale_pcs, key=lambda pc: min(abs(midi%12-pc), 12-abs(midi%12-pc)))
            midi = (midi // 12) * 12 + best

        notes_out.append({
            "midi_note":      midi,
            "note_name":      _midi_to_note_name(midi),
            "start_beat":     round(beat, 4),
            "duration_beats": round(duration_beats, 4),
            "velocity":       80,
        })
        beat += duration_beats

    return notes_out


# ---------------------------------------------------------------------------
# Style prompt builder
# ---------------------------------------------------------------------------

def _build_style_prompt(key: str, tempo_bpm: float, style: str = "") -> str:
    parts   = key.strip().split()
    root    = parts[0] if parts else "C"
    quality = parts[1].lower() if len(parts) > 1 else "major"
    mode    = "minor" if "min" in quality else "major"
    tempo_desc = "slow" if tempo_bpm < 70 else ("medium tempo" if tempo_bpm < 100 else ("upbeat" if tempo_bpm < 140 else "fast"))

    if style and style.strip():
        base = style.strip()
        if not base.endswith("."): base += "."
        return f"This is a {tempo_desc} melody. {base} The key is {root} {mode}."

    if mode == "minor":
        return f"This is a {tempo_desc} melody in {root} minor. It has an expressive and emotive character."
    return f"This is a {tempo_desc} melody in {root} major. It has a bright and flowing character."


# ---------------------------------------------------------------------------
# ABC model
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _load_abc_model():
    if "model" in _model_cache:
        return _model_cache["tokenizer"], _model_cache["model"]
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print("[melody] Loading text-to-music model (first run ~500 MB download)...")
    tokenizer = AutoTokenizer.from_pretrained("sander-wood/text-to-music")
    model     = AutoModelForSeq2SeqLM.from_pretrained("sander-wood/text-to-music")
    model.eval()
    _model_cache["tokenizer"] = tokenizer
    _model_cache["model"]     = model
    print("[melody] Model loaded.")
    return tokenizer, model


def _generate_abc(prompt: str, max_length: int = 512, top_p: float = 0.9, temperature: float = 1.0, seed: int = 0) -> str:
    import torch
    tokenizer, model = _load_abc_model()
    if seed > 0:
        torch.manual_seed(seed)

    input_ids   = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]
    decoder_ids = torch.tensor([[model.config.decoder_start_token_id]])
    eos_id      = model.config.eos_token_id

    with torch.no_grad():
        for _ in range(max_length):
            outputs  = model(input_ids=input_ids, decoder_input_ids=decoder_ids)
            logits   = outputs.logits[:, -1, :]
            probs    = torch.softmax(logits / max(0.01, temperature), dim=-1)
            probs_np = probs[0].cpu().numpy()

            sorted_idx = np.argsort(probs_np)[::-1]
            cumsum, keep = 0.0, []
            for idx in sorted_idx:
                cumsum += probs_np[idx]
                keep.append(idx)
                if cumsum >= top_p:
                    break
            mask = np.zeros_like(probs_np)
            mask[keep] = probs_np[keep]
            if mask.sum() > 0:
                mask /= mask.sum()
            next_token = int(np.random.choice(len(mask), p=mask))

            if next_token == eos_id:
                break
            decoder_ids = torch.cat([decoder_ids, torch.tensor([[next_token]])], dim=-1)

    return tokenizer.decode(decoder_ids[0].tolist()[1:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_melody(
    key:         str   = "C major",
    tempo_bpm:   float = 90.0,
    num_steps:   int   = 128,
    temperature: float = 1.0,
    primer:      Optional[list[int]] = None,
    seed:        int   = 0,
    style:       str   = "",
    preset_name: str   = "",
) -> list[dict]:
    total_beats = num_steps / 4.0

    # Indian presets get raga-aware melody generation
    if _RAGA_AVAILABLE and preset_name and is_indian_preset(preset_name):
        try:
            raga = get_raga_for_key(key)
            notes = generate_raga_melody(
                key=key,
                tempo_bpm=tempo_bpm,
                total_beats=total_beats,
                raga=raga,
                temperature=temperature,
                seed=seed,
            )
            if notes:
                print(f"[melody] Raga melody generated ({len(notes)} notes)")
                return notes
        except Exception as e:
            print(f"[melody] Raga generation failed ({e}), falling back")

    # Western presets: try ABC model then Markov
    if abc_model_available():
        try:
            return _abc_generate(key, tempo_bpm, num_steps, temperature, seed, style)
        except Exception as e:
            print(f"[melody] ABC model failed ({e}), falling back to Markov")
    return _markov_generate(key, tempo_bpm, num_steps, temperature, primer, seed)


def _abc_generate(key, tempo_bpm, num_steps, temperature, seed, style) -> list[dict]:
    prompt   = _build_style_prompt(key, tempo_bpm, style)
    print(f"[melody] Prompt: {prompt}")
    abc_text = _generate_abc(prompt=prompt, max_length=512, top_p=0.9, temperature=temperature, seed=seed)
    print(f"[melody] ABC output ({len(abc_text)} chars)")
    notes = _parse_abc_to_notes(abc_text, key, tempo_bpm, snap_to_key=True)
    if not notes:
        raise ValueError("ABC parser returned no notes")
    target_beats = num_steps / 4.0
    notes = [n for n in notes if n["start_beat"] < target_beats]
    if not notes:
        raise ValueError("No notes within target beat range")
    return notes


# ---------------------------------------------------------------------------
# Markov fallback
# ---------------------------------------------------------------------------

_MAJOR_TRANSITIONS = np.array([
    [0.05,0.15,0.20,0.10,0.25,0.15,0.10],
    [0.20,0.05,0.30,0.15,0.10,0.15,0.05],
    [0.15,0.20,0.05,0.25,0.15,0.10,0.10],
    [0.10,0.10,0.20,0.05,0.30,0.15,0.10],
    [0.25,0.10,0.15,0.15,0.05,0.20,0.10],
    [0.15,0.10,0.10,0.15,0.20,0.05,0.25],
    [0.30,0.10,0.10,0.10,0.15,0.15,0.10],
], dtype=np.float32)

_MINOR_TRANSITIONS = np.array([
    [0.05,0.15,0.15,0.20,0.20,0.10,0.15],
    [0.20,0.05,0.25,0.15,0.15,0.10,0.10],
    [0.15,0.20,0.05,0.20,0.15,0.15,0.10],
    [0.15,0.10,0.15,0.05,0.30,0.15,0.10],
    [0.25,0.10,0.10,0.15,0.05,0.20,0.15],
    [0.10,0.15,0.15,0.10,0.20,0.05,0.25],
    [0.30,0.10,0.10,0.10,0.20,0.15,0.05],
], dtype=np.float32)

_CONTOUR_SHAPES = ["arch","valley","ascending","descending","plateau","wave"]
_RHYTHM_PATTERNS = {
    "melodic":  [(0.25,0.10),(0.50,0.30),(0.75,0.10),(1.00,0.25),(1.50,0.10),(2.00,0.15)],
    "ballad":   [(0.50,0.15),(1.00,0.35),(1.50,0.20),(2.00,0.20),(3.00,0.10)],
    "rhythmic": [(0.25,0.20),(0.50,0.40),(0.75,0.15),(1.00,0.20),(1.50,0.05)],
}


def _contour_bias(contour, position, n):
    bias = np.ones(n, dtype=np.float32)
    mid  = n // 2
    if contour == "arch":
        peak = int(position*2*mid) if position < 0.5 else int((1-position)*2*mid)
        for i in range(n): bias[i] = 1.0+0.8*np.exp(-0.3*abs(i-mid-peak//2))
    elif contour == "ascending":
        t = int(position*(n-1))
        for i in range(n): bias[i] = 1.0+0.6*np.exp(-0.4*abs(i-t))
    elif contour == "descending":
        t = int((1-position)*(n-1))
        for i in range(n): bias[i] = 1.0+0.6*np.exp(-0.4*abs(i-t))
    elif contour == "valley":
        t = mid - int(abs(position-0.5)*mid)
        for i in range(n): bias[i] = 1.0+0.8*np.exp(-0.3*abs(i-t))
    elif contour == "plateau":
        t = int(0.7*n) if position > 0.2 else int(position*n*3)
        for i in range(n): bias[i] = 1.0+0.7*np.exp(-0.3*abs(i-t))
    elif contour == "wave":
        t = int(((position*2)%1.0)*n)
        for i in range(n): bias[i] = 1.0+0.5*np.exp(-0.4*abs(i-t))
    s = bias.sum()
    return bias/s if s > 0 else bias


def _markov_generate(key, tempo_bpm, num_steps, temperature, primer, seed) -> list[dict]:
    rng         = np.random.default_rng(seed if seed > 0 else None)
    scale_midi  = _build_scale_midi(key, low=52, high=81)
    n           = len(scale_midi)
    quality     = key.strip().split()[1].lower() if len(key.strip().split()) > 1 else "major"
    transitions = _MINOR_TRANSITIONS if "min" in quality else _MAJOR_TRANSITIONS
    pcs         = _key_to_scale(key)
    root_pc     = _NOTE_TO_PC.get(key.strip().split()[0], 0)

    total_beats  = num_steps / 4.0
    rhythm_style = "ballad" if tempo_bpm < 70 else ("rhythmic" if tempo_bpm > 130 else "melodic")
    rhythm       = _RHYTHM_PATTERNS[rhythm_style]
    durations    = [d for d,_ in rhythm]
    dur_probs    = np.array([p for _,p in rhythm], dtype=np.float32)
    dur_probs   /= dur_probs.sum()

    phrase_beats = 8.0
    num_phrases  = max(1, int(total_beats/phrase_beats))
    contours     = rng.choice(_CONTOUR_SHAPES, size=num_phrases)

    tonic_cands = [i for i,m in enumerate(scale_midi) if m%12 == root_pc]
    current_idx = min(tonic_cands, key=lambda i: abs(scale_midi[i]-65)) if tonic_cands else n//2
    stable = {0,2,4}

    all_notes: list[dict] = []
    offset = 0.0

    for phrase_idx in range(num_phrases):
        remaining_total = total_beats - offset
        phrase_len      = min(phrase_beats, remaining_total)
        if phrase_len < 0.5: break

        beat = 0.0
        idx  = current_idx

        while beat < phrase_len - 0.124:
            remaining  = phrase_len - beat
            valid_mask = np.array([d <= remaining+0.01 for d in durations], dtype=bool)
            if not valid_mask.any(): break
            vp  = dur_probs * valid_mask
            vp /= vp.sum()
            dur = float(rng.choice(durations, p=vp))

            position = beat / phrase_len
            end_soon = remaining <= max(durations)*1.5

            if end_soon:
                si = [i for i in range(n) if i%7 in stable]
                if si: idx = min(si, key=lambda i: abs(i-idx))
            else:
                deg      = idx % 7
                trans    = transitions[deg].copy()
                cb       = _contour_bias(str(contours[phrase_idx]), position, n)
                db       = np.zeros(7, dtype=np.float32)
                for i in range(n): db[i%7] += cb[i]
                if db.sum() > 0: db /= db.sum()
                combined = trans * db
                if combined.sum() == 0: combined = trans.copy()
                combined = np.power(combined, 1.0/max(0.1, temperature))
                combined /= combined.sum()
                next_deg = int(rng.choice(7, p=combined))
                cands = [i for i in range(n) if i%7 == next_deg]
                if cands: idx = min(cands, key=lambda i: abs(i-idx))

            midi = scale_midi[min(idx, n-1)]
            all_notes.append({
                "midi_note":      midi,
                "note_name":      _midi_to_note_name(midi),
                "start_beat":     round(beat+offset, 4),
                "duration_beats": round(dur, 4),
                "velocity":       int(rng.integers(68, 92)),
            })
            beat += dur

        if all_notes:
            last_pc = all_notes[-1]["midi_note"] % 12
            current_deg = pcs.index(last_pc) if last_pc in pcs else 0
            tc = [i for i in range(n) if i%7 == current_deg]
            if tc: current_idx = min(tc, key=lambda i: abs(scale_midi[i]-65))

        offset += phrase_len

    return all_notes


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def abc_model_available() -> bool:
    try:
        import transformers, torch, samplings  # noqa: F401
        return True
    except ImportError:
        return False


def magenta_available() -> bool:
    return False


def status() -> str:
    if abc_model_available():
        return (
            "✓ text-to-music model ready\n"
            "  Generates melodies from style descriptions (jazz, blues, folk, classical…)"
        )
    return (
        "✓ Markov melody generator active\n"
        "  For better melodies: py -3.11 -m pip install transformers torch samplings"
    )