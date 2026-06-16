"""
src/vocal_processor.py
======================
Full vocal enhancement pipeline. Call process_vocal() on any audio file
before transcription to get clean, in-tune, studio-quality audio.

Pipeline order (matters — each step assumes the previous is done):
  1. Load audio
  2. Noise gate         — silence between words / breath noise
  3. Noise reduction    — remove room hiss / background hum
  4. EQ                 — shape the frequency spectrum for voice
  5. Compression        — even out loud/quiet dynamics
  6. Tuning fix         — shift audio to A=440 Hz reference
  7. Pitch correction   — snap notes to nearest scale degree
  8. Reverb             — add space / room ambience
  9. Normalise          — bring final level to -1 dBFS
 10. Save & return path to processed file

Dependencies (add to requirements.txt):
  noisereduce>=2.0.0
  scipy>=1.11.0       (already present)
  numpy>=1.24.0       (already present)
  librosa>=0.10.2     (already present)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve

# noisereduce is the only new dependency
try:
    import noisereduce as nr
    _HAS_NR = True
except ImportError:
    _HAS_NR = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ReverbPreset = Literal["none", "studio", "small_room", "hall", "cathedral"]

def process_vocal(
    audio_path: str,
    *,
    # --- noise ---
    noise_gate_db: float = -40.0,       # dB below peak → silence (−40 is gentle)
    noise_reduce: bool = True,          # subtract learned room noise profile
    noise_reduce_strength: float = 0.7, # 0.0–1.0, higher = more aggressive
    # --- EQ ---
    eq: bool = True,
    highpass_hz: float = 80.0,          # cut rumble / AC hum below this
    presence_boost_db: float = 2.5,     # boost 2–5 kHz (vocal clarity)
    mud_cut_db: float = -2.0,           # cut 300–500 Hz (muddiness)
    # --- compression ---
    compress: bool = True,
    comp_threshold_db: float = -18.0,   # above this level, gain is reduced
    comp_ratio: float = 3.0,            # 3:1 — gentle, natural
    comp_attack_ms: float = 10.0,
    comp_release_ms: float = 80.0,
    # --- pitch ---
    fix_tuning: bool = True,            # shift to A=440 Hz
    pitch_correct: bool = True,         # snap notes to key
    pitch_strength: float = 0.8,        # 0.0 = off, 1.0 = hard snap
    key: str = "C major",               # e.g. "D minor", "G major"
    # --- reverb ---
    reverb_preset: ReverbPreset = "studio",
    reverb_mix: float = 0.18,           # 0.0 = dry, 1.0 = full wet
    # --- output ---
    output_path: str | None = None,     # None → temp file
    sr: int = 22050,
) -> tuple[str, dict]:
    """
    Process a vocal/instrument recording through the full enhancement chain.

    Returns
    -------
    (output_path, info_dict)
        output_path  — path to the processed WAV file
        info_dict    — dict with tuning_offset_cents, notes_corrected, etc.
    """
    info: dict = {}

    # 1. Load
    audio, file_sr = librosa.load(audio_path, sr=sr, mono=True)

    # 2. Noise gate
    audio = _noise_gate(audio, threshold_db=noise_gate_db)

    # 3. Noise reduction
    if noise_reduce and _HAS_NR:
        # Use the first 0.5 s as the noise profile (assumes recording starts
        # with a moment of silence / room tone — common practice)
        noise_sample_len = int(sr * 0.5)
        noise_clip = audio[:noise_sample_len] if len(audio) > noise_sample_len else audio
        audio = nr.reduce_noise(
            y=audio,
            y_noise=noise_clip,
            sr=sr,
            prop_decrease=noise_reduce_strength,
            stationary=False,   # handles varying background noise better
        )
    elif noise_reduce and not _HAS_NR:
        info["noise_reduce_warning"] = (
            "noisereduce not installed — skipped. "
            "Run: pip install noisereduce"
        )

    # 4. EQ
    if eq:
        audio = _apply_eq(audio, sr, highpass_hz, presence_boost_db, mud_cut_db)

    # 5. Compression
    if compress:
        audio = _compress(
            audio, sr,
            threshold_db=comp_threshold_db,
            ratio=comp_ratio,
            attack_ms=comp_attack_ms,
            release_ms=comp_release_ms,
        )

    # 6. Tuning fix
    if fix_tuning:
        offset_cents = _estimate_tuning_offset(audio, sr)
        info["tuning_offset_cents"] = round(offset_cents, 1)
        if abs(offset_cents) > 5:           # only shift if meaningfully off
            semitones = offset_cents / 100.0
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

    # 7. Pitch correction
    if pitch_correct and pitch_strength > 0:
        audio, n_corrected = _pitch_correct(audio, sr, key, pitch_strength)
        info["notes_corrected"] = n_corrected

    # 8. Reverb
    if reverb_preset != "none" and reverb_mix > 0:
        audio = _apply_reverb(audio, sr, reverb_preset, reverb_mix)

    # 9. Normalise to -1 dBFS
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.99

    # 10. Save
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_processed.wav")
        os.close(fd)
    sf.write(output_path, audio, sr)

    info["output_path"] = output_path
    return output_path, info


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _noise_gate(audio: np.ndarray, threshold_db: float) -> np.ndarray:
    """
    Zero out samples below threshold_db relative to peak.
    Uses a short smoothed envelope so the gate opens/closes gradually
    (no clicks at boundaries).
    """
    peak = np.max(np.abs(audio)) + 1e-9
    threshold_linear = peak * (10 ** (threshold_db / 20))

    # Smooth the envelope with a short window so transitions aren't clicks
    envelope = np.abs(audio)
    window = int(0.01 * 22050)   # 10 ms smoothing
    if window > 1:
        kernel = np.ones(window) / window
        envelope = np.convolve(envelope, kernel, mode="same")

    gate_mask = (envelope > threshold_linear).astype(np.float32)

    # Smooth the mask edges (fade in/out over 5 ms)
    fade = max(1, int(0.005 * 22050))
    gate_mask = np.convolve(gate_mask, np.ones(fade) / fade, mode="same")
    gate_mask = np.clip(gate_mask, 0, 1)

    return audio * gate_mask


def _apply_eq(
    audio: np.ndarray,
    sr: int,
    highpass_hz: float,
    presence_boost_db: float,
    mud_cut_db: float,
) -> np.ndarray:
    """
    Three-band EQ tailored for voice:
      • High-pass at highpass_hz    — remove rumble
      • Peak cut at ~400 Hz         — reduce mud / boxiness
      • Peak boost at ~3 kHz        — add vocal presence / clarity
    All filters use second-order sections (SOS) for numerical stability.
    """
    # High-pass (removes rumble below 80 Hz)
    sos_hp = butter(4, highpass_hz / (sr / 2), btype="high", output="sos")
    audio = sosfilt(sos_hp, audio)

    # Mud cut: gentle bell at 400 Hz
    if abs(mud_cut_db) > 0.1:
        audio = _peak_filter(audio, sr, freq=400, gain_db=mud_cut_db, q=1.0)

    # Presence boost: gentle bell at 3 kHz
    if abs(presence_boost_db) > 0.1:
        audio = _peak_filter(audio, sr, freq=3000, gain_db=presence_boost_db, q=1.2)

    return audio


def _peak_filter(
    audio: np.ndarray, sr: int, freq: float, gain_db: float, q: float
) -> np.ndarray:
    """
    Second-order peak (bell) EQ filter.
    gain_db > 0 → boost, gain_db < 0 → cut.
    """
    A  = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)

    b0 =  1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 =  1 - alpha * A
    a0 =  1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 =  1 - alpha / A

    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    return sosfilt(sos, audio)


def _compress(
    audio: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    """
    Feed-forward dynamic range compressor.
    Reduces gain above threshold by (ratio-1)/ratio dB per dB of excess.
    Attack/release are first-order smoothed to avoid pumping artefacts.
    """
    threshold_linear = 10 ** (threshold_db / 20)
    attack_coef  = np.exp(-1 / (sr * attack_ms  / 1000))
    release_coef = np.exp(-1 / (sr * release_ms / 1000))

    envelope = np.abs(audio)
    gain      = np.ones_like(audio)
    env_state = 0.0

    for i in range(len(audio)):
        level = envelope[i]
        if level > env_state:
            env_state = attack_coef  * env_state + (1 - attack_coef)  * level
        else:
            env_state = release_coef * env_state + (1 - release_coef) * level

        if env_state > threshold_linear:
            excess_db  = 20 * np.log10(env_state / threshold_linear)
            gain_db    = -excess_db * (1 - 1 / ratio)
            gain[i]    = 10 ** (gain_db / 20)

    # Apply 3 dB make-up gain to compensate for overall level reduction
    makeup = 10 ** (3 / 20)
    return audio * gain * makeup


def _estimate_tuning_offset(audio: np.ndarray, sr: int) -> float:
    """
    Estimate how many cents the recording is off from A=440 Hz.
    Returns a value in cents (e.g. +15 means the recording is 15 cents sharp).
    """
    # librosa.estimate_tuning returns offset in fractions of a bin;
    # each bin is 1 semitone, so multiply by 100 for cents
    offset = librosa.estimate_tuning(y=audio, sr=sr)   # returns value in [-0.5, 0.5] semitones
    return float(offset * 100)


def _pitch_correct(
    audio: np.ndarray,
    sr: int,
    key: str,
    strength: float,
) -> tuple[np.ndarray, int]:
    """
    Snap detected notes to the nearest scale degree in `key`.
    strength=1.0 → hard snap (Auto-Tune), 0.5 → half-way nudge.

    Returns (corrected_audio, number_of_notes_corrected).
    """
    scale_midi = _key_to_midi_classes(key)

    # Detect pitch frame-by-frame
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )

    hop = 512
    frame_len = 2048
    output = audio.copy()
    n_corrected = 0
    i = 0

    while i < len(f0):
        # Skip unvoiced / low-confidence frames
        if not voiced_flag[i] or voiced_probs[i] < 0.6 or np.isnan(f0[i]):
            i += 1
            continue

        # Find a run of voiced frames with similar pitch
        run_start = i
        run_f0    = [f0[i]]
        i += 1
        while (i < len(f0)
               and voiced_flag[i]
               and voiced_probs[i] >= 0.6
               and not np.isnan(f0[i])
               and abs(librosa.hz_to_midi(f0[i]) - librosa.hz_to_midi(run_f0[0])) < 1.5):
            run_f0.append(f0[i])
            i += 1
        run_end = i

        # Average MIDI pitch of this note
        midi_avg = float(np.mean([librosa.hz_to_midi(x) for x in run_f0]))
        midi_class = midi_avg % 12
        nearest_class = min(scale_midi, key=lambda c: _midi_distance(midi_class, c))
        diff = _midi_distance_signed(midi_class, nearest_class)

        if abs(diff) < 0.05:
            continue   # already in tune

        # Shift this segment by diff * strength semitones
        shift = -diff * strength
        start_sample = run_start * hop
        end_sample   = min(run_end * hop + frame_len, len(audio))
        segment      = audio[start_sample:end_sample]
        corrected    = librosa.effects.pitch_shift(segment, sr=sr, n_steps=shift)
        corrected    = corrected[:end_sample - start_sample]
        output[start_sample:end_sample] = corrected
        n_corrected += 1

    return output, n_corrected


def _apply_reverb(
    audio: np.ndarray,
    sr: int,
    preset: ReverbPreset,
    mix: float,
) -> np.ndarray:
    """
    Convolutional reverb using a synthetic impulse response.
    Generates a simple exponential-decay IR that approximates each preset.
    For production use, replace _make_ir() with a real .wav impulse response.
    """
    ir = _make_ir(preset, sr)
    wet = fftconvolve(audio, ir)[:len(audio)]
    return audio * (1 - mix) + wet * mix


def _make_ir(preset: ReverbPreset, sr: int) -> np.ndarray:
    """
    Synthetic impulse response. A real IR (sampled from an actual room
    with a starter pistol or sine sweep) sounds dramatically better — 
    but this is free, requires no files, and is good enough to hear the effect.

    Preset parameters:
      pre_delay_ms  — gap before reverb starts (room size feel)
      decay_ms      — how long the tail lasts
      diffusion     — randomness of the decay (higher = smoother)
    """
    params = {
        "studio":     dict(pre_delay_ms=8,  decay_ms=400,  diffusion=0.6),
        "small_room": dict(pre_delay_ms=5,  decay_ms=250,  diffusion=0.5),
        "hall":       dict(pre_delay_ms=20, decay_ms=1800, diffusion=0.8),
        "cathedral":  dict(pre_delay_ms=40, decay_ms=4000, diffusion=0.9),
    }
    p = params.get(preset, params["studio"])
    pre  = int(sr * p["pre_delay_ms"] / 1000)
    size = int(sr * p["decay_ms"]     / 1000)
    t    = np.linspace(0, 1, size)
    rng  = np.random.default_rng(42)   # fixed seed → deterministic
    noise = rng.standard_normal(size)
    decay = np.exp(-6 * t)             # exponential tail
    # Add diffuse scatter
    scatter = rng.standard_normal(size) * p["diffusion"] * 0.1
    ir_body = noise * decay + scatter * decay
    ir = np.concatenate([np.zeros(pre), ir_body])
    ir /= np.max(np.abs(ir)) + 1e-9
    return ir.astype(np.float32)


# ---------------------------------------------------------------------------
# Music theory helpers
# ---------------------------------------------------------------------------

_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]

_NOTE_NAME_TO_MIDI: dict[str, int] = {
    "C": 0,  "C#": 1,  "Db": 1,
    "D": 2,  "D#": 3,  "Eb": 3,
    "E": 4,  "F": 5,   "F#": 6,  "Gb": 6,
    "G": 7,  "G#": 8,  "Ab": 8,
    "A": 9,  "A#": 10, "Bb": 10,
    "B": 11,
}


def _key_to_midi_classes(key: str) -> list[int]:
    """
    Parse a key string like "D minor" or "F# major" into a list of
    MIDI pitch classes (0–11) belonging to that scale.
    """
    parts   = key.strip().split()
    root    = parts[0] if parts else "C"
    quality = parts[1].lower() if len(parts) > 1 else "major"
    root_pc = _NOTE_NAME_TO_MIDI.get(root, 0)
    intervals = _MINOR_INTERVALS if "min" in quality else _MAJOR_INTERVALS
    return [(root_pc + i) % 12 for i in intervals]


def _midi_distance(a: float, b: float) -> float:
    """Minimum circular distance between two pitch classes (0–11)."""
    d = abs(a - b) % 12
    return min(d, 12 - d)


def _midi_distance_signed(source: float, target: float) -> float:
    """
    Signed distance from source to nearest target (positive = source is sharp).
    Returns value in [-6, 6].
    """
    d = (source - target) % 12
    if d > 6:
        d -= 12
    return d


# ---------------------------------------------------------------------------
# Convenience: process and return info string for UI display
# ---------------------------------------------------------------------------

def process_vocal_ui(
    audio_path: str,
    key: str = "C major",
    reverb_preset: ReverbPreset = "studio",
    reverb_mix: float = 0.18,
    pitch_strength: float = 0.8,
    noise_reduce_strength: float = 0.7,
) -> tuple[str, str]:
    """
    Thin wrapper for Gradio. Returns (output_path, status_message).
    """
    if not audio_path:
        return audio_path, "No audio provided."
    try:
        out_path, info = process_vocal(
            audio_path,
            key=key,
            reverb_preset=reverb_preset,
            reverb_mix=reverb_mix,
            pitch_strength=pitch_strength,
            noise_reduce_strength=noise_reduce_strength,
        )
        lines = ["✓ Vocal processing complete"]
        if "tuning_offset_cents" in info:
            c = info["tuning_offset_cents"]
            lines.append(f"  Tuning offset corrected: {c:+.1f} cents")
        if "notes_corrected" in info:
            lines.append(f"  Notes pitch-corrected: {info['notes_corrected']}")
        if "noise_reduce_warning" in info:
            lines.append(f"  ⚠ {info['noise_reduce_warning']}")
        lines.append(f"  Reverb: {reverb_preset}  mix={int(reverb_mix*100)}%")
        return out_path, "\n".join(lines)
    except Exception as e:
        return audio_path, f"✗ Processing failed: {e}"
