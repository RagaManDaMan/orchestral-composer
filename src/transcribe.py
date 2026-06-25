"""
Audio → melody MIDI transcription.

Two backends are available:
  - "pyin":  librosa's probabilistic YIN — fast, no GPU needed, works well for
             clear monophonic sources. Struggles with vibrato, glides, and soft
             note onsets common in Indian classical / expressive singing.
  - "crepe": torchcrepe (PyTorch port of CREPE neural pitch tracker) — slower
             but dramatically more accurate on ornamental / expressive melodic
             lines. Handles gamakas, meends, and vibrato well. Requires
             `pip install torchcrepe`.
"""

import os
import sys
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SAMPLE_RATE, GRID_SIZE_BEATS, MIN_NOTE_DURATION_SEC
from src.vocal_processor import process_vocal

# Deferred — librosa triggers scipy DLL load; only import when actually transcribing
librosa = None

def _ensure_librosa():
    global librosa
    if librosa is None:
        import librosa as _l
        librosa = _l

_CREPE_SR        = 16000   # torchcrepe requires 16 kHz input
_CREPE_HOP       = 160     # 10 ms per frame at 16 kHz
_CREPE_VOICED_TH = 0.21    # periodicity threshold; lower = more notes detected

# Krumhansl-Schmuckler tonal profiles (pitch class → tonal weight)
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NOTE_NAMES    = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_note: int) -> str:
    """62 → 'D4'"""
    octave = midi_note // 12 - 1
    return f"{_NOTE_NAMES[midi_note % 12]}{octave}"


def transcribe_audio(
    audio_path: str,
    sr: Optional[int] = None,
    backend: str = "crepe",
    key: str = "C major",
    reverb_preset: str = "studio",
    reverb_mix: float = 0.18,
    pitch_strength: float = 0.8,
    noise_reduce_strength: float = 0.7,
) -> list[dict]:
    """
    Transcribe a monophonic audio file to a list of MIDI note events.

    Args:
        audio_path:            Path to the audio file.
        sr:                    Sample rate override (None = use config default).
        backend:               "crepe" (default, more accurate) or "pyin" (faster, no GPU).
        key:                   Musical key for pitch correction (e.g. "D minor").
        reverb_preset:         Reverb room type ("studio", "small_room", "hall", "cathedral", "none").
        reverb_mix:            Wet/dry mix for reverb (0.0 = dry, 1.0 = full wet).
        pitch_strength:        How hard to snap pitch to scale (0.0 = off, 1.0 = hard Auto-Tune).
        noise_reduce_strength: How aggressively to remove background noise (0.0 = off, 1.0 = max).

    Returns:
        List of dicts: [{"midi_note": int, "start_sec": float, "duration_sec": float}, ...]
    """
    _ensure_librosa()
    # Run the full vocal enhancement pipeline before transcription:
    # noise gate → noise reduction → EQ → compression →
    # tuning fix → pitch correction → reverb → normalise
    audio_path, _ = process_vocal(
        audio_path,
        key=key,
        reverb_preset=reverb_preset,
        reverb_mix=reverb_mix,
        pitch_strength=pitch_strength,
        noise_reduce_strength=noise_reduce_strength,
    )

    if backend == "crepe":
        try:
            return _transcribe_crepe(audio_path)
        except Exception as e:
            import warnings
            warnings.warn(f"CREPE transcription failed ({e}), falling back to pyin.")

    return _transcribe_pyin(audio_path, sr)


def _transcribe_pyin(audio_path: str, sr: Optional[int] = None) -> list[dict]:
    """pyin pitch tracker (librosa). Fast; struggles with ornamental singing."""
    _ensure_librosa()
    target_sr = sr or SAMPLE_RATE
    y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    hop_length = 512  # ~23 ms at 22050 Hz

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=target_sr,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=target_sr, hop_length=hop_length)

    notes: list[dict] = []
    current_midi: Optional[int] = None
    current_start: Optional[float] = None

    for freq, voiced, t in zip(f0, voiced_flag, times):
        if voiced and freq is not None and not np.isnan(freq) and freq > 0:
            midi = int(round(librosa.hz_to_midi(freq)))
            if current_midi is None:
                current_midi, current_start = midi, t
            elif midi != current_midi:
                _flush_note(notes, current_midi, current_start, t)
                current_midi, current_start = midi, t
        else:
            if current_midi is not None:
                _flush_note(notes, current_midi, current_start, t)
                current_midi, current_start = None, None

    if current_midi is not None:
        _flush_note(notes, current_midi, current_start, times[-1])

    return [n for n in notes if n["duration_sec"] >= MIN_NOTE_DURATION_SEC]


def _transcribe_crepe(audio_path: str) -> list[dict]:
    """
    CREPE neural pitch tracker via torchcrepe. Significantly more accurate than
    pyin on ornamental / expressive singing (gamakas, meends, vibrato).
    Uses the 'full' model at 16 kHz with Viterbi decoding for smooth pitch tracks.
    """
    import torch
    import torchcrepe

    y, _ = librosa.load(audio_path, sr=_CREPE_SR, mono=True)
    audio_tensor = torch.from_numpy(y).unsqueeze(0)  # (1, T)

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    freqs, periodicity = torchcrepe.predict(
        audio_tensor,
        _CREPE_SR,
        hop_length=_CREPE_HOP,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        model="full",
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
        device=device,
    )

    freqs_np  = freqs[0].cpu().numpy()
    period_np = periodicity[0].cpu().numpy()
    times     = np.arange(len(freqs_np)) * _CREPE_HOP / _CREPE_SR

    notes: list[dict] = []
    current_midi: Optional[int] = None
    current_start: Optional[float] = None

    for freq, period, t in zip(freqs_np, period_np, times):
        voiced = (period > _CREPE_VOICED_TH) and not np.isnan(freq) and freq > 0
        if voiced:
            midi = int(round(librosa.hz_to_midi(float(freq))))
            if current_midi is None:
                current_midi, current_start = midi, t
            elif midi != current_midi:
                _flush_note(notes, current_midi, current_start, t)
                current_midi, current_start = midi, t
        else:
            if current_midi is not None:
                _flush_note(notes, current_midi, current_start, t)
                current_midi, current_start = None, None

    if current_midi is not None:
        _flush_note(notes, current_midi, current_start, times[-1])

    return [n for n in notes if n["duration_sec"] >= MIN_NOTE_DURATION_SEC]


def _flush_note(notes: list, midi: int, start: float, end: float) -> None:
    notes.append({"midi_note": midi, "start_sec": start, "duration_sec": end - start})


def quantize_to_grid(
    notes: list[dict],
    tempo_bpm: float,
    grid_size: float = GRID_SIZE_BEATS,
) -> list[dict]:
    """
    Convert note timings from seconds to beats and snap to the nearest grid.

    Args:
        notes:      Raw note list from transcribe_audio()
        tempo_bpm:  Tempo in BPM — user-supplied or from detect_tempo()
        grid_size:  Grid resolution in beats (default from config: 8th note)

    Returns:
        List of dicts with beat-based timing and note names.
    """
    beat_dur = 60.0 / tempo_bpm

    quantized = []
    for n in notes:
        start_b    = n["start_sec"]    / beat_dur
        duration_b = n["duration_sec"] / beat_dur

        q_start    = round(start_b    / grid_size) * grid_size
        q_duration = max(grid_size, round(duration_b / grid_size) * grid_size)

        quantized.append({
            "midi_note":      n["midi_note"],
            "note_name":      midi_to_note_name(n["midi_note"]),
            "start_beat":     round(q_start,    4),
            "duration_beats": round(q_duration, 4),
        })

    return quantized


def detect_tempo(audio_path: str, sr: Optional[int] = None) -> float:
    """
    Estimate tempo from an audio file using librosa's beat tracker.
    Returns BPM as a float. Falls back to DEFAULT_TEMPO_BPM if detection fails.
    """
    _ensure_librosa()
    from config import DEFAULT_TEMPO_BPM
    try:
        target_sr = sr or SAMPLE_RATE
        y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=target_sr)
        bpm = float(np.atleast_1d(tempo)[0])
        if 40 <= bpm <= 240:
            return round(bpm, 1)
    except Exception:
        pass
    return DEFAULT_TEMPO_BPM


def detect_key(notes: list[dict]) -> str:
    """
    Estimate the musical key of a note sequence using the
    Krumhansl-Schmuckler key-finding algorithm.

    Builds a pitch-class histogram weighted by note duration, then
    correlates against all 24 major/minor profiles. Returns the
    best-matching key as a string, e.g. 'D major'.
    """
    if not notes:
        return "C major"

    histogram = [0.0] * 12
    duration_key = "duration_beats" if "duration_beats" in notes[0] else "duration_sec"
    for n in notes:
        pc = n["midi_note"] % 12
        histogram[pc] += n[duration_key]

    total = sum(histogram)
    if total == 0:
        return "C major"
    histogram = [h / total for h in histogram]

    best_key, best_score = "C major", -float("inf")

    for tonic in range(12):
        for profile, mode in [(_MAJOR_PROFILE, "major"), (_MINOR_PROFILE, "minor")]:
            rotated = profile[tonic:] + profile[:tonic]
            score = sum(a * b for a, b in zip(histogram, rotated))
            if score > best_score:
                best_score = score
                best_key = f"{_NOTE_NAMES[tonic]} {mode}"

    return best_key