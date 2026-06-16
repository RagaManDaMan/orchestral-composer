"""
src/indian_audio_engine.py
==========================
Sample-based audio rendering engine for Carnatic and Hindustani classical presets.

Pipeline:
  1. Tanpura — Sa-Pa-Sa-Sa drone in the raga key
  2. Veena — main melodic body
  3. Violin — occasional harmonic pops at phrase peaks
  4. Bansuri / sitar — sparse exciting embellishments
  5. Mridangam — continuous adi tala with fills
  6. Ensemble master bus → 24-bit stereo WAV

Pitch shifting uses Rubber Band (pyrubberband) when available — much cleaner
than librosa on sustained Indian timbres. Falls back to librosa automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt, fftconvolve

_SRC_DIR     = Path(__file__).parent
_PROJECT_DIR = _SRC_DIR.parent
_SAMPLES_DIR = _PROJECT_DIR / "samples"

if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

try:
    from config import USE_NEURAL_INSTRUMENTS
except ImportError:
    USE_NEURAL_INSTRUMENTS = False

from src.carnatic_ensemble import (
    make_embellishments,
    make_violin_harmony_pops,
    melody_activity_envelope,
    mridangam_pattern_for_beat,
    tag_phrase_boundaries,
)
from src.raga_engine import resolve_raga, raga_scale_pcs, snap_notes_to_raga

SR = 44100

# Pitch-shift backend detection (logged once)
_PITCH_BACKEND: str | None = None


def _detect_pitch_backend() -> str:
    global _PITCH_BACKEND
    if _PITCH_BACKEND is not None:
        return _PITCH_BACKEND
    try:
        import pyrubberband as pyrb  # noqa: F401
        _test = np.zeros(1024, dtype=np.float32)
        pyrb.pitch_shift(_test, SR, 0)
        _PITCH_BACKEND = "rubberband"
        print("[indian_audio] Pitch shift: Rubber Band (high quality)")
    except Exception:
        _PITCH_BACKEND = "librosa"
        print("[indian_audio] Pitch shift: librosa fallback (install pyrubberband + rubberband for better quality)")
    return _PITCH_BACKEND


def _pitch_shift(y: np.ndarray, n_steps: float) -> np.ndarray:
    """High-quality pitch shift — Rubber Band preferred, librosa fallback."""
    if abs(n_steps) < 0.05:
        return y
    backend = _detect_pitch_backend()
    if backend == "rubberband":
        try:
            import pyrubberband as pyrb
            return pyrb.pitch_shift(y, SR, n_steps).astype(np.float32)
        except Exception:
            pass
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps).astype(np.float32)


# ---------------------------------------------------------------------------
# Sample loader
# ---------------------------------------------------------------------------

def _load(path: str | Path, mono: bool = True) -> np.ndarray:
    path = str(path)
    y, _ = librosa.load(path, sr=SR, mono=mono)
    return y.astype(np.float32)


def _find_sample(folder: str, keywords: list[str] | None = None) -> Path | None:
    d = _SAMPLES_DIR / folder
    if not d.exists():
        return None
    exts = {".wav", ".mp3", ".ogg", ".flac"}
    candidates = [f for f in d.iterdir() if f.suffix.lower() in exts]
    if not candidates:
        return None
    if keywords:
        for kw in keywords:
            for c in candidates:
                if kw.lower() in c.name.lower():
                    return c
    return candidates[0]


def _find_all(folder: str) -> list[Path]:
    d = _SAMPLES_DIR / folder
    if not d.exists():
        return []
    exts = {".wav", ".mp3", ".ogg", ".flac"}
    return [f for f in sorted(d.iterdir()) if f.suffix.lower() in exts]


# ---------------------------------------------------------------------------
# Tanpura drone
# ---------------------------------------------------------------------------

_NOTE_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

_TANPURA_SAMPLE_ROOT = "C"


def _detect_tanpura_root(y: np.ndarray) -> str:
    try:
        y_short = y[:SR * 15]
        f0, voiced, _ = librosa.pyin(
            y_short,
            fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C4"),
            sr=SR,
        )
        voiced_f0 = f0[voiced & ~np.isnan(f0)]
        if len(voiced_f0) > 0:
            low_f0 = voiced_f0[voiced_f0 < float(np.percentile(voiced_f0, 50))]
            sa_hz = float(np.median(low_f0)) if len(low_f0) > 0 else float(np.median(voiced_f0))
            sa_note = librosa.hz_to_note(sa_hz, unicode=False)
            pc = sa_note.rstrip("0123456789")
            print(f"[indian_audio] Tanpura Sa detected: {sa_hz:.1f}Hz = {sa_note} (pitch class: {pc})")
            return pc
    except Exception as e:
        print(f"[indian_audio] Tanpura pitch detection failed: {e}")
    return _TANPURA_SAMPLE_ROOT


def _make_tanpura_drone(key_root: str, total_seconds: float, volume: float = 0.55) -> np.ndarray:
    """
    Sa–Pa–Sa–Sa drone in key_root. Uses a recording only when already in key;
    otherwise synthesises exact Do–Fa–Do–Do tuning (avoids heavy pitch-shift).
    """
    sample_path = _find_sample("tanpura")
    if sample_path is not None:
        print(f"[indian_audio] Checking tanpura sample: {sample_path.name}")
        drone = _load(sample_path)
        detected_root  = _detect_tanpura_root(drone)
        root_pc        = _NOTE_TO_PC.get(key_root, 0)
        sample_root_pc = _NOTE_TO_PC.get(detected_root, _NOTE_TO_PC.get(_TANPURA_SAMPLE_ROOT, 0))
        semitones      = (root_pc - sample_root_pc) % 12
        if semitones > 6:
            semitones -= 12

        if abs(semitones) == 0:
            print(f"[indian_audio] Tanpura sample in {key_root} — looping recording")
            return _crossfade_loop(drone, total_seconds) * volume

    print(f"[indian_audio] Synthesising Sa-Pa-Sa-Sa tanpura in {key_root}")
    return _synth_tanpura(key_root, total_seconds, volume)


def _synth_tanpura(key_root: str, total_seconds: float, volume: float = 0.5) -> np.ndarray:
    root_pc  = _NOTE_TO_PC.get(key_root, 0)
    sa_freq  = 261.63 * (2 ** (root_pc / 12))
    pa_freq  = sa_freq * 1.5
    sa2_freq = sa_freq * 0.5

    n_samples = int(total_seconds * SR)
    t = np.linspace(0, total_seconds, n_samples, endpoint=False)
    out = np.zeros(n_samples, dtype=np.float32)

    for freq, amp in [(sa2_freq, 0.4), (pa_freq, 0.3), (sa_freq, 0.3)]:
        string = np.zeros(n_samples, dtype=np.float32)
        for h in range(1, 9):
            h_amp = amp / (h ** 0.7)
            string += h_amp * np.sin(2 * np.pi * freq * h * t)
        pluck_len = int(SR * 0.8)
        env = np.ones(n_samples, dtype=np.float32)
        env[:pluck_len] *= np.linspace(0, 1, pluck_len) ** 0.3
        out += string * env

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * volume
    return out


# ---------------------------------------------------------------------------
# Melody instrument
# ---------------------------------------------------------------------------

_MELODY_CACHE: dict[str, np.ndarray] = {}
_NOTE_SAMPLE_CACHE: dict[str, np.ndarray] = {}


def _load_melody_sample(instrument: str) -> np.ndarray | None:
    folder = "veena" if "veena" in instrument else "bansuri"
    path   = _find_sample(folder)
    if path is None:
        alt = "bansuri" if folder == "veena" else "veena"
        path = _find_sample(alt)
    if path is None:
        return None

    key = str(path)
    if key not in _MELODY_CACHE:
        print(f"[indian_audio] Loading melody sample: {path.name}")
        y = _load(path)
        y_trimmed, _ = librosa.effects.trim(y, top_db=40)
        _MELODY_CACHE[key] = y_trimmed[:int(SR * 3.0)]

    return _MELODY_CACHE[key]


def _detect_sample_pitch(y: np.ndarray) -> float:
    try:
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=SR,
        )
        voiced_f0 = f0[voiced & ~np.isnan(f0)]
        if len(voiced_f0) > 0:
            return float(np.median(voiced_f0))
    except Exception:
        pass
    return 261.63


def _load_chromatic_library(instrument: str) -> dict[str, np.ndarray]:
    folder = "veena" if "veena" in instrument or "sitar" in instrument else "bansuri"
    if instrument == "violin":
        folder = "violin"
    d = _SAMPLES_DIR / folder
    library: dict[str, np.ndarray] = {}

    if not d.exists():
        return library

    for f in sorted(d.glob("*.wav")):
        note_name = f.stem
        cache_key = str(f)
        if cache_key not in _NOTE_SAMPLE_CACHE:
            _NOTE_SAMPLE_CACHE[cache_key] = _load(f)
        library[note_name] = _NOTE_SAMPLE_CACHE[cache_key]

    if library:
        print(f"[indian_audio] Loaded {len(library)} {instrument} samples: {sorted(library.keys())}")
    return library


def _chromatic_distance(semitones: float) -> float:
    """Circular distance in semitones (0 = unison, 6 = tritone)."""
    s = abs(semitones) % 12
    return min(s, 12 - s)


def _find_nearest_sample(
    target_note: str,
    library: dict[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    """
    Find the best sample for target_note.
    Prefers exact matches, then small chromatic shifts, then octave shifts
    (±12) which sound much cleaner than large chromatic pitch-shifts.
    """
    if target_note in library:
        return library[target_note], 0.0

    try:
        target_midi = librosa.note_to_midi(target_note)
    except Exception:
        return next(iter(library.values())), 0.0

    best_note: str | None = None
    best_shift = 0.0
    best_score = float("inf")

    for note_name in library:
        try:
            midi = librosa.note_to_midi(note_name)
        except Exception:
            continue
        shift = target_midi - midi
        chrom = _chromatic_distance(shift)
        # Octave shifts score lower than chromatic shifts of the same size
        score = chrom + abs(shift) * 0.02
        if score < best_score:
            best_score = score
            best_note = note_name
            best_shift = shift

    if best_note is None:
        return next(iter(library.values())), 0.0

    return library[best_note], float(best_shift)


def _find_in_scale_sample(
    target_note: str,
    library: dict[str, np.ndarray],
    scale_pcs: frozenset[int],
) -> tuple[np.ndarray, float, str]:
    """
    Pick a sample that minimises pitch-shift — prefer exact in-scale match.
    Returns (audio, semitones, resolved_note_name).
    """
    if target_note in library:
        return library[target_note], 0.0, target_note

    try:
        target_midi = librosa.note_to_midi(target_note)
    except Exception:
        first = next(iter(library.keys()))
        return library[first], 0.0, first

    # Exact in-scale notes available in library
    in_scale: list[tuple[int, str, int]] = []
    for note_name in library:
        try:
            midi = librosa.note_to_midi(note_name)
        except Exception:
            continue
        if midi % 12 in scale_pcs:
            in_scale.append((abs(midi - target_midi), note_name, midi))

    if in_scale:
        in_scale.sort()
        _, note_name, midi = in_scale[0]
        shift = float(target_midi - midi)
        return library[note_name], shift, note_name

    sample, shift = _find_nearest_sample(target_note, library)
    return sample, shift, target_note


def _envelope_note(note_audio: np.ndarray, attack_ms: float = 8.0, release_ms: float = 150.0) -> np.ndarray:
    """Apply smooth attack and release to avoid clicks."""
    out = note_audio.copy()
    attack = min(int(SR * attack_ms / 1000), len(out) // 6)
    release = min(int(SR * release_ms / 1000), len(out) // 3)
    if attack > 1:
        out[:attack] *= np.linspace(0, 1, attack) ** 0.5
    if release > 1:
        out[-release:] *= np.linspace(1, 0, release)
    return out


def _place_note(
    out: np.ndarray,
    start: int,
    note_audio: np.ndarray,
    gain: float,
    crossfade_ms: float = 30.0,
    monophonic: bool = False,
) -> None:
    """Place a note into the mix buffer. Monophonic = replace (no muddy overlaps)."""
    if start >= len(out) or len(note_audio) == 0:
        return

    end = min(start + len(note_audio), len(out))
    n = end - start
    chunk = _envelope_note(note_audio[:n]) * gain

    if monophonic:
        out[start:end] = chunk
        return

    xf = min(int(SR * crossfade_ms / 1000), n // 3, start)
    if xf > 1:
        fade_out = np.linspace(1, 0, xf)
        fade_in  = np.linspace(0, 1, xf)
        region   = out[start:start + xf]
        out[start:start + xf] = region * fade_out + chunk[:xf] * fade_in
        out[start + xf:end] += chunk[xf:]
    else:
        out[start:end] += chunk


def _role_gain(note: dict) -> float:
    """Volume scale by ensemble role."""
    role = note.get("role", "main")
    if role == "grace":
        return 0.30
    if role == "harmony":
        return 0.58
    if role == "embellishment":
        return 0.40
    if note.get("phrase_end"):
        return 1.05
    return 1.0


def _prepare_note_audio(sample: np.ndarray, note: dict, beat_dur: float) -> np.ndarray:
    """Trim or extend sample with natural decay grace period."""
    note_dur_samples = int(note["duration_beats"] * beat_dur * SR)
    grace = int(SR * (1.6 if note.get("phrase_end") else 1.2))

    if len(sample) <= note_dur_samples + grace:
        return sample.copy()

    use_len = note_dur_samples + grace
    note_audio = sample[:use_len].copy()
    fade_len = min(int(SR * 0.25), use_len // 3)
    if fade_len > 1:
        note_audio[-fade_len:] *= np.linspace(1, 0, fade_len)
    return note_audio


def _render_melody(
    melody_notes: list[dict],
    instrument: str,
    tempo_bpm: float,
    volume: float = 0.7,
    scale_pcs: frozenset[int] | None = None,
) -> np.ndarray:
    """Render melody — neural model → chromatic samples → synthesis fallback."""
    if not melody_notes:
        return np.array([], dtype=np.float32)

    beat_dur      = 60.0 / tempo_bpm
    total_beats   = max(n["start_beat"] + n["duration_beats"] for n in melody_notes)
    total_samples = int(total_beats * beat_dur * SR) + SR
    out = np.zeros(total_samples, dtype=np.float32)

    # Neural synthesis — optional, disabled by default (heavy to train)
    if USE_NEURAL_INSTRUMENTS:
        try:
            from train_instrument import load_model, model_available
            if model_available(instrument):
                model = load_model(instrument)
                if model is not None:
                    print(f"[indian_audio] Using neural model for {instrument}")
                    for note in melody_notes:
                        try:
                            midi      = librosa.note_to_midi(note["note"])
                            dur       = note["duration_beats"] * beat_dur
                            vel_scale = note.get("velocity", 75) / 90.0
                            dynamic   = vel_scale * 0.8 + 0.1
                            audio     = model.synthesize(midi, dur, dynamic=dynamic)
                            if audio is not None and len(audio) > 0:
                                start = int(note["start_beat"] * beat_dur * SR)
                                gain = vel_scale * volume * _role_gain(note)
                                _place_note(out, start, audio, gain)
                        except Exception:
                            continue
                    if np.max(np.abs(out)) > 0:
                        return out
                    print("[indian_audio] Neural model returned silence — falling back to samples")
        except ImportError:
            pass

    library = _load_chromatic_library(instrument)
    if library:
        print(f"[indian_audio] Rendering {instrument} from {len(library)} sliced samples")
    if not library:
        fallback = _load_melody_sample(instrument)
        if fallback is None:
            print(f"[indian_audio] No samples for {instrument} — synthesizing")
            return _synth_melody(melody_notes, tempo_bpm, volume)
        fallback_note = librosa.hz_to_note(_detect_sample_pitch(fallback), unicode=False)
        library = {fallback_note: fallback}
        print(f"[indian_audio] Using single sample fallback: {fallback_note}")

    for note in melody_notes:
        target_note = note.get("note", "C4")
        if scale_pcs is not None:
            sample, semitones, _ = _find_in_scale_sample(target_note, library, scale_pcs)
        else:
            sample, semitones = _find_nearest_sample(target_note, library)

        chrom = _chromatic_distance(semitones)
        if abs(semitones) > 0.05:
            try:
                sample = _pitch_shift(sample.copy(), semitones)
            except Exception:
                pass

        note_audio = _prepare_note_audio(sample, note, beat_dur)
        vel_scale = note.get("velocity", 75) / 90.0
        start = int(note["start_beat"] * beat_dur * SR)
        gain = vel_scale * volume * _role_gain(note)
        is_main = note.get("role") in ("main", "melody", None)
        _place_note(out, start, note_audio, gain, monophonic=is_main)

    return out


def _synth_melody(melody_notes: list[dict], tempo_bpm: float, volume: float = 0.6) -> np.ndarray:
    beat_dur = 60.0 / tempo_bpm
    total_beats = max(n["start_beat"] + n["duration_beats"] for n in melody_notes)
    total_samples = int(total_beats * beat_dur * SR) + SR
    out = np.zeros(total_samples, dtype=np.float32)

    for note in melody_notes:
        try:
            freq = librosa.note_to_hz(note["note"])
        except Exception:
            continue

        dur_samples = int(note["duration_beats"] * beat_dur * SR)
        t = np.linspace(0, note["duration_beats"] * beat_dur, dur_samples, endpoint=False)
        tone = np.zeros(dur_samples, dtype=np.float32)
        for h in range(1, 7):
            decay = np.exp(-3.0 * h * t / (note["duration_beats"] * beat_dur))
            tone += (1.0 / h) * np.sin(2 * np.pi * freq * h * t) * decay

        vel_scale = note.get("velocity", 75) / 90.0
        start = int(note["start_beat"] * beat_dur * SR)
        _place_note(out, start, tone, vel_scale * volume)

    return out


# ---------------------------------------------------------------------------
# Mridangam rhythm
# ---------------------------------------------------------------------------

_PERC_BY_TYPE: dict[str, list[np.ndarray]] = {}


def _load_perc_samples() -> dict[str, list[np.ndarray]]:
    global _PERC_BY_TYPE
    if _PERC_BY_TYPE:
        return _PERC_BY_TYPE

    by_type: dict[str, list[np.ndarray]] = {
        "thom": [], "ta": [], "din": [], "fill": [], "extra": []
    }
    for p in _find_all("mridangam"):
        stem = p.stem.lower()
        for stroke in ["thom", "ta", "din", "fill", "extra"]:
            if stem.startswith(stroke):
                try:
                    y = _load(p)
                    y_t, _ = librosa.effects.trim(y, top_db=35)
                    if len(y_t) > int(SR * 0.05):
                        by_type[stroke].append(y_t.astype(np.float32))
                except Exception:
                    pass
                break

    _PERC_BY_TYPE = {k: v for k, v in by_type.items() if v}
    if _PERC_BY_TYPE:
        print(f"[indian_audio] Mridangam: {{{', '.join(f'{k}:{len(v)}' for k, v in _PERC_BY_TYPE.items())}}}")
    return _PERC_BY_TYPE


def _pick_stroke(stroke_type: str, velocity: float = 1.0) -> np.ndarray:
    by_type = _load_perc_samples()
    fallbacks = {
        "thom": ["thom", "extra", "din", "ta"],
        "ta":   ["ta",   "extra", "din", "thom"],
        "din":  ["din",  "extra", "ta",  "thom"],
        "fill": ["fill", "extra", "ta",  "din"],
    }
    for t in fallbacks.get(stroke_type, [stroke_type]):
        variants = by_type.get(t, [])
        if variants:
            y = variants[np.random.randint(len(variants))].copy()
            return (y * velocity * (0.9 + np.random.random() * 0.2)).astype(np.float32)
    return np.zeros(int(SR * 0.1), dtype=np.float32)


def _render_mridangam(
    total_beats: float,
    tempo_bpm: float,
    beats_per_bar: float = 4.0,
    volume: float = 0.65,
) -> np.ndarray:
    """Continuous adi tala with fills every cycle — plays throughout."""
    beat_dur = 60.0 / tempo_bpm
    total_samples = int(total_beats * beat_dur * SR) + SR
    out = np.zeros(total_samples, dtype=np.float32)

    by_type = _load_perc_samples()
    if not by_type:
        print("[indian_audio] No mridangam samples — skipping percussion")
        return out

    pattern = mridangam_pattern_for_beat()

    def place(stroke: str, abs_beat: float, stroke_vel: float) -> None:
        sample = _pick_stroke(stroke, stroke_vel)
        start = int(abs_beat * beat_dur * SR)
        _place_note(out, start, sample, volume, crossfade_ms=5.0)

    beat = 0.0
    cycle = 0
    while beat < total_beats:
        pos_in_cycle = int(beat % 8.0)

        for stroke, offset, stroke_vel in pattern.get(pos_in_cycle, []):
            place(stroke, beat + offset, stroke_vel)

        if pos_in_cycle == 7 and "fill" in by_type:
            fills = by_type["fill"]
            fill = fills[np.random.randint(len(fills))].copy()
            _place_note(out, int((beat + 0.35) * beat_dur * SR), fill, volume * 0.72, crossfade_ms=5.0)
            if cycle % 2 == 0:
                fill2 = fills[np.random.randint(len(fills))].copy()
                _place_note(out, int((beat + 0.65) * beat_dur * SR), fill2, volume * 0.55, crossfade_ms=5.0)
            cycle += 1

        beat += 1.0

    return out


def _crossfade_loop(audio: np.ndarray, total_seconds: float) -> np.ndarray:
    target = int(total_seconds * SR)
    if len(audio) == 0:
        return np.zeros(target, dtype=np.float32)

    fade_len = min(int(SR * 2.0), len(audio) // 4)
    out = np.zeros(target, dtype=np.float32)
    pos = 0
    while pos < target:
        remaining = target - pos
        chunk = audio[:min(remaining, len(audio))].copy()
        if pos > 0 and fade_len > 0 and len(chunk) >= fade_len:
            chunk[:fade_len] *= np.linspace(0, 1, fade_len)
        end = pos + len(chunk)
        out[pos:end] += chunk
        pos = end

    return out


# ---------------------------------------------------------------------------
# Master bus DSP
# ---------------------------------------------------------------------------

ReverbPreset = Literal["studio", "hall"]


def _highpass(audio: np.ndarray, cutoff_hz: float = 80.0) -> np.ndarray:
    sos = butter(2, cutoff_hz, btype="high", fs=SR, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _lowpass(audio: np.ndarray, cutoff_hz: float = 8000.0) -> np.ndarray:
    sos = butter(2, cutoff_hz, btype="low", fs=SR, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _compress(
    audio: np.ndarray,
    threshold_db: float = -18.0,
    ratio: float = 2.5,
    attack_ms: float = 15.0,
    release_ms: float = 120.0,
) -> np.ndarray:
    threshold = 10 ** (threshold_db / 20)
    attack_coef  = np.exp(-1 / (SR * attack_ms  / 1000))
    release_coef = np.exp(-1 / (SR * release_ms / 1000))

    envelope = np.abs(audio)
    gain = np.ones_like(audio)
    env_state = 0.0

    for i in range(len(audio)):
        level = envelope[i]
        env_state = (
            attack_coef * env_state + (1 - attack_coef) * level
            if level > env_state
            else release_coef * env_state + (1 - release_coef) * level
        )
        if env_state > threshold:
            excess_db = 20 * np.log10(env_state / threshold + 1e-9)
            gain[i] = 10 ** (-excess_db * (1 - 1 / ratio) / 20)

    return (audio * gain * 10 ** (2 / 20)).astype(np.float32)


def _make_reverb_ir(preset: ReverbPreset) -> np.ndarray:
    params = {
        "studio": dict(pre_delay_ms=10, decay_ms=500,  diffusion=0.6),
        "hall":   dict(pre_delay_ms=25, decay_ms=2000, diffusion=0.85),
    }
    p = params[preset]
    pre  = int(SR * p["pre_delay_ms"] / 1000)
    size = int(SR * p["decay_ms"] / 1000)
    t = np.linspace(0, 1, size)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(size)
    decay = np.exp(-5 * t)
    scatter = rng.standard_normal(size) * p["diffusion"] * 0.12
    ir = np.concatenate([np.zeros(pre), (noise + scatter) * decay])
    ir /= np.max(np.abs(ir)) + 1e-9
    return ir.astype(np.float32)


def _apply_reverb(audio: np.ndarray, preset: ReverbPreset, mix: float) -> np.ndarray:
    if mix <= 0:
        return audio
    wet = fftconvolve(audio, _make_reverb_ir(preset))[:len(audio)]
    return (audio * (1 - mix) + wet * mix).astype(np.float32)


def _pan_mono(mono: np.ndarray, pan: float) -> tuple[np.ndarray, np.ndarray]:
    """Pan mono signal: pan -1.0 = full left, +1.0 = full right."""
    angle = (pan + 1.0) * np.pi / 4.0
    return mono * np.cos(angle), mono * np.sin(angle)


def _duck_under(audio: np.ndarray, envelope: np.ndarray, amount: float = 0.22) -> np.ndarray:
    """Gently reduce level when melody is active — tanpura and mridangam step back."""
    if len(envelope) != len(audio):
        envelope = envelope[:len(audio)]
    gain = 1.0 - envelope * amount
    return (audio * gain).astype(np.float32)


def _stereo_track(
    mono: np.ndarray,
    pan: float,
    reverb_preset: ReverbPreset,
    reverb_mix: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pan a mono track and apply its own reverb send."""
    dry_l, dry_r = _pan_mono(mono, pan)
    wet = _apply_reverb(mono, reverb_preset, reverb_mix)
    wet_l, wet_r = _pan_mono(wet, pan)
    return dry_l + wet_l * 0.7, dry_r + wet_r * 0.7


def _master_bus(
    drone: np.ndarray,
    veena: np.ndarray,
    violin: np.ndarray,
    bansuri: np.ndarray,
    sitar: np.ndarray,
    perc: np.ndarray,
    melody_envelope: np.ndarray | None = None,
) -> np.ndarray:
    """
    Ensemble mix:
      veena   — main body, centre-front
      violin  — harmony pops, right, wetter
      bansuri/sitar — embellishments, wide and bright
      tanpura — back, dry
      mridangam — continuous, slightly ducked under veena
    """
    if melody_envelope is None:
        melody_envelope = np.abs(veena).astype(np.float32)
        if np.max(melody_envelope) > 0:
            melody_envelope /= np.max(melody_envelope)

    drone_soft  = _duck_under(_lowpass(drone, 6000), melody_envelope, 0.15) * 0.95
    perc_live   = _duck_under(_highpass(perc, 65), melody_envelope, 0.10)
    veena_clean = _highpass(veena, 120)
    violin_h    = _highpass(violin, 180)
    bansuri_b   = _highpass(bansuri, 200)
    sitar_b     = _highpass(sitar, 150)

    dl, dr = _stereo_track(drone_soft,   0.00, "studio", 0.04)
    vl, vr = _stereo_track(veena_clean, -0.12, "studio", 0.08)
    hl, hr = _stereo_track(violin_h,     0.35, "hall",   0.18)
    bl, br = _stereo_track(bansuri_b,    0.55, "hall",   0.14)
    sl, sr = _stereo_track(sitar_b,     -0.50, "hall",   0.12)
    pl, pr = _stereo_track(perc_live,    0.05, "studio", 0.06)

    left  = dl + vl + hl + bl + sl + pl
    right = dr + vr + hr + br + sr + pr
    stereo = np.stack([left, right], axis=1)

    for ch in range(2):
        stereo[:, ch] = _compress(stereo[:, ch], threshold_db=-20.0, ratio=2.2)

    peak = np.max(np.abs(stereo))
    if peak > 0.88:
        stereo = stereo / peak * 0.84

    return stereo.astype(np.float32)


def _pad_or_trim(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) < target:
        return np.pad(audio, (0, target - len(audio)))
    return audio[:target]


def _save_wav(audio: np.ndarray, path: str) -> str:
    """Save as 24-bit PCM WAV (stereo if shape is (n, 2))."""
    sf.write(path, audio, SR, subtype="PCM_24")
    return path


def _wav_to_mp3(wav_path: str, mp3_path: str) -> str | None:
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate="320k")
        return mp3_path
    except Exception as e:
        print(f"[indian_audio] MP3 conversion failed: {e}")
        return None


def _align_melody_to_key(melody_notes: list[dict], key_root: str) -> list[dict]:
    if not melody_notes:
        return melody_notes

    result = []
    for note in melody_notes:
        note_name = note.get("note", "C4")
        try:
            midi = librosa.note_to_midi(note_name)
            while midi < 55:
                midi += 12
            while midi > 84:
                midi -= 12
            result.append({**note, "note": librosa.midi_to_note(midi, unicode=False)})
        except Exception:
            result.append(note)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_indian_audio(
    melody_notes: list[dict],
    key: str,
    tempo_bpm: float,
    total_beats: float,
    preset_name: str,
    output_stem: str,
    beats_per_bar: float = 4.0,
) -> tuple[str | None, str | None]:
    key_root = key.strip().split()[0]
    total_seconds = total_beats * 60.0 / tempo_bpm
    target_samples = int(total_seconds * SR)

    print(f"[indian_audio] Rendering {preset_name} — {key} — {total_seconds:.1f}s")
    _detect_pitch_backend()

    is_ensemble = preset_name in (
        "Carnatic Ensemble", "Hindustani Classical", "Sitar & Strings", "Sufi / Qawwali",
    )

    raga = resolve_raga(key)
    scale_pcs = raga_scale_pcs(key, raga)
    print(f"[indian_audio] Raga: {raga.name} — scale PCs: {sorted(scale_pcs)}")

    if melody_notes:
        melody_notes = snap_notes_to_raga(melody_notes, key, raga)
        melody_notes = _align_melody_to_key(melody_notes, key_root)
        melody_notes = snap_notes_to_raga(melody_notes, key, raga)
        melody_notes = tag_phrase_boundaries(melody_notes)
        for n in melody_notes:
            if n.get("role") not in ("grace", "harmony", "embellishment"):
                n["role"] = "main"

    print("[indian_audio] Building Sa-Pa-Sa-Sa tanpura drone...")
    drone = _pad_or_trim(_make_tanpura_drone(key_root, total_seconds, volume=0.26), target_samples)

    # Veena — main melodic body (always veena for ensemble when samples exist)
    veena_inst = "veena" if _find_all("veena") else _melody_instrument_for_preset(preset_name)
    print(f"[indian_audio] Rendering veena (main body)...")
    veena_track = _pad_or_trim(
        _render_melody(melody_notes, veena_inst, tempo_bpm, volume=0.78, scale_pcs=scale_pcs),
        target_samples,
    )

    violin_track = np.zeros(target_samples, dtype=np.float32)
    bansuri_track = np.zeros(target_samples, dtype=np.float32)
    sitar_track = np.zeros(target_samples, dtype=np.float32)

    if is_ensemble and melody_notes:
        harmony_notes = snap_notes_to_raga(
            make_violin_harmony_pops(melody_notes, key, raga), key, raga
        )
        if harmony_notes:
            v_inst = "violin" if _load_chromatic_library("violin") else veena_inst
            print(f"[indian_audio] Violin harmony in gaps ({len(harmony_notes)} notes)...")
            violin_track = _pad_or_trim(
                _render_melody(harmony_notes, v_inst, tempo_bpm, volume=0.38, scale_pcs=scale_pcs),
                target_samples,
            )

        has_bansuri = bool(_load_chromatic_library("bansuri") or _find_all("bansuri"))
        has_sitar   = bool(_load_chromatic_library("sitar") or _find_all("sitar"))
        # Alternate embellishment instrument so bansuri and sitar never clash
        if has_bansuri:
            b_notes = snap_notes_to_raga(
                make_embellishments(melody_notes, "bansuri", key, raga, max_flourishes=3), key, raga
            )
            if b_notes:
                print(f"[indian_audio] Bansuri flourishes in gaps ({len(b_notes)} notes)...")
                bansuri_track = _pad_or_trim(
                    _render_melody(b_notes, "bansuri", tempo_bpm, volume=0.40, scale_pcs=scale_pcs),
                    target_samples,
                )
        elif has_sitar:
            s_notes = snap_notes_to_raga(
                make_embellishments(melody_notes, "sitar", key, raga, max_flourishes=3), key, raga
            )
            if s_notes:
                print(f"[indian_audio] Sitar flourishes in gaps ({len(s_notes)} notes)...")
                sitar_track = _pad_or_trim(
                    _render_melody(s_notes, "sitar", tempo_bpm, volume=0.38, scale_pcs=scale_pcs),
                    target_samples,
                )

    print("[indian_audio] Rendering mridangam (continuous adi tala + fills)...")
    perc = _pad_or_trim(
        _render_mridangam(total_beats, tempo_bpm, beats_per_bar, volume=0.48),
        target_samples,
    )

    mel_env = melody_activity_envelope(melody_notes, target_samples, tempo_bpm, SR)
    print("[indian_audio] Ensemble master bus...")
    mixed = _master_bus(drone, veena_track, violin_track, bansuri_track, sitar_track, perc, mel_env)

    wav_path = output_stem + "_indian.wav"
    mp3_path = output_stem + "_indian.mp3"

    _save_wav(mixed, wav_path)
    print(f"[indian_audio] WAV saved (24-bit stereo): {wav_path} ({Path(wav_path).stat().st_size // 1024} KB)")

    mp3 = _wav_to_mp3(wav_path, mp3_path)
    return wav_path, mp3


def _melody_instrument_for_preset(preset_name: str) -> str:
    mapping = {
        "Carnatic Ensemble":      "veena",
        "Hindustani Classical":   "bansuri",
        "Sitar & Strings":        "sitar",
        "Sufi / Qawwali":         "bansuri",
    }
    inst = mapping.get(preset_name, "veena")
    if not _find_all(inst):
        for fallback in ["veena", "bansuri", "sitar", "violin"]:
            if _find_all(fallback):
                print(f"[indian_audio] {inst} has no samples — using {fallback}")
                return fallback
    return inst


def samples_status() -> str:
    lines = ["Indian audio samples:"]
    for folder in ["tanpura", "mridangam", "veena", "bansuri", "sitar", "violin"]:
        files = _find_all(folder)
        wav_notes = len(list((_SAMPLES_DIR / folder).glob("*.wav"))) if (_SAMPLES_DIR / folder).exists() else 0
        if files:
            detail = f"{len(files)} file(s)"
            if wav_notes:
                detail += f", {wav_notes} chromatic note(s)"
            lines.append(f"  [ok] {folder}: {detail}")
        else:
            lines.append(f"  [--] {folder}: no samples found")

    if USE_NEURAL_INSTRUMENTS:
        models_dir = _PROJECT_DIR / "models"
        neural = []
        for inst in ["veena", "bansuri", "sitar", "violin", "carnatic_violin"]:
            if (models_dir / inst / "model_best.pt").exists() or (models_dir / inst / "model_final.pt").exists():
                neural.append(inst)
        if neural:
            lines.append(f"  [ok] neural models (enabled): {', '.join(neural)}")
        else:
            lines.append("  [--] neural models: none found")
    else:
        lines.append("  [ok] render mode: sliced samples (neural disabled in config.py)")
        partial = []
        models_dir = _PROJECT_DIR / "models"
        for inst in ["veena", "bansuri", "sitar", "violin"]:
            if (models_dir / inst / "model_best.pt").exists():
                partial.append(inst)
        if partial:
            lines.append(f"  [i] partial models on disk (ignored): {', '.join(partial)}")

    return "\n".join(lines)


def setup_sample_folders() -> None:
    for folder in ["tanpura", "mridangam", "veena", "bansuri", "sitar", "violin"]:
        (_SAMPLES_DIR / folder).mkdir(parents=True, exist_ok=True)
    (_PROJECT_DIR / "models").mkdir(parents=True, exist_ok=True)
