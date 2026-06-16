"""
setup_samples.py
================
Run this ONCE (or re-run when you add new recordings) to build
the Indian instrument sample library and train neural synthesizers.

PHASES:
  Phase 1 — Source separation + slicing (runs every time)
  Phase 2 — DDSP neural instrument training (run after gathering 30+ min audio)

Usage:
    py -3.11 setup_samples.py            # Phase 1 only
    py -3.11 setup_samples.py --train    # Phase 1 + Phase 2 (DDSP training)

Install dependencies first:
    py -3.11 -m pip install demucs ddsp librosa soundfile numpy

Place raw recordings anywhere under samples/ — script finds them automatically.
Mixed recordings (veena + tabla + drone) are fine — Demucs will isolate the
melodic instrument before slicing.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SR        = 44100
MIN_DUR   = 0.15
MAX_SLICE_SECS = 300   # only slice the first 5 minutes of any recording
MIN_CONF  = 0.38
PROJECT   = Path(__file__).parent
SAMPLES   = PROJECT / "samples"
MODELS    = PROJECT / "models"   # trained DDSP models go here
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".aiff", ".m4a", ".webm", ".opus", ".wma"}

OUT_DIRS = {
    "tanpura":   SAMPLES / "tanpura",
    "veena":     SAMPLES / "veena",
    "bansuri":   SAMPLES / "bansuri",
    "mridangam": SAMPLES / "mridangam",
    "sitar":     SAMPLES / "sitar",
    "violin":    SAMPLES / "violin",
}

INSTRUMENT_KEYWORDS = {
    "tanpura":   ["tanpura", "tambura", "tanpuri", "miraj", "drone"],
    "mridangam": ["mridangam", "mridanga", "thom", "ta_", "_ta", "din", "fill", "kanjira", "parai"],
    "veena":     ["veena", "vina", "saraswati"],
    "bansuri":   ["bansuri", "flute", "bamboo", "indian_flute"],
    "sitar":     ["sitar"],
    "violin":    ["violin", "fiddle", "carnatic_violin", "sarangi"],
}

MRIDANGAM_STROKES = {
    "thom": ["thom", "bass", "low", "_1", "-1", "bayan"],
    "ta":   ["ta_",  "_ta", "treble", "high", "_4", "-4", "dayan"],
    "din":  ["din",  "resonant", "_3", "-3"],
    "fill": ["fill", "karvai", "_2", "-2", "kanjira"],
}

# Instruments we train DDSP models for (melodic only)
DDSP_INSTRUMENTS = ["veena", "bansuri", "sitar", "violin"]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route(path: Path) -> str | None:
    name = path.stem.lower()
    full = str(path).lower()
    for inst, kws in INSTRUMENT_KEYWORDS.items():
        if any(kw in name or kw in full for kw in kws):
            return inst
    return None


def _stroke(path: Path) -> str | None:
    name = path.stem.lower()
    for stroke, kws in MRIDANGAM_STROKES.items():
        if any(kw in name for kw in kws):
            return stroke
    return None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _convert_to_wav(path: Path) -> Path:
    """
    Convert non-WAV/MP3 formats (webm, opus, wma) to WAV using pydub.
    Returns path to converted WAV file in a temp location.
    """
    suffix = path.suffix.lower()
    if suffix in {".wav", ".mp3"}:
        return path   # librosa handles these natively

    wav_path = path.parent / (path.stem + "_converted.wav")
    if wav_path.exists():
        return wav_path

    print(f"    Converting {path.name} → WAV...")
    try:
        from pydub import AudioSegment

        fmt_map = {".webm": "webm", ".opus": "ogg",
                   ".wma": "asf", ".ogg": "ogg",
                   ".flac": "flac", ".aiff": "aiff", ".m4a": "mp4"}
        fmt = fmt_map.get(suffix, suffix.lstrip("."))

        audio = AudioSegment.from_file(str(path), format=fmt)
        audio = audio.set_frame_rate(SR).set_channels(1)
        audio.export(str(wav_path), format="wav")
        print(f"    Converted: {wav_path.name}")
        return wav_path
    except Exception as e:
        print(f"    Conversion failed: {e}")
        print(f"    Install ffmpeg from https://www.gyan.dev/ffmpeg/builds/")
        print(f"    Or run: py -3.11 -m yt_dlp --ffmpeg-location PATH ...")
        return path   # return original, librosa will try its best


def _load(path: Path) -> np.ndarray:
    path = _convert_to_wav(path)
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


def _fade(y: np.ndarray, ms: float = 15.0) -> np.ndarray:
    fade = min(int(SR * ms / 1000), len(y) // 8)
    if fade > 0:
        y = y.copy()
        y[:fade]  *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    return y


def _detect_pitch(seg: np.ndarray) -> tuple[float, float]:
    try:
        f0, voiced, probs = librosa.pyin(
            seg, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C8"), sr=SR,
        )
        mask = voiced & ~np.isnan(f0) & (probs > 0.3)
        voiced_f0 = f0[mask]
        conf = float(np.mean(probs[mask])) if len(probs[mask]) > 0 else 0.0
        if len(voiced_f0) >= 3:
            return float(np.median(voiced_f0)), conf
    except Exception:
        pass
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Phase 1a — Demucs source separation
# ---------------------------------------------------------------------------

def _demucs_available() -> bool:
    try:
        import demucs
        return True
    except ImportError:
        return False


def separate_stems(src: Path, tmp_dir: Path) -> dict[str, Path]:
    """
    Run Demucs on a mixed recording.
    Returns dict of stem name → path: {"vocals", "drums", "bass", "other"}
    "other" is typically the melodic instrument (veena, sitar, etc.)
    """
    print(f"    Separating stems: {src.name}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "demucs",
             "--two-stems", "other",   # splits into "other" + "no_other"
             "--out", str(tmp_dir),
             "--mp3",
             str(src)],
            capture_output=True, text=True, timeout=300,
        )
        # Demucs outputs to tmp_dir/htdemucs/stem_name/
        out_base = tmp_dir / "htdemucs" / src.stem
        stems = {}
        for stem_name in ["other", "no_other", "vocals", "drums", "bass"]:
            for ext in [".wav", ".mp3"]:
                p = out_base / f"{stem_name}{ext}"
                if p.exists():
                    stems[stem_name] = p
                    break
        if stems:
            print(f"    Separated: {list(stems.keys())}")
        else:
            print(f"    Demucs produced no output — using original")
        return stems
    except subprocess.TimeoutExpired:
        print(f"    Demucs timed out — using original")
        return {}
    except Exception as e:
        print(f"    Demucs failed: {e} — using original")
        return {}


def get_melodic_stem(src: Path, tmp_dir: Path) -> Path:
    """
    Return the best audio file to slice for a melodic instrument.
    Uses Demucs separation if available, otherwise returns original.
    """
    if not _demucs_available():
        return src

    stems = separate_stems(src, tmp_dir)
    # "other" stem is the melodic instrument in most Indian music
    if "other" in stems:
        return stems["other"]
    return src


# ---------------------------------------------------------------------------
# Phase 1b — Slice melodic recordings into chromatic notes
# ---------------------------------------------------------------------------

def _find_natural_end(y: np.ndarray, start: int, next_onset: int) -> int:
    """
    Find where the note naturally ends by tracking amplitude envelope.
    Extends past the next onset if the current note is still sustaining.
    Returns sample index of natural end.
    """
    # Look at RMS energy in 50ms windows
    win = int(SR * 0.05)
    pos = start

    # Find peak amplitude in first 500ms
    peak_region = y[start:start + int(SR * 0.5)]
    if len(peak_region) == 0:
        return next_onset
    peak_amp = float(np.max(np.abs(peak_region))) if len(peak_region) > 0 else 0.01
    if peak_amp < 0.01:
        return next_onset

    # Threshold: note ends when amplitude drops below 2% of peak
    threshold = peak_amp * 0.02

    # Search from next onset onward for natural decay
    search_end = min(start + int(SR * 8.0), len(y))  # max 8 seconds
    pos = next_onset

    while pos + win < search_end:
        rms = float(np.sqrt(np.mean(y[pos:pos + win] ** 2)))
        if rms < threshold:
            # Found end — add small tail
            return min(pos + win, search_end)
        pos += win // 2

    return min(search_end, len(y))


def _extract_with_slides(
    y: np.ndarray,
    onset: float,
    next_onset: float,
    pre_roll: float = 0.08,
) -> np.ndarray:
    """
    Extract a note segment that includes:
    - pre_roll seconds BEFORE the onset (captures the approach slide)
    - The full sustained note
    - The natural decay tail (even past next onset)

    pre_roll=0.08 = 80ms before onset, enough to capture the gamaka approach.
    """
    start_with_slide = max(0, int((onset - pre_roll) * SR))
    next_onset_samp  = int(next_onset * SR)

    # Find where note naturally ends (may extend past next onset)
    natural_end = _find_natural_end(y, int(onset * SR), next_onset_samp)

    seg = y[start_with_slide:natural_end]
    return seg


def slice_into_notes(
    src: Path,
    out_dir: Path,
    existing: dict[str, tuple[str, float]],
    use_separation: bool = True,
) -> dict[str, tuple[str, float]]:
    """
    Slice a recording into individual note samples with full character:
    - 80ms pre-roll before onset (approach slide / gamaka)
    - Full sustained note
    - Natural resonance decay (not cut at next onset)
    - Normalized consistently
    existing: {note_name: (path, duration)} — keeps longest sample per note.
    """
    print(f"    Processing: {src.name}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Source separation if available and file is long (likely mixed)
        duration = librosa.get_duration(path=str(src))
        if use_separation and duration > 10.0 and _demucs_available():
            audio_path = get_melodic_stem(src, tmp_path)
        else:
            audio_path = src

        try:
            y = _load(audio_path)
            max_samples = int(MAX_SLICE_SECS * SR)
            if len(y) > max_samples:
                print(f"    Trimming to first {MAX_SLICE_SECS//60} min ({len(y)/SR:.0f}s total)")
                y = y[:max_samples]
        except Exception as e:
            print(f"    Could not load: {e}")
            return existing

        # Onset detection — slightly looser to catch soft approach notes
        try:
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=SR, backtrack=True,
                delta=0.04,   # more sensitive than before
                wait=3,
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=SR)
        except Exception as e:
            print(f"    Onset detection failed: {e}")
            return existing

        if len(onset_times) == 0:
            print(f"    No onsets detected")
            return existing

        saved = 0
        for i, t in enumerate(onset_times):
            # Next onset or end of file
            next_t = onset_times[i + 1] if i + 1 < len(onset_times) else len(y) / SR

            # Skip very short gaps between onsets (ornaments, not separate notes)
            gap = next_t - t
            if gap < MIN_DUR:
                continue

            # Detect pitch on the core note (skip first 50ms which may be slide)
            core_start = int((t + 0.05) * SR)
            core_end   = int(min(t + 1.5, next_t) * SR)
            core_seg   = y[core_start:core_end]

            if len(core_seg) < int(SR * 0.1):
                continue

            hz, conf = _detect_pitch(core_seg)
            if hz < 60 or conf < MIN_CONF:
                continue

            note_name = librosa.hz_to_note(hz, unicode=False)

            # Extract with slide + full decay
            seg = _extract_with_slides(y, t, next_t, pre_roll=0.08)

            if len(seg) < int(SR * MIN_DUR):
                continue

            actual_dur = len(seg) / SR

            # Keep the longest clean sample per note
            if note_name in existing and existing[note_name][1] >= actual_dur:
                continue

            # Normalize to consistent peak level (-3dBFS)
            peak = float(np.max(np.abs(seg)))
            if peak > 0.001:
                seg = seg * (0.7 / peak)
            else:
                continue

            # Gentle fade-in at start (5ms) to smooth any click from pre-roll join
            fade_in = min(int(SR * 0.005), len(seg) // 10)
            seg[:fade_in] *= np.linspace(0, 1, fade_in)

            # NO fade-out at end — let it decay naturally
            seg = seg.astype(np.float32)

            out_path = out_dir / f"{note_name}.wav"
            sf.write(str(out_path), seg, SR)
            existing[note_name] = (str(out_path), actual_dur)
            print(f"      {note_name:<6} {hz:6.1f}Hz  {actual_dur:.2f}s  (slide+decay)")
            saved += 1

        if saved == 0:
            print(f"    No usable notes extracted")
    return existing


# ---------------------------------------------------------------------------
# Phase 2 — DDSP Neural Instrument Training
# ---------------------------------------------------------------------------

def _ddsp_available() -> bool:
    try:
        import ddsp
        return True
    except ImportError:
        return False


def _build_training_audio(instrument: str, out_dir: Path) -> Path | None:
    """
    Concatenate all sliced WAV samples for an instrument into one
    long training audio file.
    """
    sample_dir = SAMPLES / instrument
    wav_files  = sorted(sample_dir.glob("*.wav"))

    if len(wav_files) < 5:
        print(f"  Not enough samples for {instrument} ({len(wav_files)} notes, need ≥5)")
        return None

    print(f"  Building training audio from {len(wav_files)} {instrument} samples...")
    segments = []
    silence  = np.zeros(int(SR * 0.3), dtype=np.float32)  # 300ms silence between notes

    for wav in wav_files:
        try:
            y = _load(wav)
            # Pad/trim each note to 2 seconds
            target = int(SR * 2.0)
            if len(y) > target:
                y = y[:target]
            else:
                y = np.pad(y, (0, target - len(y)))
            segments.append(y)
            segments.append(silence)
        except Exception:
            continue

    if not segments:
        return None

    audio = np.concatenate(segments)
    out_path = out_dir / f"{instrument}_training.wav"
    sf.write(str(out_path), audio, SR)
    duration = len(audio) / SR
    print(f"  Training audio: {out_path.name} ({duration:.1f}s)")
    return out_path


def train_ddsp_instrument(instrument: str) -> bool:
    """
    Train a DDSP harmonic+noise synthesizer model for one instrument.

    The model learns:
    - How the instrument's harmonics are distributed at each pitch
    - How quickly notes attack and decay
    - The noise character (breath, string buzz, resonance)

    After training, synthesize any note by calling:
        synthesize_note(model_path, note_midi, duration_sec)
    """
    if not _ddsp_available():
        print(f"\n  DDSP not installed. Run: py -3.11 -m pip install ddsp")
        return False

    import ddsp
    import ddsp.training
    import tensorflow as tf

    model_dir  = MODELS / instrument
    model_dir.mkdir(parents=True, exist_ok=True)
    audio_dir  = model_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Build training audio
    train_audio = _build_training_audio(instrument, audio_dir)
    if train_audio is None:
        return False

    print(f"\n  Training DDSP model for {instrument}...")
    print(f"  This takes 2-3 hours on CPU. Leave it running.")
    print(f"  Model will be saved to: {model_dir}")

    # Load audio
    y = _load(train_audio)

    # DDSP expects audio in chunks — 4 second frames at 16kHz
    DDSP_SR     = 16000
    FRAME_SECS  = 4
    FRAME_LEN   = DDSP_SR * FRAME_SECS

    # Resample to DDSP sample rate
    y_16k = librosa.resample(y, orig_sr=SR, target_sr=DDSP_SR)

    # Split into overlapping frames
    hop = FRAME_LEN // 2
    frames = []
    for start in range(0, len(y_16k) - FRAME_LEN, hop):
        frame = y_16k[start:start + FRAME_LEN]
        frames.append(frame)

    if len(frames) < 4:
        print(f"  Not enough audio for training ({len(frames)} frames, need ≥4)")
        print(f"  Find more recordings and re-run with --train")
        return False

    audio_batch = np.stack(frames).astype(np.float32)   # [batch, samples]
    print(f"  Training on {len(frames)} audio frames ({len(frames) * FRAME_SECS}s total)")

    # Build DDSP model — Harmonic + Noise + Reverb
    n_harmonics  = 60
    n_noise_mags = 65

    model = ddsp.training.models.Autoencoder(
        preprocessor=ddsp.training.preprocessing.DefaultPreprocessor(time_steps=250),
        encoder=ddsp.training.encoders.MfccTimeDistributedRnnEncoder(
            rnn_channels=512, rnn_type="gru",
            z_dims=16, z_time_steps=125,
        ),
        decoder=ddsp.training.decoders.RnnFcDecoder(
            rnn_channels=512, rnn_type="gru",
            ch=512, layers_per_stack=3,
            input_keys=("ld_scaled", "f0_scaled", "z"),
            output_splits=(
                ("amps",        1),
                ("harmonic_distribution", n_harmonics),
                ("noise_magnitudes",      n_noise_mags),
            ),
        ),
        processor_group=ddsp.training.train_util.get_processor_group(
            n_harmonics=n_harmonics, n_noise_mags=n_noise_mags,
        ),
        losses=[
            ddsp.losses.SpectralLoss(loss_type="L1", mag_weight=1.0, delta_time_weight=0.5),
        ],
    )

    # Training loop
    optimizer  = tf.keras.optimizers.Adam(learning_rate=1e-3)
    n_epochs   = 100    # ~2-3 hours on CPU; increase to 300 for better quality
    batch_size = 4

    best_loss  = float("inf")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_mgr   = tf.train.CheckpointManager(checkpoint, str(model_dir), max_to_keep=3)

    dataset = tf.data.Dataset.from_tensor_slices(audio_batch)
    dataset = dataset.batch(batch_size).shuffle(100).prefetch(2)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        batches    = 0
        for audio_b in dataset:
            with tf.GradientTape() as tape:
                outputs = model({"audio": audio_b}, training=True)
                total_loss = sum(outputs["losses"].values())
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += float(total_loss)
            batches    += 1

        avg_loss = epoch_loss / max(batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_mgr.save()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  best={best_loss:.4f}")

    # Save final model weights
    model.save_weights(str(model_dir / "model_weights"))
    print(f"\n  ✓ {instrument} model saved to {model_dir}")
    print(f"    Best loss: {best_loss:.4f}")
    return True


# ---------------------------------------------------------------------------
# Synthesis — use trained model to generate notes
# ---------------------------------------------------------------------------

def synthesize_note(
    instrument: str,
    midi_note:  int,
    duration_sec: float = 2.0,
) -> np.ndarray | None:
    """
    Synthesize a note using a trained DDSP model.
    Returns audio array at SR, or None if model not available.

    This is called by indian_audio_engine.py when a trained model exists,
    replacing the sample-based approach entirely.
    """
    model_dir = MODELS / instrument
    weights   = model_dir / "model_weights.index"

    if not weights.exists():
        return None

    if not _ddsp_available():
        return None

    import ddsp
    import tensorflow as tf

    try:
        # Recreate model architecture
        model = ddsp.training.models.Autoencoder(
            preprocessor=ddsp.training.preprocessing.DefaultPreprocessor(time_steps=250),
            encoder=None,   # no encoder needed for synthesis
            decoder=ddsp.training.decoders.RnnFcDecoder(
                rnn_channels=512, rnn_type="gru",
                ch=512, layers_per_stack=3,
                input_keys=("ld_scaled", "f0_scaled"),
                output_splits=(("amps", 1), ("harmonic_distribution", 60), ("noise_magnitudes", 65)),
            ),
            processor_group=ddsp.training.train_util.get_processor_group(
                n_harmonics=60, n_noise_mags=65,
            ),
            losses=[],
        )
        model.load_weights(str(model_dir / "model_weights"))

        # Build synthesis inputs
        DDSP_SR    = 16000
        time_steps = 250
        hz = librosa.midi_to_hz(midi_note)

        # F0 trajectory — constant at target pitch with slight vibrato
        f0 = np.full(time_steps, hz, dtype=np.float32)
        # Add natural vibrato (5Hz, ±8 cents)
        t  = np.linspace(0, duration_sec, time_steps)
        f0 *= (1.0 + 0.005 * np.sin(2 * np.pi * 5 * t))

        # Loudness — natural attack/decay envelope
        ld = np.zeros(time_steps, dtype=np.float32)
        attack  = int(time_steps * 0.05)
        release = int(time_steps * 0.2)
        ld[:attack]  = np.linspace(-80, -10, attack)
        ld[attack:time_steps - release] = -10
        ld[time_steps - release:] = np.linspace(-10, -80, release)

        inputs = {
            "f0_hz":   tf.constant(f0[np.newaxis, :, np.newaxis]),
            "loudness_db": tf.constant(ld[np.newaxis, :, np.newaxis]),
        }

        outputs = model(inputs, training=False)
        audio_16k = outputs["audio_synth"].numpy().squeeze()

        # Resample to 44100
        audio = librosa.resample(audio_16k, orig_sr=DDSP_SR, target_sr=SR)

        # Trim to requested duration
        target_len = int(duration_sec * SR)
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))

        return audio.astype(np.float32)

    except Exception as e:
        print(f"[synthesize_note] Failed for {instrument} MIDI {midi_note}: {e}")
        return None


def model_available(instrument: str) -> bool:
    """Return True if a trained DDSP model exists for this instrument."""
    return (MODELS / instrument / "model_weights.index").exists()


# ---------------------------------------------------------------------------
# Mridangam processing — slicing + fill synthesis
# ---------------------------------------------------------------------------

def _slice_percussion_hits(src: Path, out_dir: Path, prefix: str, existing_count: int = 0) -> int:
    """Slice a percussion recording into individual hit WAV files."""
    try:
        y = _load(src)
        y = y[:int(MAX_SLICE_SECS * SR)]
    except Exception as e:
        print(f"    Could not load {src.name}: {e}")
        return 0

    try:
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=SR, backtrack=False, delta=0.08, wait=2,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=SR)
    except Exception:
        return 0

    saved = 0
    for i, t in enumerate(onset_times):
        next_t = onset_times[i + 1] if i + 1 < len(onset_times) else t + 2.0
        dur    = min(next_t - t, 2.5)
        if dur < 0.05:
            continue
        start = int(t * SR)
        seg   = y[start:int((t + dur) * SR)]
        peak  = float(np.max(np.abs(seg))) if len(seg) > 0 else 0
        if peak < 0.02:
            continue
        seg_t, _ = librosa.effects.trim(seg, top_db=35)
        if len(seg_t) < int(SR * 0.05):
            continue
        seg_t = seg_t * (0.8 / peak)
        fade_in = min(int(SR * 0.003), len(seg_t) // 10)
        seg_t[:fade_in] *= np.linspace(0, 1, fade_in)
        out_path = out_dir / f"{prefix}_{existing_count + saved + 1:03d}.wav"
        sf.write(str(out_path), seg_t.astype(np.float32), SR)
        saved += 1

    print(f"    Sliced {saved} hits from {src.name}")
    return saved


def _synthesize_mridangam_fills(out_dir: Path) -> int:
    """Generate fill patterns by combining existing strokes."""
    strokes: dict[str, list[np.ndarray]] = {}
    for stroke in ["thom", "ta", "din"]:
        variants = []
        for p in sorted(out_dir.glob(f"{stroke}*.wav")):
            try:
                variants.append(_load(p))
            except Exception:
                pass
        if variants:
            strokes[stroke] = variants

    if len(strokes) < 2:
        print("    Not enough strokes to synthesize fills")
        return 0

    def pick(s: str) -> np.ndarray:
        v = strokes.get(s, strokes.get("ta", [np.zeros(int(SR * 0.3), dtype=np.float32)]))
        y = v[np.random.randint(len(v))].copy()
        return (y * 10 ** (np.random.uniform(-3, 3) / 20)).astype(np.float32)

    def make_fill(pattern: list, bpm: float = 120) -> np.ndarray:
        beat_dur = 60.0 / bpm
        total_beats = max(b for _, b in pattern) + 0.5
        total_samples = int(total_beats * beat_dur * SR)
        out = np.zeros(total_samples, dtype=np.float32)
        for stroke_name, beat_offset in pattern:
            hit   = pick(stroke_name)
            start = int(beat_offset * beat_dur * SR)
            end   = min(start + len(hit), total_samples)
            if end > start:
                out[start:end] += hit[:end - start]
        peak = np.max(np.abs(out))
        return (out / peak * 0.8).astype(np.float32) if peak > 0 else out

    patterns = [
        [("thom", 0.0), ("ta", 0.5), ("din", 1.0), ("ta", 1.5)],
        [("ta", 0.0), ("din", 0.33), ("ta", 0.67), ("thom", 1.0), ("ta", 1.33), ("din", 1.67)],
        [("thom", 0.0), ("thom", 0.75), ("ta", 1.0), ("ta", 1.25), ("thom", 1.5)],
        [("ta", 0.0), ("ta", 0.25), ("ta", 0.5), ("din", 0.75), ("thom", 1.0)],
        [("thom", 0.0), ("ta", 0.25), ("din", 0.5), ("ta", 0.75),
         ("thom", 1.0), ("ta", 1.25), ("din", 1.5), ("ta", 1.75), ("thom", 2.0)],
        [("din", 0.0), ("ta", 0.5), ("din", 1.0), ("thom", 1.5)],
        [("ta", 0.0), ("din", 0.375), ("ta", 0.75), ("thom", 1.125), ("ta", 1.5)],
        [("thom", 0.0), ("ta", 0.125), ("ta", 0.25), ("din", 0.5), ("ta", 0.75), ("thom", 1.0)],
    ]

    saved = 0
    for i, pattern in enumerate(patterns):
        fill = make_fill(pattern)
        sf.write(str(out_dir / f"fill_{i+1:03d}.wav"), fill, SR)
        print(f"    Generated fill pattern {i+1}: {len(fill)/SR:.2f}s")
        saved += 1
    return saved


def _process_mridangam(files: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not files:
        print("  No new mridangam files — synthesizing fills from existing strokes")
        if any(out_dir.glob("*.wav")):
            n = _synthesize_mridangam_fills(out_dir)
            print(f"  Generated {n} fill patterns")
        return

    # Route by stroke keyword
    assigned: dict[str, Path] = {}
    unmatched: list[Path] = []
    for f in files:
        s = _stroke(f)
        if s and s not in assigned:
            assigned[s] = f
        else:
            unmatched.append(f)

    # Assign unmatched files to missing stroke names
    for stroke in [s for s in ["thom", "ta", "din"] if s not in assigned]:
        if unmatched:
            assigned[stroke] = unmatched.pop(0)

    # Process each named stroke
    for stroke, src_path in assigned.items():
        print(f"  Processing {stroke}: {src_path.name}")
        try:
            y = _load(src_path)
            y_t, _ = librosa.effects.trim(y, top_db=30)
            if len(y_t) / SR < 0.5:
                # Single hit — save directly
                peak = np.max(np.abs(y_t))
                if peak > 0:
                    y_t = y_t * (0.8 / peak)
                sf.write(str(out_dir / f"{stroke}.wav"), y_t.astype(np.float32), SR)
                print(f"    Saved: {stroke}.wav ({len(y_t)/SR:.2f}s)")
            else:
                # Long recording — slice into multiple hits
                existing = len(list(out_dir.glob(f"{stroke}_*.wav")))
                n = _slice_percussion_hits(src_path, out_dir, stroke, existing)
                if n == 0:
                    sf.write(str(out_dir / f"{stroke}.wav"), y_t.astype(np.float32), SR)
                    print(f"    Saved as single: {stroke}.wav")
        except Exception as e:
            print(f"    Failed ({e}) — copying as-is")
            try:
                shutil.copy2(str(src_path), str(out_dir / f"{stroke}.wav"))
            except Exception:
                pass

    # Slice any extra unmatched files
    for f in unmatched:
        existing = len(list(out_dir.glob("extra_*.wav")))
        _slice_percussion_hits(f, out_dir, "extra", existing)

    # Synthesize fill patterns
    print("  Synthesizing fill patterns...")
    n = _synthesize_mridangam_fills(out_dir)
    print(f"  Generated {n} fill patterns")
    print(f"  Total mridangam samples: {len(list(out_dir.glob('*.wav')))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Run DDSP training after sample setup (takes 2-3h per instrument)")
    parser.add_argument("--instrument", type=str, default=None,
                        help="Train only this instrument (veena/bansuri/sitar)")
    parser.add_argument("--no-separate", action="store_true",
                        help="Skip Demucs source separation")
    args = parser.parse_args()

    print("=" * 60)
    print("  Orchestral Composer — Indian Sample Setup")
    print("=" * 60)

    use_sep = not args.no_separate
    if use_sep and _demucs_available():
        print("  [ok] Demucs available -- will separate mixed recordings")
    elif use_sep:
        print("  [warning] Demucs not installed -- skipping separation")
        print("    Run: py -3.11 -m pip install demucs")
        use_sep = False
    else:
        print("  Skipping Demucs separation (--no-separate)")

    # Create output folders
    for d in OUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # Find all unprocessed audio files
    output_paths = set()
    for d in OUT_DIRS.values():
        output_paths.update(str(f) for f in d.glob("*"))

    all_files = [f for f in SAMPLES.rglob("*")
                 if f.suffix.lower() in AUDIO_EXTS
                 and str(f) not in output_paths]

    print(f"\nFound {len(all_files)} unprocessed file(s)")

    # Route
    routed: dict[str, list[Path]] = {k: [] for k in OUT_DIRS}
    unrouted: list[Path] = []
    for f in all_files:
        inst = _route(f)
        if inst:
            routed[inst].append(f)
        else:
            unrouted.append(f)

    if unrouted:
        print(f"\n[warning] Unrouted files (rename to include instrument keyword):")
        for f in unrouted:
            print(f"  {f.name}")

    # ── Tanpura ──────────────────────────────────────────────────────────
    print("\n[1/4] Tanpura")
    for f in routed["tanpura"]:
        dest = OUT_DIRS["tanpura"] / f.name
        shutil.copy2(str(f), str(dest))
        print(f"  Copied: {f.name}")
    if not routed["tanpura"]:
        print("  No new tanpura files")

    # ── Veena ────────────────────────────────────────────────────────────
    print("\n[2/4] Veena")
    vn_ex = {p.stem: (str(p), librosa.get_duration(path=str(p)))
             for p in OUT_DIRS["veena"].glob("*.wav")}
    for f in routed["veena"]:
        vn_ex = slice_into_notes(f, OUT_DIRS["veena"], vn_ex, use_sep)
    total_vn = len(list(OUT_DIRS["veena"].glob("*.wav")))
    print(f"  Veena: {total_vn} notes -- {sorted(p.stem for p in OUT_DIRS['veena'].glob('*.wav'))}")

    # ── Bansuri ──────────────────────────────────────────────────────────
    print("\n[3/4] Bansuri")
    bn_ex = {p.stem: (str(p), librosa.get_duration(path=str(p)))
             for p in OUT_DIRS["bansuri"].glob("*.wav")}
    for f in routed["bansuri"]:
        bn_ex = slice_into_notes(f, OUT_DIRS["bansuri"], bn_ex, use_sep)
    total_bn = len(list(OUT_DIRS["bansuri"].glob("*.wav")))
    print(f"  Bansuri: {total_bn} notes -- {sorted(p.stem for p in OUT_DIRS['bansuri'].glob('*.wav'))}")

    # ── Sitar ────────────────────────────────────────────────────────────
    if routed["sitar"]:
        print("\n[3b/4] Sitar")
        st_ex = {p.stem: (str(p), librosa.get_duration(path=str(p)))
                 for p in OUT_DIRS["sitar"].glob("*.wav")}
        for f in routed["sitar"]:
            st_ex = slice_into_notes(f, OUT_DIRS["sitar"], st_ex, use_sep)

    # ── Violin ───────────────────────────────────────────────────────────
    print("\n[3c/4] Violin")
    vl_ex = {p.stem: (str(p), librosa.get_duration(path=str(p)))
             for p in OUT_DIRS["violin"].glob("*.wav")}
    for f in routed["violin"]:
        vl_ex = slice_into_notes(f, OUT_DIRS["violin"], vl_ex, use_sep)
    total_vl = len(list(OUT_DIRS["violin"].glob("*.wav")))
    print(f"  Violin: {total_vl} notes -- {sorted(p.stem for p in OUT_DIRS['violin'].glob('*.wav'))}")

    # ── Mridangam ────────────────────────────────────────────────────────
    print("\n[4/4] Mridangam")
    _process_mridangam(routed["mridangam"], OUT_DIRS["mridangam"])

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 1 Complete -- Sample Library")
    print("=" * 60)
    for inst, d in OUT_DIRS.items():
        files = list(d.glob("*"))
        icon  = "[ok]" if files else "[x]"
        print(f"  {icon} {inst:<12} {len(files)} file(s)")

    # ── Phase 2: DDSP Training ────────────────────────────────────────────
    if args.train:
        print("\n" + "=" * 60)
        print("  Phase 2 -- DDSP Neural Instrument Training")
        print("=" * 60)

        if not _ddsp_available():
            print("\n  DDSP not installed. Run:")
            print("    py -3.11 -m pip install ddsp")
            print("  Then re-run: py -3.11 setup_samples.py --train")
            return

        instruments_to_train = (
            [args.instrument] if args.instrument else DDSP_INSTRUMENTS
        )

        for inst in instruments_to_train:
            n_samples = len(list((SAMPLES / inst).glob("*.wav")))
            if n_samples < 5:
                print(f"\n  Skipping {inst} -- only {n_samples} samples (need >=5)")
                print(f"  Find more {inst} recordings and re-run setup first")
                continue
            print(f"\n  Training {inst} ({n_samples} note samples available)...")
            success = train_ddsp_instrument(inst)
            if success:
                print(f"  [ok] {inst} model ready")
                print(f"    The audio engine will use this model automatically")
            else:
                print(f"  [error] {inst} training failed")

        print("\n" + "=" * 60)
        print("  Phase 2 Complete")
        print("=" * 60)
        for inst in DDSP_INSTRUMENTS:
            icon = "[ok]" if model_available(inst) else "[.]"
            print(f"  {icon} {inst:<12} {'model ready' if model_available(inst) else 'not trained yet'}")

    else:
        print("\nTo train neural instrument models, run:")
        print("  py -3.11 -m pip install ddsp")
        print("  py -3.11 setup_samples.py --train")
        print("\nThis takes 2-3 hours per instrument on CPU.")
        print("After training, the audio engine uses the model instead of samples.")

    print("\nRun py -3.11 app.py to start the app.")


if __name__ == "__main__":
    main()