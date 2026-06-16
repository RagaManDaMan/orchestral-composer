"""
src/audio_render.py
===================
Render MIDI files to WAV or MP3 using FluidSynth + a SoundFont.

FluidSynth path and SoundFont path are configured in config.py.
MP3 conversion uses pydub (which requires ffmpeg) if available,
otherwise falls back to WAV only.

Usage:
    from src.audio_render import render_midi_to_audio

    wav_path, mp3_path = render_midi_to_audio(
        midi_path="outputs/my_song.mid",
        output_stem="outputs/my_song",
    )
    # wav_path is always set, mp3_path may be None if ffmpeg unavailable
"""

from __future__ import annotations

import os
import subprocess
import shutil
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Locate FluidSynth
# ---------------------------------------------------------------------------

def _find_fluidsynth() -> str | None:
    """
    Return path to fluidsynth executable.
    Checks config.py first, then PATH, then common Windows locations.
    """
    # 1. config.py override
    try:
        from config import FLUIDSYNTH_PATH
        if FLUIDSYNTH_PATH and Path(FLUIDSYNTH_PATH).exists():
            return FLUIDSYNTH_PATH
    except (ImportError, AttributeError):
        pass

    # 2. PATH
    found = shutil.which("fluidsynth")
    if found:
        return found

    # 3. Common Windows install locations
    common = [
        r"C:\fluidsynth\bin\fluidsynth.exe",
        r"C:\Program Files\FluidSynth\bin\fluidsynth.exe",
        str(Path.home() / "Downloads" /
            "fluidsynth-v2.5.4-win10-x64-cpp11" /
            "fluidsynth-v2.5.4-win10-x64-cpp11" /
            "bin" / "fluidsynth.exe"),
    ]
    for p in common:
        if Path(p).exists():
            return p

    return None


def _find_soundfont() -> str | None:
    """
    Return path to .sf2 soundfont file.
    Checks config.py first, then the project soundfonts/ folder.
    """
    try:
        from config import SOUNDFONT_PATH
        if SOUNDFONT_PATH and Path(SOUNDFONT_PATH).exists():
            return SOUNDFONT_PATH
    except (ImportError, AttributeError):
        pass

    # Look in project soundfonts/ folder — prefer SGM for Indian instruments
    project_root = Path(__file__).parent.parent
    sf_dir = project_root / "soundfonts"
    if sf_dir.exists():
        # Preferred order
        try:
            from config import SOUNDFONT_PREFERENCE
            pref = SOUNDFONT_PREFERENCE
        except Exception:
            pref = ["SGM-v2.01", "GeneralUser", "FluidR3"]
        for name in pref:
            for sf in sf_dir.glob("*.sf2"):
                if name.lower() in sf.name.lower():
                    return str(sf)
        # Fallback: first sf2 found
        for sf in sorted(sf_dir.glob("*.sf2")):
            return str(sf)

    return None


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_midi_to_audio(
    midi_path: str,
    output_stem: str | None = None,
    sample_rate: int = 44100,
    gain: float = 0.8,
) -> tuple[str | None, str | None]:
    """
    Render a MIDI file to WAV (and optionally MP3) using FluidSynth.

    Parameters
    ----------
    midi_path   : Path to the .mid file
    output_stem : Output path without extension (e.g. "outputs/my_song").
                  Defaults to midi_path without .mid extension.
    sample_rate : Audio sample rate (default 44100)
    gain        : FluidSynth gain (0.0–5.0, default 0.8)

    Returns
    -------
    (wav_path, mp3_path)
        wav_path  — path to rendered WAV file, or None if render failed
        mp3_path  — path to MP3 file, or None if conversion unavailable
    """
    midi_path = str(midi_path)
    if output_stem is None:
        output_stem = midi_path.replace(".mid", "")

    wav_path = output_stem + ".wav"
    mp3_path = output_stem + ".mp3"

    fluidsynth = _find_fluidsynth()
    soundfont  = _find_soundfont()

    if not fluidsynth:
        print("[audio_render] FluidSynth not found. Set FLUIDSYNTH_PATH in config.py")
        return None, None

    if not soundfont:
        print("[audio_render] No .sf2 soundfont found. Add one to soundfonts/ folder.")
        return None, None

    # Build FluidSynth command
    cmd = [
        fluidsynth,
        "-ni",                      # no interactive mode
        "-g", str(gain),            # gain
        "-r", str(sample_rate),     # sample rate
        "-F", wav_path,             # output WAV file
        soundfont,
        midi_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"[audio_render] FluidSynth error:\n{result.stderr[:500]}")
            return None, None
        if not Path(wav_path).exists():
            print("[audio_render] FluidSynth ran but WAV file not created.")
            return None, None
        print(f"[audio_render] WAV rendered: {wav_path} ({Path(wav_path).stat().st_size//1024} KB)")
    except subprocess.TimeoutExpired:
        print("[audio_render] FluidSynth timed out (>120s).")
        return None, None
    except Exception as e:
        print(f"[audio_render] Render failed: {e}")
        return None, None

    # Convert WAV → MP3 using pydub if available
    mp3_result = _wav_to_mp3(wav_path, mp3_path)
    return wav_path, mp3_result


def _wav_to_mp3(wav_path: str, mp3_path: str) -> str | None:
    """Convert WAV to MP3 using pydub. Returns mp3_path or None."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate="192k")
        print(f"[audio_render] MP3 exported: {mp3_path} ({Path(mp3_path).stat().st_size//1024} KB)")
        return mp3_path
    except ImportError:
        print("[audio_render] pydub not installed — WAV only. Run: pip install pydub")
        return None
    except Exception as e:
        print(f"[audio_render] MP3 conversion failed: {e}")
        return None


def fluidsynth_available() -> bool:
    """Return True if FluidSynth and a soundfont are both available."""
    return bool(_find_fluidsynth() and _find_soundfont())


def status() -> str:
    """Human-readable status for UI display."""
    fs = _find_fluidsynth()
    sf = _find_soundfont()

    if not fs:
        return (
            "⚠ FluidSynth not found.\n"
            "  Set FLUIDSYNTH_PATH in config.py to:\n"
            r"  C:\Users\Aditi Sriram\Downloads\fluidsynth-v2.5.4-win10-x64-cpp11"
            r"\fluidsynth-v2.5.4-win10-x64-cpp11\bin\fluidsynth.exe"
        )
    if not sf:
        return (
            "⚠ No soundfont found.\n"
            "  Add a .sf2 file to the soundfonts/ folder."
        )
    sf_name = Path(sf).name
    return f"✓ FluidSynth ready · Soundfont: {sf_name}"