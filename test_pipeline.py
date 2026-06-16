"""
End-to-end pipeline test — no UI, no microphone.

Generates a synthetic sine-wave melody (D major scale), runs it through
every stage of the pipeline, and writes a MIDI file to examples/test_output.mid.

Run:  python test_pipeline.py
"""

import sys
import os
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))

from src.transcribe import transcribe_audio, quantize_to_grid, detect_key
from src.orchestrate import orchestrate, check_ollama
from src.midi_builder import build_midi


def make_sine_melody(
    notes_hz: list[float],
    duration_sec: float = 0.4,
    sr: int = 22050,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Synthesise a simple melody as a numpy audio array."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    silence = np.zeros(int(sr * 0.05))   # 50 ms gap between notes
    segments = []
    for freq in notes_hz:
        tone = amplitude * np.sin(2 * np.pi * freq * t)
        # Simple fade in/out to avoid clicks
        fade = int(sr * 0.01)
        tone[:fade]  *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        segments.extend([tone, silence])
    return np.concatenate(segments).astype(np.float32)


# D major scale: D4 E4 F#4 G4 A4 B4 C#5 D5
D_MAJOR_HZ = [293.66, 329.63, 369.99, 392.00, 440.00, 493.88, 554.37, 587.33]


def main():
    print("=== Orchestral Composer — Pipeline Test ===\n")

    # --- 1. Create synthetic audio ---
    print("[1/5] Generating synthetic D major scale melody...")
    audio = make_sine_melody(D_MAJOR_HZ, duration_sec=0.5)
    os.makedirs("examples", exist_ok=True)
    audio_path = "examples/test_melody.wav"
    sf.write(audio_path, audio, 22050)
    print(f"      Saved: {audio_path}  ({len(audio)/22050:.1f} s)")

    # --- 2. Transcribe ---
    print("[2/5] Transcribing with pyin...")
    raw_notes = transcribe_audio(audio_path)
    if not raw_notes:
        print("      ERROR: No notes detected. Check that soundfile/librosa are installed.")
        sys.exit(1)
    print(f"      Detected {len(raw_notes)} raw note segments")

    # --- 3. Quantise and detect key ---
    print("[3/5] Quantising to beat grid and detecting key...")
    tempo = 120.0
    notes = quantize_to_grid(raw_notes, tempo)
    key   = detect_key(notes)
    total_beats = max(n["start_beat"] + n["duration_beats"] for n in notes)
    print(f"      Key: {key}  |  Tempo: {tempo} BPM  |  Duration: {total_beats:.1f} beats")
    for n in notes:
        print(f"        {n['note_name']:4s}  start={n['start_beat']:.2f}  dur={n['duration_beats']:.2f}")

    # --- 4. Check Ollama ---
    print("[4/5] Checking Ollama...")
    ok, msg = check_ollama()
    print(f"      {msg}")
    if not ok:
        print("\n      Skipping LLM step — run 'ollama serve' and 'ollama pull llama3.1:8b' first.")
        print("      Writing a stub MIDI from the melody only instead.")
        _write_stub_midi(notes, tempo, "examples/test_output_stub.mid")
        return

    # --- 5. Orchestrate ---
    print("[5/5] Orchestrating via LLM (may take 30–90 s)...")
    instruments = ["strings_melody", "strings_harmony", "bass"]
    orchestration = orchestrate(notes, key, tempo, total_beats, instruments)

    output_path = "examples/test_output.mid"
    build_midi(orchestration, output_path)
    size = os.path.getsize(output_path)

    print(f"\n=== Done ===")
    print(f"MIDI written to: {output_path}  ({size} bytes)")
    print(f"Parts generated:")
    for part, part_notes in orchestration["parts"].items():
        print(f"  {part}: {len(part_notes)} notes")
    print(f"\nDrag {output_path} into Logic Pro to hear the result.")


def _write_stub_midi(notes, tempo, output_path):
    """Write a single-track MIDI from just the melody — no LLM needed."""
    orchestration = {
        "tempo": tempo,
        "parts": {
            "strings_melody": [
                {"note": n["note_name"], "start_beat": n["start_beat"],
                 "duration_beats": n["duration_beats"], "velocity": 80}
                for n in notes
            ]
        }
    }
    build_midi(orchestration, output_path)
    print(f"      Stub MIDI written: {output_path}")


if __name__ == "__main__":
    main()
