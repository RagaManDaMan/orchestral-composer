import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.raga_engine import generate_raga_melody, resolve_raga
import librosa

def test_melody():
    key = "D Kalyani"
    tempo = 90.0
    total_beats = 32.0
    raga = resolve_raga(key)
    
    print(f"Raga: {raga.name}")
    print(f"Arohana: {raga.arohana}")
    print(f"Avarohana: {raga.avarohana}")
    
    notes = generate_raga_melody(key=key, tempo_bpm=tempo, total_beats=total_beats, raga=raga)
    
    print(f"\nGenerated {len(notes)} notes:")
    prev_midi = None
    max_jump = 0
    step_count = 0
    jump_count = 0
    
    for i, n in enumerate(notes):
        note_name = n["note"]
        start = n["start_beat"]
        dur = n["duration_beats"]
        role = n.get("role", "main")
        
        midi = librosa.note_to_midi(note_name)
        if role == "grace":
            print(f"  [{i:2d}] {note_name:4s} (MIDI {midi:3d}) | start={start:5.2f} dur={dur:5.2f} | grace")
            continue
            
        if prev_midi is not None:
            dist = abs(midi - prev_midi)
            max_jump = max(max_jump, dist)
            if dist <= 2:
                step_count += 1
            else:
                jump_count += 1
            dist_str = f" | dist={dist:2d}"
        else:
            dist_str = " | dist=--"
            
        print(f"  [{i:2d}] {note_name:4s} (MIDI {midi:3d}) | start={start:5.2f} dur={dur:5.2f}{dist_str}")
        prev_midi = midi
        
    print(f"\nMelody Statistics:")
    print(f"  Total Melodic Steps: {step_count + jump_count}")
    print(f"  Stepwise / Small Skips (<= 2 semitones): {step_count} ({step_count / max(1, step_count + jump_count) * 100:.1f}%)")
    print(f"  Larger Jumps (> 2 semitones): {jump_count} ({jump_count / max(1, step_count + jump_count) * 100:.1f}%)")
    print(f"  Max Jump: {max_jump} semitones")

if __name__ == "__main__":
    test_melody()
