import sys
sys.path.insert(0, ".")

from src.indian_audio_engine import (
    _make_tanpura_drone, _render_melody, _render_mridangam,
    _load_chromatic_library, samples_status
)

print(samples_status())
print()

# Test tanpura
print("Testing tanpura...")
drone = _make_tanpura_drone("C", 10.0)
print(f"Drone: {len(drone)} samples, max={drone.max():.3f}")

# Test melody library
print("\nTesting veena library...")
lib = _load_chromatic_library("veena")
print(f"Loaded {len(lib)} notes: {sorted(lib.keys())}")

# Test mridangam
print("\nTesting mridangam...")
from src.indian_audio_engine import _render_mridangam
perc = _render_mridangam(16.0, 90.0)
print(f"Mridangam: {len(perc)} samples, max={perc.max():.3f}")

print("\nAll OK — if you see this, samples are loading fine")