"""
prepare_indian_audio.py
=======================
One-command pipeline to build high-quality Indian instrument audio.

Phase 1 — Slice raw recordings into chromatic note libraries (setup_samples.py)
Phase 2 — (OPTIONAL, heavy) Train neural synthesis models — needs a powerful machine

For most laptops, Phase 1 alone is enough. The engine uses your real sliced
samples + Rubber Band pitch shift. Set USE_NEURAL_INSTRUMENTS = True in
config.py only after fully training a model on a desktop/GPU machine.

Usage:
    py -3.11 prepare_indian_audio.py              # slice samples (recommended)
    py -3.11 prepare_indian_audio.py --status     # show sample/model status
    py -3.11 prepare_indian_audio.py --train      # optional: train (heavy)

Before running:
  1. Place raw recordings under samples/ (any subfolder — auto-routed)
  2. Install deps:  py -3.11 -m pip install -r requirements.txt
  3. For best pitch quality: install Rubber Band library + pyrubberband
     Windows: download rubberband from https://breakfastquay.com/rubberband/
     Then:    py -3.11 -m pip install pyrubberband
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).parent
INSTRUMENTS = ["veena", "bansuri", "sitar", "violin"]


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=PROJECT)
    if result.returncode != 0:
        print(f"[prepare] FAILED: {label}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Indian audio samples and neural models")
    parser.add_argument("--train", action="store_true", help="Train neural models after slicing")
    parser.add_argument("--instrument", type=str, choices=INSTRUMENTS,
                        help="Train only this instrument (requires --train)")
    parser.add_argument("--status", action="store_true", help="Show sample and model status")
    parser.add_argument("--skip-slice", action="store_true", help="Skip sample slicing phase")
    args = parser.parse_args()

    if args.status:
        sys.path.insert(0, str(PROJECT))
        from src.indian_audio_engine import samples_status, setup_sample_folders
        setup_sample_folders()
        print(samples_status())
        return 0

    if not args.skip_slice:
        ok = _run(
            [sys.executable, str(PROJECT / "setup_samples.py")],
            "Phase 1: Slice recordings into chromatic note libraries",
        )
        if not ok:
            return 1

    if args.train:
        targets = [args.instrument] if args.instrument else INSTRUMENTS
        for inst in targets:
            samples_dir = PROJECT / "samples" / inst
            wav_count = len(list(samples_dir.glob("*.wav"))) if samples_dir.exists() else 0
            if wav_count < 5:
                print(f"[prepare] Skipping {inst}: only {wav_count} note WAVs (need 5+). Add recordings and re-run.")
                continue
            ok = _run(
                [sys.executable, str(PROJECT / "train_instrument.py"), "--instrument", inst],
                f"Phase 2: Train neural model for {inst} ({wav_count} samples)",
            )
            if not ok:
                print(f"[prepare] Training failed for {inst} — continuing with other instruments")

    sys.path.insert(0, str(PROJECT))
    from src.indian_audio_engine import samples_status
    print("\n" + samples_status())
    print("\nDone. Re-run the app and generate an Indian preset to hear the improvements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
