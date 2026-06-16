"""
Humanize MIDI note lists with subtle timing, velocity, and duration variations.

A "humanize amount" of 0.0 = perfectly mechanical (no change).
A value of 1.0 = noticeable looseness — still musical, not sloppy.
"""

from __future__ import annotations
import copy
import random


def humanize_notes(
    notes: list[dict],
    amount: float = 0.3,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Return a new note list with human-like imperfections applied.

    Variations applied (scaled by amount):
    - Velocity jitter (gaussian, ±10 units at amount=1)
    - Timing offset (gaussian, ±30 ms equivalent at 120 BPM for amount=1)
    - Duration variation (gaussian, ±4% per note at amount=1)
    - Occasional inner-voice drop (simulates a pianist missing a chord note)
    """
    if amount <= 0.0 or not notes:
        return list(notes)

    if rng is None:
        rng = random.Random()

    timing_std  = 0.025 * amount   # beats  (~12ms at 120 BPM per unit)
    vel_std     = 9     * amount   # MIDI velocity units
    dur_std     = 0.04  * amount   # fraction of note duration
    # Inner-voice drops only kick in above 50% humanize
    drop_prob   = max(0.0, (amount - 0.5) * 0.12)

    result = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        n = dict(note)

        n["velocity"] = max(20, min(127,
            int(n.get("velocity", 72) + rng.gauss(0, vel_std))))

        jitter = rng.gauss(0, timing_std)
        n["start_beat"] = max(0.0, round(n.get("start_beat", 0.0) + jitter, 4))

        dur = n.get("duration_beats", 1.0)
        n["duration_beats"] = max(0.05, round(dur * (1.0 + rng.gauss(0, dur_std)), 4))

        # Drop quiet inner-voice notes occasionally at higher amounts
        if drop_prob > 0 and n.get("velocity", 72) < 60 and rng.random() < drop_prob:
            continue

        result.append(n)

    return result


def humanize_orchestration(
    orchestration: dict,
    amount: float = 0.3,
    melody_instruments: frozenset[str] | None = None,
    seed: int | None = None,
) -> dict:
    """
    Apply humanization to every part in an orchestration dict.
    Melody instruments get slightly more timing freedom (×1.2).
    """
    result = copy.deepcopy(orchestration)
    rng = random.Random(seed)

    for inst, notes in result.get("parts", {}).items():
        if not isinstance(notes, list):
            continue
        is_melody = melody_instruments and inst in melody_instruments
        inst_amount = min(1.0, amount * 1.2) if is_melody else amount
        result["parts"][inst] = humanize_notes(notes, amount=inst_amount, rng=rng)

    return result
