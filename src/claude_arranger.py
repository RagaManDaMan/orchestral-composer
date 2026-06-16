"""
Claude API-based harmony enhancement.

When the user rates an output as bad, this module sends the chord progression
to Claude for reharmonization — secondary dominants, tritone substitutions,
passing chords. Claude returns an enhanced chord timeline; the algo arranger
then handles all voice leading from there.

Training examples (prompt+response pairs) are saved to training_examples/ so
they can later be injected as few-shot context into Ollama prompts.
"""

from __future__ import annotations
import json
import os
import re
from datetime import datetime
from pathlib import Path

TRAINING_DIR = Path(__file__).parent.parent / "training_examples"
TRAINING_DIR.mkdir(exist_ok=True)

_REHARMONIZE_PROMPT = """\
You are an expert jazz and symphonic arranger. The composer has provided a chord progression \
and wants tasteful reharmonization.

Key: {key}
Style context: {style}

Original chord timeline (beat → chord):
{chords_str}

Total duration: {total_beats:.0f} beats

Your task:
1. Analyze each chord's harmonic function (T=tonic, SD=subdominant, D=dominant, X=chromatic/borrowed)
2. Insert secondary dominants (V7/X) before non-tonic chords that last long enough (≥ 2 beats)
3. Consider tritone substitutions on dominant chords where they add chromatic interest
4. Consider passing chords between root-motion leaps of a 4th or more
5. Keep all the composer's original chords — only INSERT preparation chords

Constraints:
- Maximum 30% of original chords should get a preparation — avoid over-spicing
- A secondary dominant inserted before a chord takes half the original chord's duration
- The remaining half goes to the original chord at the beat where it would now start
- Tritone substitution REPLACES a dominant chord (same beat, different root +6 semitones)
- Never insert a chord that lasts less than 1 beat

Respond with ONLY this JSON (no markdown fences, no explanation outside JSON):
{{
  "enhanced_timeline": [
    {{"beat": 0.0, "chord": "Dm7"}},
    {{"beat": 2.0, "chord": "A7"}},
    {{"beat": 3.0, "chord": "Dm7"}}
  ],
  "changes": [
    "Inserted A7 (V7/ii = secondary dominant) before Dm7 at beat 2; split its 4-beat slot in half",
    "..."
  ]
}}"""


def reharmonize_with_claude(
    chord_timeline: list[tuple[float, str]],
    key: str,
    style: str = "",
    save_example: bool = True,
    model: str = "claude-haiku-4-5-20251001",
    total_beats: float | None = None,
) -> tuple[list[tuple[float, str]], str]:
    """
    Ask Claude to reharmonize a chord progression with tasteful substitutions.

    Returns:
        (enhanced_timeline, explanation)
        enhanced_timeline — potentially extended chord list
        explanation      — bullet-point description of what changed and why
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable not set.\n"
            "Export it in your shell:  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=api_key)

    chords_str = "\n".join(f"  beat {b:.1f} → {sym}" for b, sym in chord_timeline)
    if total_beats is None:
        total_beats = chord_timeline[-1][0] + 4.0 if chord_timeline else 16.0

    prompt = _REHARMONIZE_PROMPT.format(
        key=key,
        style=style or "general orchestral/jazz",
        chords_str=chords_str,
        total_beats=total_beats,
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    json_str = fence.group(1) if fence else raw

    data = json.loads(json_str)
    enhanced: list[tuple[float, str]] = [
        (float(item["beat"]), str(item["chord"]))
        for item in data["enhanced_timeline"]
    ]
    enhanced.sort(key=lambda x: x[0])

    changes = data.get("changes", [])
    explanation = "\n".join(f"  • {c}" for c in changes) if changes else "  • No changes applied"

    if save_example:
        _save_training_example(chord_timeline, key, style, enhanced, explanation)

    return enhanced, explanation


def _save_training_example(
    original_timeline: list[tuple[float, str]],
    key: str,
    style: str,
    enhanced_timeline: list[tuple[float, str]],
    explanation: str,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    example = {
        "timestamp": timestamp,
        "key": key,
        "style": style,
        "original_chords": [[b, sym] for b, sym in original_timeline],
        "enhanced_chords": [[b, sym] for b, sym in enhanced_timeline],
        "explanation": explanation,
    }
    path = TRAINING_DIR / f"harmony_{timestamp}.json"
    path.write_text(json.dumps(example, indent=2))


def load_few_shot_examples(max_examples: int = 3) -> list[dict]:
    """
    Load recent saved training examples. Used to inject few-shot context
    into future Ollama prompts so the local model learns from Claude's choices.
    """
    files = sorted(TRAINING_DIR.glob("harmony_*.json"), reverse=True)[:max_examples]
    return [json.loads(f.read_text()) for f in files]


def format_few_shot_for_prompt(examples: list[dict]) -> str:
    """Format training examples as a section to prepend to Ollama prompts."""
    if not examples:
        return ""
    lines = ["REHARMONIZATION EXAMPLES (from expert arranger — follow this style):\n"]
    for ex in examples:
        orig = ", ".join(sym for _, sym in ex["original_chords"])
        enhanced = ", ".join(sym for _, sym in ex["enhanced_chords"])
        lines.append(f"Key: {ex['key']} | Original: {orig}")
        lines.append(f"Enhanced: {enhanced}")
        lines.append(ex["explanation"])
        lines.append("")
    return "\n".join(lines)
