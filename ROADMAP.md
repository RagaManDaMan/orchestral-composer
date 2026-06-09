# Orchestral Composer — Integration Plan & Feature Roadmap

_Ragavan Manian · June 2026_

---

## Context

My intern has been working in parallel on this codebase for several weeks. Her changes are
substantial — a new raga engine, Indian audio engine, role-based voicing, 22 genre presets,
vocal enhancement pipeline, and a text-to-music model integration. Meanwhile I have been
making significant improvements to the live accompaniment engine: BPM floor extended to 20
for vilambit laya, drum density tiers, guitar reverb and humanizing, half-time redesign, Markov
chord model fixes for Indian modes, bass/chord range selectors, and a melodic response
engine (call and response). She now needs to integrate her work on top of my changes.

---

## Part 1 — Code Integration Plan

### Conflict Map

| File | Her status | My status | Risk |
|------|-----------|----------|------|
| `src/vocal_processor.py` | New | Untouched | None — additive |
| `src/melody_rnn.py` | New | Untouched | None — additive |
| `src/raga_engine.py` | New | Untouched | None — additive |
| `src/indian_audio_engine.py` | New, has circular import bug | Untouched | Medium — bug blocks Indian presets entirely |
| `src/drum_engine.py` | New (MIDI export drums) | Untouched (my drums are live JS) | Low — parallel systems, need shared pattern names |
| `src/prompts.py` | Expanded 15 → 40+ instruments | Untouched | Low — additive only |
| `src/algo_arranger.py` | **Rewritten** with role-based voicing | Untouched | **High** — core pipeline file, regression risk |
| `config.py` | **Rewritten** with 8 → 22 presets | Untouched | **Medium** — my runtime constants must survive her reorganization |
| `app.py` | Untouched by her | Heavy changes by me | No conflict from her side |

### Step-by-step merge sequence

**Step 1 — She branches from today's HEAD (`676ef1a`):**
```bash
git checkout main && git pull
git checkout -b intern/feature-integration
```
This gives her my full live accompaniment engine as a baseline — drum density tiers, melodic
response engine, BPM floor at 20, range selectors, all the Markov fixes.

**Step 2 — She commits the zero-conflict new files first:**
In a single commit, she adds the files that touch nothing existing:
`vocal_processor.py`, `melody_rnn.py`, `raga_engine.py`, `drum_engine.py`.
She verifies clean imports before committing:
```bash
python -c "from src import vocal_processor, melody_rnn, raga_engine, drum_engine"
```

**Step 3 — She fixes the circular import before touching `indian_audio_engine.py`:**
The defect report says `indian_audio_engine.py` causes intermittent circular dependency
faulting between pipeline entry points and arrangement modules. The fix: she extracts any
shared types or constants (raga definitions, instrument names, sample paths) into a new
`src/types.py` that neither `app.py` nor `indian_audio_engine.py` imports back from. Both
can then import from the neutral module without creating a cycle. She confirms the fix, then
commits `indian_audio_engine.py`.

**Step 4 — She merges `prompts.py` (additive, low risk):**
Her 25 new instruments are pure additions to existing dicts. She hand-merges from a
`git diff src/prompts.py` against `main` rather than overwriting, so any future changes I
make to the base instrument list are not silently dropped.

**Step 5 — She merges `config.py` (medium risk):**
She merges her 22 presets and style hints wholesale, then verifies that these constants from
my side are present and unchanged:
`OLLAMA_MODEL`, `SAMPLE_RATE`, `GRID_SIZE_BEATS`, `MIN_NOTE_DURATION_SEC`,
`DEFAULT_TEMPO_BPM`, `DEFAULT_MAX_TOKENS`, `SOUNDFONT_PATH`.
She also checks that the `INSTRUMENT_PRESETS` dict structure still matches what `app.py`
expects — I reference it by preset name in the pipeline.

**Step 6 — She merges `algo_arranger.py` last (high risk):**
This is the most dangerous merge because it is a full rewrite of a core file. Before touching
anything, she reads both versions side by side. She then verifies that her new
`inject_algo_parts()` signature is identical — `app.py` calls it by name with specific
arguments. She runs the integration test:
_Harmony Only → Jazz Quartet → C major → 8 bars → MIDI has chords, bass, correct note
ranges_.
If it passes, she commits. If not, I keep her rewrite on a sub-branch until she stabilises it.

**Step 7 — She opens a PR from `intern/feature-integration` → `main`.**
The PR description explicitly flags `algo_arranger.py` and `config.py` as the items requiring
my review before merge. I review, we discuss, and I merge.

### One naming coordination issue to resolve now

Her `drum_engine.py` (used for MIDI export) and my live JS drum library share the same
conceptual patterns but use different names. As the project grows, a producer will hear "Blues
Shuffle" in the live engine and get a different pattern in the exported MIDI. We agree on a
canonical name set early and map both systems to it, even though the synthesis path differs.

---

## Part 2 — Feature Roadmap

### Week 1 — Fix broken things and quick wins

**Fix the circular import in `indian_audio_engine.py`.**
This blocks the Indian presets entirely. It is the first task before anything else because it
gates a large portion of her other work.

**Wire `render_midi_to_audio` to the UI.**
Her report says the backend function already exists but lacks a Gradio binding. I wire a
"Render to Audio" button that calls the function and returns a downloadable WAV/MP3. This
is half a day of work and immediately gives all users a way to hear the arrangement without a
DAW.

**Add tap tempo to the live accompaniment panel.**
A simple button in the panel measures inter-tap intervals and sets the BPM slider. She listed
it as scheduled. It is 20 lines of JavaScript.

**SGM-v2.01 soundfont.**
I add download instructions to the README and auto-detection logic in `config.py`. Immediate
and noticeable improvement to sitar, bansuri, and tabla patches with zero code risk.

---

### Week 2 — Musical intelligence

**Raga engine → melodic response.**
I wire her `raga_engine.py` data — vadi, samvadi, pakad, arohana, avarohana — directly
into my `_freePhrase` and `_answerPhrase` functions in the live accompaniment JS. Right now
my response engine treats all scale degrees equally. With raga data, the free phrase knows
which swara to emphasize, which movements are characteristic, and where to land. This is
the single most impactful improvement to the melodic response feature and directly addresses
my own assessment that it is "in very early stages."

**Chord-aware melody generation.**
She correctly identified that the ABC transformer generates melodies that fit the key but ignore
beat-by-beat chord changes. I add a post-processing layer that looks at which chord is active
at each beat and biases note selection toward chord tones at that moment. The melody then
actually lands on the right notes when the harmony changes.

**Vocal processor integration.**
She says `vocal_processor.py` is functional. I wire it into the Upload / Record pipeline so
the six-stage chain — noise gate, noise reduction, EQ, compression, tuning fix, pitch
correction — runs automatically before transcription. No new UI needed; it runs inline.

**Mridangam velocity layering.**
Her report flags this as a limitation: hits use accurate transients but sound static because
there is no dynamic velocity layering. I add two additional velocity layers per stroke
(soft/medium/loud, either using samples or synthesis variation) so the mridangam responds
to the velocity values already in the pattern.

---

### Week 3 — New surfaces

**Pitch bend and gamaka for raga melodies.**
Real Indian classical music requires continuous pitch movement between notes — meend
(glide), andolan (oscillation), gamaka (ornament). MIDI pitch bend messages approximate
this. I post-process the raga melody note list in `midi_builder.py`, inserting pitch bend
curves between adjacent notes where the raga's characteristic gamaka patterns call for a
slide. Her `raga_engine.py` already has the gamaka data per raga; I use it here. This is the
single biggest gap between what MIDI can express and what Indian classical music requires,
and closing it is the most musically meaningful work on the roadmap.

**Melody variations.**
The app already has a Variations slider (1–4). I wire her ABC model to generate variations at
different temperatures and style biases rather than running the same Ollama path with
different random seeds. I present them side by side in the player and let the user pick which
one to arrange.

**Piano roll editor (read-only first).**
I render the generated MIDI as an HTML Canvas grid inside a `gr.HTML()` component — notes
as coloured blocks, time on the x-axis, pitch on the y-axis. Read-only first so the user can
see exactly what was generated. I add click-to-edit in a second pass: click a block to remove
it, drag to move, click blank space to add. Changes write back to the orchestration dict before
MIDI export.

---

### Longer term

**Melodic response — CREPE capture.**
I replace the WebAudio autocorrelation pitch detection in the Melodic Response panel with a
round-trip to the Python CREPE backend already in `src/transcribe.py`. The accuracy gain
for gamaka and meend singing is significant. The latency overhead is acceptable because the
capture phase is not real-time — the user plays, stops, then triggers the response.

**Soundfont manager in the UI.**
I add a panel that lets the user browse, download, and switch soundfonts without editing
`config.py`. The immediate target is SGM-v2.01 for Indian instruments, with the ability to
add others.

**Per-section style override.**
I let A sections use jazz comping while K sections use korvai stabs, without the style setting
being global. This has been in my notes since the korvai engine was first built.

**Eduppu offset control.**
The K section entry point within the tala cycle. Already planned in the korvai roadmap.

---

## Part 3 — ACE-Step Integration

### Why ACE-Step is relevant

My motivation for building this tool in the first place was the limitations of ACE-step's
repaint and remix: it did not preserve enough of my musical intent. When I asked it to repaint
a section of a Carnatic piece, it drifted from the raga. When I used remix, it lost the tala
structure. ACE-step is a powerful audio model but it does not understand musical structure —
it has no concept of arohana/avarohana, of a korvai resolving to sam, of a specific raga's
characteristic phrases.

Orchestral Composer gives ACE-step something it lacks: **structured musical intent encoded
as MIDI with full theoretical metadata** — raga, tala, korvai form, voice leading, instrument
roles. ACE-step gives Orchestral Composer something it lacks: **audio realism beyond what
MIDI and FluidSynth can produce**.

The combination is more powerful than either alone.

### ACE-Step 1.5 capabilities relevant here

ACE-Step 1.5 (released January 2026) runs locally on Apple Silicon and supports:
- **Text + audio conditioning** — I provide a style description and an audio clip; it generates
  audio that matches both
- **singing2accompaniment** — I provide a vocal stem; it generates a complementary
  backing arrangement
- **Repaint** — I provide a full track and mark a region; it regenerates only that region while
  preserving the rest
- **Stem generation** — given a reference track, it generates a single instrument stem that
  complements it

### Integration path

**Phase A — ACE-Step as a downstream renderer (low risk, high value).**
My current pipeline ends at MIDI. I render that MIDI to audio via FluidSynth, then pass the
rendered audio to ACE-step with a text prompt derived from the preset and mode:
_"Carnatic Ensemble, Bhairavi raga, slow vilambit tempo, tanpura drone, mridangam"_.
ACE-step uses the MIDI-rendered audio as its conditioning signal (melody reference) and
generates a full mix in the correct style. Because my MIDI carries the correct pitch content
and rhythm, ACE-step's output stays in the raga and respects the structure — which is exactly
what straight ACE-step text prompting failed to do.

The implementation: after MIDI export, a "Render with ACE-Step" button runs the FluidSynth
render, then calls the ACE-step local API with the audio + a generated prompt, and returns
the WAV alongside the MIDI outputs.

**Phase B — ACE-Step repaint, section by section.**
Because I export per-section MIDI files (A.mid, K1.mid, B.mid), I can render each section
separately and call ACE-step repaint on individual sections rather than the whole piece. If the
K1 korvai section sounds wrong, I repaint just that 8-bar clip without touching A and B. This
is the specific use case where ACE-step repaint failed me before — I was asking it to repaint
within a track it had generated, with no structural grounding. Now I am giving it a
FluidSynth-rendered section with exact MIDI pitch content as the conditioning signal, so the
repaint stays in the raga.

**Phase C — singing2accompaniment into my pipeline.**
I sing a phrase into the Melodic Response panel. Instead of — or alongside — my algorithmic
response generator, I pass the captured vocal audio to ACE-step's singing2accompaniment
mode with the current preset as the style prompt. ACE-step generates a short backing phrase
in the correct genre and instrumentation. I mix it back into the live accompaniment session at
the Melody track fader level.

This closes the loop between my singing and a realistic Indian instrument response — bansuri,
veena, or sarangi — rather than the triangle+LFO synthesiser I use today.

**Phase D — Gamaka audio synthesis.**
For gamakas that MIDI pitch bend cannot adequately express (especially gamakas on veena
and bansuri), I use ACE-step to synthesise short ornament clips — one per gamaka type per
raga — and splice them into the rendered audio at the correct positions. Her `raga_engine.py`
already has gamaka patterns per raga; I use that data to drive which clips to generate and
where to place them.

### What this is not

I am not replacing my pipeline with ACE-step. The MIDI pipeline gives me something ACE-step
will never give on its own: a structurally correct, theoretically grounded, editable score that
I can take into Logic Pro, fix, and build on. ACE-step is a rendering and enhancement layer —
a way to turn my structured MIDI intent into audio that sounds like real musicians, without
losing control of the structure.

---

_Sources consulted for ACE-Step capabilities:_  
_[ACE-Step project page](https://ace-step.github.io/) · [ACE-Step 1.5 on GitHub](https://github.com/ace-step/ACE-Step-1.5) · [ACE-Step 1.5 paper](https://arxiv.org/html/2602.00744v1)_
