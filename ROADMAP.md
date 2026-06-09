# Orchestral Composer — Vision, Integration Plan & Feature Roadmap

_Ragavan Manian · June 2026_

---

## Why This Exists

Every serious AI music tool today — Suno, Udio, ACE-Step, Soundraw — shares the same
blind spot: they generate audio, not music. They have no concept of a raga's characteristic
phrase, of a korvai resolving to sam, of a meend that is specific to Bhairavi and wrong in
Yaman. They produce convincing sound but structurally empty output. You cannot take what
they give you into a DAW and build on it, because there is nothing structurally there to build on.

Commercial tools also assume a subscription, an internet connection, and a Western frame of
reference. A music student in Chennai or Varanasi cannot afford Suno. They also should not
have to translate their raga into "A minor scale" to get something usable out of a tool.

Orchestral Composer is built to close both gaps. It runs entirely on-device via Ollama — free,
private, no cloud dependency. And it treats Indian classical music as a first-class system, not
an afterthought. Carnatic ragas, Hindustani ragas, korvai, tala, vilambit laya, gamaka — these
are not edge cases. They are the primary design target.

The competitive moat is not any single feature. It is the combination of three things that
no other tool has together:

1. **Structured musical intent** — theoretically correct, editable MIDI with full metadata
   (raga, tala, korvai form, voice leading, instrument roles). This is what AI audio generators
   cannot produce on their own, and what makes this tool a composer's instrument rather than
   a sound generator.

2. **Indian classical music as a first-class system** — raga engine, korvai engine, tala
   awareness, gamaka, vilambit BPM down to 20. Built for a student in India who wants to
   hear what their composition sounds like before they perform it.

3. **A live accompanist that understands the raga** — the melodic response engine listens to
   what you play or sing, identifies the scale degrees within the active raga, and responds in
   kind. No commercial tool offers call-and-response that is raga-aware.

Everything on this roadmap is a move toward making those three things stronger.

---

## Current State

The live accompaniment engine is the most mature part of the system. It supports 9 chord
styles, 12 drum patterns with proper density tiers, MIDI clock sync to Logic Pro, passing
chords, Markov chord transitions corrected for Indian modes (accidental-prefix degrees no
longer get a Western V→I pull), bass and chord range selectors, and a melodic response
engine that captures a sung or played phrase and generates a call-and-response in the active
raga's scale. BPM goes down to 20 for vilambit laya.

The MIDI export pipeline generates multi-track Type-1 MIDI with per-section files, korvai
bridges, and Synfire Pro compatibility.

My intern has been building in parallel: a raga engine with 15 ragas and full musicological
data, an Indian audio engine with real tanpura drone and chromatic sample arrays for veena
and bansuri, a role-based voicing system, a text-to-music model integration, a vocal
enhancement pipeline, and 22 genre presets spanning Western and Indian styles. Her work
is the foundation for making the Indian classical side match the structural correctness of the
Western side.

---

## Part 1 — Code Integration Plan

### Conflict Map

| File | Her status | My status | Risk |
|------|-----------|----------|------|
| `src/vocal_processor.py` | New | Untouched | None |
| `src/melody_rnn.py` | New | Untouched | None |
| `src/raga_engine.py` | New | Untouched | None |
| `src/indian_audio_engine.py` | New, circular import bug | Untouched | Medium — blocks Indian presets entirely |
| `src/drum_engine.py` | New (MIDI export drums) | Untouched | Low — parallel to my live JS drums |
| `src/prompts.py` | Expanded 15 → 40+ instruments | Untouched | Low — additive |
| `src/algo_arranger.py` | **Rewritten** (role-based voicing) | Untouched | **High** — core pipeline file |
| `config.py` | **Rewritten** (8 → 22 presets) | Untouched | **Medium** — my runtime constants must survive |
| `app.py` | Untouched by her | Heavy changes by me | None from her side |

### Merge Sequence

**Step 1 — Branch from today's HEAD (`676ef1a`):**
```bash
git checkout main && git pull
git checkout -b intern/feature-integration
```
She gets my full live accompaniment engine, drum density tiers, melodic response engine,
BPM floor at 20, range selectors, and corrected Markov model as her baseline.

**Step 2 — Commit zero-conflict new files first:**
`vocal_processor.py`, `melody_rnn.py`, `raga_engine.py`, `drum_engine.py` in one commit.
Verify imports cleanly:
```bash
python -c "from src import vocal_processor, melody_rnn, raga_engine, drum_engine"
```

**Step 3 — Fix the circular import before committing `indian_audio_engine.py`:**
The defect report says `indian_audio_engine.py` causes intermittent circular dependency
faulting between pipeline entry points and arrangement modules. The fix: extract shared
types and constants into a new `src/types.py` that neither `app.py` nor
`indian_audio_engine.py` imports back from. Fix first, commit after.

**Step 4 — Merge `prompts.py`:**
Hand-merge from a `git diff` against `main` — do not overwrite. Her 25 new instruments
are purely additive.

**Step 5 — Merge `config.py`:**
Her 22 presets wholesale. Then verify these constants survive unchanged:
`OLLAMA_MODEL`, `SAMPLE_RATE`, `GRID_SIZE_BEATS`, `MIN_NOTE_DURATION_SEC`,
`DEFAULT_TEMPO_BPM`, `DEFAULT_MAX_TOKENS`, `SOUNDFONT_PATH`.
Verify `INSTRUMENT_PRESETS` dict structure still matches what `app.py` expects.

**Step 6 — Merge `algo_arranger.py` last:**
Full rewrite, highest risk. Verify `inject_algo_parts()` signature is identical — `app.py`
calls it by name with specific arguments. Run the integration test:
_Harmony Only → Jazz Quartet → C major → 8 bars → MIDI has chords, bass, correct ranges_.
If it passes, commit. If not, hold her rewrite on a sub-branch until stable.

**Step 7 — PR from `intern/feature-integration` → `main`.**
PR description explicitly flags `algo_arranger.py` and `config.py` as review items.

### One naming issue to fix now

Her `drum_engine.py` (MIDI export) and my live JS drum library use different pattern names
for the same grooves. A producer will hear "Blues Shuffle" in the live engine and get a
different pattern in the exported MIDI. We agree on a canonical name set now and map
both systems to it before this divergence calcifies.

---

## Part 2 — Feature Roadmap

The sequencing principle: fix what blocks Indian music first, then deepen the raga
intelligence, then add surfaces that expose the structural correctness to users and
downstream tools.

### Week 1 — Unblock Indian presets and quick wins

**Fix the circular import in `indian_audio_engine.py`.**
This blocks the Indian presets entirely and gates the most distinctive part of the tool.
It is the first task, before anything else.

**Wire `render_midi_to_audio` to the UI.**
Her backend function already exists but has no Gradio binding. I add a "Render to Audio"
button. This is half a day of work and immediately gives users a way to hear a realistic
render without needing Logic Pro or a DAW. It is also what makes the ACE-step integration
possible (see Part 3).

**Add tap tempo.**
A button in the live accompaniment panel that measures inter-tap intervals and sets the BPM
slider. Twenty lines of JavaScript. Important for Indian classical where the student may not
know the exact BPM of their laya — they feel it and tap it.

**SGM-v2.01 soundfont.**
Add download instructions to the README and auto-detection in `config.py`. Immediate and
noticeable improvement to sitar, bansuri, and tabla patches with zero code risk.

---

### Week 2 — Deepen the raga intelligence

**Raga engine → melodic response.**
Right now my `_freePhrase` and `_answerPhrase` treat all scale degrees equally within the
active mode. I wire her `raga_engine.py` data — vadi, samvadi, pakad, arohana, avarohana
— directly into the response generator. The free phrase then knows which swara to
emphasize, which movements are characteristic to the raga, and where to land. This is the
most important single improvement to the live accompanist feature, and it is what makes it
distinctively Indian rather than just "scale-aware."

**Chord-aware melody generation.**
The ABC transformer generates melodies that fit the key but ignore beat-by-beat chord
changes. I add a post-processing layer that biases note selection toward chord tones at each
beat boundary. The melody lands on the right notes when the harmony changes. This
improves the AI Compose mode for all genres.

**Vocal processor integration.**
Her six-stage chain — noise gate, noise reduction, EQ, compression, tuning fix, pitch
correction — runs automatically before transcription in the Upload / Record pipeline. No new
UI needed. Better input means better transcription means better arrangement.

**Mridangam velocity layering.**
Her report flags this as static. I add soft/medium/loud velocity layers per stroke so the
mridangam responds dynamically to the velocity values already encoded in the pattern.

---

### Week 3 — Gamaka and new surfaces

**Pitch bend and gamaka for raga melodies.**
This is the most musically significant item on the roadmap. Real Indian classical music
requires continuous pitch movement between notes — meend (glide), andolan (oscillation),
gamaka (ornament). MIDI pitch bend messages approximate this. I post-process the raga
melody note list in `midi_builder.py`, inserting pitch bend curves between notes where the
raga's characteristic gamaka patterns call for a slide. Her `raga_engine.py` already has
gamaka data per raga. I use it here. This is the gap between a MIDI file that is structurally
correct and one that actually sounds Indian.

**Melody variations.**
The app already has a Variations slider (1–4). I wire her ABC model to generate variations at
different temperatures and style biases, present them side by side in the player, and let the
user pick which one to arrange.

**Piano roll — read-only first.**
I render the generated MIDI as an HTML Canvas grid inside a `gr.HTML()` component. Notes
as coloured blocks, time on the x-axis, pitch on the y-axis. Read-only first so the user can
see exactly what was generated. Click-to-edit in a second pass — add, remove, move notes,
write back to the orchestration dict before MIDI export.

---

### Longer term

**Melodic response — CREPE capture.**
I replace the WebAudio autocorrelation in the Melodic Response panel with a round-trip to
the Python CREPE backend already in `src/transcribe.py`. The accuracy gain for gamaka
and meend singing is significant. The latency is acceptable because capture is not
real-time — the user plays, stops, then triggers the response.

**Per-section style override.**
A sections use jazz comping, K sections use korvai stabs, without the style setting being
global. In the notes since the korvai engine was first built.

**Eduppu offset control.**
K section entry point within the tala cycle. In the korvai roadmap.

**Soundfont manager in the UI.**
Browse, download, switch soundfonts without editing `config.py`.

---

## Part 3 — ACE-Step as a Rendering Layer

### The problem with ACE-Step as a composition tool

My motivation for building this project was the failure of ACE-step's repaint and remix. When
I asked it to repaint a section of a Carnatic piece, it drifted from the raga. When I used remix,
it lost the tala structure. ACE-step is a powerful audio model that generates convincing
sound — but it has no concept of arohana, of a korvai resolving to sam, of a specific raga's
pakad. Ask it for "Bhairavi raga" and it produces something that sounds vaguely Indian. Ask it
again and you get something different. The structure is not there.

This is not a fixable limitation of ACE-step — it is a category difference. ACE-step is an
audio model. I am building a composition model. They are complementary, not competing.

### The integration opportunity

Orchestral Composer gives ACE-step what it lacks: **structured musical intent encoded as
MIDI with full theoretical metadata** — raga, tala, korvai form, voice leading, gamaka
positions, instrument roles. ACE-step 1.5 (released January 2026, runs locally on Apple
Silicon) gives this tool what it lacks: **audio realism beyond FluidSynth**. The combination
is more powerful than either alone.

The key insight is that when ACE-step is conditioned on a FluidSynth-rendered audio signal
derived from my structurally correct MIDI, it stays in the raga and respects the structure —
because the pitch content and rhythm are already baked into the conditioning signal. The
problem I had with repaint was that I was asking ACE-step to regenerate structure it had
generated freely. Now I am giving it structure to follow.

### Four integration phases

**Phase A — ACE-Step as downstream renderer.**
My pipeline ends at MIDI. I render that MIDI to audio via FluidSynth, then pass the rendered
audio to ACE-step with a text prompt derived from the preset and mode:
_"Carnatic ensemble, Bhairavi raga, slow vilambit tempo, tanpura drone, mridangam"_.
ACE-step uses the audio as its melody conditioning signal and generates a full mix in the
correct style. The output sounds like real musicians playing my structurally correct
composition. I add a "Render with ACE-Step" button alongside the existing "Render to Audio"
button.

**Phase B — Per-section repaint.**
Because I already export per-section MIDI files — `A.mid`, `K1.mid`, `B.mid` — I render
each section separately and call ACE-step repaint on individual sections. If the K1 korvai
sounds wrong, I repaint just that 8-bar clip without touching A and B. This is the specific
scenario where ACE-step repaint failed me before. The difference now: instead of asking
ACE-step to repaint within a track it generated freely, I am giving it a FluidSynth-rendered
section grounded in correct raga pitch content as the conditioning signal.

**Phase C — singing2accompaniment into the live accompanist.**
I sing a phrase into the Melodic Response panel. Alongside my algorithmic response
generator, I pass the captured vocal audio to ACE-step's singing2accompaniment mode with
the current preset as the style prompt. ACE-step generates a short backing phrase in the
correct genre and instrumentation — bansuri, veena, or sarangi — and I mix it back into the
live session at the Melody track fader. This closes the loop between my singing and a
realistic Indian instrument response. The algorithmic response engine stays as the fast
fallback; ACE-step is the high-quality path when latency is acceptable.

**Phase D — Gamaka audio synthesis.**
For gamakas that MIDI pitch bend cannot fully express — particularly on veena and bansuri —
I use ACE-step to synthesise short ornament clips, one per gamaka type per raga, and splice
them into the rendered audio at the correct positions. Her `raga_engine.py` already carries
gamaka patterns per raga; I use that data to drive which clips to generate and where to
place them. This is the final step in making the rendered output sound like Carnatic music
rather than a faithful but sterile MIDI render.

### What this is not

I am not replacing this pipeline with ACE-step. The MIDI pipeline gives me something
ACE-step will never give on its own: a structurally correct, theoretically grounded, editable
score that I can take into Logic Pro, fix, publish, and build on. ACE-step is a rendering and
enhancement layer — a way to turn my structured intent into audio that sounds like real
musicians, without surrendering control of the structure.

The student in Chennai who uses this tool should be able to compose a Bhairavi alaap with
correct gamaka, export it as MIDI, render it through ACE-step conditioned on that MIDI, and
have something that sounds like a bansuri recording — produced entirely on their laptop,
for free, without a cloud subscription, in a way that no commercial tool currently offers.

That is the moat.

---

_Sources: [ACE-Step project page](https://ace-step.github.io/) · [ACE-Step 1.5 on GitHub](https://github.com/ace-step/ACE-Step-1.5) · [ACE-Step 1.5 paper](https://arxiv.org/html/2602.00744v1)_
