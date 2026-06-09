# Orchestral Composer

Record a melody (or let the AI generate one), define a song form, add a korvai
bridge — and get a multi-track MIDI arrangement ready to drag into Logic Pro or
any DAW.

**Runs entirely on-device.** No cloud API required for core features (Ollama
drives the LLM melody generation; Anthropic API is optional for reharmonization
only).

---

## What it does

```
Voice / instrument / text idea
          │
          ▼
    Three pipeline modes
    ┌─────────────────────────────────────────────────────────┐
    │  Upload / Record   → pitch-detect melody → LLM arrange  │
    │  Harmony only      → instant chord arrangement, no LLM  │
    │  AI Compose        → LLM writes melody + arranges it    │
    └─────────────────────────────────────────────────────────┘
          │
          ▼
    Song structure   A · K1 · B · K2 · C
    (A/B/C sections + korvai bridges)
          │
          ▼
    Harmony engine   jazz comping · arpeggio · korvai stabs
    + bass line      walking bass · korvai root/fifth
          │
          ▼
    Multi-track MIDI  ── one file per instrument part
    Section MIDIs     ── one file per song section (A.mid, K1.mid …)
          │
          ▼
    Logic Pro / GarageBand / any DAW
```

---

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU / SoC | Apple Silicon M1 | M3 or later |
| RAM | 16 GB | 24 GB+ |
| Storage | 10 GB free (model + app) | — |

Intel Macs and Linux (NVIDIA GPU) also work — Ollama handles both.

---

## Installation

### 1 · Install Ollama

Download from **https://ollama.com/download** and install it.

### 2 · Run the setup script

```bash
cd orchestral-composer
bash setup.sh
```

This installs Python dependencies (`pip install -r requirements.txt`) and
pulls `llama3.1:8b` (~4.7 GB, downloaded once and cached).

---

## Running the app

**Start Ollama** (once per session):
```bash
ollama serve
```

**Launch the app:**
```bash
python app.py
```

Open **http://127.0.0.1:7861** in your browser.

---

## Usage guide

### Pipeline modes

| Mode | When to use |
|------|-------------|
| **Upload / Record** | Sing or play a melody → LLM arranges around it |
| **Harmony only** | Instant chord arrangement with no LLM (fast iteration) |
| **AI Compose** | LLM generates both melody and arrangement |

For fast harmonic experimentation — including all korvai work — use
**Harmony only**. No Ollama required.

---

### Song structure (A / B / C / K sections)

Set the **Song form** field to a space- or letter-separated sequence of section
labels:

```
A B A        — standard ABA (verse–chorus–verse)
A K1 B K2 A  — sections with korvai bridges between them
K1           — korvai only (test / loop a single pattern)
```

- **A / B / C** — regular sections. Set chord progression and number of bars
  for each in the *Song Structure* panel.
- **K1 / K2** — korvai bridge sections. Configured in the *Korvai* panel.

When a form with 2+ distinct labels is used, the app writes a **separate MIDI
file for each section** alongside the main file.

```
outputs/
├── 2026-06-09_Jazz_Quartet_C_major_harmony.mid   ← full arrangement
├── 2026-06-09_Jazz_Quartet_C_major_harmony_A.mid  ← section A only
├── 2026-06-09_Jazz_Quartet_C_major_harmony_K1.mid ← section K1 only
└── …
```

---

### Harmony styles

| Style | Character |
|-------|-----------|
| **Jazz comping** | Syncopated shell voicings, rhythmically varied |
| **Arpeggio** | Broken chords, configurable rhythm density |
| **Block chords** | Straight homophonic stabs on every beat |
| **Korvai pattern** | Stabs placed at solkattu syllable attack points (K sections only) |

---

### Chord progressions

- Pick from 30+ named progressions (ii–V–I, 12-bar blues, Andalusian, Bhairav…)
- Or type chords directly: `Cmaj7 Am7 Dm7 G7`
- Roman numeral input is supported: `I-IV-V`, `i-iv-V7-i`, `I-♭VII-IV-I`
- Harmonic transformations available: secondary dominants, tritone subs,
  back-cycling, passing chords, auto-seventh extension

---

### Korvai (K sections)

A **korvai** is a Carnatic rhythmic composition in **A B A B A** form:

- **A** = main phrase, repeated 3×, building in density and velocity
  - Phrase 1: single-note attacks, soft
  - Phrase 2: two-voice shell, medium
  - Phrase 3 (mukthāyi): three voices, full velocity
- **B** = connector phrase, repeated 2×, quieter, rhythmically distinct
- Explicit rests: `,` (1 matra) · `;` (2 matras)

#### Configuring korvai

| Parameter | What it sets |
|-----------|-------------|
| **A phrase** | Solkattu syllables, e.g. `ta ka dhi mi ta ki ta` |
| **Connector** | Solkattu syllables for the B phrase, e.g. `ta ka ta ,` |
| **Gati (nadai)** | Subdivision: Chatusram (4), Tisram (3), Khandam (5) |
| **Kalam (speed)** | Syllable duration: Normal (♬ 1/16), Keezh (♪ 1/8), Vilambita (♩ 1/4) |
| **Chord (opt)** | Override the harmony for K sections; leave blank to inherit |

Numeric shorthand: enter `5` → expands to `ta ka dhi mi ta`; enter `5, 4, 3`
→ gopuccha yati with rests between groups.

#### Solkattu syllable reference

Short (1 matra): `ta` `ka` `dhi` `mi` `ki` `ti` `na` `din` `gin` `tin` …  
Long (2 matras): `taam` `dheem` `thaam` `daam` …  
Rests: `,` = 1 matra · `;` = 2 matras

---

## Live accompaniment engine

The **Live Accompaniment** panel streams a real-time chord loop — useful for
improvising, practising, demoing a progression, or feeding notes into a DAW
while you play.

### Transport controls

| Control | Effect |
|---------|--------|
| **BPM** | 20–240 BPM. Goes down to 20 for Indian classical vilambit laya |
| **Time sig** | 4/4 · 3/4 · 6/8 · 5/4 · 7/8 |
| **Style** | Chord instrument timbre (see table below) |
| **Bars/chord** | How long each chord lasts — 0.5–8 bars in 0.5 steps |
| **Feel** | 0 = metronomic · 100 = loose human timing |
| **Passing chords** | 0–50% probability of inserting a ii/V or chromatic approach chord between changes |

### Chord instrument styles

| Style | Timbre |
|-------|--------|
| Ballad | Strings (slow-attack ensemble) |
| Jazz | Guitar (warm triangle harmonics, lowpass) |
| Blues | Piano (triangle+sine, matched to palette hover preview) |
| Bossa Nova | Guitar — nylon (brighter, longer decay with plate reverb) |
| Honky Tonk | Detuned dual-voice piano (~15 cents apart) |
| Waltz | Accordion (beating sawtooth reeds) |
| Pad | Synth pad (very slow attack, detuned sines) |
| Arpeggio Up / Down | Piano (voiced across octaves, same chain as chord palette) |

Guitar styles (Jazz and Bossa Nova) include **plate reverb** and **strum humanizing**
(per-harmonic ±0.2% tuning detune, 0–6ms strum jitter). Decay: Bossa 1.8s, Jazz 2.4s.

### Mixer

Each track has an independent volume slider and a **density** selector:

| Density | Effect |
|---------|--------|
| Sparse | Core skeleton only — kick+snare+quarter hi-hats, bass on downbeats |
| Moderate | Fills in 8th-note subdivisions + syncopated extras |
| Dense | All subdivisions covered — no gaps |
| Off | Track silent |

The **Melody** track (see Melodic Response below) also has its own fader here.

### Range selectors

A **🎹 Range** panel below the mixer sets the MIDI note boundaries for bass
and chord voices independently:

| Track | Default | Notes |
|-------|---------|-------|
| Bass | C2 – G3 | Root note always octave-snapped within range |
| Chord | C3 – G5 | Voicing anchors at Lo, stacks upward clamped to Hi |

Changes take effect immediately while playing.

### Drum library — 12 patterns

| Pattern | Category | Character |
|---------|----------|-----------|
| Ballad | Soft | Brushed feel, 8th hi-hats |
| Rock Basic | Rock | Straight 4/4, density-tiered |
| Rock Groove | Rock | Syncopated 16th kicks, open hi-hat |
| Rock Heavy | Rock | Double kick runs, toms at bar end |
| Half-Time | Half-Time | Snare on beat 3 only, quarter hi-hats (sparse) → 8ths (moderate) |
| Half-Time Heavy | Half-Time | Fat snare + crash + pickup kicks |
| Funk Light | Funk | 16th hi-hats, sparse ghost snares |
| Funk Heavy | Funk | Dense 16ths, claps, heavy ghost layer |
| Jazz Swing | Jazz | Triplet ride, hi-hat foot on 2+4 |
| Bossa Nova | Latin | Rim click clave pattern |
| Double Time | Groove | Quarter hi-hats (sparse) → 8ths (moderate) → full 16ths (dense) |
| Reggae | Groove | Rockers kick pattern, claps |

Density tiers are pattern-aware: at **Sparse** the groove breathes; at **Dense**
every available subdivision is filled.

### Recording

Press **⏺ Record** before playing, then **■ Stop**. A Type-1 MIDI file
(`accompaniment_<timestamp>.mid`) downloads automatically with separate tracks
for Chords, Bass, and Drums.

### Streaming to a DAW (Logic Pro / Ableton)

The **MIDI Out → DAW** dropdown streams notes in real time to a virtual MIDI
port as the loop plays.

**macOS setup (IAC driver):**
1. Open **Audio MIDI Setup** → Window → Show MIDI Studio
2. Double-click **IAC Driver** → check **Device is online** → add a bus if none exists
3. In the browser, click **↺** next to MIDI Out — "IAC Bus 1" should appear
4. Select it; the **● LIVE** badge confirms

**MIDI channel assignments:**

| Ch | Part |
|----|------|
| 1 | Chords / strings / guitar / piano |
| 2 | Bass |
| 10 | Drums (GM — kick 36, snare 38, hi-hat 42, ride 51 …) |

**BPM sync:** MIDI Clock (24 ppq via setInterval) is streamed alongside notes
so Logic can follow the BPM slider in real time. Enable in Logic's synchronisation
settings by setting sync source to the IAC Bus.

### Markov chord transitions

The accompaniment picks the next chord via a Markov model weighted by functional
harmony (tonic / subdominant / dominant). The pool is the checked checkboxes in
the chord palette — uncheck specific chords to exclude them.

Accidental-prefix degrees (♭VII, ♭VI, ♭II, etc.) are treated as neutral weight,
so modal and Carnatic palettes don't get an unwanted Western V→I pull.

### Listen In

| Mode | Effect |
|------|--------|
| **MIDI** | Detects chords from a connected MIDI device and drives the chord changes live |
| **Mic** | Autocorrelation pitch detection — sing into the mic to drive chord changes and tempo detection |
| Respond: Chord | Best-match chord from palette changes accompaniment immediately |
| Respond: Beat | Detected tempo updates the BPM slider |
| Respond: Both | Both simultaneously |

---

## Melodic Response (call and response)

The **🎵 Melodic Response** panel captures a melodic phrase you sing or play,
then generates a response phrase through the same scale as the active palette mode.

### Workflow

1. Generate the palette (sets the key and mode — the scale source)
2. Enable **Listen In** (MIDI or Mic)
3. Click **📡 Arm** — button turns green "Listening…"
4. Play or sing a phrase
5. Stop playing — 1.2s silence triggers phrase complete automatically
6. Click **▶ Play** to hear back what was captured (quantized to 8th-note grid)
7. Or click **✦ Generate** to hear an algorithmic response

Tick **Auto after phrase** to skip step 6/7 — response fires immediately when
the phrase ends.

### Response styles

| Style | What it does |
|-------|-------------|
| **Answer** | Starts at the end note of your phrase, inverts the step direction, converges to the tonic |
| **Imitate** | Transposes your interval sequence up 2–4 scale degrees, preserving rhythm |
| **Free** | Ascending arch then descending back to tonic — independent of input, good as filler |

All three use only the pitch classes from the active palette — raga-safe.

### Phrase capture notes

- **MIDI** — captures full note-on/off events with timestamps; note count is exact
- **Mic** — 3-frame median smoothing on raw pitch reduces vibrato triggering false note boundaries
- All captured phrases are quantized to the 8th-note grid before playback or generation
- **▶ Play** replays the literal captured phrase (quantized); **✦ Generate** creates a new response
- **✕ Clear** resets the capture buffer

The melody voice plays through a triangle+LFO vibrato synth (flute-like) with
its own volume fader in the Mixer.

---

## Project structure

```
orchestral-composer/
├── app.py                  — Gradio UI, live accompaniment engine (JS), pipeline wiring
├── config.py               — Tuneable parameters (model, BPM, grid, presets)
├── requirements.txt
├── setup.sh                — One-command setup
└── src/
    ├── transcribe.py       — Audio → melody MIDI (librosa pyin / CREPE + key detection)
    ├── prompts.py          — LLM prompt templates, instrument definitions, GM patches
    ├── orchestrate.py      — Ollama integration and JSON parsing
    ├── claude_arranger.py  — Anthropic API reharmonization (optional)
    ├── midi_builder.py     — Multi-track MIDI construction + section slicing
    ├── algo_arranger.py    — Algorithmic harmony and bass generation
    ├── korvai_engine.py    — Solkattu parser, korvai frame builder, MIDI generator
    ├── song_structure.py   — Chord progressions, form parser, timeline builder
    ├── harmony.py          — Scale library, chord palette, voice-leading helpers
    └── voice_leading.py    — Functional harmony analysis, smooth voice movement
```

---

## Customisation

### Swap the LLM

Edit `config.py`:
```python
OLLAMA_MODEL = "qwen2.5:7b"   # smaller / faster
OLLAMA_MODEL = "gemma3:12b"   # larger / more musical
```
Then pull it: `ollama pull qwen2.5:7b`

### Add a chord progression

In `src/song_structure.py → COMMON_PROGRESSIONS`:
```python
"My progression": [(1, "maj7"), (4, "7"), (3, "m7"), (6, "7")],
```

### Extend the solkattu vocabulary

In `src/korvai_engine.py`, add entries to `_SHORT` or `_LONG`:
```python
_SHORT: list[tuple[str, float]] = [
    ...
    ("jham", 1.0),
]
```

### Add an instrument preset

In `config.py → INSTRUMENT_PRESETS`:
```python
"Sitar + Tabla": ["sitar_melody", "tabla_rhythm", "tanpura_drone"],
```
Add GM program numbers to `src/prompts.py → INSTRUMENT_PROGRAMS`.

---

## Outputs

All generated files land in `outputs/`:

| File | Contents |
|------|----------|
| `…_harmony.mid` | Full arrangement (all instruments) |
| `….mid` | Full arrangement including transcribed melody track |
| `…_A.mid` | Section A only, beat-baselined to 0 |
| `…_K1.mid` | Korvai K1 section only |
| `…_player.html` | In-browser MIDI player (auto-opened in the UI) |
| `accompaniment_<ts>.mid` | Recorded live accompaniment session |

---

## Carnatic note on the korvai

The korvai engine implements the classical **mukthāyi korvai** form: three
statements of the main phrase (A) separated by two connector phrases (B),
resolving to *sam* (the first beat) on the third A. Density and velocity
build across the three A statements, so the mukthāyi resolution lands with
full harmonic weight.

Planned: srotaswaha yati (ascending phrase lengths), per-section harmony style
overrides (A=block, B=jazz, K=korvai engine), eduppu offset control.
