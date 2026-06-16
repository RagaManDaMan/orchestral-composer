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
- **K1 / K2** — korvai bridge sections. Configured in the *Korvai (K1)* and
  *Korvai (K2)* panels.

When a form with 2+ distinct labels is used, the app writes a **separate MIDI
file for each section** alongside the main file. Drag them individually into
Logic Pro to loop or rearrange sections freely.

```
outputs/
├── 2026-05-26_Jazz_Quartet_C_major_harmony.mid   ← full arrangement
├── 2026-05-26_Jazz_Quartet_C_major_harmony_A.mid  ← section A only
├── 2026-05-26_Jazz_Quartet_C_major_harmony_K1.mid ← section K1 only
├── 2026-05-26_Jazz_Quartet_C_major_harmony_B.mid  ← section B only
└── …
```

Each section file starts at beat 0, with the same tempo and instrument
assignments as the full file.

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
- Or select a raga palette and check *Use palette as flat progression*
- Harmonic transformations: secondary dominants, tritone subs, back-cycling,
  passing chords, auto-seventh extension

---

### Korvai (K sections)

A **korvai** is a Carnatic rhythmic composition in **A B A B A** form:

- **A** = main phrase, repeated 3×, building in density and velocity
  - Phrase 1: single-note attacks, soft
  - Phrase 2: two-voice shell, medium
  - Phrase 3 (mukthāyi): three voices, full velocity
- **B** = connector phrase, repeated 2×, quieter, rhythmically distinct
- Explicit rests: `,` (1 matra) · `;` (2 matras)

#### Configuring K1 / K2

| Parameter | What it sets |
|-----------|-------------|
| **A phrase** | Solkattu syllables for the main phrase, e.g. `ta ka dhi mi ta ki ta` |
| **Connector phrase** | Solkattu syllables for the B phrase, e.g. `ta ka ta ,` |
| **Gati (nadai)** | Subdivision feel: Chatusram (4), Tisram (3), Khandam (5) |
| **Kalam (speed)** | Syllable duration: Normal (♬ 1/16), Keezh (♪ 1/8), Vilambita (♩ 1/4) |

#### Solkattu syllable reference

Short (1 matra): `ta` `ka` `dhi` `mi` `ki` `ti` `na` `din` `gin` `tin` `tha` `thu` `num` …  
Long (2 matras): `taam` `dheem` `thaam` `daam` …  
Rests: `,` = 1 matra · `;` = 2 matras

The matra count indicator beneath the phrase fields shows total matras and
whether the pattern fits a standard cycle (32 matras = 2 bars of 4/4 at
Normal kalam).

---

### Useful controls

| Control | Where | Effect |
|---------|-------|--------|
| BPM | Top | Global tempo |
| Duration bars | Top | Total piece length (K-only forms tile to fill it) |
| Humanize | Advanced | Adds timing/velocity variation |
| Beats per chord | Advanced | Chord change rate in A/B/C sections |
| Use palette as flat progression | Chord panel | Ignores section chords; uses raga palette across everything (K sections still play) |

---

## Project structure

```
orchestral-composer/
├── app.py                  — Gradio UI and pipeline wiring
├── config.py               — Tuneable parameters
├── requirements.txt
├── setup.sh                — One-command setup
└── src/
    ├── transcribe.py       — Audio → melody MIDI (librosa pyin + key detection)
    ├── prompts.py          — LLM prompt templates, instrument definitions, GM patches
    ├── orchestrate.py      — Ollama integration and JSON parsing
    ├── claude_arranger.py  — Anthropic API reharmonization (optional)
    ├── midi_builder.py     — Multi-track MIDI construction + section slicing
    ├── algo_arranger.py    — Algorithmic harmony and bass generation
    ├── korvai_engine.py    — Solkattu parser, korvai frame builder, MIDI generator
    ├── song_structure.py   — Chord progressions, form parser, timeline builder
    ├── harmony.py          — Voice leading, chord tones, harmonic transformations
    ├── voice_leading.py    — Smooth voice-leading between chord changes
    └── audio_render.py     — Optional: MIDI → WAV via soundfont
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

### Add an instrument preset

In `config.py`:
```python
INSTRUMENT_PRESETS = {
    ...
    "Sitar + Tabla": ["sitar_melody", "tabla_rhythm", "tanpura_drone"],
}
```
Add corresponding GM program numbers to `src/prompts.py → INSTRUMENT_PROGRAMS`.

### Extend the solkattu vocabulary

In `src/korvai_engine.py`, add entries to `_SHORT` or `_LONG`:
```python
_SHORT: list[tuple[str, float]] = [
    ...
    ("jham", 1.0),  # new syllable
]
```

### Add a chord progression

In `src/song_structure.py → COMMON_PROGRESSIONS`:
```python
"My progression": [(1, "maj7"), (4, "7"), (3, "m7"), (6, "7")],
```

---

## Carnatic note on the korvai

The korvai engine implements the classical **mukthāyi korvai** form: three
statements of the main phrase (A) separated by two connector phrases (B),
resolving to *sam* (the first beat) on the third A. Density and velocity
build across the three A statements, so the mukthāyi resolution lands with
full harmonic weight.

Future forms planned: A B A′ B′ A″ (incremental variation),
srotaswaha yati (ascending phrase lengths), gopuccha yati (descending).

---

## Outputs

All generated files land in `outputs/`:

| File | Contents |
|------|----------|
| `…_harmony.mid` | Full arrangement (all instruments) |
| `….mid` | Full arrangement including transcribed melody track |
| `…_A.mid` | Section A only, beat-baselined to 0 |
| `…_K1.mid` | Korvai K1 section only |
| `…_B.mid` | Section B only |
| `…_player.html` | In-browser MIDI player (auto-opened in the UI) |
