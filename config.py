"""
Central configuration — change these to adapt the tool to your setup.
Students: start here if you want to swap models or tune behaviour.
"""

# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------

# Ollama model for orchestration reasoning.
# Pull first: ollama pull llama3.2:3b
# Good alternatives: "mistral:7b", "qwen2.5:7b", "gemma3:12b", "llama3.1:8b"
OLLAMA_MODEL    = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Audio sample rate for transcription (22050 is standard for pitch analysis)
SAMPLE_RATE = 22050

# Quantisation grid in beats (0.5 = 8th note, 0.25 = 16th note)
GRID_SIZE_BEATS = 0.5

# Notes shorter than this are discarded as noise
MIN_NOTE_DURATION_SEC = 0.08

# Default tempo when not specified
DEFAULT_TEMPO_BPM = 90

# Generation defaults (user-adjustable in the UI)
DEFAULT_NOTES_PER_BEAT = 1.0
DEFAULT_MAX_TOKENS     = 16384

# Indian audio: neural instrument synthesis (train_instrument.py).
# Training is CPU/GPU-heavy (~30+ min per instrument). Leave False to use
# real sliced samples + Rubber Band pitch shift — much lighter and usually
# sounds better unless you have a fully trained model.
USE_NEURAL_INSTRUMENTS = False

# ---------------------------------------------------------------------------
# FluidSynth path — set for your machine, leave None for auto-detect
# ---------------------------------------------------------------------------
# Windows example (Aditi's machine):
#   FLUIDSYNTH_PATH = r"C:\Users\Aditi Sriram\Downloads\fluidsynth-v2.5.4-win10-x64-cpp11\bin\fluidsynth.exe"
# Mac (after: brew install fluid-synth):
#   FLUIDSYNTH_PATH = "/opt/homebrew/bin/fluidsynth"
# Leave as None to let the app find it automatically.
FLUIDSYNTH_PATH: str | None = None

# Path to a .sf2 soundfont file. None = use pretty_midi's bundled soundfont.
# SGM-v2.01 is recommended for Indian instruments — download separately.
SOUNDFONT_PATH: str | None = None
SOUNDFONT_PREFERENCE = ["SGM-v2.01", "GeneralUser", "FluidR3"]

# ---------------------------------------------------------------------------
# Instrument presets
# ---------------------------------------------------------------------------

INSTRUMENT_PRESETS = {

    # ── Western Orchestral ─────────────────────────────────────────────────
    "String Trio": [
        "strings_melody", "strings_harmony", "bass",
    ],
    "Chamber (Strings + Flute)": [
        "flute_melody", "strings_harmony", "bass",
    ],
    "Brass + Strings": [
        "trumpet_melody", "strings_harmony", "bass",
    ],
    "Baroque Ensemble": [
        "oboe_melody", "strings_harmony", "cello",
    ],
    "Piano Trio": [
        "piano_melody", "strings_harmony", "cello",
    ],

    # ── Jazz ──────────────────────────────────────────────────────────────
    "Jazz Quartet": [
        "alto_sax", "piano_harmony", "acoustic_bass",
    ],
    "Jazz Big Band": [
        "trumpet_melody", "tenor_sax", "trombone",
        "piano_harmony", "acoustic_bass",
    ],
    "Jazz Trio (Brush)": [
        "piano_melody", "acoustic_bass",
    ],
    "Bebop Quintet": [
        "alto_sax", "trumpet_melody",
        "piano_harmony", "acoustic_bass",
    ],

    # ── Blues ─────────────────────────────────────────────────────────────
    "Blues Band": [
        "harmonica", "electric_guitar", "piano_harmony", "acoustic_bass",
    ],
    "Blues Trio": [
        "electric_guitar", "piano_harmony", "acoustic_bass",
    ],

    # ── Funk / Soul ────────────────────────────────────────────────────────
    "Funk Band": [
        "tenor_sax", "electric_guitar", "electric_piano", "electric_bass",
    ],

    # ── Pop ───────────────────────────────────────────────────────────────
    "Pop Band": [
        "synth_lead", "electric_guitar", "synth_pad", "electric_bass",
    ],
    "Retro Pop (80s)": [
        "dx7_epiano", "synth_pad", "electric_bass",
    ],
    "City Pop": [
        "dx7_epiano", "electric_guitar", "synth_pad", "slap_bass",
    ],

    # ── Hip-Hop ────────────────────────────────────────────────────────────
    "Hip-Hop Beat": [
        "synth_lead", "synth_pad", "electric_bass",
    ],

    # ── Rock ──────────────────────────────────────────────────────────────
    "Rock Band": [
        "electric_guitar", "overdriven_guitar", "electric_bass",
    ],

    # ── Bossa Nova ────────────────────────────────────────────────────────
    "Bossa Nova": [
        "flute_melody", "nylon_guitar", "acoustic_bass",
    ],

    # ── Indian Classical ──────────────────────────────────────────────────
    "Carnatic Ensemble": [
        "veena",
        "carnatic_violin",
        "bansuri",
        "tambura",
    ],
    "Hindustani Classical": [
        "sitar",
        "sarangi",
        "tambura",
    ],
    "Sitar & Strings": [
        "sitar", "strings_harmony", "acoustic_bass",
    ],
    "Sufi / Qawwali": [
        "harmonium",
        "sarangi",
        "acoustic_bass",
    ],

    # ── Indian Folk / Fusion ──────────────────────────────────────────────
    "Bollywood Golden Era": [
        "strings_melody",
        "strings_harmony",
        "bansuri",
        "acoustic_bass",
    ],
    "Bollywood Modern": [
        "synth_lead",
        "strings_harmony",
        "mandolin",
        "electric_bass",
    ],
    "Koothu / Folk": [
        "nadaswaram",
        "strings_harmony",
        "acoustic_bass",
    ],
    "Dholak Party": [
        "strings_melody",
        "harmonium",
        "acoustic_bass",
    ],
}

# ---------------------------------------------------------------------------
# Style hints injected into LLM prompts
# ---------------------------------------------------------------------------

PRESET_STYLE_HINTS: dict[str, str] = {

    # Western orchestral
    "String Trio":               "Western classical; smooth voice leading; avoid parallel fifths",
    "Chamber (Strings + Flute)": "Classical chamber; lyrical flute melody over supporting strings",
    "Brass + Strings":           "Romantic orchestral; bold brass melody with rich string harmony",
    "Baroque Ensemble":          "Baroque counterpoint; ornamental style; figured-bass harmony",
    "Piano Trio":                "Classical piano trio; balanced voices; idiomatic piano writing",

    # Jazz
    "Jazz Quartet":     "Swing feel; jazz harmony with 7ths and extensions; ii-V-I progressions; walking bass",
    "Jazz Big Band":    "Big band swing; brass punches on 2 and 4; piano shell voicings; four-bar melodic phrases",
    "Jazz Trio (Brush)":"Intimate brush trio; sparse piano voicings; melodic bass lines; lots of space",
    "Bebop Quintet":    "Bebop; fast chromatic lines; dense substitutions; call and response between horns",

    # Blues
    "Blues Band":  "12-bar blues; harmonica bends; shuffle rhythm guitar; piano comping behind; walking bass",
    "Blues Trio":  "Delta blues; sparse and raw; guitar bends and vibrato; slow shuffle; space between notes",

    # Funk
    "Funk Band":   "Tight funk; syncopated horn punches on off-beats; chicken-scratch guitar; slap bass; 16th note feel",

    # Pop
    "Pop Band":        "Modern pop; clean punchy synth lead; four-on-the-floor; polished and bright",
    "Retro Pop (80s)": "80s pop production; DX7 FM electric piano; lush Juno-60 pads; crisp LinnDrum feel",
    "City Pop":        "Japanese City Pop 80s; DX7 keys; funky Strat guitar; sophisticated chord extensions; slap bass",

    # Hip-Hop
    "Hip-Hop Beat": "Boom bap; 808 sub kick; dusty sampled melody; sparse atmospheric pad; slow groove",

    # Rock
    "Rock Band": "Classic rock; power chord rhythm guitar; lead guitar melody with sustain; driving bass",

    # Bossa Nova
    "Bossa Nova": "Brazilian bossa nova; gentle nylon guitar syncopation; flute sings the melody; walking bass; laid-back",

    # Indian classical
    "Carnatic Ensemble":    "South Indian Carnatic classical; strict raga grammar; gamaka ornaments; veena leads; mridangam tala; no Western harmony",
    "Hindustani Classical": "North Indian classical; sitar with meend and gamaka; sarangi responses; slow alap development; tanpura drone; no chord changes",
    "Sitar & Strings":      "Raga-flavoured fusion; sitar ornamental melody; string harmony wash; drone bass",
    "Sufi / Qawwali":       "Sufi devotional qawwali; harmonium drone and melody; sarangi melisma; building intensity; spiritual ecstasy",

    # Indian folk / fusion
    "Bollywood Golden Era": "1970s Bollywood orchestral; RD Burman style; sweeping string melody; bansuri colour; congas and dholak rhythm",
    "Bollywood Modern":     "Contemporary Bollywood; Anirudh/Pritam production; electronic lead with acoustic textures; hook-driven; energetic",
    "Koothu / Folk":        "North Chennai street folk; nadaswaram piercing lead; raw aggressive energy; parai and thavil percussion; festive",
    "Dholak Party":         "North Indian festive; dholak bhangra groove; harmonium sustained chords; celebratory and bright",
}