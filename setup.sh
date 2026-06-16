#!/bin/bash
# Setup script for Orchestral Composer
# Run once: bash setup.sh

set -e

echo "=== Orchestral Composer — Setup ==="
echo ""

# 1. Python dependencies
echo "[1/2] Installing Python dependencies..."
pip install -r requirements.txt
echo "      Done."
echo ""

# 2. Ollama model
echo "[2/2] Pulling Ollama model (llama3.1:8b, ~4.7 GB)..."
echo "      This downloads once and is cached locally."
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "ERROR: Ollama is not installed."
    echo "Install it from https://ollama.com/download (free, open source)."
    echo "Then re-run this script."
    exit 1
fi
ollama pull llama3.1:8b
echo "      Done."
echo ""

echo "=== Setup complete ==="
echo ""
echo "Start Ollama (once per session, if not already running):"
echo "    ollama serve"
echo ""
echo "Launch the app:"
echo "    python app.py"
echo ""
echo "Then open http://127.0.0.1:7861 in your browser."
