#!/bin/bash
# Runs ON the box under nohup: full eval-gated stage-5 render.
set -euo pipefail
source /workspace/venv/bin/activate
cd "${PDF2AB_ROOT:-/workspace/pdf2audiobook}"
book=${1:?book id}
voice=${2:?voice name}
stages=${3:-5}

python3 main.py build "library/$book.pdf" --stages "$stages" \
    --engine qwen3tts --voice "voices/$voice.wav" --keep-going

touch "artifacts/$book/RENDER_DONE"
echo "RENDER JOB COMPLETE"
