#!/bin/bash
# Provision the vast.ai box (idempotent; provisioning only — no job launch).
set -euo pipefail
ROOT="${PDF2AB_ROOT:-/workspace/pdf2audiobook}"
cd "$ROOT"

apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq sox ffmpeg libsndfile1 >/dev/null 2>&1

if [ ! -d /workspace/venv ]; then
  python3 -m venv --system-site-packages /workspace/venv
fi
source /workspace/venv/bin/activate
python3 -m pip install -q --upgrade pip
pip install -q -r requirements-cuda.txt
python3 -c "import qwen_tts" 2>/dev/null || \
  pip install -q "git+https://github.com/QwenLM/Qwen3-TTS.git"
if [ "${VAST_FLASH_ATTN:-0}" = "1" ]; then
  pip install -q flash-attn --no-build-isolation || echo "flash-attn build failed (optional)"
fi

# Fail fast before any render burns GPU-hours.
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA torch'"
sox --version >/dev/null
python3 -c "from qwen_tts import Qwen3TTSModel"
python3 -c "from faster_whisper import WhisperModel"

# Prefetch weights so renders never stall on downloads.
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
from faster_whisper import WhisperModel
WhisperModel("small.en", device="cpu", compute_type="int8")  # triggers download
print("weights prefetched")
EOF

touch /workspace/BOOTSTRAP_OK
echo "BOOTSTRAP OK"
