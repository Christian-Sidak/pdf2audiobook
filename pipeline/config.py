"""Central pipeline configuration, loaded from config.yaml at the repo root.

Modules import the derived constants below; the raw tree is available as CFG
for section access (CFG["structural"]["header_zone"], ...).
"""
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as _f:
    CFG: dict = YAML(typ="safe").load(_f)

LIBRARY_DIR = ROOT / CFG["paths"]["library"]
ARTIFACTS_DIR = ROOT / CFG["paths"]["artifacts"]
OUTPUT_DIR = ROOT / CFG["paths"]["output"]

# Pauses are literal silence inserted by the assembler, never requested from
# the TTS. Eval-enforced (pause_policy_bounds, silence_gaps_vs_policy).
PAUSE_POLICY = {k: dict(v) for k, v in CFG["pause_policy"].items()}
SEGMENT_TYPES = tuple(PAUSE_POLICY.keys())
SEGMENT_LENGTH_LIMITS = {k: tuple(v) for k, v in CFG["segment_length_limits"].items()}

OLLAMA_URL = CFG["narration"]["ollama_url"]
REWRITE_MODEL = CFG["narration"]["rewrite_model"]
JUDGE_MODEL = CFG["narration"]["judge_model"]

TTS_ENGINES = {k: dict(v) for k, v in CFG["tts"]["engines"].items()}
DEFAULT_ENGINE = CFG["tts"]["default_engine"]
SAMPLE_RATE = int(CFG["tts"]["sample_rate"])
TTS_BATCH = int(CFG["tts"]["batch"])

LOUDNESS_TARGET_LUFS = float(CFG["mastering"]["lufs_target"])
TRUE_PEAK_MAX_DBTP = float(CFG["mastering"]["true_peak_max_dbtp"])
AAC_BITRATE = str(CFG["mastering"]["aac_bitrate"])


def book_id_for(pdf_path: str | Path) -> str:
    """Stable book id from a PDF filename."""
    import re
    stem = Path(pdf_path).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
