"""Filesystem layout. Set the data root with $STREAMDUO_DATA (default: ./data next to the repo)."""
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = pathlib.Path(os.environ.get("STREAMDUO_DATA", REPO_ROOT / "data"))

# Proact-VL game commentary data (HF: oaaoaa/game_commentary_sft, oaaoaa/LiveGamingBenchmark)
SFT_ROOT = DATA_ROOT / "hf" / "oaaoaa__game_commentary_sft" / "DATA"
BENCH_ROOT = DATA_ROOT / "hf" / "oaaoaa__LiveGamingBenchmark" / "DATA"
SFT_ANNS = SFT_ROOT / "anns"
BENCH_ANNS = BENCH_ROOT / "anns"

MANIFEST_ROOT = DATA_ROOT / "manifests"
CACHE_ROOT = DATA_ROOT / "cache"
