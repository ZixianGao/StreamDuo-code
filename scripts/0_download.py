"""Download the Proact-VL game commentary data and the LiveCC-7B-Base weights into $STREAMDUO_DATA/hf.

    HF_TOKEN=hf_... python scripts/0_download.py
The two Proact-VL datasets are gated on the Hugging Face Hub: request access first.
"""
import os
import pathlib

from huggingface_hub import snapshot_download

DATA_ROOT = pathlib.Path(os.environ.get("STREAMDUO_DATA", pathlib.Path(__file__).resolve().parents[1] / "data"))
REPOS = [("oaaoaa/game_commentary_sft", "dataset"), ("oaaoaa/LiveGamingBenchmark", "dataset"),
         ("chenjoya/LiveCC-7B-Base", "model")]

for repo, kind in REPOS:
    dst = DATA_ROOT / "hf" / repo.replace("/", "__")
    print("->", dst, flush=True)
    snapshot_download(repo, repo_type=kind, local_dir=dst, token=os.environ.get("HF_TOKEN"), max_workers=8)
