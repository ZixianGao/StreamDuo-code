#!/bin/bash
# Data: splits, wiki snapshot + entity ontology, clean test protocol, K/V/S labels, game dossiers, agent index.
# Needs the Proact-VL data under $STREAMDUO_DATA/hf (see README) and a running scripts/serve_llm.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
$PY_TRAIN -m streamduo.data.video_ids                         # video_fingerprints.json (duplicated / mislabeled videos)
$PY_TRAIN -m streamduo.data.build_splits                       # manifests/<fold>/{train_*,test}.jsonl, train_recipe.json
$PY_TRAIN -m streamduo.wiki.crawl --all                         # wiki/<game>/pages.jsonl (MediaWiki API)
$PY_TRAIN -m streamduo.wiki.ontology                            # wiki/<game>/ontology.jsonl
$PY_TRAIN -m streamduo.data.clean --fold $FOLD                  # manifests/<fold>/test_clean.jsonl (LLM)
$PY_TRAIN -m streamduo.data.kvs --fold all                      # K-strict labels (LLM)
for G in baldurs_gate_3 black_myth_wukong csgo cyberpunk_2077 elden_ring lol minecraft starcraft2 \
         streetfighter6 tears_of_the_kingdom yu_gi_oh; do       # game dossiers written by the research agent (LLM)
  $PY_TRAIN -m streamduo.research.agent --game $G --max-queries 48 --max-tokens 2048 --max-entities 60 \
    --sweep --no-web --run-name $DOSSIER
done
$PY_TRAIN -m streamduo.interact.build_test_lookups candidates   # interact/candidates.json (agent search index)
