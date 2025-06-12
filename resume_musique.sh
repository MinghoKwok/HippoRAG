#!/bin/bash

mkdir -p logs

python main.py \
  --dataset musique \
  --llm_base_url https://api.openai.com/v1 \
  --llm_name gpt-4o-mini \
  --embedding_name nvidia/NV-Embed-v2 \
  --openie_result_path outputs/musique/openie_results_ner_gpt-4o-mini.json \
  > logs/musique_resume.log 2>&1
