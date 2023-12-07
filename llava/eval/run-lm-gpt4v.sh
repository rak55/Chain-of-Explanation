#!/bin/bash

BASE_DIR=/shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD

python llava/eval/model_vqa_med_predict_demos_gpt4v.py \
    --config configs/gpt-4v-3.yaml \
    --dataset flaviagiammarino/vqa-rad \
    --demos ${BASE_DIR}/image-rationale-demos-gpt4v.jsonl \
    --output ${BASE_DIR}/test-image-rationale-gpt4v-3-predictions-gpt4v.jsonl

python llava/eval/run_eval.py \
    --dataset flaviagiammarino/vqa-rad \
    --pred ${BASE_DIR}/test-image-rationale-gpt4v-3-predictions-gpt4v.jsonl \
    > ${BASE_DIR}/test-image-rationale-gpt4v-3-predictions-gpt4v.results

