#!/bin/bash

# LLaVA-MED KM GPT-4V QAM
BASE_DIR=/shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD

declare -a search=("image" "question")
# declare -a index=("rationale" "image" "question")
declare -a index=("image" "question")

for s in "${search[@]}"
do
    for i in "${index[@]}"
    do
        echo "Search mode: $s, Index mode: $i"
        python llava/eval/model_vqa_med_predict_demos_gpt4v.py \
            --config configs/gpt-4v-3.yaml \
            --dataset flaviagiammarino/vqa-rad \
            --demos ${BASE_DIR}/$s-$i-demos-multi.jsonl \
            --output ${BASE_DIR}/test-$s-$i-3-predictions-gpt4v.jsonl

        python llava/eval/run_eval.py \
            --dataset flaviagiammarino/vqa-rad \
            --pred ${BASE_DIR}/test-$s-$i-3-predictions-gpt4v.jsonl \
            > ${BASE_DIR}/test-$s-$i-3-gpt4v.results

        cat ${BASE_DIR}/test-$s-$i-3-gpt4v.results
    done
done
