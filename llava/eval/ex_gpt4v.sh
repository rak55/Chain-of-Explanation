#!/bin/bash

# GPT-4V KM GPT-4V QAM
BASE_DIR=/shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD

declare -a search=("image" "question")
declare -a index=("rationale" "image" "question")

for s in "${search[@]}"
do
    for i in "${index[@]}"
    do
        echo "Search mode: $s, Index mode: $i"
        # skip $s=image, $i=rationale because already done
        if [ "$s" == "image" ] && [ "$i" == "rationale" ]; then
            echo "Skipping"
            continue
        fi
        python llava/eval/model_vqa_med_predict_demos_gpt4v.py \
            --config configs/gpt-4v-3.yaml \
            --dataset flaviagiammarino/vqa-rad \
            --demos ${BASE_DIR}/$s-$i-demos-gpt4v.jsonl \
            --output ${BASE_DIR}/test-$s-$i-gpt4v-3-predictions-gpt4v.jsonl

        python llava/eval/run_eval.py \
            --dataset flaviagiammarino/vqa-rad \
            --pred ${BASE_DIR}/test-$s-$i-gpt4v-3-predictions-gpt4v.jsonl \
            > ${BASE_DIR}/test-$s-$i-gpt4v-3-gpt4v.results

        cat ${BASE_DIR}/test-$s-$i-gpt4v-3-gpt4v.results
    done
done
