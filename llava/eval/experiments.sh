pip install tqdm datasets openai tenacity ujson nltk tabulate
pip install open-clip-torch faiss-gpu transformers


python llava/eval/model_vqa_med_predict_gpt4v.py \
    --config configs/gpt-4v-pred.yaml \
    --dataset flaviagiammarino/vqa-rad \
    --output /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/train-predictions-gpt4v.jsonl

python llava/eval/model_vqa_med_find_demos.py \
    --dataset flaviagiammarino/vqa-rad \
    --rationales /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/train-predictions-gpt4v.jsonl \
    --search_mode image \
    --index_mode rationale \
    --output /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/image-rationale-demos-gpt4v.jsonl

python llava/eval/model_vqa_med_find_demos.py \
    --dataset flaviagiammarino/vqa-rad \
    --rationales /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/train-predictions-gpt4v.jsonl \
    --search_mode question \
    --index_mode rationale \
    --output /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/question-rationale-demos-gpt4v.jsonl

python llava/eval/model_vqa_med_find_demos.py \
    --dataset flaviagiammarino/vqa-rad \
    --rationales /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/train-predictions-gpt4v.jsonl \
    --search_mode image \
    --index_mode image \
    --output /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/image-image-demos-gpt4v.jsonl

python llava/eval/model_vqa_med_find_demos.py \
    --dataset flaviagiammarino/vqa-rad \
    --rationales /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/train-predictions-gpt4v.jsonl \
    --search_mode question \
    --index_mode image \
    --output /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/question-image-demos-gpt4v.jsonl

python llava/eval/model_vqa_med_find_demos.py \
    --dataset flaviagiammarino/vqa-rad \
    --rationales /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/train-predictions-gpt4v.jsonl \
    --search_mode image \
    --index_mode question \
    --output /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/image-question-demos-gpt4v.jsonl

python llava/eval/model_vqa_med_find_demos.py \
    --dataset flaviagiammarino/vqa-rad \
    --rationales /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/train-predictions-gpt4v.jsonl \
    --search_mode question \
    --index_mode question \
    --output /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/question-question-demos-gpt4v.jsonl


python llava/eval/model_vqa_med_predict_demos_gpt4v.py \
    --config configs/gpt-4v-3.yaml \
    --dataset flaviagiammarino/vqa-rad \
    --demos /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/image-rationale-demos-gpt4v.jsonl \
    --output /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/test-image-rationale-gpt4v-3-predictions-gpt4v.jsonl

python llava/eval/run_eval.py \
    --dataset flaviagiammarino/vqa-rad \
    --pred /shared/aifiles/disk1/media/artifacts/LLaVA-Med/VQA-RAD/test-image-rationale-gpt4v-3-predictions-gpt4v.jsonl