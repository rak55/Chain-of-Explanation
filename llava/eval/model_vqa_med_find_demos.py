import argparse
import torch
import os
import json
from tqdm import tqdm
from datasets import load_dataset

from open_clip import create_model_from_pretrained, get_tokenizer


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def format_rationale(rationale: str):
    # If the rationale starts with "The answer is correct because" filter it out
    if rationale.startswith("The answer is correct because"):
        rationale = rationale[len("The answer is correct because") :].strip()
    return rationale


def eval_model(args):
    model_name = args.model
    print(f"Loading model: {model_name}")

    model, preprocess = create_model_from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)
    model.cuda()
    model.eval()

    context_length = 256

    def get_text_embeddings(text_list):
        texts = tokenizer(text_list, context_length=context_length).cuda()
        text_features = model.encode_text(texts)
        return text_features

    def get_image_embeddings(image_list):
        images = torch.stack([preprocess(img) for img in image_list]).cuda()
        image_features = model.encode_image(images)
        return image_features

    print(f"Loading dataset: {args.dataset} ({args.split})")
    dataset = load_dataset(args.dataset)
    data_split = dataset[args.split]
    print(f"Loading demos: {args.dataset} ({args.demo_split})")
    demo_split = dataset[args.demo_split]
    rationales = read_jsonl(args.rationales)
    rationales = {r["id"]: format_rationale(r["text"]) for r in rationales}
    demo_split = demo_split.add_column("id", [idx for idx in range(len(demo_split))])
    demo_split = demo_split.add_column(
        "rationale", [rationales[idx] for idx in range(len(demo_split))]
    )

    with torch.inference_mode():
        # TODO could batch this
        if args.index_mode == "rationale":
            demo_split = demo_split.map(
                lambda x: {
                    "embeddings": get_text_embeddings(x["rationale"])
                    .detach()
                    .cpu()
                    .numpy()[0]
                },
            )
        elif args.index_mode == "image":
            demo_split = demo_split.map(
                lambda x: {
                    "embeddings": get_image_embeddings([x["image"]])
                    .detach()
                    .cpu()
                    .numpy()[0]
                },
            )
        elif args.index_mode == "question":
            demo_split = demo_split.map(
                lambda x: {
                    "embeddings": get_text_embeddings(x["question"])
                    .detach()
                    .cpu()
                    .numpy()[0]
                },
            )
        else:
            raise ValueError(f"Unknown index mode: {args.index_mode}")
        demo_split.add_faiss_index(column="embeddings")

    answers_file = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    seen_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                line = json.loads(line)
                seen_ids.add(line["id"])

    with open(answers_file, "a") as f:
        for idx in tqdm(range(len(data_split))):
            if idx in seen_ids:
                continue
            ex = data_split[idx]
            image = ex["image"]
            question = ex["question"]

            demos = []
            with torch.inference_mode():
                if args.search_mode == "image":
                    question_embedding = (
                        get_image_embeddings([image]).cpu().detach().numpy()[0]
                    )
                elif args.search_mode == "question":
                    question_embedding = (
                        get_text_embeddings([question]).cpu().detach().numpy()[0]
                    )
                else:
                    raise ValueError(f"Unknown search mode: {args.search_mode}")
                scores, samples = demo_split.get_nearest_examples(
                    "embeddings", question_embedding, k=args.top_k
                )

                for score, s_id, s_rationale in zip(
                    scores, samples["id"], samples["rationale"]
                ):
                    demos.append(
                        {
                            "id": s_id,
                            "score": float(score),
                            "rationale": s_rationale,
                        }
                    )

            f.write(
                json.dumps(
                    {
                        "id": idx,
                        "demos": demos,
                    }
                )
                + "\n"
            )


# requires open_clip_torch==2.23.0 and transformers==4.35.2
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="flaviagiammarino/vqa-rad")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--demo_split", type=str, default="train")
    parser.add_argument("--rationales", type=str, required=True)
    parser.add_argument(
        "--index_mode",
        type=str,
        default="rationale",
        choices=["rationale", "image", "question"],
    )
    parser.add_argument(
        "--search_mode", type=str, default="image", choices=["image", "question"]
    )
    args = parser.parse_args()

    eval_model(args)
