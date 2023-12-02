import argparse
from transformers import AutoTokenizer, AutoConfig
import torch
import os
import json
from tqdm import tqdm
from datasets import load_dataset

from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from transformers import CLIPImageProcessor, StoppingCriteria


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024,
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(
            f"`mm_vision_tower` not found in `{config}`, applying patch and save to disk."
        )
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len :], skip_special_tokens=True
            )[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def eval_model(args):
    model_name = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    patch_config(model_name)

    print(model_name)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True
    ).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16
    )
    vision_tower = model.model.vision_tower[0]
    vision_tower.to(device="cuda", dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # import pdb; pdb.set_trace()
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    print(f"Loading dataset: {args.dataset} ({args.split})")
    dataset = load_dataset(args.dataset)
    data_split = dataset[args.split]

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
            answer = ex["answer"]
            idx = line["id"]

            qs = question
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, "mm_use_im_start_end", False):
                qs = (
                    qs
                    + "\n"
                    + DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                    + DEFAULT_IM_END_TOKEN
                )
            else:
                qs = qs + "\n" + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

            qs = (
                qs
                + '\nIf the answer is "'
                + answer
                + '" then explain why in great detail, thinking step-by-step.'
            )

            conv = conv_templates["multimodal"].copy()
            conv.append_message(conv.roles[0], qs)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            keywords = ["###"]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria],
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                (input_ids != output_ids[:, :input_token_len]).sum().item()
            )
            if n_diff_input_output > 0:
                print(
                    f"[Warning] Sample {idx}: {n_diff_input_output} output_ids are not the same as the input_ids"
                )
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]

            try:
                index = outputs.index(conv.sep)
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep)

            outputs = outputs[:index].strip()

            f.write(
                json.dumps(
                    {
                        "id": idx,
                        "text": outputs,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="flaviagiammarino/vqa-rad")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    eval_model(args)
