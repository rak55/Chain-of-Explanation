import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

from llava.conversation import conv_templates
from llava.utils import disable_torch_init

#extra in cot
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoConfig
from transformers import CLIPImageProcessor, CLIPVisionModel
from transformers import StoppingCriteria, BitsAndBytesConfig
from llava import LlavaLlamaForCausalLM

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

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

#added old func.
def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex
                

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

def load_pretrained_model(
    model_name,
    load_8bit=False,
    load_4bit=False,
    load_bf16=False,
    device_map="auto",
    device="cuda",
):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if load_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if load_bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, **kwargs
    )                                                                            #took out low_mem_usga=False.

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    vision_tower = model.model.vision_tower[0]
    if vision_tower.device.type == "meta":
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path,
            torch_dtype=torch.bfloat16 if load_bf16 else torch.float16,
            low_cpu_mem_usage=False,
        ).cuda()
        model.model.vision_tower[0] = vision_tower
    else:
        vision_tower.to(
            device="cuda", dtype=torch.bfloat16 if load_bf16 else torch.float16
        )

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
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower,
        torch_dtype=torch.bfloat16 if load_bf16 else torch.float16,
    )
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    return model, tokenizer, image_processor, image_token_len

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, use_cache=True
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
    return model, tokenizer, image_processor, image_token_len

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model)
    patch_config(model_name)

    print(model_name)

    #model, tokenizer, image_processor, image_token_len = load_pretrained_model(model_name, args.load_8bit, args.load_4bit, args.load_bf16)
    model, tokenizer, image_processor, image_token_len = load_model(model_name)                   #see which to add load_model or load_pretrained model.

    print(f"Loading dataset: {args.dataset}")
    if args.dataset[0] == "/":
        dataset = list(read_jsonl(args.dataset))                                                                  #change this. lod jsonl file. no split. make a demos jsonl with 20 demos and remove them from train jsonl and reduce the no of memes.
    else:
        dataset = load_dataset(args.dataset)
        
    #data_split = list(read_jsonl(args.input_path))
    #demo_split = dataset[args.demo_split]
    demos = read_jsonl(args.demos)
    demos = demos[: args.num_demos]
    ex_demos = []
    for d_item in demos:
                    
        dlabels=[]                                               #get labels.
                
        if d_item["shaming"]==1:
            dlabels.append("shaming")
        if d_item["stereotype"]==1:
            dlabels.append("stereotype")
        if d_item["objectification"]==1:
            dlabels.append("objectification")
        if d_item["violence"]==1:
            dlabels.append("violence")
        ex_demos.append(
            {
                "image": load_image(os.path.join(args.images_path, d_item["file_name"])),
                "question": "What does the man in the meme feel entitled to?",                          #again same question for every demo.
                "answer": d_item["Entitlement"],
                "rationale": d_item["Entitlement Reason"],
                "text": d_item["Text"],
                "labels": dlabels,                               
            }
        )

    answers_file = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    seen_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                line = json.loads(line)
                seen_ids.add(line["file_name"])

    r_prompt = "Explain why your answer is correct in great detail, referencing the provided image. Think step-by-step, and make sure to only draw conclusions from evidence present in the provided image."
    a_prompt = "What is the final answer to the question? Be short and concise."

    def add_r_turn(conv, text, labels, question: str, rationale: str | None = None):
        label= "The meme is classified among the following categories: " + labels[:]
        qs = f"{text} {label} {question} {r_prompt}"

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

        conv.append_message(conv.roles[0], qs)
        if rationale is not None:
            #rationale = format_rationale(rationale)
            conv.append_message(
                conv.roles[1],
                rationale + "\n",
            )
        else:
            conv.append_message(
                conv.roles[1],
                None,
            )

    def add_a_turn(conv, answer: str | None = None):
        qs = a_prompt

        conv.append_message(conv.roles[0], qs)
        if answer is not None:
            conv.append_message(
                conv.roles[1],
                answer + "\n",
            )
        else:
            conv.append_message(
                conv.roles[1],
                None,
            )

    def run(conv, images):
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        keywords = [conv.sep]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True,
                top_p=0.7,
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
        return outputs
#continue from here.
    for idx in tqdm(range(len(dataset))):
        if dataset[idx]["file_name"] in seen_ids:
            continue
        ex = dataset[idx]
        image_path = ex["file_name"]                 #i think we have to load the actual image.
        image = load_image(os.path.join(args.images_path, image_path))
        question = "What does the man in the meme feel entitled to?"              #same question for every ex. change according to output expected here.
        text=ex["Text"]
        
        labels=[]                                               #get labels.
            
        if ex["shaming"]==1:
            labels.append("shaming")
        if ex["stereotype"]==1:
            labels.append("stereotype")
        if ex["objectification"]==1:
            labels.append("objectification")
        if ex["violence"]==1:
            labels.append("violence")
        
        image_tensor = image_processor.preprocess(
            [d["image"] for d in ex_demos] + [image], return_tensors="pt"
        )["pixel_values"]
        # removed .unsqueeze(0)
        images = image_tensor.half().cuda()
        conv = conv_templates["multimodal"].copy()

        for d in ex_demos:
            add_r_turn(
                conv,text=d["text"],labels=d["labels"],
                question=d["question"],
                rationale=d["rationale"],            #changed near demos. 
            )
            add_a_turn(
                conv,
                answer=d["answer"],                   #change category wrt output.
            )

        final_conv = conv.copy()

        add_r_turn(
            conv,text=text,
            labels=labels,
            question=question,
            
        )

        rationale = run(conv, images)

        add_r_turn(
            final_conv,
            text=text,
            labels=labels,
            question=question,
            rationale=rationale,
                                       #shud we add text and labels twice?
        )
        full_conv = final_conv.copy()
        add_a_turn(final_conv)

        pred = run(final_conv, images)
        add_a_turn(full_conv, answer=pred)

        # print(
        #     full_conv.get_prompt().replace(
        #         DEFAULT_IM_START_TOKEN
        #         + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        #         + DEFAULT_IM_END_TOKEN,
        #         DEFAULT_IMAGE_TOKEN,
        #     )
        # )

        with open(answers_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "id": ex["file_name"],
                        "rationale": rationale,
                        "pred": pred,
                    }
                )
                + "\n"
            )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--demos", type=str, required=True)
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--num_demos", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="flaviagiammarino/vqa-rad")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-bf16", action="store_true")
    
    args = parser.parse_args()
    
    eval_model(args)
