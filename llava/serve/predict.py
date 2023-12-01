import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from datasets import load_dataset

from tqdm import tqdm
import json

from PIL import Image

import requests
from PIL import Image
from io import BytesIO

def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


def write_jsonl(path, examples):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

def main(args):
    # Model
    disable_torch_init()
    if args.model_name is None:
        model_name = get_model_name_from_path(args.model_path)
    else:
        model_name = args.model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(
      args.model_path, 
      args.model_base, 
      model_name, 
      args.load_8bit, 
      args.load_4bit, 
      args.load_bf16, 
      device=args.device, 
      lora_path=args.lora_path
    )

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    dataset = load_dataset(args.dataset)
    seen_ids = set()
    if os.path.exists(args.output_path):
        for p in read_jsonl(args.output_path):
            seen_ids.add(p['id'])
    
    with open(args.output_path, 'a') as f:
        for ex in tqdm(data):
            if ex['id'] in seen_ids:
                continue
            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles
            image_path = ex['images'][0]
            try:
                image = load_image(os.path.join(args.images_path, image_path))
                # Similar operation in model_worker.py
                image_tensor = process_images([image], image_processor, model.config)
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.bfloat16 if args.load_bf16 else torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16 if args.load_bf16 else torch.float16)
            
                inp = args.prompt
        
                if image is not None:
                    # first message
                    if model.config.mm_use_im_start_end:
                        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                    else:
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                    conv.append_message(conv.roles[0], inp)
                    image = None
                else:
                    # later messages
                    conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
        
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
                with torch.inference_mode():
                    output_ids = model.generate(
                        inputs=input_ids,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        images=image_tensor,
                    )
            
            except Exception as e:
                print(e)
                continue
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            f.write(json.dumps(
                {
                    "id": ex["id"],
                    "response": outputs
                }
            ) + "\n")
            f.flush()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True, help='huggingface dataset')
    parser.add_argument("--split", type=str, default='train', help='huggingface dataset split')
    parser.add_argument("--output_path", type=str, required=True, help='output path for .jsonl file')
    parser.add_argument("--prompt", type=str, default='Describe this image in great detail.', help='Prompt to use along with image')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-bf16", action="store_true")
    args = parser.parse_args()
    main(args)
