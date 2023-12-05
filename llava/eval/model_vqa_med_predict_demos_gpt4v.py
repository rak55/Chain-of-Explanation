import argparse
import base64
from io import BytesIO
import os
import time
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI, BadRequestError
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential
import dataclasses
import ujson as json
import yaml


class MinimumDelay:
    def __init__(self, delay: int):
        self.delay = delay
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        seconds = end - self.start
        if self.delay > seconds:
            time.sleep(self.delay - seconds)


@dataclasses.dataclass
class Config:
    seed: int
    delay: int
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    system_prompt: str
    num_demos: int


@retry(wait=wait_random_exponential(min=1, max=90), stop=stop_after_attempt(3))
def chat(client, **kwargs) -> ChatCompletion | None:
    try:
        return client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        print(f"Bad Request: {e}")
        if "safety" in e.message:
            return None
        raise e
    except Exception as e:
        print(f"Exception: {e}")
        raise e


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def format_rationale(rationale: str):
    # If the rationale starts with "The answer is correct because" filter it out
    if rationale.startswith("The answer is correct because"):
        rationale = rationale[len("The answer is correct because") :].strip()
    return rationale


class ContextCreator:
    def __init__(self, config: Config, demos: dict[str, str], demo_split):
        self.config = config
        self.demos = demos
        self.demo_split = demo_split

    def create_prompt(self, ex):
        image = ex["image"]
        question = ex["question"]
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("ascii")

        image_url = f"data:image/jpeg;base64,{base64_image}"

        return {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": image_url,
                },
            ],
        }

    def create_demo(self, d):
        d_idx = d["id"]
        ex = self.demo_split[d_idx]
        rationale = format_rationale(d["rationale"])
        answer = ex["answer"]
        return [
            self.create_prompt(ex),
            {"role": "assistant", "content": f"{rationale} {answer}"},
        ]

    def create_context(self, ex_id: int, ex: dict):
        system_prompt = self.config.system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        for demo in self.demos[ex_id][: self.config.num_demos]:
            messages.extend(self.create_demo(demo))
        messages.append(self.create_prompt(ex))
        return messages


def print_messages(messages):
    for message in messages:
        if isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "text":
                    print(f"{message['role']}: {content['text']}")
                elif content["type"] == "image_url":
                    print(f"{message['role']}: [IMAGE_URL]")
        else:
            print(f"{message['role']}: {message['content']}")
        print()


@dataclasses.dataclass
class CompletionUsageEstimate:
    completion_tokens: int
    prompt_tokens: int
    completion_cost: float
    prompt_cost: float
    total_cost: float


class CompletionUsageEstimator:
    def __init__(
        self,
        completion_cost: float = 0.03,
        completion_cost_tokens: int = 1000,
        prompt_cost: float = 0.01,
        prompt_cost_tokens: int = 1000,
    ):
        self.config = config
        self.running_completion_tokens = 0
        self.running_prompt_tokens = 0
        self.prompts = 0
        self.remaining_prompts = 0
        self.completion_cost = completion_cost / completion_cost_tokens
        self.prompt_cost = prompt_cost / prompt_cost_tokens

    def init(self, remaining_prompts: int):
        self.running_completion_tokens = 0
        self.running_prompt_tokens = 0
        self.prompts = 0
        self.remaining_prompts = remaining_prompts

    def update(self, completion: ChatCompletion):
        usage = completion.usage
        if usage is not None:
            self.running_completion_tokens += usage.completion_tokens
            self.running_prompt_tokens += usage.prompt_tokens
            self.prompts += 1
            self.remaining_prompts -= 1

    def estimate(self):
        # linearly interpolate by computing average tokens per prompt and multiplying by remaining prompts
        completion_tokens_per_prompt = self.running_completion_tokens / self.prompts
        prompt_tokens_per_prompt = self.running_prompt_tokens / self.prompts
        total_completion_tokens = (
            self.running_completion_tokens
            + self.remaining_prompts * completion_tokens_per_prompt
        )
        total_prompt_tokens = (
            self.running_prompt_tokens
            + self.remaining_prompts * prompt_tokens_per_prompt
        )
        completion_cost = total_completion_tokens * self.completion_cost
        prompt_cost = total_prompt_tokens * self.prompt_cost
        total_cost = completion_cost + prompt_cost
        return CompletionUsageEstimate(
            completion_tokens=total_completion_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_cost=completion_cost,
            prompt_cost=prompt_cost,
            total_cost=total_cost,
        )


def eval_model(
    config: Config,
    dataset_name: str,
    split_name: str,
    demo_split_name: str,
    demos_path: str,
    output_path: str,
):
    print(f"Loading dataset: {dataset_name} ({split_name})")
    dataset = load_dataset(dataset_name)
    data_split = dataset[split_name]
    demo_split = dataset[demo_split_name]
    demos = {d["id"]: d["demos"] for d in read_jsonl(demos_path)}
    creator = ContextCreator(config, demos, demo_split)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=os.getenv("OPENAI_TIMEOUT", 90),
    )

    answers_file = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    seen_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                line = json.loads(line)
                seen_ids.add(line["id"])

    usage = CompletionUsageEstimator()
    usage.init(len(data_split) - len(seen_ids))

    with open(answers_file, "a") as f:
        with tqdm(total=len(data_split)) as pbar:
            pbar.update(len(seen_ids))
            for idx in range(len(data_split)):
                if idx in seen_ids:
                    continue
                ex = data_split[idx]

                messages = creator.create_context(idx, ex)
                # print_messages(messages)
                with MinimumDelay(config.delay):
                    completion = chat(
                        client,
                        model=config.model,
                        messages=messages,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        seed=config.seed,
                    )
                    if completion is None:
                        continue
                    content = completion.choices[0].message.content
                    usage.update(completion)
                messages.append({"role": "assistant", "content": content})
                # print_messages([messages[-1]])
                # print("---")
                use = usage.estimate()

                f.write(
                    json.dumps(
                        {
                            "id": idx,
                            "rationale": "",
                            "pred": content,
                        }
                    )
                    + "\n"
                )
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "cost": f"${use.total_cost:.2f}",
                    }
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--demos", type=str, required=True)
    parser.add_argument("--demo_split", type=str, default="train")
    parser.add_argument("--dataset", type=str, default="flaviagiammarino/vqa-rad")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = Config(**config)
    eval_model(
        config=config,
        dataset_name=args.dataset,
        split_name=args.split,
        demo_split_name=args.demo_split,
        demos_path=args.demos,
        output_path=args.output,
    )
