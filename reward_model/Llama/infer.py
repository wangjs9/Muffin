import os
import random
import sys
import re

import fire
import json
from tqdm import tqdm

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, set_seed

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter


def main(
        load_8bit: bool = True,
        batch_size: int = 12,
        input_data_dir: str = "",
        base_model: str = "",
        lora_weights: str = "",
        prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    set_seed(1)
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    tokenizer.padding_side = "left"

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    step_tokens = {
        13: [8241, 3782, 16107, 10403, 4806, 5015, 1939], 10403: [2087, 10050], 1939: [7361], 2087: [2276], 2276: [296],
        4806: [557], 5015: [549], 557: [7361], 549: [7361], 7361: [493], 493: [29891], 29891: [13], 3782: [7361]
    }

    def prefix_allowed_tokens_fn(batch_id, inputs):
        if inputs[-2] == 29901 and inputs[-1] == 13:
            return step_tokens[13]
        elif inputs[-1] != 13 and int(inputs[-1]) in step_tokens:
            return step_tokens[int(inputs[-1])]
        else:
            return [i for i in range(tokenizer.vocab_size)]

    def evaluate(
            prompt,
            temperature=0.1,
            top_p=0.75,
            top_k=30,
            num_beams=4,
            max_new_tokens=8,
            stream_output=False,
            **kwargs,
    ):
        inputs = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            # "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    return prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s)
        return [prompter.get_response(o) for o in output]

    data_paths = ["instruction_empathy.jsonl", "instruction_coherence.jsonl", "instruction_strategy.jsonl"]
    for data_path in data_paths:
        input_data_path = os.path.join(input_data_dir, data_path)
        output_data_path = os.path.join(
            input_data_dir, re.sub(".jsonl", ".txt", re.sub("instruction", "feedback", data_path))
        )
        if os.path.exists(output_data_path):
            continue
        with open(input_data_path, "r") as file:
            reader = [json.loads(line) for line in file.readlines()]

        fp = open(output_data_path, "w", encoding="utf-8")
        options = {
            "strategy": ["Other", "MI Adherent", "MI Non-Adherent"],
            "coherence": ["Yes", "No"],
            "empathy": ["Weak Empathy", "Strong Empathy", "No Empathy"]
        }
        prompt_batch = []
        task = ""
        for line in tqdm(reader, total=len(reader)):
            prompt = prompter.generate_prompt(**line)
            prompt_batch.append(prompt)
            if len(prompt_batch) == batch_size:
                answers = evaluate(prompt_batch, num_beams=1)
                for answer in answers:
                    answer = re.sub("[^A-Za-z \n-]", "", answer).strip()
                    task = line["task"]
                    for opt in options[line["task"]]:
                        if answer.startswith(opt):
                            answer = opt
                            break
                    fp.write(answer + "\n")
                prompt_batch = []
            torch.cuda.empty_cache()

        if len(prompt_batch) > 0:
            answers = evaluate(prompt_batch, num_beams=1)
            for idx, answer in enumerate(answers):
                answer = re.sub("[^A-Za-z \n-]", "", answer).strip()
                counter = 1
                while True:
                    for opt in options[task]:
                        if answer.startswith(opt):
                            answer = opt
                            break
                    if answer not in options[task]:
                        answer = evaluate(
                            [prompt_batch[idx]],
                            num_beams=counter + 1,
                            temperature=round(random.random(), 1)
                        )[0]
                    elif counter > 4:
                        answer = options[task][0]
                        break
                    else:
                        break
                fp.write(answer + "\n")
            torch.cuda.empty_cache()
        fp.close()


if __name__ == "__main__":
    fire.Fire(main)
