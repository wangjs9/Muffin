import os
import sys
import re

import fire
import json
import torch
import transformers
from tqdm import tqdm
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, set_seed

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass
torch.cuda.empty_cache()


def main(
        load_8bit: bool = False,
        batch_size: int = 8,
        base_model: str = "",
        # lora_weights: str = "",
        lora_weights: str = "",
        prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    set_seed(0)
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
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
            model.merge_and_unload()
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if lora_weights != "":
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
        if lora_weights != "":
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
        4806: [557], 5015: [549], 557: [7361], 549: [7361], 7361: [493], 493: [29891], 29891: [13],
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
            temperature=0.7,
            top_p=0.3,
            top_k=30,
            num_beams=4,
            max_new_tokens=6,
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

    with open('../dataset/finetune_test.jsonl', "r") as file:
        reader = [json.loads(line) for line in file.readlines()]

    fp = open("./test_LLaMA_1920.txt", "w", encoding="utf-8")
    # fp = open("./test_LLaMA_tuned.txt", "w", encoding="utf-8")
    options = {
        "strategy": ["MI Adherent", "MI Non-Adherent", "Other"],
        "coherence": ["Yes", "No"],
        "empathy": ["No Empathy", "Weak Empathy", "Strong Empathy"]
    }

    prompt_batch = []
    output_batch = []
    task_batch = []
    pbar = tqdm(total=len(reader) // batch_size)
    for line in reader:
        prompt = prompter.generate_prompt(**line)
        prompt_batch.append(prompt)
        output_batch.append(line["output"])
        task_batch.append(line["task"])
        if len(prompt_batch) == batch_size:
            answers = evaluate(prompt_batch, num_beams=1)
            for answer, output, task in zip(answers, output_batch, task_batch):
                answer = re.sub("[^A-Za-z \n-]", "", answer).strip()
                for opt in options[task]:
                    if answer.startswith(opt):
                        answer = opt
                        break
                fp.write(answer + " ## " + output)
            prompt_batch = []
            output_batch = []
            task_batch = []
            pbar.update(1)

    if len(prompt_batch) > 0:
        answers = evaluate(prompt_batch, num_beams=1)
        for answer, output, task in zip(answers, output_batch, task_batch):
            answer = re.sub("[^A-Za-z \n-]", "", answer).strip()
            for opt in options[task]:
                if answer.startswith(opt):
                    answer = opt
                    break
            fp.write(answer + " ## " + output)
    fp.close()


if __name__ == "__main__":
    fire.Fire(main)
