import os
import fire
import json
import torch
import torch.nn as nn
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
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


def main(
        load_8bit: bool = True,
        base_model: str = "",
        lora_weights: str = "./lora-alpaca",
        prompt_template: str = "alpaca",  # The prompt template to use
        output_path: str = "./infer_output.txt",
        cutoff_len: int = 1024,
):
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

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    EMPATHY = json.load(open("./templates/empathy.json", "r"))
    STRATEGY = json.load(open("./templates/strategy.json", "r"))
    COHERENCE = json.load(open("./templates/coherence.json", "r"))
    TEMPLATE = {"empathy": EMPATHY, "strategy": STRATEGY, "coherence": COHERENCE}

    def tokenize(prompt, mask_only=False):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        if mask_only:
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            return torch.tensor([len(mask) for mask in result["attention_mask"]], device=device).reshape(-1, 1)
        else:
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=True,
                return_tensors="pt",
            )
            result["labels"] = result["input_ids"].clone()
            return {key: value.to(device) for key, value in result.items()}

    def empty_value():

        full_prompts, mask_prompts = [], []
        for key, value in TEMPLATE.items():
            options = [opt.strip() for opt in value["option"].split(",")]
            for opt in options:
                full_prompts.append(
                    prompter.generate_prompt(value["instruction"], "seeker says:\nsupporter says:", value["option"],
                                             f"{opt}\n", is_train=False))
                mask_prompts.append(
                    prompter.generate_prompt(value["instruction"], "seeker says:\nsupporter says:", value["option"],
                                             is_train=False))
        tokenized_full_prompt = tokenize(full_prompts)
        labels = tokenized_full_prompt["labels"]
        prompt_mask_lengths = tokenize(mask_prompts, mask_only=True)
        prompt_arange = torch.arange(0, labels.size(1)).expand(labels.size(0), -1).to(device)
        prompt_mask = (prompt_arange >= prompt_mask_lengths).long() * (labels != 0).long()
        with torch.no_grad():
            logits = model(**tokenized_full_prompt)[1]
            loss_fct = nn.CrossEntropyLoss(ignore_index=0, reduction="none")  # <- Defined without the weight parameter
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = (loss.reshape(labels.size(0), -1) * prompt_mask).sum(-1) / prompt_mask.sum(-1)
            prob = torch.exp(-loss / 100).detach().cpu().numpy()
        count = 0
        MODEL_BIAS = {"empathy": {}, "strategy": {}, "coherence": {}}
        for key, value in TEMPLATE.items():
            options = [opt.strip() for opt in value["option"].split(",")]
            for opt in options:
                MODEL_BIAS[key][opt] = prob[count]
                count += 1

        return MODEL_BIAS

    def evaluate(
            instruction,
            input=None,
            option=None,
    ):
        full_prompts, mask_prompts = [], []
        for opt in option.split(", "):
            full_prompts.append(prompter.generate_prompt(instruction, input, option, f"{opt}\n", is_train=False))
            mask_prompts.append(prompter.generate_prompt(instruction, input, option, is_train=False))
        model_inputs = tokenize(full_prompts)
        prompt_mask_lengths = tokenize(mask_prompts, mask_only=True)
        labels = model_inputs["labels"]
        prompt_arange = torch.arange(0, labels.size(1), device=device).expand(labels.size(0), -1)
        prompt_mask = (prompt_arange >= prompt_mask_lengths).long() * (labels != 0).long()
        with torch.no_grad():
            logits = model(**model_inputs)[1]
            loss_fct = nn.CrossEntropyLoss(ignore_index=0, reduction="none")  # <- Defined without the weight parameter
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = (loss.reshape(labels.size(0), -1) * prompt_mask).sum(-1) / prompt_mask.sum(-1)
            # prob = torch.exp(-loss)
            prob = torch.exp(-loss / 100).detach().cpu().numpy()
        MODEL_PREDICT = {}
        count = 0
        for opt in option.split(", "):
            MODEL_PREDICT[opt] = prob[count]
            count += 1

        return MODEL_PREDICT

    # testing code for readme
    with open('../dataset/finetune_test.jsonl', "r") as file:
        reader = [json.loads(line) for line in file.readlines()]
    MODEL_BIAS = empty_value()
    binary_label = {
        "MI Adherent": "Yes", "MI Non-Adherent": "No", "Other": "Yes",
        "No Empathy": "No", "Weak Empathy": "Yes", "Strong Empathy": "Yes",
        "Yes": "Yes", "No": "No",
    }
    fp = open(output_path, "w", encoding="utf-8")
    for line in tqdm(reader, total=len(reader)):
        predict = evaluate(line["instruction"], line["input"], line["option"])
        predict = {key: value - 0.01 * MODEL_BIAS[line["task"]][key] for key, value in predict.items()}
        answer = max(predict, key=lambda x: predict[x])
        fp.write(f"{binary_label[answer]} ## {binary_label[line['output'].strip()]}\n")
    fp.close()


if __name__ == "__main__":
    fire.Fire(main)
