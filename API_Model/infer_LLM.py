import sys

sys.path.append("../")
sys.path.append("../reward_model/Llama")
import os
import random
import json
from tqdm import tqdm
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--model_framework", type=str, required=True, choices=["vanilla", "muffin"])
parse.add_argument("--openai_api_key", type=str, default="sk-GIZ7FxZAcRIYtkUihDVwrKfG98HNT85Q0PzEUQcs4wQME8AY")
# parse.add_argument("--openai_api_key", type=str, default="sk-YYpAQgm6ntC5RNIA3a85E35238D845Fa807f087dDf61D4Da")
parse.add_argument("--openai_api_base", type=str, default="https://api.chatanywhere.com.cn/v1")
# parse.add_argument("--openai_api_base", type=str, default="https://lonlie.plus7.plus/v1")
parse.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
# parse.add_argument("--model_name", type=str, default=g4f.models.gpt_4)
parse.add_argument("--base_framework", type=str, default="muffin")
parse.add_argument("--new_framework", type=str, default="muffin2.0")
args = parse.parse_args()
MODEL_NAME = args.model_name
import logging

logging.getLogger().setLevel(logging.INFO)
logging.info(f"Using model: {MODEL_NAME}")
os.environ["OPENAI_API_KEY"] = args.openai_api_key
os.environ["OPENAI_API_BASE"] = args.openai_api_base

from reward_model.chatarena.agent import Player
from reward_model.chatarena.backends import OpenAIResponse

system_role = """Imagine that you are expert in emotional support. You are expected provide some emotional support to a user have 
emotional distress and assistant them in overcoming their challenge."""
muffin_role = """Imagine that you are expert in emotional support. Here is a conversation between a supporter and
a help-seeker: %s.\n\nIt is found that the supporter's last response ``%s`` is not supportive enough, especially in the 
aspect of %s. Please provide a more supportive response and avoid the aforementioned problem."""

speaker_map = {"usr": "help-seeker", "sys": "supporter"}


def _norm(s):
    return ' '.join(s.strip().split())


def convert_data_to_inputs(data):
    dialog = data['dialog']
    inputs = []
    context = []
    speaker = []  # added line

    for i in range(len(dialog)):
        text = _norm(dialog[i]['text'])
        speaker = dialog[i]['speaker']  # added line
        if i > 0 and speaker == 'sys':
            res = {
                'context': context[-10:].copy(),
                'response': text,
            }
            inputs.append(res)

        context = context + [f"{speaker_map[speaker]}: " + text]

    return inputs


def get_infer_batch(infer_input_file, infer_batch_size=1):
    with open(infer_input_file, "r") as reader:
        data_reader = reader.readlines()

    sample_ids = []
    posts = []
    references = []
    for sample_id, json_line in tqdm(
            enumerate(data_reader), total=len(data_reader), desc="reading data", position=0, leave=True):
        line = json.loads(json_line)
        inputs = convert_data_to_inputs(line)
        for i in range(len(inputs)):
            ipt = inputs[i]
            if len(ipt['context']) < 3:
                continue
            posts.append("\n".join(ipt['context']))
            references.append(ipt['response'])
            sample_ids.append(sample_id)

            if len(posts) == infer_batch_size:
                yield posts, references, sample_ids
                sample_ids = []
                posts = []
                references = []

    if len(sample_ids) > 0:
        yield posts, references, sample_ids


def get_random_example_from_trainingset(train_file):
    """

    Args:
        train_file:

    Returns:
        a conversation snippet like:
        conversation context:\n
        help-seeker: xxx
        supporter: xxx
        help-seeker: xxx

        supporter response:\n
        supporter: xxx
    """
    with open(train_file, "r", encoding="utf-8") as reader:
        data_reader = reader.readlines()

    contexts = []
    responses = []
    for line in data_reader:
        line = json.loads(line)
        conv_list = convert_data_to_inputs(line)
        conv_list = [conv for conv in conv_list[:-2] if len(conv['context']) > 2]
        for i in range(len(conv_list)):
            conv_list[i]['context'] = conv_list[i]['context'][-5:]
        for conv in conv_list:
            context = "\n".join(conv['context'])
            response = conv['response']
            contexts.append(context)
            responses.append(response)
    while True:
        idx = random.randint(0, len(contexts) - 1)
        context = contexts[idx]
        response = responses[idx]
        yield idx, "conversation context:\n" + context + "\n\nsupporter response:\nsupporter: " + response


def process_raw(input_data, output_dir):
    template = {
        "empathy": json.load(open("../reward_model/Llama/templates/empathy.json", "r")),
        "strategy": json.load(open("../reward_model/Llama/templates/strategy.json", "r")),
        "coherence": json.load(open("../reward_model/Llama/templates/coherence.json", "r"))
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    processed = {"empathy": [], "strategy": [], "coherence": []}
    for line in input_data:
        line = json.loads(line)
        context = line["context"]
        response = line["response"]
        conv = context.split("\n")[-1].split(":")[-1].strip()

        empathy_inst = {
            "task": "empathy",
            "instruction": template["empathy"]["instruction"],
            "input": template["empathy"]["input"].format(context=conv, response=response),
        }
        skill_inst = {
            "task": "strategy",
            "instruction": template["strategy"]["instruction"],
            "input": template["strategy"]["input"].format(context=conv, response=response),
        }
        coherence_inst = {
            "task": "coherence",
            "instruction": template["coherence"]["instruction"],
            "input": template["coherence"]["input"].format(context=context, response=response),
        }
        processed["empathy"].append(empathy_inst)
        processed["strategy"].append(skill_inst)
        processed["coherence"].append(coherence_inst)

    for key, value in processed.items():
        with open(os.path.join(output_dir, f"instruction_{key}.jsonl"), "w") as f:
            for line in value:
                f.write(json.dumps(line) + "\n")


def compute_feedback(output_dir):
    from reward_model.Llama.infer import main as llama_feedback
    prompt_template = "/home/jiashuo/codes/Muffin/reward_model/Llama/templates/alpaca"
    lora_weights = "/home/jiashuo/codes/Muffin/reward_model/Llama/lora-7b"

    llama_feedback(
        input_data_dir=output_dir,
        prompt_template=prompt_template,
        lora_weights=lora_weights
    )


if __name__ == "__main__":
    random.seed(42)
    system = Player(
        name="supporter",
        role_desc=system_role,
        backend=OpenAIResponse(temperature=0, model=MODEL_NAME, max_tokens=128)
    )
    train_file = "../ESConv/DATA/train.txt"
    test_file = "../ESConv/DATA/test.txt"
    example_generator = get_random_example_from_trainingset(train_file)
    infer_dataloader = get_infer_batch(test_file)

    if args.model_framework == "vanilla":
        if os.path.exists("vanilla_generations.jsonl"):
            exit(0)
        if not os.path.exists("vanilla_generations.jsonl"):
            writer = open("vanilla_generations.jsonl", "a+")
            counter = len(open("vanilla_generations.jsonl", "r").readlines())
        else:
            writer = open("vanilla_generations.jsonl", "w")
            counter = 0
        cur_id = 0
        for (post, reference, sample_id), (example_id, example) in zip(infer_dataloader, example_generator):
            cur_id += 1
            if cur_id <= counter:
                continue
            messages = [{
                'role': 'user',
                'content': system_role + "\n\nHere is one snippet of an emotional support conversation for your reference:\n"
                           + example + "\n\nThe response doesn't necessarily have to be lengthy.\n\nNow, here is a conversation context:\n"
                           + post[0] + "\n\nwhat should be the supporter response?"
            }]
            response = system(messages)
            generation_line = {
                "response": response,
                "reference": reference[0],
                "context": post[0],
                "conversation sample": example
            }
            writer.write(json.dumps(generation_line) + "\n")
            counter += 1
        writer.close()
    else:
        assert os.path.exists(f"{args.base_framework}_generations.jsonl")
        with open(f"{args.base_framework}_generations.jsonl", "r") as reader:
            data_reader = reader.readlines()
        if not os.path.exists(f"{args.base_framework}_generations_feedback.jsonl"):
            os.makedirs(f"{args.base_framework}_feedback", exist_ok=True)
            process_raw(data_reader, f"{args.base_framework}_feedback")
            compute_feedback("base_feedback")
            with open(f"{args.base_framework}_feedback/feedback_strategy.txt", "r") as reader:
                strategy_feedback = reader.readlines()
            with open(f"{args.base_framework}_feedback/feedback_empathy.txt", "r") as reader:
                empathy_feedback = reader.readlines()
            with open(f"{args.base_framework}_feedback/feedback_coherence.txt", "r") as reader:
                coherence_feedback = reader.readlines()
            with open(f"{args.base_framework}_generations_feedback.jsonl", "w") as writer:
                for line in data_reader:
                    line = json.loads(line)
                    strategy = strategy_feedback.pop(0)
                    empathy = empathy_feedback.pop(0)
                    coherence = coherence_feedback.pop(0)
                    required_muffin = False
                    if strategy.strip() == "MI Non-Adherent" or empathy.strip() == "No Empathy" or coherence.strip() == "No":
                        required_muffin = True
                    line.update({
                        "strategy_feedback": strategy,
                        "empathy_feedback": empathy,
                        "coherence_feedback": coherence,
                        "required_muffin": required_muffin,
                    })
                    writer.write(json.dumps(line) + "\n")

        if not os.path.exists(f"{args.new_framework}_generations.jsonl"):
            with open(f"{args.base_framework}_generations_feedback.jsonl", "r") as reader:
                data_reader = reader.readlines()
            writer = open(f"{args.new_framework}_generations.jsonl", "w")
            for line in tqdm(data_reader):
                line = json.loads(line)
                required_muffin = line["required_muffin"]
                if not required_muffin:
                    writer.write(json.dumps(line) + "\n")
                    continue
                strategy = line["strategy_feedback"]
                empathy = line["empathy_feedback"]
                coherence = line["coherence_feedback"]
                non_helpful = ""
                if strategy == "MI Non-Adherent":
                    non_helpful += "the communication skill, which is MI Non-Adherent"
                if empathy == "No Empathy":
                    if non_helpful != "":
                        non_helpful += ", and "
                    non_helpful += "the empathetic expression, which is No Empathy"
                if coherence == "No":
                    if non_helpful != "":
                        non_helpful += ", and "
                    non_helpful += "the response coherence, which is not coherent"
                context = line["context"]
                response = line["response"]
                conversation = context + "\n" + response
                messages = [{
                    'role': 'user',
                    'content': muffin_role % (conversation, response, non_helpful)
                }]
                response = system(messages)
                line["response"] = response
                writer.write(json.dumps(line) + "\n")
            writer.close()

        if not os.path.exists(f"{args.new_framework}_generations_feedback.jsonl"):
            with open(f"{args.new_framework}_generations.jsonl", "r") as reader:
                data_reader = reader.readlines()
            os.makedirs(f"{args.new_framework}_feedback", exist_ok=True)
            process_raw(data_reader, f"{args.new_framework}_feedback")
            compute_feedback(f"{args.new_framework}_feedback")
            with open(f"{args.new_framework}_feedback/feedback_strategy.txt", "r") as reader:
                strategy_feedback = reader.readlines()
            with open(f"{args.new_framework}_feedback/feedback_empathy.txt", "r") as reader:
                empathy_feedback = reader.readlines()
            with open(f"{args.new_framework}_feedback/feedback_coherence.txt", "r") as reader:
                coherence_feedback = reader.readlines()
            with open(f"{args.new_framework}_generations_feedback.jsonl", "w") as writer:
                for line in data_reader:
                    line = json.loads(line)
                    strategy = strategy_feedback.pop(0)
                    empathy = empathy_feedback.pop(0)
                    coherence = coherence_feedback.pop(0)
                    required_muffin = False
                    if strategy.strip() == "MI Non-Adherent" or empathy.strip() == "No Empathy" or coherence.strip() == "No":
                        required_muffin = True
                    line.update({
                        "strategy_feedback": strategy,
                        "empathy_feedback": empathy,
                        "coherence_feedback": coherence,
                        "required_muffin": required_muffin,
                    })
                    writer.write(json.dumps(line) + "\n")
