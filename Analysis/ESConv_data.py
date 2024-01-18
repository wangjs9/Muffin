import os
import sys
import json

sys.path.append("../")
sys.path.append("../reward_model/Llama")
from reward_model.Llama.infer import main as llama_feedback

EMPATHY = json.load(open("../reward_model/Llama/templates/empathy.json", "r"))
STRATEGY = json.load(open("../reward_model/Llama/templates/strategy.json", "r"))
COHERENCE = json.load(open("../reward_model/Llama/templates/coherence.json", "r"))
TEMPLATE = {
    "empathy": EMPATHY,
    "strategy": STRATEGY,
    "coherence": COHERENCE
}
prompt_template = "/home/jiashuo/codes/Muffin/reward_model/Llama/templates/alpaca_option"
lora_weights = "/home/jiashuo/codes/Muffin/reward_model/Llama/checkpoint-initial"


def human_feedback():
    data = json.load(open("../dataset/ESConv.json", "r"))
    feedback = []
    total_num = 0
    for idx, conv in enumerate(data):
        for line in conv["dialog"]:
            total_num += 1
            if "feedback" in line["annotation"]:
                feedback.append(int(line["annotation"]["feedback"]))

    print(len(feedback) / total_num)
    print(len([x for x in feedback if x == 3]) / len(feedback))


def process_raw(input_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    processed = {
        "coherence": [],
        "empathy": [],
        "strategy": []
    }
    for line in input_data:
        line = json.loads(line)
        for key, value in TEMPLATE.items():
            context_list = line["context"]
            Seeker = ""
            Supporter = ""
            for uttr in context_list[::-1]:
                if uttr.startswith("Seeker: "):
                    Seeker = uttr[8:] + Seeker
                else:
                    Supporter = uttr[10:] + Supporter
                if Seeker != "":
                    break
            response = line["response"]
            processed_line = {
                "task": value["task"],
                "instruction": value["instruction"],
                "input": value["input"].format(context=Seeker, response=Supporter + " " + response),
                "option": value["option"]
            }
            processed[key].append(processed_line)

    for key, value in processed.items():
        with open(os.path.join(output_dir, f"instruction_{key}.jsonl"), "w") as file:
            for line in value:
                file.write(json.dumps(line) + "\n")


def AI_feedback():
    with open("./ESConv/train_context.txt", "r") as file:
        train_context = file.readlines()
    process_raw(train_context, "./ESConv/feedback_data")
    llama_feedback(
        input_data_dir="./ESConv/feedback_data",
        prompt_template=prompt_template,
        lora_weights=lora_weights
    )


AI_feedback()
