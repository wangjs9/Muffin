import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import sys

sys.path.append("../")
sys.path.append("Llama")

empathy_transfer = {"No Empathy": 0, "Weak Empathy": 1, "Strong Empathy": 1}


def process_raw(input_data, output_dir):
    template = {
        "empathy": json.load(open("../reward_model/Llama/templates/empathy.json", "r")),
        "strategy": json.load(open("../reward_model/Llama/templates/strategy.json", "r")),
        "coherence": json.load(open("../reward_model/Llama/templates/coherence.json", "r"))
    }
    processed_empathy = []
    for line in input_data:
        empathy_inst = {
            "task": "empathy",
            "instruction": template["empathy"]["instruction"],
            "input": template["empathy"]["input"].format(context="", response=line),
        }
        processed_empathy.append(empathy_inst)

    with open(os.path.join(output_dir, "instruction_empathy.jsonl"), "w") as f:
        for line in processed_empathy:
            f.write(json.dumps(line) + "\n")


def compute_feedback(output_dir):
    from Llama.infer import main as llama_feedback
    prompt_template = "/home/jiashuo/codes/Muffin/reward_model/Llama/templates/alpaca"
    lora_weights = "/home/jiashuo/codes/Muffin/reward_model/Llama/lora-7b"
    # lora_weights = None

    llama_feedback(
        input_data_dir=output_dir,
        prompt_template=prompt_template,
        lora_weights=lora_weights
    )


def main(utterances):
    process_raw(utterances, "./rebuttal_experiments")
    compute_feedback("./rebuttal_experiments")


if __name__ == "__main__":
    data_path = "./dataset/messages.csv"
    # # data_path = "./dataset/additional_empathetic_dataset.csv"
    # # data = pd.read_csv(data_path, header=0, encoding='windows-1252')
    data = pd.read_csv(data_path, header=0)
    # utterances = []
    # real_labels = []
    # for idx, line in data.iterrows():
    #     utterances.append(line["essay"])
    #     real_labels.append(line["empathy_bin"])
    # os.makedirs("rebuttal_experiments", exist_ok=True)
    # with open("rebuttal_experiments/real_labels.txt", "w") as f:
    #     for label in real_labels:
    #         f.write(str(label) + "\n")
    # main(utterances)
    real_labels = []
    trush = []
    for idx, line in data.iterrows():
        if int(line["distress_bin"]) == 1:
            trush.append(idx)
    print(len(trush))
    with open("rebuttal_experiments/real_labels.txt", "r") as f:
        for idx, line in enumerate(f.readlines()):
            label = line.strip()
            real_labels.append(int(label))
            # real_labels.append(empathy_transfer.get(label, 1))
    # real_labels = [x for idx, x in enumerate(real_labels) if idx not in trush]
    with open("rebuttal_experiments/feedback_empathy_vanilla.txt", "r") as f:
        vanilla_data = [empathy_transfer.get(line.strip(), 1) for line in f.readlines()]
        # vanilla_data = [x for idx, x in enumerate(vanilla_data) if idx not in trush]
    with open("rebuttal_experiments/feedback_empathy_finetuned.txt", "r") as f:
        finetuned_data = [empathy_transfer.get(line.strip(), 1) for line in f.readlines()]
        # finetuned_data = [x for idx, x in enumerate(finetuned_data) if idx not in trush]
    with open("rebuttal_experiments/feedback_GPT3.5.jsonl", "r") as f:
        gpt_data = [empathy_transfer.get(json.loads(line)["answer"], 1) for line in f.readlines()]
        # gpt_data = [x for idx, x in enumerate(gpt_data) if idx not in trush]
    with open("rebuttal_experiments/feedback_GPT4.jsonl", "r") as f:
        gpt4_data = [empathy_transfer.get(json.loads(line)["answer"], 2) for line in f.readlines()]
        # gpt4_data = [x for idx, x in enumerate(gpt4_data) if idx not in trush]

    print("F1: ", f1_score(real_labels, vanilla_data, average="macro"))
    print("Acc: ", accuracy_score(real_labels, vanilla_data))
    print("Recall: ", recall_score(real_labels, vanilla_data, average="macro"))
    print("Precision: ", precision_score(real_labels, vanilla_data, average="macro"))
    print()
    print("F1: ", f1_score(real_labels, finetuned_data, average="macro"))
    print("Acc: ", accuracy_score(real_labels, finetuned_data))
    print("Recall: ", recall_score(real_labels, finetuned_data, average="macro"))
    print("Precision: ", precision_score(real_labels, finetuned_data, average="macro"))
    print()
    print("F1: ", f1_score(real_labels, gpt_data, average="macro"))
    print("Acc: ", accuracy_score(real_labels, gpt_data))
    print("Recall: ", recall_score(real_labels, gpt_data, average="macro"))
    print("Precision: ", precision_score(real_labels, gpt_data, average="macro"))
    print()
    real_labels = [x for x, y in zip(real_labels, gpt4_data) if y != 2]
    gpt4_data = [x for x in gpt4_data if x != 2]
    print("F1: ", f1_score(real_labels, gpt4_data, average="macro"))
    print("Acc: ", accuracy_score(real_labels, gpt4_data))
    print("Recall: ", recall_score(real_labels, gpt4_data, average="macro"))
    print("Precision: ", precision_score(real_labels, gpt4_data, average="macro"))
