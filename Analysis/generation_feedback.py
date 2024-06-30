"""
This file is obtain AI feedback of generated responses, including
ESConv training data
ESConv test data

Base models' generated responses (vanilla, joint, MultiESC, transesc, kemi)
Muffin models' generated responses.

Therefore, there will be 12 result_path.
"""

import os
import sys
import json

sys.path.append("../")
sys.path.append("../reward_model/Llama")
from reward_model.Llama.infer import main as llama_feedback

MAP = {
    "No": 0,
    "No Empathy": 0,
    "MI Non-Adherent": 0,
}

EMPATHY = json.load(open("../reward_model/Llama/templates/empathy.json", "r"))
STRATEGY = json.load(open("../reward_model/Llama/templates/strategy.json", "r"))
COHERENCE = json.load(open("../reward_model/Llama/templates/coherence.json", "r"))
TEMPLATE = {
    "empathy": EMPATHY,
    "strategy": STRATEGY,
    "coherence": COHERENCE
}

SAVE_DIR = "./generation_feedback"
prompt_template = "/home/jiashuo/codes/Muffin/reward_model/Llama/templates/alpaca"
lora_weights = "/home/jiashuo/codes/Muffin/reward_model/Llama/lora-7b"


def process_raw(input_data, output_dir, sort="muffin"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert sort in ["muffin", "real", "base"]
    processed = {
        "empathy": [],
        "strategy": [],
        "coherence": []
    }
    for line in input_data:
        context = line["context"]
        response = line["generation"]
        if len(context) > 0:
            conv = context[-1]
        else:
            conv = "(Nothing)"
        empathy_inst = {
            "task": "empathy",
            "instruction": TEMPLATE["empathy"]["instruction"],
            "input": TEMPLATE["empathy"]["input"].format(context=conv, response=response),
        }
        skill_inst = {
            "task": "strategy",
            "instruction": TEMPLATE["strategy"]["instruction"],
            "input": TEMPLATE["strategy"]["input"].format(context=conv, response=response),
        }
        if len(context) == 3:
            conv = f"Help-seeker: {context[-3]}\nSupporter: {context[-2]}\nHelp-seeker: {context[-1]}"
        elif len(context) == 2:
            conv = f"Supporter: {context[-2]}\nHelp-seeker: {context[-1]}"
        elif len(context) == 1:
            conv = f"Help-seeker: {context[-1]}"
        elif len(context) == 0:
            conv = ""
        coherence_inst = {
            "task": "coherence",
            "instruction": TEMPLATE["coherence"]["instruction"],
            "input": TEMPLATE["coherence"]["input"].format(context=conv, response=response),
        }
        processed["empathy"].append(empathy_inst)
        processed["strategy"].append(skill_inst)
        processed["coherence"].append(coherence_inst)

    for key, value in processed.items():
        with open(os.path.join(output_dir, f"instruction_{key}.jsonl"), "w") as f:
            for line in value:
                f.write(json.dumps(line) + "\n")


def compute_feedback():
    models = ["vanilla", "strat", "MultiESC", "TransESC", "KEMI"]
    # models = ["TransESC"]
    modes = ["base", "muffin"]
    facets = ["empathy", "strategy", "coherence"]
    for model in models:
        print("=====================================")
        print(f"Model {model}:")
        for mode in modes:
            feedback = []
            data_dir = f"generation_feedback/{model}-{mode}"
            for facet in facets:
                data_path = os.path.join(data_dir, f"feedback_{facet}.txt")
                with open(data_path, "r") as f:
                    data = f.readlines()
                    data = [MAP.get(line.strip(), 1) for line in data]
                    print(f"Mode {mode}, facet {facet}: {sum(data) / len(data)}")
                    if feedback == []:
                        feedback = data.copy()
                    else:
                        feedback = list(map(lambda x, y: x & y, feedback, data))
            print(f"Overall, average: {sum(feedback) / len(feedback)}")
            print("*" * 20)
        print("=====================================")


def main():
    result_path = {
        "vanilla": "./output/vanilla_strat.json",
        "strat": "./output/vanilla_strat.json",
        "MultiESC": "./output/MultiESC.json",
        "TransESC": "./output/TransESC.json",
        "KEMI": "./output/KEMI.json"
    }
    for key, path in result_path.items():
        data_reader = json.load(open(path, "r"))
        muffin_data, base_data = [], []
        for line in data_reader:
            base_data.append({
                "context": line["context"],
                # "response": line["response"],
                "generation": line[f"{key} base"]
            })
            muffin_data.append({
                "context": line["context"],
                # "response": line["response"],
                "generation": line[f"{key} muffin"]
            })

        if not os.path.exists(os.path.join(SAVE_DIR, f"{key}-base")):
            process_raw(base_data, os.path.join(SAVE_DIR, f"{key}-base"), sort="base")
        if not os.path.exists(os.path.join(SAVE_DIR, f"{key}-muffin")):
            process_raw(muffin_data, os.path.join(SAVE_DIR, f"{key}-muffin"), sort="muffin")
        # if not os.path.exists(os.path.join(SAVE_DIR, f"{key}-real")):
        #     process_raw(base_data, os.path.join(SAVE_DIR, f"{key}-real"), sort="real")

        print(f"Generating feedback for {key} base")
        llama_feedback(
            input_data_dir=os.path.join(SAVE_DIR, f"{key}-base"),
            prompt_template=prompt_template,
            lora_weights=lora_weights
        )
        print(f"Generating feedback for {key} muffin")
        llama_feedback(
            input_data_dir=os.path.join(SAVE_DIR, f"{key}-muffin"),
            prompt_template=prompt_template,
            lora_weights=lora_weights
        )
        # print(f"Generating feedback for {key} real")
        # llama_feedback(
        #     input_data_dir=os.path.join(SAVE_DIR, f"{key}-real"),
        #     prompt_template=prompt_template,
        #     lora_weights=lora_weights
        # )


def ablation_feedback():
    mode_set = {"coherence", "empathy", "strategy"}
    for mode in mode_set:
        data_reader = json.load(open(f"./output/strat_{mode}.json", "r"))
        process_raw(data_reader, os.path.join(SAVE_DIR, f"strat_{mode}"), sort="muffin")
        print(f"Generating feedback for strat_{mode}")
        llama_feedback(
            input_data_dir=os.path.join(SAVE_DIR, f"strat_{mode}"),
            prompt_template=prompt_template,
            lora_weights=lora_weights
        )


def compute_ablation_feedback():
    path_dict = {
        "empathy": "./generation_feedback/strat_empathy",
        "strategy": "./generation_feedback/strat_strategy",
        "coherence": "./generation_feedback/strat_coherence"
    }
    facets = {"empathy", "strategy", "coherence"}
    for key, path in path_dict.items():
        feedback = []
        for facet in facets:
            data_path = os.path.join(path, f"feedback_{facet}.txt")
            with open(data_path, "r") as f:
                data = f.readlines()
                data = [MAP.get(line.strip(), 1) for line in data]
                print(f"Mode {key}, facet {facet}: {sum(data) / len(data)}")
                if feedback == []:
                    feedback = data.copy()
                else:
                    feedback = list(map(lambda x, y: x & y, feedback, data))
        print(f"Overall, average: {sum(feedback) / len(feedback)}")
        print("*" * 20)


if __name__ == "__main__":
    # main()
    # ablation_feedback()
    compute_feedback()
    # compute_ablation_feedback()
