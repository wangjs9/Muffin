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

SAVE_DIR = "./feedback"
prompt_template = "/home/jiashuo/codes/Muffin/reward_model/Llama/templates/alpaca_option"
lora_weights = "/home/jiashuo/codes/Muffin/reward_model/Llama/checkpoint-initial"


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
            if sort == "real":
                response = line["response"]
            else:
                response = line["generation"]
            processed_line = {
                "task": value["task"],
                "instruction": value["instruction"],
                "input": value["input"].format(context=Seeker, response=Supporter + " " + response),
                "option": value["option"]
            }
            processed[key].append(processed_line)

    for key, value in processed.items():
        with open(os.path.join(output_dir, f"instruction_{key}.jsonl"), "w") as f:
            for line in value:
                f.write(json.dumps(line) + "\n")


def compute_feedback():
    models = {"vanilla", "strat", "MultiESC", "TransESC", "KEMI"}
    modes = {"real", "base", "muffin"}
    facets = {"empathy", "strategy", "coherence"}
    for model in models:
        print("=====================================")
        print(f"Model {model}:")
        for mode in modes:
            feedback = []
            data_dir = f"feedback/{model}-{mode}"
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
                "response": line["response"],
                "generation": line[f"{key} base"]
            })
            muffin_data.append({
                "context": line["context"],
                "response": line["response"],
                "generation": line[f"{key} muffin"]
            })

        if not os.path.exists(os.path.join(SAVE_DIR, f"{key}-base")):
            process_raw(base_data, os.path.join(SAVE_DIR, f"{key}-base"), sort="base")
        if not os.path.exists(os.path.join(SAVE_DIR, f"{key}-muffin")):
            process_raw(muffin_data, os.path.join(SAVE_DIR, f"{key}-muffin"), sort="muffin")
        if not os.path.exists(os.path.join(SAVE_DIR, f"{key}-real")):
            process_raw(base_data, os.path.join(SAVE_DIR, f"{key}-real"), sort="real")

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
        print(f"Generating feedback for {key} real")
        llama_feedback(
            input_data_dir=os.path.join(SAVE_DIR, f"{key}-real"),
            prompt_template=prompt_template,
            lora_weights=lora_weights
        )


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
        "empathy": "./feedback/strat_empathy",
        "strategy": "./feedback/strat_strategy",
        "coherence": "./feedback/strat_coherence"
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
    # compute_feedback()
    compute_ablation_feedback()
