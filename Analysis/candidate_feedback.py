import json
import os
import sys

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

SAVE_DIR = "./candidate_feedback"
prompt_template = "/home/jiashuo/codes/Muffin/reward_model/Llama/templates/alpaca"
lora_weights = "/home/jiashuo/codes/Muffin/reward_model/Llama/lora-7b"


def process_raw(input_data, output_name):
    if not os.path.exists(output_name):
        os.makedirs(output_name)
    processed = {
        "empathy": [],
        "strategy": [],
        "coherence": []
    }
    for line in input_data:
        context = line["context"]
        responses = line["response"]
        for resp in responses:
            if len(context) > 0:
                conv = context[-1]
            else:
                conv = "(Nothing)"
            empathy_inst = {
                "task": "empathy",
                "instruction": TEMPLATE["empathy"]["instruction"],
                "input": TEMPLATE["empathy"]["input"].format(context=conv, response=resp),
            }
            skill_inst = {
                "task": "strategy",
                "instruction": TEMPLATE["strategy"]["instruction"],
                "input": TEMPLATE["strategy"]["input"].format(context=conv, response=resp),
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
                "input": TEMPLATE["coherence"]["input"].format(context=conv, response=resp),
            }
            processed["empathy"].append(empathy_inst)
            processed["strategy"].append(skill_inst)
            processed["coherence"].append(coherence_inst)

    for key, value in processed.items():
        with open(os.path.join(output_name, f"instruction_{key}.jsonl"), "w") as f:
            for line in value:
                f.write(json.dumps(line) + "\n")


def compute_feedback():
    models = ["vanilla", "strat", "MultiESC", "KEMI", "TransESC"]
    modes = ["base", "muffin"]
    facets = ["empathy", "strategy", "coherence"]
    for model in models:
        print("=====================================")
        print(f"Model {model}:")
        for mode in modes:
            feedback = []
            data_dir = f"candidate_feedback/{model}_{mode}"
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
        "vanilla_base": "./output_candidates/vanilla_base.json",
        "vanilla_muffin": "./output_candidates/vanilla_muffin.json",
        "strat_base": "./output_candidates/strat_base.json",
        "strat_muffin": "./output_candidates/strat_muffin.json",
        "MultiESC_base": "./output_candidates/MultiESC_base.json",
        "MultiESC_muffin": "./output_candidates/MultiESC_muffin.json",
        "TransESC_base": "./output_candidates/TransESC_base.json",
        "TransESC_muffin": "./output_candidates/TransESC_muffin.json",
        "KEMI_base": "./output_candidates/KEMI_base.json",
        "KEMI_muffin": "./output_candidates/KEMI_muffin.json"
    }
    for key, path in result_path.items():
        data_reader = json.load(open(path, "r"))
        process_raw(data_reader, os.path.join(SAVE_DIR, key))

        print(f"Generating feedback for {key} base")
        llama_feedback(
            input_data_dir=os.path.join(SAVE_DIR, key),
            prompt_template=prompt_template,
            lora_weights=lora_weights
        )


if __name__ == "__main__":
    # main()
    compute_feedback()
