import os
import json

import fire

EMPATHY = json.load(open("./templates/empathy.json", "r"))
STRATEGY = json.load(open("./templates/strategy.json", "r"))
COHERENCE = json.load(open("./templates/coherence.json", "r"))
TEMPLATE = {
    "empathy": EMPATHY,
    "strategy": STRATEGY,
    "coherence": COHERENCE
}


def process(
        input_file_dir: str = "",
):
    if not os.path.exists(input_file_dir):
        raise FileNotFoundError(f"The candidate file directory {input_file_dir} is not found.")
    try:
        data = json.load(open(os.path.join(input_file_dir, "candidates.json"), "r"))
    except FileNotFoundError:
        data = json.load(open(os.path.join(input_file_dir, "candidates.txt"), "r"))
    processed = {
        "empathy": [],
        "strategy": [],
        "coherence": []
    }
    for line in data:
        context = line.get("context", line["post"])
        responses = line.get("responses", line["generation"])

        for candidate in responses:
            for key, value in TEMPLATE.items():
                try:
                    if len(context) == 0:
                        context = [""]
                    processed_line = {
                        "task": value["task"],
                        "instruction": value["instruction"],
                        "input": value["input"].format(context=context[-1], response=candidate)
                    }
                    processed[key].append(processed_line)
                except:
                    print(value)

    for key, value in processed.items():
        with open(os.path.join(input_file_dir, f"instruction_{key}.jsonl"), "w") as f:
            for line in value:
                f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    fire.Fire(process)
