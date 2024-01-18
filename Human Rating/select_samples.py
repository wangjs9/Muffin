import random
import json
import os

random.seed(42)

test_models = ["MultiESC", "KEMI", "strat"]
data_path = {
    "strat": "../Analysis/output/vanilla_strat.json",
    "MultiESC": "../Analysis/output/MultiESC.json",
    "KEMI": "../Analysis/output/KEMI.json"
}

sample_dir = "./samples"
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)
random_index_path = "./samples/random_index.json"
if not os.path.exists(random_index_path):
    index_dict = {}
    for model in test_models:
        data = json.load(open(data_path[model], "r"))
        random_index = random.sample(range(0, len(data)), 100)
        index_dict[model] = random_index
    json.dump(index_dict, open(random_index_path, "w"))

index_dict = json.load(open(random_index_path, "r"))
for model in test_models:
    if os.path.exists(os.path.join(sample_dir, f"{model}_samples.json")):
        continue
    base_data = json.load(open(data_path[model], "r"))
    random_index = index_dict[model]
    sample_data = [base_data[i] for i in random_index]
    processed_data = []
    for index, data in zip(random_index, sample_data):
        temp_data = {}
        total_context = ""
        context = data["context"]
        sample_id = data.get("sample_id")
        if index != 0:
            last_sample_id = base_data[index - 1].get("sample_id")
            if sample_id == last_sample_id:
                total_context = "Seeker: " + base_data[index - 1]["context"] \
                                + "\nSupporter: " + base_data[index - 1]["response"] + "\n"
        total_context += "Seeker: " + context
        temp_data["context"] = total_context
        temp_data["order"] = ["base", "muffin"] if index % 2 == 0 else ["muffin", "base"]
        if index % 2 == 0:
            temp_data["AB"] = "A: " + data[f"{model} base"] + "\nB: " + data[f"{model} muffin"]
        else:
            temp_data["AB"] = "A: " + data[f"{model} muffin"] + "\nB: " + data[f"{model} base"]
        temp_data["index"] = index
        temp_data["model"] = model
        processed_data.append(temp_data)
    sample_json_path = os.path.join(sample_dir, f"{model}_samples.json")
    json.dump(processed_data, open(sample_json_path, "w"), indent=2)
    sample_txt_path = os.path.join(sample_dir, f"{model}_samples.txt")
    with open(sample_txt_path, "w") as file:
        for line in processed_data:
            file.write(json.dumps(line) + "\n")

all_data = []
for model in test_models:
    data = json.load(open(os.path.join(sample_dir, f"{model}_samples.json"), "r"))
    all_data += data
random.shuffle(all_data)
json.dump(all_data, open(os.path.join(sample_dir, "all_samples.json"), "w"), indent=2)
