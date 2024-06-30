import json
import random
import numpy as np
from tqdm import tqdm
import krippendorff
from collections import defaultdict, Counter
import logging

logging.getLogger().setLevel(logging.INFO)

unhelpful_signal = {"No Empathy": 0, "MI Non-Adherent": 0, "No": 0}


def compute_feedback(model_data):
    feedback = {
        "empathy": [],
        "strategy": [],
        "coherence": [],
        "overall": []
    }
    for line in tqdm(model_data, total=len(model_data)):
        line = json.loads(line)
        empathy = line["empathy_feedback"].strip()
        strategy = line["strategy_feedback"].strip()
        coherence = line["coherence_feedback"].strip()
        overall = line["required_muffin"]
        feedback["empathy"].append(unhelpful_signal.get(empathy, 1))
        feedback["strategy"].append(unhelpful_signal.get(strategy, 1))
        feedback["coherence"].append(unhelpful_signal.get(coherence, 1))
        feedback["overall"].append(int(overall == False))

    logging.info(f"Feedback computed:")
    logging.info(f"Empathy: {sum(feedback['empathy']) / len(feedback['empathy'])}")
    logging.info(f"Strategy: {sum(feedback['strategy']) / len(feedback['strategy'])}")
    logging.info(f"Coherence: {sum(feedback['coherence']) / len(feedback['coherence'])}")
    logging.info(f"Overall: {sum(feedback['overall']) / len(feedback['overall'])}")


def compute_unhelpfulness():
    base_model_file = "vanilla_generations_feedback.jsonl"
    muffin_model_file = "muffin2.0_generations_feedback.jsonl"
    muffin_model_once_file = "muffin_generations_feedback.jsonl"

    logging.info("Computing feedback for base model:")
    with open(base_model_file, "r") as reader:
        base_model_data = reader.readlines()
        compute_feedback(base_model_data)

    logging.info("Computing feedback for muffin model:")
    with open(muffin_model_file, "r") as reader:
        muffin_model_data = reader.readlines()
        compute_feedback(muffin_model_data)

    logging.info("Computing feedback for muffin model:")
    with open(muffin_model_once_file, "r") as reader:
        muffin_model_data = reader.readlines()
        compute_feedback(muffin_model_data)


def format_for_human_evaluation():
    base_model_file = "vanilla_generations_feedback.jsonl"
    muffin_model_file = "muffin2.0_generations_feedback.jsonl"

    with open(base_model_file, "r") as reader:
        base_model_data = reader.readlines()

    with open(muffin_model_file, "r") as reader:
        muffin_model_data = reader.readlines()

    json_list = []
    for idx, (base_line, muffin_line) in (
            tqdm(enumerate(zip(base_model_data, muffin_model_data)), total=len(base_model_data))):
        base_line = json.loads(base_line)
        muffin_line = json.loads(muffin_line)
        assert base_line["context"] == muffin_line["context"]
        base_response = base_line["response"]
        muffin_response = muffin_line["response"]
        if base_response == muffin_response:
            continue
        writen_line = {
            "context": base_line["context"].split("\n"),
            "base_response": base_response,
            "muffin_response": muffin_response
        }
        json_list.append(writen_line)

    json.dump(json_list, open("required_evaluation.json", "w"), indent=2)


def content_requires_further_evaluation():
    random.seed(42)
    json_content = json.load(open("required_evaluation.json", "r"))
    selected_data_idx = random.sample(range(len(json_content)), 100)
    human_comparison_data = []
    for idx, line in enumerate(json_content):
        if idx in selected_data_idx:
            line.update({"Helpful": ""})
            human_comparison_data.append(line)
    json.dump(human_comparison_data, open("human_evaluation.json", "w"), indent=2)


def compute_human_evaluation():
    model_results = []
    inter_results = []
    label_map = {"base": 0, "muffin": 1, "tie": 2}
    human_evaluation_data = json.load(open("human_evaluation.json", "r"))
    for line in tqdm(human_evaluation_data):
        result = line["Helpful"]
        if result == "":
            continue
        if len(inter_results) >= 50:
            break
        count = defaultdict(int)
        inter_list = [np.nan] * 4
        inter_list[0] = label_map[result[0]]
        inter_list[1] = label_map[result[1]]
        inter_list[2] = label_map[result[2]]
        inter_list[3] = label_map[result[3]]
        inter_results.append(inter_list)
        count[result[0]] += 1
        count[result[1]] += 1
        count[result[2]] += 1
        count[result[3]] += 1

        highest = max(count.values())
        final = [k for k, v in count.items() if v == highest]
        if len(final) == 1:
            model_results.append(final[0])
        elif len(final) == 2 and "tie" in final:
            final.remove("tie")
            model_results.append(final[0])
        else:
            model_results.append("tie")

    print(dict(Counter(model_results)))

    # print(np.array(inter_results))
    value = krippendorff.alpha(reliability_data=np.transpose(np.array(inter_results)), level_of_measurement='nominal')
    print(value)


if __name__ == "__main__":
    # format_for_human_evaluation()
    # content_requires_further_evaluation()
    compute_human_evaluation()
