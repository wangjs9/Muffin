import json
import os
import numpy as np

MAP = {
    "No": 0,
    "No Empathy": 0,
    "MI Non-Adherent": 0,
}


def unhelpfulness():
    file_dir = {
        "/home/jiashuo/codes/Muffin/TransESC/candidate_data",
        "/home/jiashuo/codes/Muffin/KEMI/DATA/strat.strat.esconv.sbert/2023-06-30223758.3e-05.16.1gpu/candidates_10_epoch-4.bin_train",
        "/home/jiashuo/codes/Muffin/MultiESC/final_output/lwg_whlookahead_generate_candidate_10",
        "/home/jiashuo/codes/Muffin/ESConv/DATA/vanilla.vanilla/2023-06-20204748.3e-05.16.1gpu/candidates_10_best_model.bin_train",
        "/home/jiashuo/codes/Muffin/ESConv/DATA/strat.strat/2023-06-20204057.3e-05.16.1gpu/candidates_10_best_model.bin_train"
    }
    for dir in file_dir:
        data_path = os.path.join(dir, "candidate_feedback.npy")
        coherence_path = os.path.join(dir, "feedback_coherence.txt")
        with open(coherence_path, "r") as file:
            coherence_reader = [MAP.get(line.strip(), 1) for line in file.readlines()]
        empathy_path = os.path.join(dir, "feedback_empathy.txt")
        with open(empathy_path, "r") as file:
            empathy_reader = [MAP.get(line.strip(), 1) for line in file.readlines()]
        strategy_path = os.path.join(dir, "feedback_strategy.txt")
        with open(strategy_path, "r") as file:
            strategy_reader = [MAP.get(line.strip(), 1) for line in file.readlines()]
        feedback = np.load(data_path, allow_pickle=True)
        total = len(feedback) * feedback.shape[1]
        positive = sum(list(map(lambda x: sum(x), feedback)))
        model_name = dir.split('/')[5]
        if model_name == "ESConv":
            model_name = dir.split("/")[7].split(".")[0]
        print(f"{model_name}: {1 - positive / total}")
        print(f"coherence: {1 - sum(coherence_reader) / total}")
        print(f"empathy: {1 - sum(empathy_reader) / total}")
        print(f"strategy: {1 - sum(strategy_reader) / total}")
        print("=====================================")


def statis():
    coherence_path = "../reward_model/dataset/processed_coherence.txt"
    empathy_path = "../reward_model/dataset/processed_empathy.txt"
    strategy_path = "../reward_model/dataset/processed_strategy.txt"

    train_path = "../reward_model/dataset/finetune_train.jsonl"
    eval_path = "../reward_model/dataset/finetune_eval.jsonl"
    test_path = "../reward_model/dataset/finetune_test.jsonl"

    with open(train_path, "r") as file:
        train_reader = [json.loads(line) for line in file.readlines()]
        print(f"train: {len(train_reader)}")

    with open(eval_path, "r") as file:
        eval_reader = [json.loads(line) for line in file.readlines()]
        print(f"eval: {len(eval_reader)}")

    with open(test_path, "r") as file:
        test_reader = [json.loads(line) for line in file.readlines()]
        print(f"test: {len(test_reader)}")

    with open(coherence_path, "r") as file:
        coherence_reader = [line for line in file.readlines()]
        print(f"coherence: {len(coherence_reader)}")

    with open(empathy_path, "r") as file:
        empathy_reader = [line for line in file.readlines()]
        print(f"empathy: {len(empathy_reader)}")

    with open(strategy_path, "r") as file:
        strategy_reader = [line for line in file.readlines()]
        print(f"strategy: {len(strategy_reader)}")


if __name__ == "__main__":
    unhelpfulness()
