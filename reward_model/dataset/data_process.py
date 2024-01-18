import os
import json
import pandas as pd
import random

random.seed(42)


def process_empathy():
    """
    there are three aspects related to empathy:
    1. emotional reactions; 2. explorations; 3. interpretations
    we add three scores of all three aspects, and we regard 0 as non-empathy
    :return:
    generate a .txt file with each json_string line in the format: context, response, and score
    """
    data_directory = "./empathy/"
    data_paths = os.listdir(data_directory)
    context, response, score = [], [], []
    empathy_level = {0: "No Empathy", 1: "Weak Empathy", 2: "Strong Empathy"}
    for path in data_paths:
        data = pd.read_csv(os.path.join(data_directory, path), header=0)
        for row_id, line in data.iterrows():
            if len(context) <= row_id:
                context.append(" ".join(line["seeker_post"].split()))
                response.append(" ".join(line["response_post"].split()))
                score.append(line["level"])
            else:
                score[row_id] = max(score[row_id], line["level"])
        assert row_id + 1 == len(context) == len(response) == len(score)
    options = ["No Empathy", "Weak Empathy", "Strong Empathy"]
    with open("processed_empathy.txt", "w") as f:
        for _cont, _resp, _score in zip(context, response, score):
            line = {
                "task": "empathy",
                "instruction": "In the context of empathy, there are three aspects: (1) emotional reactions: expressing emotions such as warmth, compassion, and concern, experienced by peer supporter after reading seeker's post, (2) interpretations: communicating an understanding of feelings and experiences inferred from the seeker's post, and (3) explorations: improving understanding of the seeker by exploring the feelings and experiences not stated in the post. Each aspect has no communication, weak communication, and strong communication, which depend on how the related content is expressed. Empathy level is determined by the maximum level of these three aspects.\n Identify the empathy level of the supporter's response in the input conversation.",
                "input": f"seeker says: '{_cont[-300:]}'\nsupporter says: '{_resp[:256]}'",
                "option": ", ".join(options),
                "output": f"{empathy_level[_score]}\n"
            }
            f.write(json.dumps(line))
            f.write("\n")

    with open("empathy.txt", "w") as f:
        for _cont, _resp, _score in zip(context, response, score):
            line = {"context": _cont[-300:], "response": _resp[:256], "label": empathy_level[_score]}
            f.write(json.dumps(line))
            f.write("\n")


def process_strategy():
    data_path = "./strategy/MI Dataset.csv"
    data = pd.read_csv(data_path, header=0)
    context, response, strategy = [], [], []
    current_context = ""
    for row_id, line in data.iterrows():
        author = line["author"]
        if author == "speaker":
            current_context = line["text"]
        else:
            label = line["final agreed label"]
            if type(label) == str and label != "-":
                context.append(" ".join(current_context.split()))
                response.append(" ".join(line["text"].split()))
                strategy.append(label)
    strategy2class = {
        "Advise with Permission": "MI Adherent", "Affirm": "MI Adherent", "Emphasize Autonomy": "MI Adherent",
        "Support": "MI Adherent", "Warn": "MI Non-Adherent", "Advise without Permission": "MI Non-Adherent",
        "Direct": "MI Non-Adherent", "Confront": "MI Non-Adherent", "Open Question": "Other", "Self-Disclose": "Other",
        "Other": "Other", "Closed Question": "Other", "Give Information": "Other", "Simple Reflection": "Other",
        "Complex Reflection": "Other"
    }
    options = ["MI Adherent", "MI Non-Adherent", "Other"]
    with open("processed_strategy.txt", "w") as f:
        for _cont, _resp, _strat in zip(context, response, strategy):
            line = {
                "task": "strategy",
                "instruction": "There are three strategies in Motivational Interviewing, which  can be described as follows:\n MI Adherent includes: (1) advising when the speaker asks directly for advice; in direct forms of permission can also occur, such as when the listener says to disregard the advice as appropriate. (2) encouraging the speaker by saying something positive or complimentary. (3) emphasizing the speaker's control, freedom of choice, autonomy, and ability to decide. (4) statements of compassion or sympathy.\nMI Non-Adherent includes: (1) making suggestions, offering solutions or possible actions without first obtaining permission from the speaker (2) directly and unambiguously disagreeing, arguing, blaming, criticizing, or questioning the speaker's honesty. (3) giving orders, commands, or imperatives. (4) statement or event that warns of something or that serves as a cautionary example.\nOther includes any other expression, such as (1) the supporter discloses his/her personal information or experiences. (2) questions that can be answered with a yes/no response or a very restricted range of answers. (3) questions that allow a wide range of possible answers. (4) repetition, rephrasing, or paraphrasing of the seeker's previous statement. (5) repeating or rephrasing the previous statement of the speaker but adding substantial meaning/emphasis to it. (6) educating, providing feedback, or giving an opinion without advising.\nIndentify the response strategy of the supporter in the input conversation.",
                "input": f"seeker says: '{_cont[-300:]}'\nsupporter says: '{_resp[:256]}'",
                "option": ", ".join(options),
                "output": f"{strategy2class[_strat]}\n"
            }
            f.write(json.dumps(line))
            f.write("\n")

    with open("strategy.txt", "w") as f:
        for _cont, _resp, _strat in zip(context, response, strategy):
            line = {"context": _cont[-300:], "response": _resp[:256], "label": strategy2class[_strat]}
            f.write(json.dumps(line))
            f.write("\n")


def process_coherence():
    with open("/home/jiashuo/codes/Muffin/ESConv/DATA/train.txt", "r") as f:
        # with open("/home/csjwang/Documents/Muffin/ESConv/DATA/train.txt", "r") as f:
        reader = f.readlines()
        reader = [json.loads(line)["dialog"] for line in reader]

    seeker_list, response_list, answer_list = [], [], []
    positive_reader_ids = random.sample(range(len(reader)), 150)
    for dialog_idx in positive_reader_ids:
        seeker = ""
        last_speaker = ""
        dialog = reader[dialog_idx]
        for idx, line in enumerate(dialog):
            speaker = line["speaker"]
            content = line["text"]
            if speaker == "usr":
                if last_speaker == "usr":
                    seeker += content
                else:
                    seeker = content
                last_speaker = "usr"
            elif seeker != "":
                if last_speaker == "usr":
                    response = content
                else:
                    response += content
                seeker_list.append(seeker)
                response_list.append(response)
                answer_list.append("Yes\n")
                last_speaker = "sys"

    negative_reader_ids = random.sample([x for x in range(len(reader)) if x not in positive_reader_ids], 150)
    for dialog_idx in negative_reader_ids:
        seeker = ""
        last_speaker = ""
        dialog = reader[dialog_idx]
        for idx, line in enumerate(dialog):
            speaker = line["speaker"]
            content = line["text"]
            if speaker == "usr":
                if last_speaker == "usr":
                    seeker += content
                else:
                    seeker = content
                last_speaker = "usr"
            elif seeker != "":
                while True:
                    random_ids = random.choice(range(len(reader)))
                    random_dialog = reader[random_ids]
                    random_text = random_dialog[random.choice(range(len(random_dialog)))]
                    if random_text["speaker"] == "sys":
                        break
                response = random_text["text"]
                seeker_list.append(seeker)
                response_list.append(response)
                answer_list.append("No\n")
                last_speaker = "sys"

    assert len(seeker_list) == len(response_list) == len(answer_list)
    all_data = []
    for seeker, response, answer in zip(seeker_list, response_list, answer_list):
        line = {
            "task": "coherence",
            "instruction": "Identify whether the supporter response is coherent with what the seeker post. A coherent response will maintain a logical flow of ideas that corresponds with the post. And a coherent response often includes supporting arguments or evidence that directly relate to the post's content.",
            "input": f"seeker says: '{seeker[-300:]}'\nsupporter says: '{response[:256]}'",
            "option": "Yes, No",
            "output": answer
        }
        all_data.append(line)

    random.shuffle(all_data)
    with open("processed_coherence.txt", "w") as f:
        for line in all_data:
            f.write(json.dumps(line))
            f.write("\n")


def create_data_json_file():
    with open("processed_strategy.txt", "r") as f:
        strategy_reader = f.readlines()

    with open("processed_empathy.txt", "r") as f:
        empathy_reader = f.readlines()

    with open("processed_coherence.txt", "r") as f:
        coherence_reader = f.readlines()

    all_data = strategy_reader + empathy_reader + coherence_reader
    random.shuffle(all_data)
    data_num = len(all_data)
    eval_num, test_num = int(data_num * 0.1), int(data_num * 0.1)
    train_num = data_num - eval_num - test_num
    with open("finetune_train.jsonl", "w") as f:
        for line in all_data[:train_num]:
            for i in range(6):
                f.write(line)
    with open("finetune_eval.jsonl", "w") as f:
        for line in all_data[train_num:train_num + eval_num]:
            f.write(line)
    with open("finetune_test.jsonl", "w") as f:
        for line in all_data[train_num + eval_num:]:
            f.write(line)


if __name__ == "__main__":
    print("Specify the function...")
    # process_empathy()
    # process_strategy()
    # process_coherence()
    create_data_json_file()
