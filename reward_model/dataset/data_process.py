import os
import sys

sys.path.append("../")
import json
import logging

logging.getLogger().setLevel(logging.INFO)
import pandas as pd
import random
from gpt4free import g4f
import argparse
from tqdm import tqdm

parse = argparse.ArgumentParser()
parse.add_argument("--openai_api_key", type=str, default="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")
parse.add_argument("--openai_api_base", type=str, default="http://localhost:1337/v1")
parse.add_argument("--model_name", type=str, default=g4f.models.gpt_4)
args = parse.parse_args()

os.environ["OPENAI_API_KEY"] = args.openai_api_key
os.environ["OPENAI_API_BASE"] = args.openai_api_base
MODEL_NAME = args.model_name

from chatarena.message import Message
from chatarena.agent import Player
from chatarena.backends import OpenAIChat

proxy = [
    "http://198.204.225.74:17042",
    "http://62.210.214.60:17037",
    "http://204.12.211.114:17018",
    "http://198.204.225.74:17027",
    "http://173.208.196.210:17051"
]

os.environ["G4F_PROXY"] = proxy.pop(0)


def process_empathy():
    """
    there are three aspects related to empathy:
    1. emotional reactions; 2. explorations; 3. interpretations
    we add three scores of all three aspects, and we regard 0 as non-empathy
    :return:
    generate a .txt file with each json_string line in the format: context, response, and score
    """
    random.seed(42)
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
            processed_cont = " ".join(_cont.split()[-128:])
            processed_resp = " ".join(_resp.split()[:128])
            line = {
                "task": "empathy",
                "instruction":
                    "In the context of empathy, there are three key aspects to consider: (1) Emotional "
                    "Reactions – expressing emotions like warmth, compassion, and concern that the peer "
                    "supporter feels after reading the seeker's post; (2) Interpretations – conveying an "
                    "understanding of the feelings and experiences inferred from the seeker's post; (3) "
                    "Explorations – seeking a deeper understanding of the seeker by delving into feelings "
                    "and experiences not explicitly stated in the post. Each aspect can exhibit varying "
                    "degrees of communication—none, weak, or strong—based on the manner in which related "
                    "content is expressed. The overall level of empathy is determined by the highest level "
                    "achieved across these three aspects.\n\nYour task is to identify the level of empathy "
                    "in the Supporter's response within the provided conversation.",
                "input": f"Help-Seeker says: '{processed_cont}'\nSupporter says: '{processed_resp}'\nIdentify "
                         f"the empathy level of the Supporter's response. Choose one of the following options:"
                         f" {', '.join(options)}.",
                "output": f"{empathy_level[_score]}\n"
            }
            f.write(json.dumps(line))
            f.write("\n")

    with open("empathy.txt", "w") as f:
        for _cont, _resp, _score in zip(context, response, score):
            line = {
                "context": " ".join(_cont.split()[-128:]),
                "response": " ".join(_resp.split()[-128:]),
                "label": empathy_level[_score]
            }
            f.write(json.dumps(line))
            f.write("\n")


def process_strategy():
    random.seed(42)
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
            processed_cont = " ".join(_cont.split()[-128:])
            processed_resp = " ".join(_resp.split()[:128])
            line = {
                "task": "strategy",
                "instruction":
                    "Motivational Interviewing involves three distinct strategies. Each strategy can be described as "
                    "follows:\n\n1. MI Adherent Strategies:\n* Advising: Providing advice when directly requested by "
                    "the Help-seeker. This may include indirect forms of permission, such as when the Supporter says "
                    "to disregard the advice as appropriate.\n* Encouraging: Offering positive remarks or compliments "
                    "to the Help-seeker.\n* Emphasizing Autonomy: Highlighting the Supporter's control, freedom of "
                    "choice, and ability to make decisions.\n* Compassion Statements: Expressing sympathy or "
                    "understanding.\n\n2. MI Non-Adherent Strategies:\n* Unsolicited Suggestions: Offering solutions "
                    "or actions without the Supporter's prior consent.\n* Direct Disagreement: Explicitly disagreeing, "
                    "arguing, blaming, criticizing, or questioning the Supporter's honesty.\n * Commands: Issuing "
                    "orders or imperatives.\n* Cautionary Statements: Warning of potential consequences or serving as "
                    "a caution.\n\n3. Other Strategies:\n* Open Questions:\n* Personal Disclosure: The supporter "
                    "shares their own information or experiences.\n* Close-ended Questions: Inquiries answerable with "
                    "a simple 'yes' or 'no' or a limited set of responses.\n* Open-ended Questions: Questions that "
                    "allow for a broad range of answers.\n* Repetition/Rephrasing: Echoing, rewording, or paraphrasing "
                    "the seeker's statements.\n* Enhanced Repetition: Repeating or rephrasing the Supporter's "
                    "statement with added emphasis or meaning.\n* Educational Feedback: Providing information, "
                    "feedback, or opinions without giving direct advice.\n\nYour task is to determine the category of "
                    "the strategy of the Supporter's response.",
                "input": f"Help-seeker says: '{processed_cont}'\nSupporter says: '{processed_resp}'\nIdentify the strategy "
                         f"of the Supporter's response. Choose one of the following options: {', '.join(options)}.",
                "output": f"{strategy2class[_strat]}\n"
            }
            f.write(json.dumps(line))
            f.write("\n")

    with open("strategy.txt", "w") as f:
        for _cont, _resp, _strat in zip(context, response, strategy):
            line = {
                "context": " ".join(_cont.split()[-128:]),
                "response": " ".join(_resp.split()[-128:]),
                "label": strategy2class[_strat]
            }
            f.write(json.dumps(line))
            f.write("\n")


def get_hard(response):
    while True:
        modifier = Player(
            name="modifier",
            role_desc="You are master in language and communication. Change a few words of a sentence making it to "
                      "talk about a totally different or opposite thing.",
            backend=OpenAIChat(temperature=0, model=MODEL_NAME)
        )
        request = Message(
            agent_name="requester",
            content=f"Modify the sentence:\n{response}\n\nChange a few words, especially adjective, verb, and noun in "
                    f"the response, so that the original and the modified one talk about totally different or opposite "
                    f"things/topics. Output the modified sentence only.",
            turn=1
        )
        modified_response = modifier([request])
        if "Sending signal to end the conversation." in modified_response:
            os.environ["G4F_PROXY"] = proxy.pop(0)
        elif modified_response != "":
            modified_response = modified_response.split(":")[-1].strip()
            break
    logging.info(modified_response)
    return modified_response


def process_coherence():
    random.seed(42)
    with open("/home/csjwang/Documents/Muffin/ESConv/DATA/train.txt", "r") as f:
        # with open("/home/jiashuo/codes/Muffin/ESConv/DATA/train.txt", "r") as f:
        reader = f.readlines()
        reader = [json.loads(line)["dialog"] for line in reader]
    writer = open("processed_coherence.txt", "r")
    written_lines = writer.readlines()
    written_num = len(written_lines)
    if written_num % 3 != 0:
        logging.info(
            f"The number of lines in the file is not a multiple of 3. Remove the last {written_num % 3} line(s).")
        written_lines = written_lines[:(-written_num % 3)]
    writer = open("processed_coherence.txt", "w")
    written_num = len(written_lines) // 3
    for line in written_lines:
        writer.write(line)
    context_list, response_list, answer_list = [], [], []
    index_list = random.sample(range(len(reader)), 300)
    line_counter = 0
    for dialog_idx in tqdm(index_list, total=300):
        conv_context_list, last_speaker = [], ""
        dialog = reader[dialog_idx]
        for idx, line in enumerate(dialog):
            speaker = line["speaker"]
            content = line["text"]
            if speaker == "usr":
                if last_speaker != "usr":
                    conv_context = "Help-seeker says: "
                else:
                    conv_context = conv_context_list.pop(-1) + " "
                conv_context += content
                conv_context_list.append(conv_context)
                last_speaker = "usr"
            else:
                if len(conv_context_list) > 0:
                    line_counter += 1
                    ############################# Coherent Response #############################
                    if last_speaker == "usr":
                        pos_response = content
                    else:
                        pos_response += content
                    # context_list.append("\n".join(conv_context_list[-3:]))
                    # response_list.append(pos_response)
                    # answer_list.append("Yes\n")
                    current_context = "\n".join(conv_context_list[-3:])
                    if line_counter > written_num:
                        line = {
                            "task": "coherence",
                            "instruction":
                                "Determine if the supporter's response aligns coherently with the seeker's post. A "
                                "coherent response should maintain a logical flow of ideas in correspondence with the "
                                "post, often including supporting arguments or evidence directly related to the post's "
                                "content.",
                            "input": f"Conversation Context:\n{current_context}\n\nThe last supporter statements is:\n"
                                     f"Supporter says: '{pos_response[:256]}'\nIdentify whether the supporter's last "
                                     f"statement is coherent with the help-seeker's post. Answer 'Yes' if the supporter's "
                                     f"response is coherent with the help-seeker's post, otherwise answer 'No'.",
                            "output": "Yes\n"
                        }
                        writer.write(json.dumps(line))
                        writer.write("\n")
                    ############################# Easy Incoherent Response #############################
                    while True:
                        random_ids = random.choice(range(len(reader)))
                        random_dialog = reader[random_ids]
                        random_text = random_dialog[random.choice(range(len(random_dialog)))]
                        if random_text["speaker"] == "sys":
                            break
                    easy_neg_response = random_text["text"]
                    # context_list.append("\n".join(conv_context_list[-3:]))
                    # response_list.append(easy_neg_response)
                    # answer_list.append("No\n")
                    if line_counter > written_num:
                        line = {
                            "task": "coherence",
                            "instruction":
                                "Determine if the supporter's response aligns coherently with the seeker's post. A "
                                "coherent response should maintain a logical flow of ideas in correspondence with the "
                                "post, often including supporting arguments or evidence directly related to the post's "
                                "content.",
                            "input": f"Conversation Context:\n{current_context}\n\nThe last supporter statements is:\n"
                                     f"Supporter says: '{easy_neg_response[:256]}'\nIdentify whether the supporter's last "
                                     f"statement is coherent with the help-seeker's post. Answer 'Yes' if the supporter's "
                                     f"response is coherent with the help-seeker's post, otherwise answer 'No'.",
                            "output": "No\n"
                        }
                        writer.write(json.dumps(line))
                        writer.write("\n")
                    ############################# Hard Response #############################
                    if line_counter > written_num:
                        hard_neg_response = get_hard(pos_response)
                        # context_list.append("\n".join(conv_context_list[-3:]))
                        # response_list.append(hard_neg_response)
                        # answer_list.append("No\n")
                        line = {
                            "task": "coherence",
                            "instruction":
                                "Determine if the supporter's response aligns coherently with the seeker's post. A "
                                "coherent response should maintain a logical flow of ideas in correspondence with the "
                                "post, often including supporting arguments or evidence directly related to the post's "
                                "content.",
                            "input": f"Conversation Context:\n{current_context}\n\nThe last supporter statements is:\n"
                                     f"Supporter says: '{hard_neg_response[:256]}'\nIdentify whether the supporter's last "
                                     f"statement is coherent with the help-seeker's post. Answer 'Yes' if the supporter's "
                                     f"response is coherent with the help-seeker's post, otherwise answer 'No'.",
                            "output": "No\n"
                        }
                        writer.write(json.dumps(line))
                        writer.write("\n")

                if last_speaker != "sys":
                    conv_context = "Supporter says: "
                else:
                    conv_context = conv_context_list.pop(-1) + " "
                conv_context += content
                conv_context_list.append(conv_context)
                last_speaker = "sys"
    writer.close()

    # assert len(context_list) == len(response_list) == len(answer_list)
    # all_data = []
    # for conv_context, response, answer in zip(context_list, response_list, answer_list):
    #     line = {
    #         "task": "coherence",
    #         "instruction":
    #             "Determine if the supporter's response aligns coherently with the seeker's post. A "
    #             "coherent response should maintain a logical flow of ideas in correspondence with the "
    #             "post, often including supporting arguments or evidence directly related to the post's "
    #             "content.",
    #         "input": f"Conversation Context:\n{conv_context}\n\nThe last supporter statements is:\n"
    #                  f"Supporter says: '{response[:256]}'\nIdentify whether the supporter's last "
    #                  f"statement is coherent with the help-seeker's post. Answer 'Yes' if the supporter's "
    #                  f"response is coherent with the help-seeker's post, otherwise answer 'No'.",
    #         "output": answer
    #     }
    #     all_data.append(line)
    #
    # random.shuffle(all_data)
    # with open("processed_coherence.txt", "w") as f:
    #     for line in all_data:
    #         f.write(json.dumps(line))
    #         f.write("\n")


def create_data_json_file():
    with open("processed_strategy.txt", "r") as f:
        strategy_reader = f.readlines()

    with open("processed_empathy.txt", "r") as f:
        empathy_reader = f.readlines()

    with open("processed_coherence.txt", "r") as f:
        coherence_reader = f.readlines()

    strat_train = strategy_reader[:int(len(strategy_reader) * 0.8)]
    strat_eval = strategy_reader[int(len(strategy_reader) * 0.8):int(len(strategy_reader) * 0.9)]
    strat_test = strategy_reader[int(len(strategy_reader) * 0.9):]

    empathy_train = empathy_reader[:int(len(empathy_reader) * 0.8)]
    empathy_eval = empathy_reader[int(len(empathy_reader) * 0.8):int(len(empathy_reader) * 0.9)]
    empathy_test = empathy_reader[int(len(empathy_reader) * 0.9):]

    coherence_train = coherence_reader[:int(len(coherence_reader) * 0.8)]
    coherence_eval = coherence_reader[int(len(coherence_reader) * 0.8):int(len(coherence_reader) * 0.9)]
    coherence_test = coherence_reader[int(len(coherence_reader) * 0.9):]

    with open("finetune_train.jsonl", "w") as f:
        all_train = strat_train + empathy_train + coherence_train
        random.shuffle(all_train)
        for line in all_train:
            f.write(line)

    with open("finetune_eval.jsonl", "w") as f:
        all_eval = strat_eval + empathy_eval + coherence_eval
        random.shuffle(all_eval)
        for line in all_eval:
            f.write(line)

    with open("finetune_test.jsonl", "w") as f:
        all_test = strat_test + empathy_test + coherence_test
        random.shuffle(all_test)
        for line in all_test:
            f.write(line)


if __name__ == "__main__":
    logging.info("Specify the function...")
    process_empathy()
    process_strategy()
    process_coherence()
    create_data_json_file()
