import os
import sys

sys.path.append("../")
import json
import logging

logging.getLogger().setLevel(logging.INFO)
from gpt4free import g4f
from tqdm import tqdm
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--openai_api_key", type=str, default="sk-GIZ7FxZAcRIYtkUihDVwrKfG98HNT85Q0PzEUQcs4wQME8AY")
parse.add_argument("--openai_api_base", type=str, default="https://api.chatanywhere.com.cn/v1")
# parse.add_argument("--model_name", type=str, default=g4f.models.gpt_4)
parse.add_argument("--model_name", type=str, default="gpt-4-1106-preview")
# parse.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
args = parse.parse_args()

os.environ["OPENAI_API_KEY"] = args.openai_api_key
os.environ["OPENAI_API_BASE"] = args.openai_api_base
MODEL_NAME = args.model_name
logging.info(f"Using model: {MODEL_NAME}")
from chatarena.message import Message
from chatarena.agent import Player
from chatarena.backends import OpenAIChat


# proxy = [
#     "http://198.204.225.74:17042",
#     "http://62.210.214.60:17037",
#     "http://204.12.211.114:17018",
#     "http://198.204.225.74:17027",
#     "http://173.208.196.210:17051"
# ]


class ClassifyPlayer():
    def __init__(self):
        self.player = Player(
            name="Classifier",
            role_desc="Below is an instruction that describes a task, paired with an input that provides further "
                      "context. Write a response that appropriately completes the request.",
            backend=OpenAIChat(temperature=0, model=MODEL_NAME)
        )
        self.template = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\nOnly provide the answer.\n"
        # self.proxy_idx = 3
        # os.environ["G4F_PROXY"] = proxy[self.proxy_idx]

    def obtain_answer(self, instance):
        request = self.template.format(
            instruction=instance["instruction"],
            input=instance["input"]
        )
        while True:
            message = Message(
                agent_name="Task",
                content=request,
                turn=1
            )
            answer = self.player([message])
            # if "Sending signal to end the conversation." in answer:
            #     self.proxy_idx += 1
            #     self.proxy_idx = self.proxy_idx % len(proxy)
            #     os.environ["G4F_PROXY"] = proxy[self.proxy_idx]
            # elif "<PHIND_SPAN_BEGIN>" in answer:
            #     continue
            # elif answer != "":
            if answer != "":
                break
        logging.info(f"Answer: {answer}")
        return answer


def test_GPT():
    # test_file = "../dataset/finetune_test.jsonl"
    test_file = "../rebuttal_experiments/instruction_empathy.jsonl"
    save_path = f"../rebuttal_experiments/feedback_GPT4.jsonl"
    # save_path = f"test_result_GPT3.5.txt"
    with open(test_file, "r") as pf:
        data_reader = [json.loads(line) for line in pf.readlines()]
    if os.path.exists(save_path):
        with open(save_path, "r") as pf:
            data_writer = pf.readlines()
    else:
        data_writer = []

    writer = open(save_path, "w")
    for line in data_writer:
        writer.write(line)

    player = ClassifyPlayer()
    for instance in tqdm(data_reader[len(data_writer):]):
        # real_label = instance["output"]
        answer = player.obtain_answer(instance)
        # temp = {"real_label": real_label, "answer": answer}
        instance.update({"answer": answer})
        writer.write(f"{json.dumps(instance)}\n")

    writer.close()


if __name__ == "__main__":
    test_GPT()
