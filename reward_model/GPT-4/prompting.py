import openai
import asyncio
import json
import random
import time
from tqdm import tqdm
import os

openai.api_base = "https://api.chatanywhere.cn/v1"
# openai.api_base = "https://api.chatanywhere.cn/v1/chat/completions"
openai.api_key = "sk-GIZ7FxZAcRIYtkUihDVwrKfG98HNT85Q0PzEUQcs4wQME8AY"
FILE_NAME = "test_GPT3.5.txt"


# FILE_NAME = "test_GPT4.txt"


async def acreate_response(messages, retry=3):
    for _ in range(retry):
        try:
            # response = await openai.ChatCompletion.acreate(messages=messages, model="gpt-4")
            response = await openai.ChatCompletion.acreate(messages=messages, model="gpt-3.5-turbo-0613")
            return response
        except openai.error.RateLimitError:
            sleep_time = random.choice(range(1, 61))
            print(f'Rate limit error, waiting for {sleep_time} second...')
            await asyncio.sleep(sleep_time)
        except openai.error.APIError:
            sleep_time = random.choice(range(1, 61))
            print(f'API error, waiting for {sleep_time} second...')
            await asyncio.sleep(sleep_time)
        except openai.error.Timeout:
            sleep_time = random.choice(range(1, 61))
            print(f'Timeout error, waiting for {sleep_time} second...')
            await asyncio.sleep(sleep_time)
    return None


async def a_process_batch(batch, retry=3):
    tasks = [acreate_response(message, retry=retry) for message in batch]
    return await asyncio.gather(*tasks)


def create_response(messages, retry=3):
    for _ in range(retry):
        try:
            # response = openai.ChatCompletion.create(messages=messages, model="gpt-4")
            response = openai.ChatCompletion.create(messages=messages, model="gpt-3.5-turbo-0613")
            return response
        except openai.error.RateLimitError:
            sleep_time = random.choice(range(1, 61))
            print(f'Rate limit error, waiting for {sleep_time} second...')
            time.sleep(sleep_time)
        except openai.error.APIError:
            sleep_time = random.choice(range(1, 61))
            print(f'API error, waiting for {sleep_time} second...')
            time.sleep(sleep_time)
        except openai.error.Timeout:
            sleep_time = random.choice(range(1, 61))
            print(f'Timeout error, waiting for {sleep_time} second...')
            time.sleep(sleep_time)
    return None


def process_batch(batch, retry=3):
    tasks = [create_response(message, retry=retry) for message in batch]
    return tasks


# Usage
# responses = asyncio.run(process_batch([""], retry=3))
# for response in responses:
#     print(response["data"][0]["embedding"])


class GPTFeedback(object):
    def __init__(self):
        self.template = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Options:\n{option}\n\n### Answer:\n"

    def feedback_instances(self):

        if os.path.exists(FILE_NAME):
            with open(FILE_NAME, "r") as f:
                lines = f.readlines()
            current_length = len(lines)
        else:
            current_length = 0

        with open('../dataset/finetune_test.jsonl', "r") as file:
            reader = [json.loads(line) for line in file.readlines()][current_length:]

        message_batch = []
        real_labels = []
        for line in tqdm(reader, total=len(reader)):
            user_message = {
                "role": "user",
                "content": self.template.format(
                    instruction=line["instruction"],
                    input=line["input"],
                    option=line["option"]
                )
            }
            message_batch.append([user_message])
            real_labels.append(line["output"])

        with open(FILE_NAME, "a+") as f:
            for message, real in tqdm(zip(message_batch, real_labels), total=len(message_batch)):
                answer = create_response(message, retry=100)
                pred = answer.choices[0].message.content
                f.write(f"{pred} ## {real}")

    def feedback_batch(self):
        with open('../dataset/finetune_test.jsonl', "r") as file:
            reader = [json.loads(line) for line in file.readlines()]

        message_batch = []
        real_labels = []
        for line in reader:
            user_message = {
                "role": "user",
                "content": self.template.format(
                    instruction=line["instruction"],
                    input=line["input"],
                    option=line["option"]
                )
            }
            message_batch.append([user_message])
            real_labels.append(line["output"])

        answers = asyncio.run(a_process_batch(message_batch, retry=100))
        predictions = []
        for answer in answers:
            pred = answer.choices[0].message.content
            predictions.append(pred)
        with open(FILE_NAME, "w") as f:
            for pred, real in zip(predictions, real_labels):
                f.write(f"{pred} ## {real}")


func = GPTFeedback()
func.feedback_instances()
