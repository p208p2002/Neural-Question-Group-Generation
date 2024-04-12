from openai import OpenAI
from config import OPEN_API_KEY
import json
import random

PROMPT_TEMPLATE = """
Please generate a diverse set of questions from the article. These questions can take various forms, including factual questions, general questions, or fill-in-the-blank types. The question set should be answerable and cover the content of the article. Please list the output in a Markdown list.

Given article:
{article}
""".strip()


def read_jsonl(path):
    with open(path) as f:
        data = f.read().strip().split("\n")
    data = [json.loads(x) for x in data]
    return data


def parse_chatgpt_output(response: str):
    raw_sents = response.strip().split("- ")
    raw_sents.pop(0)
    raw_sents = [x.strip() for x in raw_sents]
    return raw_sents


if __name__ == "__main__":

    client = OpenAI(
        api_key=OPEN_API_KEY
    )
    high = read_jsonl("../datasets/EQG-RACE-PLUS/test/high.jsonl")
    middle = read_jsonl("../datasets/EQG-RACE-PLUS/test/middle.jsonl")
    dataset = high + middle

    random.seed(0)
    random.shuffle(dataset)

    out = []
    for data_idx, data in enumerate(dataset):

        prompt = PROMPT_TEMPLATE.format(article=data["article"])
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )

        message = response.choices[0].message.content
        questions = parse_chatgpt_output(message)

        if len(questions)<5:
            continue

        print(data_idx)
        print("\n".join(questions))
        print("-"*100)

        out.append({
            "data": data,
            "gqt_qgg": questions
        })

        if len(out) == 200:
            break

    with open("chat-gpt-questions.json", "w") as f:
        f.write(json.dumps(out, ensure_ascii=False))
