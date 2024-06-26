from openai import OpenAI
from config import OPEN_API_KEY
import json
import random

PROMPT_TEMPLATE = """
Please generate a diverse set of questions based on the article. 
These questions can vary in format, including factual questions or fill-in-the-blank types. 
For instance, "What is the capital of France?" exemplifies a factual question, 
while "The human body is composed of approximately _ percent water." represents the fill-in-the-blank type,
and ""Which of the following is the best title for the passage?"" is a summarization question.
Ensure that the question set is answerable and covers the content of the article. Please list the output in a Markdown list.

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
        gold_label = data["specific_questions"] + data["cloze_questions"]
        if len(gold_label) < 3:
            continue

        prompt = PROMPT_TEMPLATE.format(article=data["article"])
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )

        message = response.choices[0].message.content
        questions = parse_chatgpt_output(message)

        if len(questions) < 5:
            continue

        print(data_idx)
        print("\n".join(questions))
        print("-"*100)

        out.append({
            "data": data,
            "gpt_question_set": questions
        })

        if len(out) == 200:
            break

    with open("chat-gpt-questions.json", "w") as f:
        f.write(json.dumps(out, ensure_ascii=False))
