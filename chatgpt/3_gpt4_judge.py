from openai import OpenAI
from config import OPEN_API_KEY
import json
import random
import os

PROMPT_TEMPLATE = """
Please read the article and choose one question group from [Question group A] or [Question group B] that you think is better. 
These questions can take various forms, including factual questions, general questions, or fill-in-the-blank types.
"What is the capital of France?" and "The atomic number of carbon is _." are examples of the various forms questions can take.
- Questions within the group that are similar are considered deductions. 
- If the group of questions can cover the context of the article, it is considered a bonus.
- If there are multiple question types within the question group, it is considered a bonus.
- If the meaning of a question in the group is duplicated with another question, it will result in a deduction of points.
Please respond with "[Question group A]" if you prefer that group, or "[Question group B]" if you prefer the other.

Here is the content that needs to be evaluated:

{article}

[Question group A]
{first_question_group}
--------------------

[Question group B]
{second_question_group}
--------------------

Please tell me which one is better?
""".strip()


def get_question_group(path, key):
    with open(path) as f:
        out = []
        data = json.load(f)
        for x in data:
            out.append(x[key])
        return out


def get_test_cases(path):
    out = []
    with open(path) as f:
        data = json.load(f)
        for x in data:
            x = x["data"]
            gold_label = x["specific_questions"] + x["cloze_questions"]
            random.shuffle(gold_label)
            out.append({
                "article": x["article"],
                "gold_label": gold_label
            })
    return out


def run_eval(test_cases, challenger, save):
    result = []
    win_count = 0
    for idx, test_case in enumerate(test_cases):
        article = test_case["article"]
        gold_label = test_case["gold_label"]
        if len(gold_label) < 3:
            continue
        count_gold_label_quesions = len(test_case["gold_label"])
        gold_label = "\n".join(gold_label)

        challenger_data = challenger[idx][:count_gold_label_quesions]
        challenger_qgg = "\n".join(challenger_data)

        prompt = PROMPT_TEMPLATE.format(
            article=article,
            # first_question_group=gold_label,
            # second_question_group=challenger_qgg
            first_question_group=challenger_qgg,
            second_question_group=gold_label
        )

        # print(prompt)

        response = client.chat.completions.create(
            temperature=0.0,
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        message = response.choices[0].message.content

        win = None
        if "group A" in message:
            win = 1
            win_count += 1
        else:
            win = 2

        result.append(win)

        try:
            win_rate = min(win_count/len(result), 1.0)
        except ZeroDivisionError:
            win_rate = None
        print(idx, save, "winner:", win, f"{win_rate=}")
        print("-"*100)

        # if idx == 20:
        #     break
    win_rate = win_count/len(result)
    print(f"{save} {win_rate=}")
    with open(save, "w") as f:
        f.write(json.dumps({
            "win_rate": win_rate,
            "result": result
        }))
    return result


if __name__ == "__main__":
    random.seed(0)

    client = OpenAI(
        api_key=OPEN_API_KEY
    )
    chatgpt_topk_data = get_question_group(
        "./chat-gpt-questions-ga.json", "gpt_topk")
    chatgpt_ga_data = get_question_group(
        "./chat-gpt-questions-ga.json", "gpt_ga")
    qmst_topk_data = get_question_group(
        "./chat-gpt-questions-qmst.json", "qmst")
    qmst_ga_data = get_question_group(
        "./chat-gpt-questions-qmst.json", "qmst_ga")
    test_cases = get_test_cases("./chat-gpt-questions.json")

    os.makedirs("result", exist_ok=True)

    run_eval(test_cases, qmst_topk_data, "result/qmst_topk.json")
    run_eval(test_cases, qmst_ga_data, "result/qmst_ga.json")
    run_eval(test_cases, chatgpt_topk_data, "result/chatgpt_topk.json")
    run_eval(test_cases, chatgpt_ga_data, "result/chatgpt_ga.json")
