import sys
sys.path.append("../")
import json
from utils.qgg_optimizer import GAOptimizer



def load_quesiton_set(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    out_f = open("chat-gpt-questions-ga.json","w")
    out = []
    question_sets = load_quesiton_set("./chat-gpt-questions.json")
    for data in question_sets:
        question_sets = data["gpt_question_set"]
        raw_data = data["data"]
        article = raw_data["article"]
        ga_opt = GAOptimizer(len(question_sets),5)
        ga_select_question_group = ga_opt.optimize(question_sets,article)
        print(ga_select_question_group)
        data["ga_select_question_group"] = ga_select_question_group
        out.append(data)
    out_f.write(json.dumps(out,ensure_ascii=False))
