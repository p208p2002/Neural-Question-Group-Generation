import json
import statistics


def analyze_cloze(gen_data, key):
    out = []
    for data in gen_data:
        # print(data.keys())
        qg_length = len(data[key])
        cloze_count = 0
        for q in data[key]:
            if "_" in q:
                cloze_count += 1
        out.append(
            cloze_count/qg_length
        )
    return statistics.mean(out)


if __name__ == "__main__":
    chat_gpt_data = json.load(open("chat-gpt-questions-ga.json"))
    qmst_data = json.load(open("chat-gpt-questions-qmst.json"))

    gpt_topk = analyze_cloze(chat_gpt_data, "gpt_topk")
    gpt_ga = analyze_cloze(chat_gpt_data, "gpt_ga")

    qmst = analyze_cloze(qmst_data,"qmst")
    qmst_ga = analyze_cloze(qmst_data,"qmst_ga")

    print(f"{gpt_topk=}")
    print(f"{gpt_ga=}")
    print(f"{qmst=}")
    print(f"{qmst_ga=}")

    # gold label
    stat_gold_labels = []
    for data in chat_gpt_data:
        data = data["data"]
        qg_size = len(data["specific_questions"]+data["cloze_questions"])
        stat_gold_labels.append(len(data["cloze_questions"])/qg_size)

    gold_label = statistics.mean(stat_gold_labels)
    print(f"{gold_label=}")
