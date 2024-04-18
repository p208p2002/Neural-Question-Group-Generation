import json
import statistics

g_type_keywords = [
    "following",
    "according",
    "passage",
    "learn from the story",
]


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


def analyze_general(gen_data, key):
    out = []
    for data in gen_data:
        # print(data.keys())
        qg_length = len(data[key])
        cloze_count = 0
        for q in data[key]:
            for gk in g_type_keywords:
                if gk in q and not '_' in q:
                    cloze_count += 1
                    break
        out.append(
            cloze_count/qg_length
        )
    return statistics.mean(out)


if __name__ == "__main__":
    chat_gpt_data = json.load(open("chat-gpt-questions-ga.json"))
    qmst_data = json.load(open("chat-gpt-questions-qmst.json"))

    #
    gpt_topk_type_c = analyze_cloze(chat_gpt_data, "gpt_topk")
    gpt_topk_type_g = analyze_general(chat_gpt_data, "gpt_topk")
    gpt_topk_type_f = 1.0 - (gpt_topk_type_c + gpt_topk_type_g)

    #
    gpt_ga_type_c = analyze_cloze(chat_gpt_data, "gpt_ga")
    gpt_ga_type_g = analyze_general(chat_gpt_data, "gpt_ga")
    gpt_ga_type_f = 1.0 - (gpt_ga_type_c + gpt_ga_type_g)

    #
    qmst_type_c = analyze_cloze(qmst_data, "qmst")
    qmst_type_g = analyze_general(qmst_data, "qmst")
    qmst_type_f = 1.0 - (qmst_type_c + qmst_type_g)

    # qmst_ga
    qmst_ga_type_c = analyze_cloze(qmst_data, "qmst_ga")
    qmst_ga_type_g = analyze_general(qmst_data, "qmst_ga")
    qmst_ga_type_f = 1.0 - (qmst_ga_type_c + qmst_ga_type_g)

    # print(f"{gpt_topk=}")
    # print(f"{gpt_ga=}")

    print(f"gpt_topk {gpt_topk_type_c=} {gpt_topk_type_g=} {gpt_topk_type_f=}")
    print(f"gpt_ga {gpt_ga_type_c=} {gpt_ga_type_g=} {gpt_ga_type_f=}")
    print(f"qmst {qmst_type_c=} {qmst_type_g=} {qmst_type_f=}")
    print(f"qmst_ga {qmst_ga_type_c=} {qmst_ga_type_g=} {qmst_ga_type_f=}")

    # # gold label
    # stat_gold_labels = []
    # for data in chat_gpt_data:
    #     data = data["data"]
    #     qg_size = len(data["specific_questions"]+data["cloze_questions"])
    #     count_true_cloze = 0
    #     for q in data["cloze_questions"]:
    #         if '_' in q:
    #             count_true_cloze += 1
    #     stat_gold_labels.append(count_true_cloze/qg_size)

    # gold_label = statistics.mean(stat_gold_labels)
    # print(f"{gold_label=}")
