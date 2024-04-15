import sys
sys.path.append("../")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from utils.qgg_optimizer import GAOptimizer
import torch
import re

def load_quesiton_set(path):
    with open(path) as f:
        return json.load(f)

def feedback_generation(model,tokenizer, input_ids, feedback_times = 3):
    outputs = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #
    input_ids = input_ids.squeeze(0).tolist()        
    # gen_ids = None

    for i in range(feedback_times):
        gened_text = tokenizer.bos_token * (len(outputs)+1)
        gened_ids = tokenizer(gened_text,add_special_tokens=False)['input_ids']            
        input_ids = gened_ids + input_ids
        input_ids = input_ids[:384]
        
        sample_outputs = model.generate(
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device),
            attention_mask=torch.LongTensor([1]*len(input_ids)).unsqueeze(0).to(device),
            max_length=384,
            early_stopping=True,
            temperature=1.0,
            do_sample=True,
            top_p=0.9,
            top_k=10,
            num_beams=1,
            no_repeat_ngram_size=5,
            num_return_sequences=1,
        )
        sample_output = sample_outputs[0]        
        decode_questions = tokenizer.decode(sample_output, skip_special_tokens=False)
        decode_questions = re.sub(re.escape(tokenizer.pad_token),'',decode_questions)
        decode_questions = re.sub(re.escape(tokenizer.eos_token),'',decode_questions)
        decode_questions = re.sub(re.escape('[Q:]'),'',decode_questions)
        decode_questions = re.sub(re.escape('[A:]'),'',decode_questions)
        decode_questions = decode_questions.replace("  "," ")
        if tokenizer.bos_token is not None:
            decode_questions = re.sub(re.escape(tokenizer.bos_token),'',decode_questions)
        decode_questions = decode_questions.strip()
        # if args.dev: print(decode_questions)
        outputs.append(decode_questions)
    return outputs

if __name__ == "__main__":
    out_f = open("chat-gpt-questions-qmst.json","w")
    out = []

    tokenizer = AutoTokenizer.from_pretrained("p208p2002/qmst-qgg")
    model = AutoModelForSeq2SeqLM.from_pretrained("p208p2002/qmst-qgg")
    model = model.to("cuda")

    question_sets = load_quesiton_set("./chat-gpt-questions.json")
    for data in question_sets:
        #
        raw_data = data["data"]
        article = raw_data["article"]

        #
        specific_questions = raw_data["specific_questions"]
        cloze_questions = raw_data["cloze_questions"]
        count_questions = len(specific_questions) + len(cloze_questions)
        #
        encodes = tokenizer.encode(article,add_special_tokens=True)
        input_ids = torch.tensor([encodes])
        input_ids = input_ids.to("cuda")
        pred_questions = feedback_generation(model,tokenizer,input_ids,6)
        print(pred_questions)

        #
        ga_opt = GAOptimizer(len(pred_questions),min(5,count_questions))
        data["qmst"] = pred_questions[:min(5,count_questions)]
        data["qmst_ga"] = ga_opt.optimize(pred_questions,article)
        out.append(data)
    out_f.write(json.dumps(out,ensure_ascii=False))