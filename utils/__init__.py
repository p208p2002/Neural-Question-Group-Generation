import re
import torch
import torch.nn as nn
# stop_words = ['%','^','_','!','?','.',',','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
_stop_words_en = open('utils/stopwords-en.txt','r',encoding='utf-8').read().split()
_stop_words_sign = open('utils/stopwords-sign.txt','r',encoding='utf-8').read().split()
stop_words = _stop_words_en + _stop_words_sign
def make_stop_word_ids(tokenizer,stop_words=stop_words):
    stop_word_ids = []
    # uncased
    for stop_word in stop_words:
        stop_word_id = tokenizer(stop_word,add_special_tokens=False)['input_ids']
        stop_word_ids += stop_word_id

    # cased
    for stop_word in stop_words:
        stop_word = stop_word[0].upper() + stop_word[1:]
        stop_word_id = tokenizer(stop_word,add_special_tokens=False)['input_ids']
        stop_word_ids += stop_word_id

    return list(set(stop_word_ids))

def compute_coverage_score(sents:list,article:str):
    sent = ' '.join(sents)
    sent_list = re.split(r",|\.|\!|\?",sent)
    for sent in sent_list[:]:
        if sent == '': sent_list.remove(sent)
    
    # get sents keywords
    keyword_list = []
    for sent in sent_list[:]:
        sent = sent.lower()
        word_list = sent.split()
        for word in word_list:
            if word not in stop_words:
                keyword_list.append(word)
    
    # process acticle into words and compute coverage
    article_sent_list = re.split(r",|\.|\!|\?",article)
    
    count_article_sent = len(article_sent_list)
    if count_article_sent == 0:
        return 0.0
    
    count_coverage = 0
    for article_sent in article_sent_list:
        article_sent = article_sent.lower().split()
        for keyword in keyword_list:
            if keyword in article_sent:
                count_coverage += 1
                break 
    return count_coverage/count_article_sent

class NegativeCElLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction='mean'):
        super(NegativeCElLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.alpha = 1.0
        self.nll = nn.NLLLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, input, target):
        nsoftmax = self.softmax(input)
        nsoftmax = torch.clamp((1.0 - nsoftmax), min=1e-32)
        return self.nll(torch.log(nsoftmax) * self.alpha, target)

def ignore_pad_token_ids(input_ids,pad_id,ignored_id=-100):
    return [ignored_id if intput_id == pad_id else intput_id for intput_id in input_ids]