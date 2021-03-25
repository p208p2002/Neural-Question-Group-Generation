import re
import torch
import torch.nn as nn
# stop_words = ['%','^','_','!','?','.',',','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
_stop_words_en = open('utils/stopwords-en.txt','r',encoding='utf-8').read().split()
_stop_words_sign = open('utils/stopwords-sign.txt','r',encoding='utf-8').read().split()
stop_words = _stop_words_en + _stop_words_sign
def make_stop_word_ids(tokenizer,stop_words=stop_words):
    stop_word_ids = []
    for stop_word in stop_words:
        # w/o prefix
        uncased_stop_word_ids = tokenizer(stop_word,add_special_tokens=False)['input_ids']
        cased_stop_word = stop_word[0].upper() + stop_word[1:]
        cased_stop_word_ids = tokenizer(cased_stop_word,add_special_tokens=False)['input_ids']
        stop_word_ids += (uncased_stop_word_ids + cased_stop_word_ids)

        # word prefix
        prefix_uncased_word = "Ġ" + stop_word
        prefix_cased_word = "Ġ" + cased_stop_word
        
        prefix_uncased_word_id = tokenizer.convert_tokens_to_ids(prefix_uncased_word)
        prefix_cased_word_id = tokenizer.convert_tokens_to_ids(prefix_cased_word)

        assert type(prefix_uncased_word_id) == int
        assert type(prefix_cased_word_id) == int

        stop_word_ids += [prefix_uncased_word_id,prefix_cased_word_id]

    # sign prefix
    for sign in _stop_words_sign:
        prefix_sign = "Ġ"+sign
        prefix_sign_id = tokenizer.convert_tokens_to_ids(prefix_sign)
        assert type(prefix_sign_id) == int
        stop_word_ids += [prefix_sign_id]

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
