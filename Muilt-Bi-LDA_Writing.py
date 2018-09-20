import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.corpora import Dictionary
from multiprocessing import Pool
import jieba
import random
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 超参数列表
Z = 8  # 主题数
Core = 2  # 核心数
step = 10  # 训练次数
article = 'doc_info.txt'  # 训练语料


def read_article():
    _docs = []
    cnt = 0
    # 所有语料放一个文件就可以，一行代表一篇文章
    with open(article, encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            line = jieba.lcut(line.replace('\n', ''))
            new_line = []
            for word in line:
                if '\u4e00' <= word <= '\u9fff' and len(word) > 1:
                    new_line.append(word)
            _docs.append(new_line)
            cnt += 1
    return _docs


# docs = [['白天', '上班', '下班'], ['作业', '铅笔', '擦皮'], ['写', '作业']]


def make_dict():
    _dictionary = Dictionary(docs)
    max_freq = 0.4
    min_wordcount = 2
    _dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)
    _ = _dictionary[0]

    _corpus = []
    token2id = _dictionary.token2id
    for _doc in docs:
        _corp = []
        for wi in range(len(_doc) - 1):
            if _doc[wi] not in token2id:
                continue
            for wj in range(wi + 1, len(_doc)):
                if _doc[wj] not in token2id:
                    continue
                _corp.append((token2id[_doc[wi]], token2id[_doc[wj]]))
        _corpus.append(_corp)
    return _dictionary, _corpus


# step.3 初始化
def init():
    global corpus
    _doc2topic = np.zeros(shape=[M, Z])
    _topic2word = np.zeros(shape=[Z, N])

    new_corpus = []
    for _doc, _corp in enumerate(corpus):
        new_corp = []
        for w1, w2 in _corp:
            topic = random.randint(0, Z - 1)
            _doc2topic[_doc][topic] += 1
            _topic2word[topic][w1] += 1
            _topic2word[topic][w2] += 1
            new_corp.append((w1, w2, topic))
        new_corpus.append(new_corp)
    corpus = new_corpus
    _topic2word_sum = [sum(_topic2word[tp]) for tp in range(Z)]

    return _doc2topic, _topic2word, _topic2word_sum


def LDA_processing(_core, corpus, N, doc2topic, topic2word, topic2word_sum, alpha, beta):
    size = len(corpus) // Core
    begin = _core * size
    end = (_core + 1) * size if _core < Core - 1 else len(corpus)
    change_topic = np.zeros(shape=[Z, N])
    change_sum = np.zeros(shape=[Z])
    for doc in range(begin, end):
        corp = corpus[doc]
        for c, (w1, w2, old_topic) in enumerate(corp):
            # 移除旧主题
            doc2topic[doc][old_topic] -= 1
            change_topic[old_topic][w1] -= 1
            change_topic[old_topic][w2] -= 1
            change_sum[old_topic] -= 2

            # 计算新主题
            topic_prob = [0 for t in range(Z)]
            word_prob = [0 for t in range(Z)]
            for tp in range(Z):
                topic_prob[tp] = doc2topic[doc][tp] + alpha
                word_prob[tp] = (topic2word[tp][w1] + beta) / (topic2word_sum[tp] + 1)
                word_prob[tp] *= (topic2word[tp][w2] + beta) / (topic2word_sum[tp] + 1 + 1)
            predict = np.array(topic_prob) * np.array(word_prob)
            predict /= sum(predict)
            topic = np.random.choice(np.arange(len(predict)), p=predict)

            # 插入新主题
            doc2topic[doc][topic] += 1
            change_topic[topic][w1] += 1
            change_topic[topic][w2] += 1
            change_sum[topic] += 2

            # 记得更新原语料
            corp[c] = (w1, w2, topic)
    return change_topic, change_sum, begin, end, corpus[begin:end], doc2topic[begin:end]


def cache_data(tup):
    result_topic.append(tup[0])
    result_sum.append(tup[1])
    corpus[tup[2]:tup[3]] = tup[4]
    doc2topic[tup[2]:tup[3]] = tup[5]


def global_updata(_result_topic, _result_sum):
    for change_topic in _result_topic:
        for z in range(Z):
            for n in range(N):
                topic2word[z][n] += change_topic[z][n]
    for change_sum in _result_sum:
        for z in range(Z):
            topic2word_sum[z] += change_sum[z]


if __name__ == "__main__":
    # step.1 读取文章
    docs = read_article()

    # step.2 构造词典
    dictionary, corpus = make_dict()
    M = len(docs)
    N = len(dictionary.keys())
    alpha = 1 / Z
    beta = 1 / N

    # step.3 初始化
    doc2topic, topic2word, topic2word_sum = init()

    # step.4 训练
    for i in range(step):
        pool = Pool(Core)
        result_topic = []
        result_sum = []
        for core in range(Core):
            pool.apply_async(func=LDA_processing,
                             args=(core, corpus, N, doc2topic, topic2word, topic2word_sum, alpha, beta),
                             callback=cache_data)
        pool.close()
        pool.join()
        global_updata(result_topic, result_sum)
        logging.info("epoch %d compelete" % i)

    # step.5 打印结果
    dicts = [{} for i in range(Z)]
    for doc, corp in enumerate(corpus):
        for c, (w1, w2, old_topic) in enumerate(corp):
            if (w1, w2) in dicts[old_topic]:
                dicts[old_topic][(w1, w2)] += 1
            else:
                dicts[old_topic][(w1, w2)] = 1

    id2token = dictionary.id2token
    for dict in dicts:
        f = {v: k for k, v in dict.items()}
        sorted(f, reverse=True)
        n = 0
        print("topic %d:" % dicts.index(dict))
        for key in f:
            print(id2token[f[key][0]] + id2token[f[key][1]], end=', ')
            n += 1
            if n > 10:
                break
        print()
