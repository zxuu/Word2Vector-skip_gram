import jieba
import re
import logging
import sys
import gensim.models as word2vec
from gensim.models.word2vec import LineSentence, logger

def pre_do(filePath, fileSegWordDonePath):
    # 将每一行文本依次存放到一个列表
    fileTrainRead = []
    with open(filePath, encoding='utf-8') as fileTrainRaw:
        for line in fileTrainRaw:
            fileTrainRead.append(line)

    # 去除标点符号
    fileTrainClean = []
    remove_chars = '[·’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    for i in range(len(fileTrainRead)):
        string = re.sub(remove_chars, "", fileTrainRead[i])
        fileTrainClean.append(string)

    # 用jieba进行分词
    fileTrainSeg = []
    # file_userDict = 'dict.txt'  # 自定义的词典
    # jieba.load_userdict(file_userDict)
    for i in range(len(fileTrainClean)):
        fileTrainSeg.append([' '.join(jieba.cut(fileTrainClean[i]))])
        # if i % 100 == 0:  # 每处理100个就打印一次
        #     print(i)

    with open(fileSegWordDonePath, 'wb') as fW:
        for i in range(len(fileTrainSeg)):
            fW.write(fileTrainSeg[i][0].encode('utf-8'))
            fW.write('\n'.encode("utf-8"))

def train_word2vec(dataset_path, out_vector):
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # 把语料变成句子集合
    sentences = LineSentence(dataset_path)
    # 训练word2vec模型（size为向量维度，window为词向量上下文最大距离，min_count需要计算词向量的最小词频）
    model = word2vec.Word2Vec(sentences, size=50, sg=1, window=10, min_count=5, workers=4, iter=5)
    # (iter随机梯度下降法中迭代的最大次数，sg为1是Skip-Gram模型)
    # 保存word2vec模型（创建临时文件以便以后增量训练）
    model.save("word2vec.model")
    model.wv.save_word2vec_format(out_vector, binary=False)


# 加载模型
def load_word2vec_model(w2v_path):
    model = word2vec.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    return model


# 计算词语的相似词
def calculate_most_similar(model, word):
    similar_words = model.most_similar(word)
    print(word)
    for term in similar_words:
        print(term[0], term[1])


if __name__ == '__main__':
    dataset_path = "corpusSegDone2.txt"
    out_vector = 'corpusSegDone2.vector'
    filePath = './data/MSRA-ner.txt'
    pre_do(filePath, dataset_path)
    train_word2vec(dataset_path, out_vector)
