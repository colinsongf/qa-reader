import os
import sys
import gensim
import numpy as np


class Singleton(type):
    """
    reference: https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons

    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BenebotVector(metaclass=Singleton):

    model = None

    def __init__(self, model_path):
        if not self.model:
            # print('load word vector model')
            self.model = gensim.models.Word2Vec.load(model_path)
        # self.dim = len(self.getVectorByWord('中国'))
        self.dim = self.model.vector_size

    def hasWord(self, word):
        return self.model.__contains__(word.strip())

    def getVectorByWord(self, word):
        result = []
        if self.model.__contains__(word.strip()):
            vector = self.model.__getitem__(word.strip())
            result = [v for v in vector]
        else:
            result = list(np.random.randn(self.dim))
        return result

    def getSimilarWords(self, word):
        result = model.most_similar(word.strip())
        return result

    def calWordSimilarity(self, word1, word2):
        result = model.similarity(word1.strip(), word2.strip())
        return result

    def getVectorBySentence(self, sentence):
        words = sentence.strip().split(' ')
        vectors = []
        for word in words:
            vector = self.getVectorByWord(word)
            if not vector:
                vector = [0.0] * self.dim
            vectors.append(vector)
        result = [0.0] * self.dim
        if vectors:
            result = np.mean(vectors, axis=0)
        return result

    def getVectorByWeightSentence(self, weight_sentence):
        vectors = []
        value_sum = 0.0
        for key, value in weight_sentence.items():
            vector = self.getVectorByWord(key)
            if not vector:
                vector = np.array([0.0] * self.dim)
            vector = np.array(vector) * value
            vectors.append(vector)
            value_sum += value
        result = np.array([0.0] * self.dim)
        if vectors:
            result = np.sum(vectors, axis=0) / value_sum
        return result


if __name__ == '__main__':
    bv = BenebotVector('word2vec.bin')
    result = bv.getVectorBySentence('我 是 中国 人')
    print(result)
