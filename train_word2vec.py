from gensim.models import Word2Vec
import gensim


def train_word2vec(train_corpus):
    wordmodel = Word2Vec(train_corpus, size=300, iter=10, sg=1, min_count=0, hs=1)
    wordmodel.save("word2vec.model")
    return wordmodel


def train_keyword2vec(keyword_corpus):
    keywordmodel = Word2Vec(keyword_corpus, size=300 ,sg=1 , min_count=0, hs =1)
    keywordmodel.save('keyword2vec.model')
    return keywordmodel
