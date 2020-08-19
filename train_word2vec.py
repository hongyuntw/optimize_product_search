from gensim.models import Word2Vec
import gensim
import pickle

def train_word2vec():
    try:
        from globals import wordmodel
        train_corpus = []
        with open('./train_data/word2vec_train_corpus.pkl', 'rb') as f:
            train_corpus = pickle.load(f)
        new_wordmodel = Word2Vec(train_corpus, size=300, iter=10, sg=1, min_count=0, hs=1, window=10)
        new_wordmodel.save("./model/wordmodel.model")
        wordmodel = new_wordmodel
        return True
    except Exception as e:
        print(e)
        return False


# def train_keyword2vec(keyword_corpus):
#     keywordmodel = Word2Vec(keyword_corpus, size=300 ,sg=1 , min_count=0, hs =1)
#     keywordmodel.save('keyword2vec.model')
#     return keywordmodel
