# -*- coding: utf-8 -*
from ckiptagger import NER, POS, WS , data_utils , construct_dictionary
import pandas as pd
import gensim
import pickle
import csv


def initialize():
    global ws
    ws = WS("./data", disable_cuda=False)
    global pos
    pos = POS("./data", disable_cuda=False)
    global ner
    ner =  NER("./data", disable_cuda=False)
    ckip_word_dict = {}
    with open('./train_data/ckip_word_dict.pkl', 'rb') as f:
        ckip_word_dict = pickle.load(f)
    global dictionary
    dictionary = construct_dictionary(ckip_word_dict)
#  model checkpoint 
    WORDMODEL_PATH = './model/wordmodel.model'
    global wordmodel
    wordmodel = gensim.models.Word2Vec.load(WORDMODEL_PATH)

    global check_point
    check_point = './model/product_weight_model.pkl'

    global train_data_filenames
    train_data_filenames = ['product_tokens1.txt']

    global same_word_dict
    same_word_dict = {}
    try:
        with open('./train_data/same_word_dict.pkl', 'rb') as f:
            same_word_dict = pickle.load(f)
    except Exception as e:
        print(e)

        with open('./train_data/同義詞.csv', newline='') as csvFile:
            rows = csv.reader(csvFile)
            count = 0
            for row in rows:
                if count == 0:
                    count += 1
                    continue
                c = 0
                keyword = ''
                for word in row:
                    if c == 0 or c == 1:
                        c += 1
                        continue
                    word = word.lower()
                    if c == 2 and word != '':
                        c += 1
                        keyword = word
                    if word not in same_word_dict and word != '':
                        same_word_dict[word] = keyword.lower()
        with open('./train_data/same_word_dict.pkl', 'wb') as f:
            pickle.dump(same_word_dict, f, pickle.HIGHEST_PROTOCOL)
