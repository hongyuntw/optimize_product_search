# -*- coding: utf-8 -*
from ckiptagger import NER, POS, WS , data_utils , construct_dictionary
import pandas as pd
import gensim
import pickle

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
