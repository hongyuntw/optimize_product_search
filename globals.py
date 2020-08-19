# -*- coding: utf-8 -*

from ckiptagger import NER, POS, WS , data_utils , construct_dictionary
import pandas as pd
import gensim

def clean_supplier_name(text):
    text = str(text)
    text = text.replace(u'\u3000', u' ').replace(u'\xa0', u' ').replace(r'\r\n','')
    text = text.lower()
    pos = text.find('(')
    if pos != -1:
        text = text[:pos]
    return text



def initialize():
    global ws
    ws = WS("./data", disable_cuda=False)
    global pos
    pos = POS("./data", disable_cuda=False)
    global ner
    ner =  NER("./data", disable_cuda=False)
    supplier_df = pd.read_excel('./關鍵字和供應商.xlsx',sheet_name = '品名和廠商')
    supplier_word = supplier_df['廠商'].apply(clean_supplier_name).tolist()
    supplier_word = list(set(supplier_word))
    mydict = dict.fromkeys(supplier_word, 1)
    global dictionary
    dictionary = construct_dictionary(mydict)
#  model checkpoint 
    WORDMODEL_PATH = './model/wordmodel.model'
    global wordmodel
    wordmodel = gensim.models.Word2Vec.load(WORDMODEL_PATH)

    global check_point
    check_point = './model/product_weight_model.pkl'
