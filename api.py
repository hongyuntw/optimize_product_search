# -*- coding: utf-8 -*
from flask import Flask, jsonify, request
from utils import tokenlize
import utils
from predict import find_product_keywords , get_synonyms
import predict
import pandas as pd
import globals
import gdown
import sys
import argparse
import prepare_data
from train import train_word2vec , train_bert
import json
from prepare_data import save_product_tokens , add_product

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/')
def hello_world():
    return 'Flask Dockerized'

@app.route('/testing')
def mytest():
    return "testing...."


@app.route("/product_keywords", methods=["POST"])
def product_keywords():
    print(request.form.get('productName'))
    product_name = request.form.get('productName')
    keywords = get_keywords(product_name)
    return jsonify(
        dict(keywords)
    )


@app.route("/product_tokens", methods=["POST"])
def product_tokens():
    print(request.form.get('productName'))
    product_name = request.form.get('productName')
    product_name = utils.process_text(product_name)
    tokens, _pos = tokenlize(product_name)

    return jsonify(
        {
            "pos": _pos,
            "tokens":tokens,
        }
    )


@app.route("/save_product_tokens", methods=["POST"])
def save_product_and_tokens_api():
    product_name = request.form.get('productName')
    tokens = request.form.get('tokens')
    tokens = json.loads(tokens)
    print(tokens,type(tokens))
    print(product_name,type(product_name))
    
    success = save_product_tokens(product_name,tokens)

    return jsonify(
        {
            "success" : success,
        }
    )


@app.route("/add_product", methods=["POST"])
def add_product_api():
    product_name = request.form.get('productName')
    keywords = request.form.get('keywords')
    isbn = request.form.get('isbn')
    category = request.form.get('category')
    print(isbn,product_name, keywords,category)
    
    success = add_product(isbn,product_name, keywords,category)

    return jsonify(
        {
            "success" : success,
        }
    )



# call this will init
@app.route("/get_ckip_dict", methods=["POST"])
def get_ckip_dict_base_on_file_api():

    ckip_word_dict = prepare_data.get_ckip_dict_base_on_file()


    return jsonify(
        dict(sorted(ckip_word_dict.items(), key=lambda x: x[1], reverse=True))
    )


@app.route("/train_word2vec", methods=["POST"])
def train_word2vec_api():
    success = train_word2vec()
    return jsonify(
        {
            "success": success
        }
    )


@app.route("/train_bert", methods=["POST"])
def train_bert_api():
    train_bert()
    return jsonify(
        {
            "success": True
        }
    )


@app.route("/get_synonyms", methods=["POST"])
def get_synonyms():

    words = request.form.get('words')
    words = json.loads(words)
    topk = request.form.get('topk')
    print(words, topk)
    synonyms = predict.get_synonyms(words,topk)

    return jsonify(
        dict(sorted(dict(synonyms).items(), key=lambda x: x[1], reverse=True))
    )





def get_keywords(product_name):
    product_name = utils.process_text(product_name)
    keywords = predict.find_product_keywords(product_name)
    print(keywords)
    return keywords


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', help='download word model' , required=False , default = False , action = 'store_true')
    args = parser.parse_args()

    if args.download:
    #   when start server , download latest model from google drive
        wordmodel_url = 'https://drive.google.com/uc?export=download&id=1-2c6BIK328VF9MKYLhyQ7ZkYOm0OZZKh'
        wordmodel_path = './model/wordmodel.model'
        gdown.download(wordmodel_url, wordmodel_path, quiet=False)

        product_weight_model_url = 'https://drive.google.com/uc?export=download&id=1-NPr3f5RgNBHDzMSdl_a8oNWrOfP6dLs'
        product_weight_path = './model/product_weight_model.pkl'
        gdown.download(product_weight_model_url, product_weight_path, quiet=False)


    globals.initialize()
    app.run(debug=False, host='0.0.0.0', port=8787)
