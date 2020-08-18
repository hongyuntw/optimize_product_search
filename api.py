# -*- coding: utf-8 -*

from flask import Flask, jsonify, request
from utils import tokenlize
import utils
from predict import find_product_keywords
import predict
import pandas as pd
import globals

app = Flask(__name__)



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
        {
            "keywords": keywords
        }
    )




@app.route("/product_tokens", methods=["POST"])
def product_tokens():
    print(request.form.get('productName'))
    product_name = request.form.get('productName')
    product_name = utils.process_text(product_name)

    tokens, _pos = tokenlize(product_name)
    tokens = tokens[0]
    _pos = _pos[0]
    return jsonify(
        {
            "pos": _pos,
            "tokens":tokens,
        }
    )


def get_keywords(product_name):
    product_name = utils.process_text(product_name)
    keywords = predict.find_product_keywords(product_name)
    print(keywords)
    return keywords


if __name__ == "__main__":
    globals.initialize()
    app.run(debug=False, host='0.0.0.0', port=8787)
