from flask import Flask, jsonify, request
import utils
from predict import find_product_keywords
from ckiptagger import NER, POS, WS , data_utils , construct_dictionary


app = Flask(__name__)



@app.route('/')
def hello_world():
    return 'Flask Dockerized'

@app.route('/testing')
def mytest():
    return "testing...."


@app.route("/product_keywords", methods=["POST"])
def product_keywords():
    data = request.get_json(force=True)

    product_name = data["productName"]
    keywords = get_keywords(product_name)
    return jsonify(
        {
            "keywords": keywords
        }
    )

# ckip part

ws = WS("./data", disable_cuda=False)
pos = POS("./data", disable_cuda=False)
ner = NER("./data", disable_cuda=False)

supplier_df = pd.read_excel('./關鍵字和供應商.xlsx',sheet_name = '品名和廠商')
supplier_word = supplier_df['廠商'].apply(clean_supplier_name).tolist()
supplier_word = list(set(supplier_word))
mydict = dict.fromkeys(supplier_word, 1)
dictionary = construct_dictionary(mydict)
def tokenlize(text):
    tokens = ws([text],recommend_dictionary = dictionary)
    word_pos = pos(tokens)
    return tokens, word_pos
    

def get_keywords(product_name):
    product_name = utils.process_text(product_name)
    return predict.find_product_keywords(p_name)



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8787)