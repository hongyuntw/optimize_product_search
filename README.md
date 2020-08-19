# search api system

## setup

```
pip install -r requirements.txt
```


## start server

```
python api.py [--download]
```

--download will downlad word2vec model and bert binary classifier model from google drive


## api

### /product_tokens

methods=["POST"]
#### parameters
productName : {string}

#### return 
pos : each token's pos
tokens : the list of ckip result 

### /product_keywords

methods=["POST"]

#### parameters

productName : {string}

#### return 
keywords : a list , every element in list contains keyword and value , the value means how close between keyword and product


### /update_ckip_dict
methods=["POST"]
#### parameters
None

#### return 
ckip_word_dict : newest ckiptagger  recommend dictionary



### /get_synonyms

methods=["POST"]
#### parameters
word : {string}
topk : {int}

#### return 
synonyms : a list contain topk elements of the word's synonyms , each element contain string and value, the value means how similar between them. If return value is empty, maybe the word is not in our vocabs or the topk is not an integer.

