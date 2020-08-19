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
```
"productName" : string
```

#### return 
```
"pos" : each token's pos,
"tokens" : the list of ckip result 
```
### /product_keywords

methods=["POST"]

#### parameters

```
"productName" : string
```

#### return 
```
"keywords" : a list , every element in list contains keyword and value , the value means how close between keyword and product
```

### /update_ckip_dict
methods=["POST"]
#### parameters
```
None
```

#### return 
```
"ckip_word_dict" : newest ckiptagger  recommend dictionary
```


### /get_synonyms

methods=["POST"]
#### parameters
```
"word" : string,
"topk" : int,
```

#### return 
```
"synonyms" : a list has topk  synonyms of the word , each element contain string and value, the value means how similar between them. If return list is empty, maybe the word is not in our vocabs or the topk is not an integer.
```

### /train_word2vec
use /train_data/word2vec_train_corpus.pkl to re-train a word2vec model and save it to /model/wordmodel.model

methods=["POST"]
#### parameters
```
None
```

#### return 
```
"success" : boolean
```