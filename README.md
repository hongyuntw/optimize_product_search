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
do tokenlize on product name and return part of speech and token

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
base on product name find it's keyword

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

use /train_data/product_tokens*.txt data construce a new ckiptagger  recommend dictionary and save it in /train_data/ckip_word_dict.pkl

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


### /train_bert
use /train_data/train_data.pkl to re-train a bert classifier and save it to /model/product_weight_model.pkl

methods=["POST"]
#### parameters
```
None
```
#### return 
```
"success" : boolean
```