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
example productNanme : 好穿的褲子
```
{
    "九分褲": 0.8544809506246054,
    "嬰兒褲": 0.8439397150800687,
    "搭褲": 0.8213980666825194,
    "內搭褲": 0.7781689287732974,
    "學習褲": 0.7780702565779052
}
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
example 
word : 長褲
topk : 10
```
{
    "打底褲": 0.9404406547546387,
    "童裝": 0.9391651153564453,
    "嬰兒褲": 0.8969577550888062,
    "兒童褲": 0.8661098480224609,
    "專用褲": 0.86543869972229,
    "毛線": 0.812707781791687,
    "針織": 0.8034014105796814,
    "兒童圍兜": 0.7960524559020996,
    "大款": 0.7885699272155762,
    "印刷": 0.7802867293357849
}
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