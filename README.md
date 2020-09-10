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

### /get_ckip_dict

回傳server端 ckip word dict，若server端也沒有會initial一個ckip word dict

methods=["POST"]
#### parameters
```
None
```

#### return 
```
a dictionary
```


### /get_synonyms

methods=["POST"]
#### parameters
```
"words" : list of string, or string
"topk" : int,
```

#### return 
example 
words : ["長褲","睡褲","童裝"]
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
對list中的每個word找出topk個相似詞，若有重複則取代
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



### /save_product_tokens

methods=["POST"]
#### parameters
```
productName : (string)比利時ACE Q軟糖48g(多口味可選)
tokens : {
    "比利時": false,
    "ace": false,
    "q": false,
    "軟糖": true,
    "48g": false,
    "多口味": false,
    "可選": false
}

會分別存入 train_data (優化 bert 權重判斷 model) call /train_bert
斷詞 ckip word dict (優化斷詞，進而優化單字找相似詞 word2vec model)
```
#### return 
```
"success" : boolean
```

### /add_product
methods=["POST"]
#### parameters
```
isbn : string
productName : string
keywords : string
category : string

存入e7Line商品.csv裡面，目的是當斷詞資料搜集夠多，要進行優化時call /train_word2vec，對裡面每個商品重新斷詞，重新train word2vec model


```
#### return 
```
"success" : boolean
```


### /re_tokenize_all
methods=["POST"]

功能是，當斷詞詞典搜集夠了，到一定程度時，可以對所有商品重新斷詞（照理來講會斷得更好），得到新的斷詞結果之後，會儲存下來。
位置在
```
pos result:
'./train_data/all_products_tokens_without_punck.pkl'

token result:
'./train_data/all_products_pos_without_punck.pkl'

此結果會被應用到 update train_corpus上面。
```

此api斷詞結束後會自行call train_word2vec，優化word2vec model。

#### parameters
```
None
```
#### return 
```
"success" : boolean
```


### /add_bad_tokens
methods=["POST"]

新增bad tokens to list

#### parameters
```
bad_tokens : list of string or single string 
ex: ['mg','cm'] or cm
```
#### return 
```
bad token list in server
[
    "顆",
    "粒",
    "入",
    "ml",
    "g",
    "cm",
    "ml3",
    "gx",
    "x6",
    "gb",
    "2l",
    "ml1",
    "x8",
    "x1",
    "kg",
    "cc",
    "km",
    "tb",
    "mml",
    "None"
]
```



### /add_bad_pos
methods=["POST"]

新增bad pos to list

#### parameters
```
bad_pos : list of string or single string
```
#### return 
```
bad pos list in server
[
    "Nf",
    "Neu",
    "Nc",
    "Nb",
    "WHITESPACE"
]
```


### /add_sameword
methods=["POST"]

新增相似詞／同義字 到詞典中

#### parameters
```
word : string
keyword : string

ex: 
word : 巧克
keyword : 巧克力

上面意思就是指巧克都會被轉換成巧克力
```
#### return 
```
"success" : boolean
```