# -*- coding: utf-8 -*

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer
from transformers import BertTokenizer , BertConfig , BertModel , XLNetTokenizer, XLNetConfig , XLNetModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertForPreTraining
import utils
from torch import nn
import gensim
from utils import tokenlize
import numpy as np
import pickle
from utils import mapping_same_word


# bad_pos_list = ['Nf','Neu','Nc','Nb','WHITESPACE']
# bad_token_list = ['顆', '粒' , '入' , 'ml' , 'g' , 'cm' , 'ml3' , 'gx' , 'x6' , 'gb' , '2l' , 'ml1' , 'x8' , 'x1' , 'kg' , 'cc' , 'km' , 'tb' ,'入組']
LM_PATH = './chinese_wwm_pytorch/'



def get_product_word_weight(t, p, test_product_name):

    bad_token_list = []
    with open('./train_data/bad_token_list.pkl', 'rb') as f:
        bad_token_list = pickle.load(f)

    bad_pos_list = []
    with open('./train_data/bad_pos_list.pkl', 'rb') as f:
        bad_pos_list = pickle.load(f)
    from globals import check_point , wordmodel
    tokenizer = BertTokenizer.from_pretrained(LM_PATH)
    max_seq_length = 50


    train_input_ids = []
    train_attention_mask = []
    train_token_types = []


    new_t = []
    new_p = []


    for i in range(len(t)):
        word = t[i]
        word = word.replace(' ','')
        _pos = p[i]

        if _pos in bad_pos_list or word == '' or word in bad_token_list or len(word)<2 or ( word.isnumeric()) :
            continue
        new_t.append(word)
        new_p.append(_pos)

        # print(new_t)
        
        input_ids = tokenizer.encode(test_product_name, word)
        if len(input_ids) >= max_seq_length:
            continue

        sep_index = input_ids.index(tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b  
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        train_input_ids.append(input_ids)
        train_token_types.append(segment_ids)
        train_attention_mask.append(input_mask)

                
    train_input_ids = np.array(train_input_ids)
    train_token_types  = np.array(train_token_types)
    train_attention_mask = np.array(train_attention_mask)

    class TestDataset(Dataset):
        def __init__(self, 
                    input_ids,
                    token_types, 
                    attention_mask):
            self.input_ids = input_ids
            self.token_type_ids = token_types
            self.attention_mask = attention_mask
        def __getitem__(self,idx):
            inputid = np.array(self.input_ids[idx])
            tokentype = np.array(self.token_type_ids[idx])
            attentionmask = np.array(self.attention_mask[idx])
            return inputid , tokentype , attentionmask
        
        def __len__(self):
            return len(self.input_ids)

    BATCH_SIZE = len(train_input_ids)
    testset = TestDataset(train_input_ids,train_token_types,train_attention_mask)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print("device:", device)

    
    NUM_LABELS = 2
    tokenizer = BertTokenizer.from_pretrained(LM_PATH)
    model = BertForSequenceClassification.from_pretrained(LM_PATH, num_labels=NUM_LABELS)
    try:
        checkpoint_state_dict = torch.load(check_point, map_location=torch.device('cpu'))
    except:
        checkpoint_state_dict = torch.jit.load(check_point, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint_state_dict)
    model = model.to(device)
    model.eval()


    with torch.no_grad():
        for data in testloader:
            tokens_tensors, segments_tensors, masks_tensors = [t.to(device) for t in data]
            outputs = model(input_ids=tokens_tensors.long(), 
                                token_type_ids=segments_tensors.long(), 
                                attention_mask=masks_tensors.long())
            pred = torch.softmax(outputs[0] , dim = -1)
            # pred = torch.softmax(pred, dim = 0)
            pred = pred.cpu().detach().numpy()
            return pred , new_t , new_p



def find_product_keywords(p_name):
    from globals import check_point , wordmodel
    similar_dict = {}
    p_name = utils.process_text(p_name)
    t, p  = tokenlize(p_name)

    # t = t[0]
    # newt = []
    # for _t in t:
    #     newt.append(_t.replace(' ',''))
    # t = newt
    # p = p[0]

    pred , t , p = get_product_word_weight(t , p , p_name )

    # print(pred)
    print(t)

    topk = 10
    for k in range(len(t)):
        print(pred[k],end='\t')
        print(t[k],end='\t' )
        print(p[k])
        
        weight = pred[k][1]
        word = t[k]
        word = mapping_same_word(word)
        # print(weight)
        
        if word in wordmodel.wv.vocab:
            top_similar = wordmodel.wv.most_similar(word,topn=topk)

            print(top_similar)
            
            top_similar_word = [x[0] for x in top_similar]
            top_similar_val = [x[1] for x in top_similar]


            for i in range(len(top_similar_word)):
                similar_word = top_similar_word[i]
                value = top_similar_val[i]
                
                if similar_word in similar_dict:
                    similar_dict[similar_word] += value * weight
                else:
                    similar_dict[similar_word] = value * weight


    return sorted(similar_dict.items(), key=lambda x: x[1], reverse=True)[:-1]




def get_synonyms(words, topk):
    from globals import wordmodel
    try:
        all_synonyms = {}
        topk = int(topk)
        for word in words:
            # print(word)
            try:
                word = utils.process_text(word)
                word = word.replace(' ', '')
                word = mapping_same_word(word)
                synonyms = wordmodel.wv.most_similar(word, topn=topk)
                for s in synonyms:
                    if s in all_synonyms:
                        all_synonyms[s[0]] += s[1]
                    else:
                        all_synonyms[s[0]] = s[1]
            except Exception as e:
                print(e)
                pass
        return all_synonyms
    except Exception as e:
        print(e)
        return []
    
    