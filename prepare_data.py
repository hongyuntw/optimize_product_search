# -*- coding: utf-8 -*
import pickle
import pandas as pd
import re
import globals

def clean_supplier_name(text):
    text = str(text)
    text = text.replace(u'\u3000', u' ').replace(u'\xa0', u' ').replace(r'\r\n','')
    text = text.lower()
    pos = text.find('(')
    if pos != -1:
        text = text[:pos]
    return text


def get_train_data():
    train_data = []
    base_path = './train_data/'
    for filename in globals.train_data_filenames:
        data_path = base_path + filename
        with open(data_path,'r',encoding='utf-8') as f:
            product_name_with_isbn = f.readline()
            product_token_result = f.readline()
            token_label = f.readline()
            while product_name_with_isbn and product_token_result and  token_label :
                product_name_with_isbn = product_name_with_isbn.replace('\n','')
                product_token_result = product_token_result.replace('\n','')
                token_label = token_label.replace('\n','')

                if token_label.replace(' ','') == '' or product_token_result.replace(' ','') == ''  or product_name_with_isbn.replace(' ','') == '':
                    continue
                
                try:
                    isbn , product_name = product_name_with_isbn.split('\t') 
                except:
                    print(product_name_with_isbn.split('\t'))
                    print('get error')
                    continue
                    
                tokens = product_token_result.split()
                token_labels = token_label.split()    
                for i in range(len(tokens)):
                    token = tokens[i]
                    try:
                        label = token_labels[i]
                    except:
                        label = 0
                        
                    train_data.append((product_name,token,label))

                product_name_with_isbn = f.readline()
                product_token_result = f.readline()
                token_label = f.readline()
            
    print(len(train_data))
    with open('./train_data/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
    return train_data
    




def update_ckip_dict():
    # with open('./train_data/ckip_word_dict.pkl', 'rb') as f:
    #     ckip_word_dict = pickle.load(f)
    ckip_word_dict = {}
    supplier_df = pd.read_excel('./train_data/關鍵字和供應商.xlsx',sheet_name = '品名和廠商')
    supplier_words = supplier_df['廠商'].apply(clean_supplier_name).tolist()
    ckip_word_dict = {}
    for word in supplier_words:
        if len(word) > 5:
            continue
        if word in ckip_word_dict:
            ckip_word_dict[word] += 1
        else:
            ckip_word_dict[word] = 1

    train_data = get_train_data()
    for data in train_data:
        product_name , token , label = data
        if token == '' or len(token)>4 or re.match("^[A-Za-z0-9]*$", token) or len(token) <2 :
            continue
        if token in ckip_word_dict:
            ckip_word_dict[token] += 1
        else:
            ckip_word_dict[token] = 1

            
    with open('./train_data/ckip_word_dict.pkl', 'wb') as f:
        pickle.dump(ckip_word_dict, f, pickle.HIGHEST_PROTOCOL)

    return ckip_word_dict
        
    


    
