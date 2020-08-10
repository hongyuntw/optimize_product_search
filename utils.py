
import re

def remove_punctuation(text):
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）」「、。：；’‘……￥·"""
    dicts={i:' ' for i in punctuation}
    punc_table=str.maketrans(dicts)
    text=text.translate(punc_table)
    return text
def process_text(text):
    text = str(text)
    text = strQ2B(text)
    text = text.replace(u'\u3000', u' ').replace(u'\xa0', u' ').replace(r'\r\n','')
    text = text.lower()
    text = remove_punctuation(text)
    return text

def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def clean_supplier_name(text):
    text = str(text)
    text = text.replace(u'\u3000', u' ').replace(u'\xa0', u' ').replace(r'\r\n','')
    text = text.lower()
    pos = text.find('(')
    if pos != -1:
        text = text[:pos]

    return text


def get_keywords(text):
    keywords = []
    text = str(text)
    text = strQ2B(text)
    text = text.replace('/','\\').replace('NULL','').replace('nan','').replace(' ','')
    text = text.lower()
    
    if text == '':
        return []
    keys = text.split("\\")
    # print(keys)
    for key in keys:
        if(key not in keywords and key != '' and key != ' ' and len(key) > 1):
            keywords.append(key)

    return keywords