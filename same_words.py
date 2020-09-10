import csv
same_word_dict = {}
with open('./train_data/同義詞.csv', newline='') as csvFile:
    rows = csv.reader(csvFile)
    count = 0
    for row in rows:
        if count == 0:
            count += 1
            continue
        c = 0
        keyword = ''
        for word in row:
            if c == 0 or c == 1:
                c += 1
                continue
            word = word.lower()
            if c == 2 and word != '':
                c += 1
                keyword = word
            if word not in same_word_dict and word != '':
                same_word_dict[word] = keyword.lower()

def mapping_same_word(list_of_word):
    if isinstance(list_of_word, list): 
        for i in range(len(list_of_word)):
            word = list_of_word[i]
            if word in same_word_dict:
                list_of_word[i] = same_word_dict[word]
            else:
                list_of_word[i] = list_of_word[i].lower()
        return list_of_word
    else:
        list_of_word = list_of_word.lower()
        if list_of_word in same_word_dict:
            return same_word_dict[list_of_word]
        return list_of_word
        
def add_word_in_same_word_dict(word, keyword):
    try:
        word = word.lower()
        keyword = keyword.lower()
        if word in same_word_dict:
            return True
        same_word_dict[word] = keyword
        return True
    except:
        return False

print(same_word_dict)
print(mapping_same_word(['摩甲刀', '修甲片']))
print(mapping_same_word('搓刀'))

print(mapping_same_word('小G麵'))



