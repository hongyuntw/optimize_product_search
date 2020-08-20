import torch
from transformers import BertTokenizer , BertConfig , BertModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from torch import nn
from prepare_data import get_train_data
import  numpy as np

def train_bert():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model_path = './chinese_wwm_pytorch/'
    tokenizer = BertTokenizer.from_pretrained(model_path)

    max_seq_length = 50

    train_data = []
    train_data = get_train_data()

    train_input_ids = []
    train_attention_mask = []
    train_token_types = []
    train_y = []

    from utils import process_text

    for data in train_data:
        product_name , token , label = data
        label = int(label)

        product_name = process_text(product_name)
        input_ids = tokenizer.encode(token, product_name)
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
        train_y.append(label)
    
    train_input_ids = np.array(train_input_ids)
    train_token_types  = np.array(train_token_types)
    train_attention_mask = np.array(train_attention_mask)
    train_y = np.array(train_y)

    print(train_y.shape)
    print(train_input_ids.shape)
    print(train_token_types.shape)
    print(train_attention_mask.shape)

    class TrainDataset(Dataset):
        def __init__(self, 
                    input_ids,
                    token_types, 
                    attention_mask,
                    labels):
            self.input_ids = input_ids
            self.token_type_ids = token_types
            self.attention_mask = attention_mask
            self.labels = labels
        def __getitem__(self,idx):
            inputid = np.array(self.input_ids[idx])
            tokentype = np.array(self.token_type_ids[idx])
            attentionmask = np.array(self.attention_mask[idx])
            labels = self.labels[idx]
            return inputid , tokentype , attentionmask,  labels
        
        def __len__(self):
            return len(self.input_ids)
    
    BATCH_SIZE = 32
    trainset = TrainDataset(train_input_ids,train_token_types,train_attention_mask,train_y)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE , shuffle=True)



    NUM_LABELS = 2
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=NUM_LABELS)
    from globals import check_point
    check_point_state_dict = torch.load(check_point , map_location=torch.device('cpu'))
    model.load_state_dict(check_point_state_dict)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    EPOCHS = 2
    for epoch in range(EPOCHS):
        running_loss = 0.0
        total = 0
        correct = 0
        for (i,data) in enumerate(trainloader):
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
            optimizer.zero_grad()
            outputs = model(input_ids=tokens_tensors, 
                                token_type_ids = segments_tensors, 
                                attention_mask = masks_tensors, 
                                labels=labels)
            loss = outputs[0]

            pred = outputs[1]
            total += pred.size()[0]
            pred = torch.argmax(pred,dim=-1)
            correct += (pred==labels).sum().item()


            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f'\rEpoch [{epoch+1}/{EPOCHS}] {i}/{len(trainloader)} Loss: {running_loss:.4f} Acc : {(correct/total):.3f}', end='')

            
        torch.save(model.state_dict(),check_point)

