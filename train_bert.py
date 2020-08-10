import torch
from transformers import RobertaTokenizer
from transformers import BertTokenizer , BertConfig , BertModel , XLNetTokenizer, XLNetConfig , XLNetModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from torch import nn

def train(trainset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)

    model_path  = './chinese_wwm_pytorch/'
    NUM_LABELS = 2
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path,num_labels=NUM_LABELS)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    EPOCHS = 20 
    for epoch in range(EPOCHS):
        running_loss = 0.0
        total = 0
        correct = 0
        for data in trainloader:
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
            
        torch.save(model.state_dict(),'./bert_product_keyword_binary_model_filter_60k' + str(epoch) + '.pkl')
        print('[epoch %d] loss: %.3f' %(epoch + 1, running_loss))
        print(correct/total)
