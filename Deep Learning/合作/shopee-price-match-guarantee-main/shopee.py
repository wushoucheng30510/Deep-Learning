# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:58:38 2021

@author: s1972
"""
import os
import math
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import spacy
import matplotlib.pyplot as plt
import sys
import tqdm as tq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import transformers
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW

import re
import nltk
# uncomment the next line if you're running for the first time
# nltk.download('popular')

from customDataset import shopeeImageDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
en = spacy.load('en_core_web_sm')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

parser = argparse.ArgumentParser()
parser.add_argument('--eval', dest='eval', action='store_true', default=False)
args = parser.parse_args()

#%%
def preprocess_data(directory):
    df = pd.read_csv(directory)
    
    # clean the title
    df2 = df['title'].apply(lambda x: preprocess_text(x, flg_stemm=False, flg_lemm=True))
    df.insert(4, "clean_title", df2)
    
    # transfomr the target columns and create a hashtable
    unique_targets = set(df['label_group'])
    dummy_list = list(range(0,len(unique_targets)))
    or_to_new = dict(zip(unique_targets, dummy_list))
    new_to_or = dict(zip(dummy_list, unique_targets))
    df['label_group'] = df['label_group'].map(or_to_new)
    
    df.to_csv('new_train.csv', index=False)
    
    return or_to_new, new_to_or

def preprocess_text(text, flg_stemm=False, flg_lemm=True):
    lst_stopwords = nltk.corpus.stopwords.words("english")
    
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()    
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

def accuracy(data_loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x1, x2, x3, y in data_loader:
            x1 = x1.to(dev)
            x2 = x2.to(dev)
            x3 = x3.to(dev)
            y = y.to(dev)
            
            scores = model(x1, x2, x3, y)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        
        print('Training accuracy:', float(num_correct)/float(num_samples) *100)

def train(model, data_loader, optimizer, crit, epoch):
    model.train()
    epoch_loss = 0
    pbar = tq.tqdm(desc="Epoch {}".format(epoch), total=len(data_loader), unit="batch")
   
    for batch, (image_data, text_seq, text_mask, targets) in enumerate(data_loader):
            
        image_data = image_data.to(dev)
        text_seq = text_seq.to(dev)
        text_mask = text_mask.to(dev)
        targets = targets.to(dev)
        
        model.zero_grad()
        preds_1, preds_2, preds_3 = model(image_data, text_seq, text_mask, targets)
          
        print(preds_1)
        print(targets)
        sys.exit()
        loss1 = crit(preds_1, targets)
        loss2 = crit(preds_2, targets)
        loss3 = crit(preds_3, targets)
        
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad()
        loss.backward()
    
        optimizer.step()
    
        epoch_loss += loss.item()
        pbar.update(1)
    
    pbar.close()
    avg_loss = epoch_loss / len(data_loader)
    return avg_loss

def evaluate(mode, data_loader, crit, epoch):
    print("evaluating validation set")
    model.eval()
    epoch_loss = 0
    
    for batch, (image_data, text_seq, text_mask, targets) in enumerate(data_loader):
        image_data = image_data.to(dev)
        text_seq = text_seq.to(dev)
        text_mask = text_mask.to(dev)
        targets = targets.to(dev)
        
        model.zero_grad()
        preds_1, preds_2, preds_3 = model(image_data, text_seq, text_mask, targets)
        
        loss1 = crit(preds_1, targets)
        loss2 = crit(preds_2, targets)
        loss3 = crit(preds_3, targets)
        
        loss = loss1 + loss2 + loss3
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(data_loader)
    return avg_loss
    
#%%
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class bert_efficientNet(nn.Module):
    def __init__(self, bert, efficient_net):
        
        super().__init__()
        
        # two main models
        self.text_backbone = bert
        self.image_backbone = efficient_net
        
        final_in_features = self.image_backbone._fc.in_features #1536
        self.image_backbone.fc = nn.Identity()
        self.image_backbone.global_pool = nn.Identity()
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.image_drouput = nn.Dropout(0.2)
        self.text_drouput = nn.Dropout(0.2)
        self.concat_drouput = nn.Dropout(0.2)
        
        self.image_fc = nn.Linear(final_in_features, 2048)
        self.text_fc = nn.Linear(768, 2048)
        self.concat_fc = nn.Linear(final_in_features + 768, 2048)
        
        self.image_bn = nn.BatchNorm1d(2048)
        self.text_bn = nn.BatchNorm1d(2048)
        self.concat_bn = nn.BatchNorm1d(2048)
        
        ##############################
        self.image_fc2 = nn.Linear(2048, 2048)
        self.text_fc2 = nn.Linear(2048, 2048)
        self.concat_fc2 = nn.Linear(2048, 2048)
        
        self.image_bn2 = nn.BatchNorm1d(2048)
        self.text_bn2 = nn.BatchNorm1d(2048)
        self.concat_bn2 = nn.BatchNorm1d(2048)
        ###############################
        
        self._init_params()
        
        #self.fc2 = nn.Linear(2048, 11014)
        final_in_features = 2048
        
        self.image_final = ArcMarginProduct(
            in_features = final_in_features,
            out_features = 11014,
            scale = 30,
            margin = 0.5,
            easy_margin = False,
            ls_eps = 0.0
        )
        
        self.text_final = ArcMarginProduct(
            in_features = final_in_features,
            out_features = 11014,
            scale = 30,
            margin = 0.5,
            easy_margin = False,
            ls_eps = 0.0
        )
        
        self.concat_final = ArcMarginProduct(
            in_features = final_in_features,
            out_features = 11014,
            scale = 30,
            margin = 0.5,
            easy_margin = False,
            ls_eps = 0.0
        )

        
    def _init_params(self):
        nn.init.xavier_normal_(self.image_fc.weight)
        nn.init.constant_(self.image_fc.bias, 0)
        nn.init.constant_(self.image_bn.weight, 1)
        nn.init.constant_(self.image_bn.bias, 0)
    
    def forward(self, image, sent_id, mask, label):
        
        # x: image backbone output
        # y: text backbone output
        # z: concatenated output
        
        x = self.image_backbone.extract_features(image)
        x = self.pooling(x).view(x.size(0), -1)
        y = self.text_backbone(sent_id, attention_mask = mask)
        z = torch.cat((x, y[1]), dim=1)

        x = self.image_drouput(x)
        y = self.text_drouput(y[1])
        z = self.concat_drouput(z)
        
        x = self.image_fc(x)
        y = self.text_fc(y)
        z = self.concat_fc(z)
        
        x = self.image_bn(x)
        y = self.text_bn(y)
        z = self.concat_bn(z)
        
        # residual
        residual1 = x
        residual2 = y
        residual3 = z
        
        x = self.image_fc2(x)
        y = self.text_fc2(y)
        z = self.concat_fc2(z)
        
        x = self.image_bn2(x)
        y = self.text_bn2(y)
        z = self.concat_bn2(z)
        
        # residual
        x += residual1
        y += residual2
        z += residual3
        
        x = self.image_final(x, label)
        y = self.text_final(y,label)
        z = self.concat_final(z,label)
        
        
        return x, y, z
    
#%%
if __name__ == "__main__":
    num_epochs = 10
    in_channel = 2
    batch_size = 16
    lr = 0.001
    
    directory = '../shopee/train.csv'
    or_to_new_hash, new_to_or_hash = preprocess_data(directory)

    
    my_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(), # range [0, 255] -> [0.0, 0.1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    
    
    # load images data
    images_dataset = shopeeImageDataset(csv_file = 'new_train.csv',
                                   root_dir = 'train_images',
                                   tokenizer = tokenizer,
                                   transform = my_transforms)
    
    train_set, test_set = torch.utils.data.random_split(images_dataset, [30000, 4250])
    train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
    test_loader =  DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)
        
    
    # import efficientnet b3 model
    image_model = EfficientNet.from_pretrained('efficientnet-b3')
    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    
    image_model.to(dev)
    bert.to(dev)
    
    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False
    for param in image_model.parameters():
        param.requires_grad = False
    
    print("Start building efficientNet+Bert Model...")
    model = bert_efficientNet(bert, image_model)
    model.to(dev)
    crit = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters())
    t_loss_history, v_loss_history = [], []
    
    best_valid_loss = float('inf')
    
    if not args.eval:
        print("\n")
        print("Start training....")
    
        for epoch in range(num_epochs):
            t_loss = train(model, train_loader, optimizer, crit, epoch + 1)
            v_loss = evaluate(model, test_loader, crit, epoch + 1)
            
            #accuracy(train_loader, model)
            
            t_loss_history.append(t_loss)
            v_loss_history.append(v_loss)
            print("\n")
            print(f'Epoch: {epoch + 1:02}\t Train Loss: {t_loss:.3f} | Valid Loss: {v_loss:.3f}')
            
            
            if v_loss < best_valid_loss:
                best_valid_loss = v_loss
                torch.save(model.state_dict(), "best-checkpoint.pt")
            
    model.load_state_dict(torch.load("best-checkpoint.pt"))
    print("\n")
    print("Running test evaluation:")
        

     