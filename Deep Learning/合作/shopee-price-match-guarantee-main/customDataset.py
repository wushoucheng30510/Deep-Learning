# -*- coding: utf-8 -*-
"""
Created on Wed May 19 00:06:36 2021

@author: s1972
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%
class shopeeImageDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, tokenizer, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        self.tokens_train = self.tokenizer.batch_encode_plus(self.annotations['clean_title'].tolist(),
                                                             max_length = 20,
                                                             pad_to_max_length=True,
                                                             truncation=True,
                                                             return_token_type_ids=False)

        self.text_train_seq = torch.tensor(self.tokens_train['input_ids'])
        self.text_train_mask = torch.tensor(self.tokens_train['attention_mask'])
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        
        # image part
        img_path = os.path.join(self.root_dir, str(self.annotations.iloc[index, 1]))
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 5]))
        
        if self.transform:
            image = self.transform(image)
        
        # text part
        t_seq = self.text_train_seq[index]
        t_mask = self.text_train_mask[index]
        
        return (image, t_seq, t_mask, y_label)