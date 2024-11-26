#from pickletools import markobject
#from pyrsistent import s
#from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k
#from spacy.tokenizer import Tokenizer
#from spacy.lang.en import English
#from spacy.lang.ar import Arabic
#import re
#from torchtext.datasets import IWSLT2016
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DataLoader1:
    def __init__(self, train_df1 ,train_df2, valid_df1, valid_df2, test_df1, test_df2, max_seq_len):
        self.train_df1 = train_df1
        self.train_df2 = train_df2
        self.valid_df1 = valid_df1
        self.valid_df2 = valid_df2
        self.test_df1 = test_df1
        self.test_df2 = test_df2
        self.DarijaBERT_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
        self.DarijaBert_model = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        self.bert = AutoModel.from_pretrained("asafaya/bert-base-arabic")
        self.max_seq_len = max_seq_len

    def  batch_encode_plus_bert(self):
        
        tokens_train_msa = self.tokenizer.batch_encode_plus(
            self.train_df1.tolist(),
            max_length = self.max_seq_len,
            padding=True,
            truncation=True,
            return_token_type_ids=False
            )
        
        tokens_train_nad = self.DarijaBERT_tokenizer.batch_encode_plus(
            self.train_df2.tolist(),
            max_length = self.max_seq_len,
            padding=True,
            truncation=True,
            return_token_type_ids=False
            )
        # tokenize and encode sequences in the validation set
        tokens_val_msa = self.tokenizer.batch_encode_plus(
            self.valid_df1.tolist(),
            max_length = self.max_seq_len,
            padding=True,
            truncation=True,
            return_token_type_ids=False
            )
        
        tokens_val_nad = self.DarijaBERT_tokenizer.batch_encode_plus(
            self.valid_df2.tolist(),
            max_length = self.max_seq_len,
            padding=True,
            truncation=True,
            return_token_type_ids=False
            )
        # tokenize and encode sequences in the test set
        tokens_test_msa = self.tokenizer.batch_encode_plus(
            self.test_df1.tolist(),
            max_length = self.max_seq_len,
            padding=True,
            truncation=True,
            return_token_type_ids=False
            )

        tokens_test_nad = self.DarijaBERT_tokenizer.batch_encode_plus(
            self.test_df2.tolist(),
            max_length = self.max_seq_len,
            padding=True,
            truncation=True,
            return_token_type_ids=False
            )
        return tokens_train_msa, tokens_train_nad, tokens_val_msa, tokens_val_nad, tokens_test_msa, tokens_test_nad

    def convert_int2tensor(self, train_df1, train_df2, valid_df1, valid_df2, test_df1, test_df2):
        train_seq_msa = torch.tensor(train_df1['input_ids'], device=device)
        #train_mask_msa = torch.tensor(train_df1['attention_mask'], device=device)
        train_seq_nad = torch.tensor(train_df2['input_ids'], device=device)
        #train_mask_nad = torch.tensor(train_df2['attention_mask'], device=device)

        # for validation set
        val_seq_msa = torch.tensor(valid_df1['input_ids'], device=device)
        #val_mask_msa = torch.tensor(valid_df1['attention_mask'], device=device)
        val_seq_nad = torch.tensor(valid_df2['input_ids'], device=device)
        #val_mask_nad = torch.tensor(valid_df2['attention_mask'], device=device)

        # for test set
        test_seq_msa = torch.tensor(test_df1['input_ids'], device=device)
        #test_mask_msa = torch.tensor(test_df1['attention_mask'], device=device)
        test_seq_nad = torch.tensor(test_df2['input_ids'], device=device)
        #test_mask_nad = torch.tensor(test_df2['attention_mask'], device=device)
    
        return train_seq_msa, train_seq_nad, val_seq_msa, val_seq_nad, test_seq_msa, test_seq_nad

    def create_data_loader(self, seq_tensor1, seq_tensor2, batch_size):
        data = TensorDataset(seq_tensor1, seq_tensor2)
        sampler = RandomSampler(data)
        data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return data_loader
    