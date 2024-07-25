from conf1 import *
from util.data_loader import DataLoader1
import pandas as pd

train = pd.read_csv(PATH + 'train' + ext)
test = pd.read_csv(PATH + 'test' + ext)
valid = pd.read_csv(PATH + 'val' + ext)

loader = DataLoader1(train_df1 = train['msa'],
                    train_df2 = train ['nad'], 
                    valid_df1 = valid['msa'], 
                    valid_df2 = valid['nad'], 
                    test_df1 = test['msa'], 
                    test_df2 = test['nad'],
                    max_seq_len = 30
                    )

tokens_train_msa, tokens_train_nad, tokens_val_msa, tokens_val_nad, tokens_test_msa, tokens_test_nad = loader.batch_encode_plus_bert()

train_seq_msa, train_mask_msa, train_seq_nad, train_mask_nad, val_seq_msa, val_mask_msa, val_seq_nad, val_mask_nad, test_seq_msa, test_mask_msa, test_seq_nad, test_mask_nad = loader.convert_int2tensor(tokens_train_msa, tokens_train_nad, tokens_val_msa, tokens_val_nad, tokens_test_msa, tokens_test_nad)

train_iter = loader.create_data_loader(train_seq_msa, train_mask_msa, train_seq_nad, train_mask_nad, batch_size) 
valid_iter = loader.create_data_loader(val_seq_msa, val_mask_msa, val_seq_nad, val_mask_nad, batch_size) 
test_iter = loader.create_data_loader(test_seq_msa, test_mask_msa, test_seq_nad, test_mask_nad, batch_size) 

for batch in train_iter:
    train_seq_msa, train_mask_msa, train_seq_nad, train_mask_nad = batch
    break
    #print(train_seq_nad)

enc_voc_size = 80000
dec_voc_size = 32000

src_pad_idx = 0
trg_pad_idx = 0
trg_sos_idx = 2