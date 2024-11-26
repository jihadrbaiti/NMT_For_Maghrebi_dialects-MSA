
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder_v2 import Encoder
from transformers import AutoTokenizer, AutoModel
DarijaBert_model = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT")
bert = AutoModel.from_pretrained("asafaya/bert-base-arabic")

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device, bert_d, bert):
        
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers, bert=bert_d)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        
        enc_src = self.encoder(src)
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_no_peak_mask(trg)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
    
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_no_peak_mask(self, trg):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(3).to(device)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=device)).type(torch.bool)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
