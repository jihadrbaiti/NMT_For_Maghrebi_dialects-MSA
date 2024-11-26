
import torch
from torch import nn

from models.model.decoder_v2 import Decoder
from models.model.encoder import Encoder
from transformers import AutoTokenizer, AutoModel
DarijaBert_model = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT")
bert = AutoModel.from_pretrained("asafaya/bert-base-arabic")

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device, bert):
        
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers, device=device, max_len=max_len, enc_voc_size = enc_voc_size)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device, bert=bert)

    def forward(self, src, trg):
        #src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        #src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        #trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * \
                   #self.make_no_peak_mask(trg, trg)
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)

        output = self.decoder(trg, enc_src, src_mask)
    
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
