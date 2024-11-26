
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device, bert):
        super().__init__()
        self.bert = bert
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, src_mask):
        
        # This is the added version
        '''for param in self.bert.embeddings.parameters():
            param.requires_grad = False'''
        
        trg_mask1 = self.make_no_peak_mask(trg)
        trg = self.bert.embeddings(trg)
        
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask1, src_mask)
        # pass to LM head
        output = self.linear(trg)
        return output
        
    def make_no_peak_mask(self, trg):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(3).to(device)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=device)).type(torch.bool)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
