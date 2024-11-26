
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding
from transformers import AutoTokenizer, AutoModel
#bert = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT")


class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob, bert):
        super().__init__()
        self.bert = bert
        #self.emb = TransformerEmbedding(d_model=d_model, max_len=max_len, vocab_size=enc_voc_size, drop_prob=drop_prob, device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x):
        # This is the added version
        '''for param in self.bert.embeddings.parameters():
            param.requires_grad = False'''
    
        s_mask = self.make_src_mask(x)
        x = self.bert.embeddings(x)
        for layer in self.layers:
            x = layer(x, s_mask.unsqueeze(1).unsqueeze(2))
        return x
    
    def make_src_mask(self, src):
        src_mask = (src != 0).int() #.unsqueeze(1).unsqueeze(2)
        return src_mask

    
