
from torch import nn

from models.blocks.encoder_layer_v2 import EncoderLayer
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

    def forward(self, x, s_mask):
        x = self.bert.embeddings(x, s_mask)
        for layer in self.layers:
            x = layer(x, s_mask)

        return x
