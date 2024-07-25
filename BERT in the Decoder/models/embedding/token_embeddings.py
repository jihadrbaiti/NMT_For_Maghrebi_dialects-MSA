
from torch import nn
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
bert = AutoModel.from_pretrained("asafaya/bert-base-arabic")

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """ 
        #print('VOCAB SIZE :',vocab_size)
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)