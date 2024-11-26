import spacy
from arabert.preprocess import ArabertPreprocessor
import re
from nltk.tokenize import word_tokenize
import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.ar import Arabic
from farasa.segmenter import FarasaSegmenter 
import pyarabic.trans as trans
from transformers import AutoTokenizer, AutoModel

class Tokenizer1:

    model_name="bert-base-arabert"
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    
    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')
        self.enNLP = English()
        self.arNLP = Arabic()
        self.enTokenizer = Tokenizer(self.enNLP.vocab)
        self.arTokenizer =  Tokenizer(self.arNLP.vocab)

    def myTokenizerEN(self, x):
        #print([word.text for word in 
               #Tokenizer(English().vocab)(re.sub(r"\s+\s+"," ",re.sub(r"[\.\'\`\"\r+\n+]"," ",x.lower())).strip())])
        return  [word.text for word in 
               Tokenizer(English().vocab)(re.sub(r"\s+\s+"," ",re.sub(r"[\.\'\`\"\r+\n+]"," ",x.lower())).strip())]
    def myTokenizerAR(self, x):
        print([word.text for word in 
                Tokenizer(Arabic().vocab)(re.sub(r"\s+\s+"," ",re.sub(r"[\.\'\`\"\r+\n+]"," ",x.lower())).strip())])
        return  [word.text for word in 
                Tokenizer(Arabic().vocab)(re.sub(r"\s+\s+"," ",re.sub(r"[\.\'\`\"\r+\n+]"," ",x.lower())).strip())]

    def remove_multiple_whitespace(string_in):
        s = string_in.strip()
        return re.sub(r"\s+", " ", s)

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        a = [tok.text for tok in self.spacy_en.tokenizer(text)]
        #print('English :',a)
        return a

    def tokenize_ar(self,text):
        model_name="bert-base-arabert"
        arabert_prep = ArabertPreprocessor(model_name=model_name)
        a = arabert_prep.preprocess(text)
        a = Tokenizer1.remove_multiple_whitespace(re.sub(r"\+", " ", a)).split(' ')
        print('Arabic :',a)
        return a

    def tokenize_ar1(self, text):
        #print('Arabic',text.split(' '))
        return text.split(' ')

    def tokenize_en1(self, text):
        #print('English :', text.split(' '))
        return text.split(' ')
    
    def tokenize_farasa(self, text):
        segmenter = FarasaSegmenter()
        segmented = segmenter.segment(text)
        #print('Arabic with Farasa :',re.sub(r"\+", " ", segmented).split(' '))
        return re.sub(r"\+", " ", trans.convert(segmented,'arabic','tim')).split(' ')

    def tokenize_bpe_ar(self, text):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        from tokenizers.trainers import BpeTrainer
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        from tokenizers.pre_tokenizers import Whitespace
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = Tokenizer.from_file("util/msa.csv")
        a = []
        output = tokenizer.encode(text)
        for i in output.tokens:
            a.append(trans.convert(i,'arabic', 'tim'))
        #print(a)
        return a
    
    '''def tokenize_bpe_nad(self, text):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        from tokenizers.trainers import BpeTrainer
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        from tokenizers.pre_tokenizers import Whitespace
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = Tokenizer.from_file("util/nad.csv")
        a = []
        output = tokenizer.encode(text)
        for i in output.tokens:
            a.append(trans.convert(i,'arabic', 'tim'))
        #print(a)
        return a'''
    
    def tokenize_bpe_nad(self,text):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file("./util/BPE/bpe_tokenizer_nad.json")
        output = tokenizer.encode(text)
        return output.tokens

    def tokenize_bpe_msa(self,text):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file("./util/BPE/bpe_tokenizer_msa.json")
        output = tokenizer.encode(text)
        return output.tokens

    def tokenize_wp_nad(self, text):
        DarijaBERT_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
        return DarijaBERT_tokenizer.tokenize(text)

    def tokenize_wp_msa(self, text):
        tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        return tokenizer.tokenize(text)
