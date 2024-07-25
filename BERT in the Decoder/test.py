

import math
from collections import Counter

import numpy as np

#from transformer11.dataa import *
from models.model.transformer_dec import Transformer
from util.bleu import *
from data import *
from transformers import AutoTokenizer, AutoModel
DarijaBERT_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
DarijaBert_model = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT")
bert = AutoModel.from_pretrained("asafaya/bert-base-arabic")
###


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=0.00,
                    device=device, bert=bert).to(device)

print()
print('***************Result of test of Config1/unprepro/bert in the decoder***************')
print()

print(f'The model has {count_parameters(model):,} trainable parameters')


def test_model(num_examples):
    iterator = test_iter
    model.load_state_dict(torch.load("/saved/model-0.096955199499388.pt", map_location=torch.device('cpu')))

    with torch.no_grad():
        batch_bleu = []
        batch_bleu2 = []
        batch_ter = []
        batch_chrf = []
        for i, batch in enumerate(iterator):
            trg, trg_mask, src, src_mask = batch
            #src = batch.src 
            #trg = batch.trg 
            output = model(src, src_mask, trg, trg_mask)

            total_bleu = []
            total_bleu2 = []
            total_ter = []
            total_chrf = []
            for j in range(num_examples):
                try:
                    src_words = decode2words1(src[j], DarijaBERT_tokenizer)
                    trg_words = decode2words1(trg[j], tokenizer)
                    output_words = output[j].max(dim=1)[1]
                    output_words = decode2words1(output_words, tokenizer)

                    print('source :', src_words)
                    print('target :', trg_words)
                    print('predicted :', output_words)
                    print()
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    bleu2 = get_bleu2([output_words.split()], [trg_words.split()])
                    ter = get_ter([output_words.split()], [trg_words.split()])
                    chrf = get_chrf([output_words.split()], [trg_words.split()])
                    
                    total_bleu.append(bleu)
                    total_bleu2.append(bleu2)
                    total_ter.append(ter)
                    total_chrf.append(chrf)
                    
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            total_bleu2 = sum(total_bleu2) / len(total_bleu2)
            total_ter = sum(total_ter) / len(total_ter)
            total_chrf = sum(total_chrf) / len(total_chrf)
            print('BLEU SCORE = {}'.format(total_bleu))
            print('BLEU SCORE 2 = {}'.format(total_bleu2))
            print('TER SCORE = {}'.format(total_ter))
            print('CHRF SCORE = {}'.format(total_chrf))
            batch_bleu.append(total_bleu)
            batch_bleu2.append(total_bleu2)
            batch_ter.append(total_ter)
            batch_chrf.append(total_chrf)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        batch_bleu2 = sum(batch_bleu2) / len(batch_bleu2)
        batch_ter = sum(batch_ter) / len(batch_ter)
        batch_chrf = sum(batch_chrf) / len(batch_chrf)
        
        print('TOTAL BLEU SCORE = {}'.format(batch_bleu))
        print('TOTAL BLEU SCORE 2 = {}'.format(batch_bleu2))
        print('TOTAL TER SCORE = {}'.format(batch_ter))
        print('TOTAL CHRF SCORE = {}'.format(batch_chrf))

if __name__ == '__main__':
    test_model(num_examples=batch_size)
