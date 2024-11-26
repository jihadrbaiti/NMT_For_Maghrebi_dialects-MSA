import torch
import torch.nn.functional as F
import math
from collections import Counter
import numpy as np
#from transformer11.dataa import *
from models.model.transformer import Transformer
from models.model.decoder_v2 import Decoder
from models.model.encoder_v2 import Encoder
from util.bleu import *
from data_inverse import *
from transformers import AutoTokenizer, AutoModel
DarijaBERT_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
DarijaBert_model = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT")
bert = AutoModel.from_pretrained("asafaya/bert-base-arabic")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(model_path, model,device):

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

print('Qualitative results of our proposed approach (M2N)')
def generate_translation(src, max_seq_len, model, device):
    """
    Generate translations using greedy decoding, feeding each generated token back to the decoder.
    """
    model.eval()
    
    with torch.no_grad():
        encoder_output = model.encoder(src).to(device)
    generated_sequence=[2]
    EOS_TOKEN_ID = 3 
    for i in range(1, max_seq_len):
        trg_tensor = torch.tensor(generated_sequence).unsqueeze(0).to(device)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        output = model.decoder(trg_tensor, encoder_output, src_mask.int()).to(device)
        probabilities = F.softmax(output[:, -1, :], dim=-1).to(device)
        next_token = torch.argmax(probabilities, dim=-1).item()
        generated_sequence.append(next_token)
        if next_token == EOS_TOKEN_ID:
            print("End of sequence token generated. Stopping.")
            break 
    return generated_sequence#, prob.to(device), logits.to(device)

def test_model(num_examples, model_path, max_len, trg_sos_idx, tokenizer, DarijaBERT_tokenizer):
    """
    Tests the model with inference and computes BLEU, TER, CHRF.
    """
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
                        device=device, bert_d=bert, bert=DarijaBert_model).to(device)
    
    model = load_model(model_path, model,device=device)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    model.eval()
    with torch.no_grad():
        batch_bleu = []
        batch_bleu2 = []
        batch_ter = []
        batch_chrf = []
        for i, batch in enumerate(test_iter):  
            src, trg = batch  
            src = src[:,1:]
            total_bleu = []
            total_bleu2 = []
            total_ter = []
            total_chrf = []
            for j in range(num_examples):
                try:
                                       
                    generated_indices = generate_translation(src[j].unsqueeze(0), max_seq_len, model, device)
                    
                    src_words = decode2words1(src[j], tokenizer)
                    trg_words = decode2words1(trg[j], DarijaBERT_tokenizer)
                    output_words = decode2words1(torch.tensor(generated_indices).squeeze(0).to(device), DarijaBERT_tokenizer)
                    
                    print('source :', src_words)
                    print('target :', trg_words)
                    print('predicted :', output_words)
                    print()
                    
                    # Calculate BLEU score
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    bleu2 = get_bleu2([output_words.split()], [trg_words.split()])
                    ter = get_ter([output_words.split()], [trg_words.split()])
                    chrf = get_chrf([output_words.split()], [trg_words.split()])
                    total_bleu.append(bleu)
                    total_bleu2.append(bleu2)
                    total_ter.append(ter)
                    total_chrf.append(chrf)
                except:
                    continue

            # Average BLEU score for this batch
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
            
        # Average BLEU score for all batches
        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        batch_bleu2 = sum(batch_bleu2) / len(batch_bleu2)
        batch_ter = sum(batch_ter) / len(batch_ter)
        batch_chrf = sum(batch_chrf) / len(batch_chrf)
        
        print('TOTAL BLEU SCORE = {}'.format(batch_bleu))
        print('TOTAL BLEU SCORE 2 = {}'.format(batch_bleu2))
        print('TOTAL TER SCORE = {}'.format(batch_ter))
        print('TOTAL CHRF SCORE = {}'.format(batch_chrf))

if __name__ == '__main__':
    
    model_path = "./saved/M2N/saved"
    
    test_model(num_examples=batch_size, model_path=model_path, max_len=max_len, trg_sos_idx=trg_sos_idx, tokenizer=tokenizer, DarijaBERT_tokenizer=DarijaBERT_tokenizer)
