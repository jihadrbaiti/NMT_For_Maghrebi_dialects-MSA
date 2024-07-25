import math
import time

from torch import nn, optim
from torch.optim import Adam

from data_inverse import *
from models.model.transformer_dec import Transformer
from util.bleu import *
from util.epoch_timer import epoch_time
from transformers import AutoTokenizer, AutoModel
DarijaBert_model = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT")
bert = AutoModel.from_pretrained("asafaya/bert-base-arabic")
DarijaBERT_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

print('Here is the MAX_LEN :', max_len)
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
                    drop_prob=drop_prob,
                    device=device, bert_d=bert, bert=DarijaBert_model).to(device)
print()
print('*************** Result of Config1/unprepro/dec_inverse ***************')
print()

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss()


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src, src_mask, trg, trg_mask = batch
        optimizer.zero_grad()
        output = model(src, src_mask, trg, trg_mask)
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg.contiguous().view(-1)
        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    batch_bleu2 = []
    batch_ter = []
    batch_chrf = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_mask, trg, trg_mask = batch
            #src = batch.src
            #trg = batch.trg
            output = model(src, src_mask, trg, trg_mask)
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg1 = trg.contiguous().view(-1)
            loss = criterion(output_reshape, trg1)
            epoch_loss += loss.item()
            total_bleu = []
            total_bleu2 = []
            total_ter = []
            total_chrf = []
            for j in range(batch_size):
                try:
                    #src_words = decode2words(src[j], tokenizer)
                    trg_words = decode2words1(trg[j], DarijaBERT_tokenizer)
                    output_words = output[j].max(dim=1)[1]
                    output_words = decode2words1(output_words, DarijaBERT_tokenizer)

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
            total_bleu = sum(total_bleu) / (len(total_bleu))
            total_bleu2 = sum(total_bleu2) / len(total_bleu2)
            total_ter = sum(total_ter) / len(total_ter)
            total_chrf = sum(total_chrf) / len(total_chrf)
            batch_bleu.append(total_bleu)
            batch_bleu2.append(total_bleu2)
            batch_ter.append(total_ter)
            batch_chrf.append(total_chrf)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    batch_bleu2 = sum(batch_bleu2) / len(batch_bleu2)
    batch_ter = sum(batch_ter) / len(batch_ter)
    batch_chrf = sum(batch_chrf) / len(batch_chrf)
    return epoch_loss / len(iterator), batch_bleu, batch_bleu2, batch_ter, batch_chrf


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus, bleus2, ters, chrfs= [], [], [], [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu, bleu2, ter, chrf = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        bleus2.append(bleu2)
        ters.append(ter)
        chrfs.append(chrf)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), '/saved/model-{0}.pt'.format(valid_loss))

        f = open('./Result_config1/M2N/unprepro/result_dec/train_loss.txt', 'w+')
        f.write(str(train_losses))
        f.close()

        f = open('./Result_config1/M2N/unprepro/result_dec/bleu.txt', 'w+')
        f.write(str(bleus))
        f.close()

        f = open('./Result_config1/M2N/unprepro/result_dec/bleu2.txt', 'w+')
        f.write(str(bleus2))
        f.close()

        f = open('./Result_config1/M2N/unprepro/result_dec/ter.txt', 'w+')
        f.write(str(ters))
        f.close()

        f = open('./Result_config1/M2N/unprepro/result_dec/chrf.txt', 'w+')
        f.write(str(chrfs))
        f.close()

        f = open('./Result_config1/M2N/unprepro/result_dec/test_loss.txt', 'w+')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)