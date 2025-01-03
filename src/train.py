import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
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
                    device=device, bert_d=DarijaBert_model, bert=bert).to(device)
print()
print('*************** Our proposed approach - Maghrebi dialects to MSA -***************')
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

criterion = nn.CrossEntropyLoss(reduction='none')


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        trg, src = batch
        src = src[:,1:]
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg_reshape = trg[:, 1:].contiguous().view(-1)
        mask = (trg_reshape != 0) & (trg_reshape != 2)
        loss_per_token = criterion(output_reshape, trg_reshape)
        masked_loss = loss_per_token * mask.float()
        loss = masked_loss.sum() / mask.sum()
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
            trg, src = batch
            src = src[:,1:]
            #src = batch.src
            #trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_reshape = trg[:, 1:].contiguous().view(-1)
            mask = (trg_reshape != 0) & (trg_reshape != 2)
            loss_per_token = criterion(output_reshape, trg_reshape)
            masked_loss = loss_per_token * mask.float()
            loss = masked_loss.sum() / mask.sum()
            #loss = criterion(output_reshape, trg1)
            epoch_loss += loss.item()
            total_bleu = []
            total_bleu2 = []
            total_ter = []
            total_chrf = []
            for j in range(batch_size):
                try:
                    trg_words = decode2words1(trg[j], tokenizer)
                    output_words = output[j].max(dim=1)[1]
                    output_words = decode2words1(output_words, tokenizer)

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
            torch.save(model.state_dict(), './saved/N2M/saved/model-{0}.pt'.format(valid_loss))

        f = open('./results_final/N2M/result_f/train_loss.txt', 'w+')
        f.write(str(train_losses))
        f.close()

        f = open('./results_final/N2M/result_f/bleu.txt', 'w+')
        f.write(str(bleus))
        f.close()

        f = open('./results_final/N2M/result_f/bleu2.txt', 'w+')
        f.write(str(bleus2))
        f.close()

        f = open('./results_final/N2M/result_f/ter.txt', 'w+')
        f.write(str(ters))
        f.close()

        f = open('./results_final/N2M/result_f/chrf.txt', 'w+')
        f.write(str(chrfs))
        f.close()

        f = open('./results_final/N2M/result_f/test_loss.txt', 'w+')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

    #torch.save(model.state_dict(), '/home/jihad.rbaiti/lustre/pt_cloud-muhqxqc6fxo/users/jihad.rbaiti/N2M/unprepro/saved_v2f_r/saved_config/model-{0}.pt'.format(valid_loss))

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)