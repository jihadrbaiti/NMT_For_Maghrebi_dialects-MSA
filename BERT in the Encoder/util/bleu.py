
import math
from collections import Counter
import sacrebleu
import numpy as np
import re


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_ter(hypothesis, reference):
  refs = [" ".join(ref) for ref in reference]
  hyps = [" ".join(hyp) for hyp in hypothesis]
  ter_score = sacrebleu.corpus_ter(hyps, [refs])
  return ter_score.score

def get_chrf(hypothesis, reference):
  refs = [" ".join(ref) for ref in reference]
  hyps = [" ".join(hyp) for hyp in hypothesis]
  chrf_score = sacrebleu.corpus_chrf(hyps, [refs])
  return chrf_score.score

def get_bleu2(hypothesis, reference):
    refs = [" ".join(ref) for ref in reference]
    hyps = [" ".join(hyp) for hyp in hypothesis]
    chrf_score = sacrebleu.corpus_bleu(hyps, [refs])
    return chrf_score.score

def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)

def idx_to_word(x, vocab):
    words = []
    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)
    arabic_pattern = re.compile(r'[\u0600-\u06FF\s]+')
    arabic_punctuation_chars = "؟،؛٬؛.،؛!،؛؛؛،"
    words_to_remove = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
    new = [word for word in words if word not in words_to_remove]
    new1 = [word for word in new if arabic_pattern.fullmatch(word)]
    cleaned_words = [''.join(char for char in word if char not in arabic_punctuation_chars) for word in new1]
    words = " ".join(cleaned_words)
    return words

def decode2words(x, bert_tok):
    
    return bert_tok.decode(x)

def decode2words1(x, bert_tok):

    token = bert_tok.decode(x)
    #arabic_pattern = re.compile(r'[\u0600-\u06FF\s]+')
    words_to_remove = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
    new = [word for word in token.split() if word not in words_to_remove]
    #new1 = [word for word in new if arabic_pattern.fullmatch(word)]
    return " ".join(new)

def decode2words11(x, bert_tok):
    token = bert_tok.decode(x)
    arabic_pattern = re.compile(r'[\u0600-\u06FF\s]+')
    arabic_punctuation_chars = "؟،؛٬؛.،؛!،؛؛؛،"
    words_to_remove = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
    new = [word for word in token.split() if word not in words_to_remove]
    new1 = [word for word in new if arabic_pattern.fullmatch(word)]
    cleaned_words = [''.join(char for char in word if char not in arabic_punctuation_chars) for word in new1]
    return " ".join(cleaned_words)