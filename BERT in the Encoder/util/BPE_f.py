from tokenizers import Tokenizer
from tokenizers.models import BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()
tokenizer = Tokenizer.from_file("data_bpe_arabic.json")
output = tokenizer.encode("الذي شارك رفقة نبيلة منيب في ندوة حول الوضعية السياسية والاقتصادية والحقوقية في المغرب")
print(output.tokens)
print('done')