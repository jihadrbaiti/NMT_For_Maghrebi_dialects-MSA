import torch
from torchtext.legacy.data import Field, TabularDataset, Dataset, Example
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
model = AutoModel.from_pretrained("asafaya/bert-base-arabic")

text = 'السلام عليكم جميعا !'
text_field = Field(tokenize=tokenizer.tokenize, lower=True)
fields = [('text', text_field)]
examples = [Example.fromlist([text], fields)]
dataset = Dataset(examples, fields)
text_field.build_vocab(dataset)

# print the size of the vocabulary
print('Vocabulary size:', len(text_field.vocab))
# print the most common words in the vocabulary
print('Most common words:', text_field.vocab.freqs.most_common(10))
# print the numericalized version of our example text
numericalized_text = [text_field.vocab.stoi[token] for token in text_field.tokenize(text)]
print('Numericalized text:', numericalized_text)

# Convert the input list into a tensor and transpose it
input_tensor = torch.tensor(numericalized_text).unsqueeze(0)


# Construct a dictionary with keys 'input_ids', 'attention_mask', and 'token_type_ids'
inputs_dict = {'input_ids': input_tensor,'attention_mask': input_tensor.ne(0),
               'token_type_ids': torch.zeros_like(input_tensor)}

with torch.no_grad():
    model.eval()
    outputs = model(**inputs_dict)
    last_hidden_states = outputs[0] # The first output contains the sequence of hidden-states

embeddings = []
for i in range(input_tensor.shape[1]):
    token_embedding = last_hidden_states[0][i]
    embeddings.append(token_embedding)


print('NUMERICALIZED_TEXT',numericalized_text)
# Show the tokens
tokens = tokenizer.convert_ids_to_tokens(numericalized_text)
print(tokens)

print(len(embeddings))