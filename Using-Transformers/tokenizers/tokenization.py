from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sequence = "Using a Transformer network is simple"

tokens = tokenizer.tokenize(sequence)

print(tokens)

'''
  ['using', 'a', 'transformer', 'network', 'is', 'simple']
'''

# 将tokens转换为ID from tokens to input IDs
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

# Decoding

'''
  decode方法不仅仅可以将索引转换为token，而且还将属于相同单词的标记组合在一起，生成可读的句子。
'''

decoded_string = tokenizer.decode(ids)


print(decoded_string)

# 将多个文本序列编码为单个张量

sequences = ["Hello, I'm a language model", "I can't do it"]

encoded_sequences = tokenizer.encode_plus(sequences, padding=True, truncation=True, return_tensors="pt")

print(encoded_sequences)