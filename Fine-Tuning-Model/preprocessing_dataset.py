from transformers import AutoTokenizer
from process_data import raw_datasets

checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

'''
  我们给分词器一个句子或一个句子列表，这样我们就可以直接对每对的所有第一句和所有第二句进行分词
'''
# tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
# tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

'''
  我们需要将两个序列作为一对处理，并做适当的预处理。
  分词器可以采用一对序列，并按照我们的BERT模型期望的方式进行准备
'''

# inputs = tokenizer("This is the first sentence", "This is the second sentence")

# print(inputs)

# '''
#   {'input_ids': [101, 1996, 1037, 1138, 102, 2023, 2005, 1037, 1138, 102], 'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]}
# '''

# # 将input_ids中的ID解码回单词
# print(tokenizer.decode(inputs["input_ids"]))

# '''
#   [CLS] this is the first sentence [SEP] this is the second sentence [SEP]
# '''

def tokenize_function(example):
  return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# 一次性在所有的数据集上应用分词函数，更快地预处理 
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)