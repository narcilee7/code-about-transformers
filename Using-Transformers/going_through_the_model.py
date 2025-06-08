from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# 下载模型
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 此架构仅包含基本的 Transformer 模块：给定一些输入，它输出我们称之为 hidden state，也称为 features。对于每个模型输入，我们将检索一个高维向量，该向量表示 Transformer 模型对该输入的上下文理解。

# 让我们看看模型在输入文本上输出的 hidden state 是什么样的。

raw_inputs = [
  "I've been waiting for a HuggingFace course my whole life.",
  "I hate this so much!"
]

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 将输入文本转换为 ID 列表
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

# 将 ID 列表转换为张量
outputs = model(**inputs)

# 打印输出
# print(outputs.logits.shape)
# print(outputs.logits)

# 使用 softmax 函数将 logits(模型最后一层输出的原始、未规范化的分数)转化为概率，经过SoftMax层后，每个样本的输出是一个包含两个值的向量，分别表示模型对两个标签的预测概率。
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(predictions)

'''
tensor([[0.9999, 0.0001],
        [0.0001, 0.9999]])
'''

# # 使用 argmax 函数来获取每个样本的预测标签。
# predictions = torch.argmax(predictions, dim=-1)

# print(predictions)

