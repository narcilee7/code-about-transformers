import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

'''
  单个小长度的序列很简单处理
  但是，我们如何处理多个序列、多个不同长度的序列、词汇索引是允许模型正常工作的唯一输入吗？
'''

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

# tokens = tokenizer.tokenize(sequence)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# input_ids = torch.tensor(ids)

# # This line will fail
# model(input_ids)

'''
  为什么会失败？
  问题是我们向模型发送了一个序列，而Transformers模型默认需要多个句子
  tokenizer 不仅将输入 ID 列表转换为张量，还在其顶部添加了一个维度：
'''

# tokenized_inputs = tokenizer(sequence, return_tensors="pt")

# print(tokenized_inputs)

# '''
#   {'input_ids': tensor([[ 101, 1045, 1005, 1045, 2129, 1012, 102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}
# '''

# 添加新维度
tokens = tokenizer.tokenize(sequence)
# 将tokens转换为ID
ids = tokenizer.convert_tokens_to_ids(tokens)
# 将ids转换为张量
input_ids = torch.tensor([ids])

# 打印输入ID
# print("Input IDs:", input_ids)

# 将输入ID传递给模型
output = model(input_ids)

# 打印输出
# print("Logits:", output.logits)

'''
  Logits: tensor([[ 0.0126, -0.0239],
                  [ 0.0126, -0.0239]], grad_fn=<AddmmBackward0>)
'''

# 批处理：通过模型一次性发送多个句子的操作。

'''
  批处理允许模型在你向它提供多个句子时工作。使用多个序列与使用单个序列构建一个批次一样简单。
  不过，还有第二个问题。当您尝试将两个（或多个）句子批处理在一起时，它们的长度可能不同。
  如果您以前使用过张量，那么您就会知道它们需要是矩形的，因此您将无法直接将输入 ID 列表转换为张量。为了解决这个问题，我们通常会填充 inputs。
'''

'''
  为了解决这个问题，我们将使用填充使我们的张量具有矩形形状。
  Padding 通过向值较少的句子添加一个名为 padding token 的特殊单词来确保我们所有的句子具有相同的长度。
'''

# sequence1_ids = [[200, 200, 200]]
# sequence2_ids = [[200, 200]]
# batched_ids = [
#   [200, 200, 200],
#   [200, 200, tokenizer.pad_token_id]
# ]

# print(batched_ids)

# print(model(torch.tensor(sequence1_ids)).logits)
# print(model(torch.tensor(sequence2_ids)).logits)
# print(model(torch.tensor(batched_ids)).logits)

# 我们的批量预测中的 logit 有问题：第二行应该与第二句的 logit 相同，但我们有完全不同的值！

'''
  这是因为Transformer模型的关键特征是将每个Token置于上下文的注意力层。
  这些将考虑padding token，因为它们会处理sequence的所有token。
  为了在通过模型传递不同长度的单个句子时，或者在传递应用了相同句子和填充的批处理时获得相同的结果，我们需要告诉这些注意力层忽略填充标记。这是通过使用注意力掩码来完成的。
'''

# 采用注意力掩码

batched_ids = [
  [200, 200, 200],
  [200, 200, tokenizer.pad_token_id]
]

# 创建注意力掩码
attention_mask = [
  [1, 1, 1],
  [1, 1, 0]
]

# 将注意力掩码传递给模型
outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))

print(outputs.logits)

# 这样获得与单个句子相同的结果


# 处理Long sequences

'''
  使用具有较长支持的序列长度的模型
  使用截断的序列
'''

