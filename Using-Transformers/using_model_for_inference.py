# 使用模型进行推理
import torch
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")

# 原始文本
sequences = ["Hello!", "Cool.", "Nice!"]

# 分词器将这些转换为通常称之为输入ID的词汇索引。每个序列都是一个数字列表
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

# 将矩阵转换为张量
inputs = torch.tensor(encoded_sequences)

# 使用张量作为输入
outputs = model(inputs)

# 打印输出
print(outputs)

# 打印输出形状
print(outputs.shape)

# 打印输出数据类型