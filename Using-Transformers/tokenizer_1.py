from transformers import AutoTokenizer

# 管道（pipeline）是 Hugging Face 提供的一个高级 API，用于简化模型的使用。
# 它将模型、tokenizer 和处理逻辑打包在一起，使得我们可以轻松地进行推理。

# 使用 AutoTokenizer 加载 tokenizer，并传入 checkpoint 参数, 我们得到一个字典，接下来要做的就是将输入ID列表转换为张量
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 将输入文本转换为 ID 列表
raw_inputs = [
  "I've been waiting for a HuggingFace course my whole life.",
  "I hate this so much!"
]

# 将 ID 列表转换为张量
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

print(inputs)

'''
  {'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
             0,     0,     0,     0,     0,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
'''

'''
  input_ids 是 tokenizer 将文本转换为 ID 列表的结果。
  attention_mask 是告诉模型哪些位置是填充的，哪些是实际的输入。
'''