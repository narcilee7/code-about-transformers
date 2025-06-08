from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)

print(model_inputs)

'''
  在这里，model_inputs 变量包含模型正常运行所需的一切。对于 DistilBERT，这包括输入 ID 以及注意力掩码。其他接受额外输入的模型也将具有 tokenizer 对象的这些输出。
'''

# 一次性处理多个序列
more_sequences = ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]

model_inputs = tokenizer(more_sequences, padding=True, truncation=True)

print(model_inputs)

# 根据几个目标填充

# 根据最大长度填充
model_inputs = tokenizer(sequence, padding="longest")

print(model_inputs)

# 根据平均长度填充
model_inputs = tokenizer(sequence, padding="max_length")

print(model_inputs)

# 截断序列
model_inputs = tokenizer(sequence, truncation=True)

print(model_inputs)

# 截断和填充
model_inputs = tokenizer(sequence, truncation=True, padding="max_length")

print(model_inputs)

'''
  tokenizer对象可以处理到特定框架张量的转换，然后可以直接将其发送到模型。
'''

# Returns Pytorch tensors
model_inputs = tokenizer(sequence, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequence, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequence, padding=True, return_tensors="np")

# Special tokens
model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))