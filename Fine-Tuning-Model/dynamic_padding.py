from transformers import DataCollatorWithPadding, AutoTokenizer
from preprocessing_dataset import tokenized_datasets

checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 最后一件事情，将元素批处理在一起时将所有示例填充到最长元素的长度-动态填充

'''
  负责将样本放在一个批次中的函数称之为collate函数。
  可以在构建 DataLoader 时传递，默认是一个函数，这个函数只会把样本转换为Pytorch张量并连接它们。
'''

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# samples = tokenized_datasets["train"][:8]
# samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

# # [len(x) for x in samples["input_ids"]]

# batch = data_collator(samples)

# {k: v.shape for k, v in batch.items()}