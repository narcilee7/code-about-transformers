import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

output = model(**tokens)

print(output.logits.shape)

'''
  torch.Size([2, 2])
'''

# 