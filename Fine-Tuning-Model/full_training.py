from preprocessing_dataset import tokenized_datasets, tokenizer, checkpoint
from dynamic_padding import data_collator
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

training_args = TrainingArguments(
  "test-trainer",
  evaluation_strategy="epoch",
  learning_rate=2e-5,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=3,
)