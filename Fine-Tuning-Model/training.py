from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
import evaluate
import numpy as np

from preprocessing_dataset import tokenized_datasets, tokenizer, checkpoint
from dynamic_padding import data_collator

# 定义一个TrainingArguments对象，它将包含我们想要传递给训练器的所有参数, 必须提供的唯一参数是保存训练模型的目录、以及沿途的检查点
training_args = TrainingArguments("test-trainer")

# 定义我们的模型
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# 评估
def compute_metrics(eval_preds):
  metric = evaluate.load("glue", "mrpc")
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

# 一个新的带有compute_metrics参数的Trainer对象
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_datasets["train"],
  eval_dataset=tokenized_datasets["validation"],
  tokenizer=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics,
  # eval_strategy="epoch"
)

trainer.train()