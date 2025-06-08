from preprocessing_dataset import tokenized_datasets
from dynamic_padding import data_collator
from torch.utils.data import DataLoader

# 打印原始的tokenized_datasets来查看
print("Original tokenized_datasets:", tokenized_datasets)

def prepare_dataset(datasets):
    # 对每个数据集分别处理
    print("Before processing:", datasets)  # 添加调试打印
    
    # Remove unwanted columns
    datasets = datasets.map(
        lambda x: {k: x[k] for k in x if k not in ["sentence1", "sentence2", "idx"]}, 
        remove_columns=["sentence1", "sentence2", "idx"]
    )
    
    # Rename label column to labels
    datasets = datasets.rename_column("label", "labels")
    
    # Set the format to pytorch
    datasets = datasets.set_format("torch")
    
    print("After processing:", datasets)  # 添加调试打印
    
    return datasets

# 调用prepare_dataset()并保存修改后的数据集
tokenized_datasets_prepared = prepare_dataset(tokenized_datasets)

# 确认返回的结果不是None
if tokenized_datasets_prepared is None:
    print("Error: tokenized_datasets is None after preparation!")
else:
    print("Processed tokenized_datasets:", tokenized_datasets_prepared)

# 确保 tokenized_datasets_prepared 有数据
train_dataloader = DataLoader(
  tokenized_datasets_prepared["train"],
  shuffle=True,
  batch_size=8,
  collate_fn=data_collator
)

eval_dataloader = DataLoader(
  tokenized_datasets_prepared["validation"],
  batch_size=8,
  collate_fn=data_collator
)

# check一下数据处理有没有错误
def check_dataloader_is_correct():
  for batch in train_dataloader:
    break
  print({k: v.shape for k, v in batch.items()})

check_dataloader_is_correct()

