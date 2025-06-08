from datasets import load_dataset

'''
  处理MRPC数据集
'''

raw_datasets = load_dataset("glue", "mrpc")

# print(raw_datasets)

'''
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
'''

'''
  数据集包含三个部分：训练集、验证集和测试集。每个部分都包含句子对以及一个标签，表示句子对是否相似。
'''

raw_train_dataset = raw_datasets["train"]
# print(raw_train_dataset[0])

'''
  {'sentence1': 'Amrozy is a good boy', 'sentence2': 'Amrozy is a bad boy', 'label': 1, 'idx': 0}
  标签已经是整数了，意味着不必再做任何的预处理
'''

# 查看数据集训练集的特征
# print(raw_train_dataset.features)

'''
  {
    'sentence1': Value(dtype='string', id=None),
    'sentence2': Value(dtype='string', id=None),
    'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], id=None),
    'idx': Value(dtype='int32', id=None)
  }
'''

'''
  数据集的标签是：
'''