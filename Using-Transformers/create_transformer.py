from transformers import BertConfig, BertModel

# # 创建一个配置对象，指定模型的架构
# config = BertConfig()

# # 创建一个模型实例，使用配置对象
# model = BertModel(config)

# # 打印模型配置
# print(config)

# # 打印模型架构
# # print(model)

# 加载已经训练过的模型
'''
  没有使用BertConfig，而是通过 bert-base-cased 标识符加载了预训练模型。这是一个由 BERT 的作者自己训练的模型检查点
  此模型现在使用检查点的所有权重进行初始化。它可以直接用于对训练它的任务进行推理，也可以对新任务进行微调。通过使用预训练的权重而不是从头开始训练，我们可以快速获得良好的结果。
'''
model = BertModel.from_pretrained("bert-base-uncased")

# 打印模型架构
# print(model)

# Saving methods
'''
  保存模型就像加载模型一样简单，使用 save_pretrained() 方法。

  保存模型时，会自动保存 tokenizer 和 config 文件。
  config.json 文件包含模型的架构信息，tokenizer_config.json 文件包含 tokenizer 的配置信息。
  pytorch_model.bin 文件称为状态字典；包含模型的所有权重
'''
model.save_pretrained("bert-base-uncased")

# # 加载模型
# model = BertModel.from_pretrained("bert-base-uncased")

# # 打印模型架构
# print(model)