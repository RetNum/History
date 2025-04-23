import torch

data = torch.load("./datasets/REDDIT-MULTI-5K.pt")
print(type(data[0]))  # 先看看数据类型
print(data[:5])
pip install pytorch-lightning==2.1c