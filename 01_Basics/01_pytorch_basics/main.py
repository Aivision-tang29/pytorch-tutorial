# coding:utf-8
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# -------------------------------------------- #
#                 内容列表                      #
# -------------------------------------------- #

# 1.autograd的例子1
# 2.autograd的例子2
# 3.从numpy 加载数据
# 4.输入
# 5.自定义数据集
# 6.预训练模型
# 7.保存和加载模型

# -------------------------------------------- #
#                 1.autograd的例子1             #
# -------------------------------------------- #

# 创建 tensor的变量 标量
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# 构建计算图
y = w * x + b

# 计算梯度
y.backward()

# 打印梯度值
print(x.grad)
print(w.grad)
print(b.grad)

# tensor(2.)
# tensor(1.)
# tensor(1.)

# -------------------------------------------- #
#                 2.autograd的例子2             #
# -------------------------------------------- #

# 构建 数据 （10,3） （10,2）
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 定义一个全连接层 Linear
layer = nn.Linear(3, 2)
print(layer.weight)
print(layer.bias)

# 定义损失函数 优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

# 前向传播
pred = layer(x)

# 计算损失
loss = criterion(pred, y)
print('loss:', loss.item())

# 反向传播，计算梯度
loss.backward()

# 打印梯度值
print('dL/dw:', layer.weight.grad)
print('dL/db:', layer.bias.grad)

# 更新一步，参数
optimizer.step()

# 显示更新一次参数后的pred
pred = layer(x)
loss = criterion(pred, y)
print('经过一次更新参数后的损失：', loss.item())

# -------------------------------------------- #
#                3.从numpy加载数据               #
# -------------------------------------------- #

# 构建np数据 shape（2,2）
x = np.array([[1, 2], [3, 4]])

# np--> tensor
y = torch.from_numpy(x)

# tensor--> np array
z = y.numpy()

# -------------------------------------------- #
#                4.输入 cifar10example         #
# -------------------------------------------- #

# 下载数据集
train_dataset = torchvision.datasets.CIFAR10(root='../../../data',
                                             train=True,
                                             transform=transforms.ToTensor()
                                             )
# 这里download 我设置为false，因为我的data里下载好了
# 如果没有的话，应该设置为true

# 读取一个数据
image, label = train_dataset[0]
print(image.size())
print(label)

# 使用 dataloader 来加载，好处是：多线程加速，shuffle，batch输入
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# 开始迭代时，队列和线程就会从硬盘读取数据
data_iter = iter(train_loader)

# Mini-batch 图像和标签
images, labels = data_iter.next()

# 在实际的使用中，我们会使用如下或者枚举的形式来加载数据
for images, labels in data_iter:
    print(images.size(), labels.size())

for idx, data in enumerate(data_iter):
    images, labels = data
    print(images.size(), labels.size())


# -------------------------------------------- #
#                5.自定义数据集                  #
# -------------------------------------------- #

# 自定义数据集需要：继承自 torch.utils.data.Dataset
# 重写 __getitem__ 和 __len__ 方法

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        # TODO
        # 1.加载数据的路径或者是传入一个list的文件名
        # pseudo code
        # import os
        # filename_list=os.list(file_dir)
        #
        pass

    def __getitem__(self, item):
        # TODO
        # 1.读取图像数据.
        # for idx,image_name in enunerate(filename_list):
        # image=Image.open(image_name) or cv2.imread(image_name) should be ok

        # 2.预处理
        # 数据变换 data-transforms
        # transforms=transforms.Compose(
        # do your transform)
        # 3.返回 图像，标签

        pass

    def __len__(self):
        return 0  # len(filename_list)

# useage:
# custom_dataset = Mydataset()
# train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
#                                            batch_size=64,
#                                            shuffle=True)

# -------------------------------------------- #
#                6.预训练模型                    #
# -------------------------------------------- #

# 加载训练好的resnet34
resnet=torchvision.models.resnet34(pretrained=True)

# 做迁移学习的话 可以冻结已经训练好的参数，只训练自加的层
for param in resnet.parameters():
    param.requires_grad=False

resnet.fc=nn.Linear(resnet.fc.in_features,10)

image=torch.randn(1,3,224,224)
outputs=resnet(image)
print(outputs.size())

# -------------------------------------------- #
#                7.保存和加载                   #
# -------------------------------------------- #

# 网络结构和参数都保留
torch.save(resnet,'model.ckpt')
model=torch.load('model.ckpt')

# 官方网站上推荐的，只保留模型的参数

torch.save(resnet.state_dict(),'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))