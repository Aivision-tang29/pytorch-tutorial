import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#超参数

input_size=784
num_classes=10
num_epoch=10
batch_size=128
lr=0.001

# 手写数字数据集MNIST
train_dataset=torchvision.datasets.MNIST(root='../../../data',
                                         train=True,
                                         transform=transforms.ToTensor(),
                                         )
test_dataset=torchvision.datasets.MNIST(root='../../../data',
                                         train=True,
                                         transform=transforms.ToTensor(),
                                         )

# DataLoader
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

# 模型
model=nn.Linear(input_size,num_classes)

# lossfunc and optimizer

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=lr)

total_step=len(train_loader)

for epoch in range(num_epoch):
    for i,(images,labels) in enumerate(train_loader):
        # input_size=784
        images=images.reshape(-1,28*28)

        # 前向传播
        output=model(images)
        loss=criterion(output,labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100==0:
            print('Epoch[{}/{}],Step[{}/{}],loss:{:.4f}'.format(epoch+1,num_epoch,i+1,total_step,loss.item()))


with torch.no_grad():
    correct=0
    total=0
    for images,labels in test_loader:
        images=images.reshape(-1,28*28)
        outputs=model(images)
        _,pred=torch.max(outputs,dim=1)
        total+=labels.size(0)
        correct+=(pred==labels).sum()
    print('acc for test 10000 images:{:.4f}'.format(100*correct/total))
