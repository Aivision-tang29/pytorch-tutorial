import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#是否使用GPU or cpu
device=torch.device('cuda' if torch.cuda.is_available() else'cpu')

#超参数
input_size=784
hidden_size=500
num_classes=10
num_epochs=10
batch_size=128
lr=0.001

# 手写数字数据集MNIST
train_dataset = torchvision.datasets.MNIST(root='../../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           )
test_dataset = torchvision.datasets.MNIST(root='../../../data',
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          )

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 模型
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet, self).__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        out=self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)
        return out

model=NeuralNet(input_size,hidden_size,num_classes).to(device)

# lossfunc and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # input_size=784
        images = images.reshape(-1, 28 * 28).to(device)
        labels=labels.to(device)

        # 前向传播
        output = model(images)
        loss = criterion(output, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch[{}/{}],Step[{}/{}],loss:{:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels=labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum()
    print('acc for test 10000 images:{:.4f}'.format(100 * correct / total))

