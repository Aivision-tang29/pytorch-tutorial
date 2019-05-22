import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 超参数
num_epochs = 10
num_classes = 10
batch_size = 128
lr = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           )

test_dataset = torchvision.datasets.MNIST(root='../../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# ConvNet two layer

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        # input shape [128,1,28,28]
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# lossfunc and optimizer

criterion = nn.CrossEntropyLoss()
# optim used Adam for fast update param
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_step = len(train_loader)

for epoch in range(num_epochs):
    # 设置model 为训练模式
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # input_size=784
        images = images.to(device)
        labels = labels.to(device)

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
    # 模型为校验模式
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum()
    print('acc for test 10000 images:{:.4f}'.format(100 * correct / total))
# Epoch[10/10],Step[400/469],loss:0.0171
# acc for test 10000 images:99.0000