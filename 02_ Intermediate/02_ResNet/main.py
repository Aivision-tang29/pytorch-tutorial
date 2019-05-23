import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 超参数
num_epochs = 50
num_classes = 10
batch_size = 128
lr = 0.001

# transforms
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])
# cifar10
train_dataset = torchvision.datasets.CIFAR10(root='../../../data',
                                             train=True,
                                             transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='../../../data',
                                            train=False,
                                            transform=transforms.ToTensor())

# data loader
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size,
                                          shuffle=False)

# conv3x3 卷积 输出大小不变
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

# 最基本的 Resnet 单元   需要注意的是 channel数的一致性
#   conv
#   bn
#   relu  x2
#   downsample 是否下采样，因为 conv stride=2 特征图大小减半
#   所以 elm add 的residual部分也是需要 downsample的
class ResidualBlock(nn.Module):
    """
    # 最基本的 Resnet 单元   需要注意的是 channel数的一致性
    #   conv
    #   bn
    #   relu  x2
    #   downsample 是否下采样，因为 conv stride=2 特征图大小减半
    #   所以 elm add 的residual部分也是需要 downsample的
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# Resnet

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        # pre conv 卷积头，为了保留大部分的信息
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avgpool=nn.AvgPool2d(8)
        self.fc=nn.Linear(64,num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        """
        :return:
        :param block: residual block
        :param out_channels: 输出通道数
        :param blocks: ResBlock 里面的嵌套的block数
        :param stride:
        :return:
        """
        downsample = None# 赋值None，是为了再次使用时清空上一次的downsample
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.conv(x)
        out=self.bn(out)
        out=self.relu(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.avgpool(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

model=ResNet(ResidualBlock,[2,2,2]).to(device)

# Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

# for update lr
def update(optimizer,lr):
    for param in optimizer.param_groups:
        param['lr']=lr

total_step = len(train_loader)
curr_lr=lr
for epoch in range(num_epochs):
    # 设置model 为训练模式
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        #
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        output = model(images)
        loss = criterion(output, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch[{}/{}],Step[{}/{}],loss:{:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    if (epoch+1)%20==0:
        curr_lr/=3
        update(optimizer,curr_lr)


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