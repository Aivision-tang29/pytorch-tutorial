# coding:utf-8
import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torchvision.utils import save_image

#
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数

latent_size=64
hidden_size=256
image_size=784
num_epochs=200
batch_size=100
sample_dir='samples'

# 创建一个文件夹
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

# 预处理
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),
                         std=(0.5,0.5,0.5))
])

#load dataset
train_dataset=torchvision.datasets.MNIST(root='../../../data/',
                                         train=True,
                                         transform=transform)



# Data loader
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# 判别器
D=nn.Sequential(
    nn.Linear(image_size,hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size,hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size,1),
    nn.Sigmoid()
)

# 生成器
G=nn.Sequential(
    nn.Linear(latent_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,image_size),
    nn.Tanh()
)

# copy model to device
D=D.to(device)
G=G.to(device)

# judge real or fake
criterion=nn.BCELoss()
d_optimizer=torch.optim.Adam(D.parameters(),lr=0.0002)
g_optimizer=torch.optim.Adam(G.parameters(),lr=0.0002)


def denorm(x):
    out=(x+1)/2
    return out.clamp(0,1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

total_step=len(data_loader)

for epoch in range(num_epochs):
    # 下划线代表，我们不关心这里的数据是什么，没用到
    for i,(images,_) in enumerate(data_loader):
        images=images.reshape(batch_size,-1).to(device)

        # 创建真假的标签 real 1   fake 0
        real_labels=torch.ones(batch_size,1).to(device)
        fake_labels=torch.zeros(batch_size,1).to(device)

        ##############################################
        #             训练判别器                       #
        ##############################################

        # BCE loss compute BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        outputs=D(images)
        d_loss_real=criterion(outputs,real_labels)
        real_score=outputs

        z=torch.randn(batch_size,latent_size).to(device)
        fake_images=G(z)
        outputs=D(fake_images)
        d_loss_fake=criterion(outputs,fake_labels)
        fake_score=outputs

        d_loss=d_loss_real+d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        ##############################################
        #             训练生成器                         #
        ##############################################

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))
    # save real image
    if (epoch+1)==1:
        images=images.reshape(images.size(0),1,28,28)
        save_image(denorm(images),os.path.join(sample_dir,'real_images.png'))
    # save fake image
    fake_images=fake_images.reshape(fake_images.size(0),1,28,28)
    save_image(denorm(fake_images),os.path.join(sample_dir,'fake_images-{}.png'.format(epoch+1)))