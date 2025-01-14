import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

# 超参数
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 15
batch_size = 128
lr = 0.001

dataset = torchvision.datasets.MNIST('../../../data',
                                     train=True,
                                     transform=transforms.ToTensor())

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparmeterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self,z):
        h=F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu,log_var =self.encode(x)
        z=self.reparmeterize(mu,log_var)
        x_reconst=self.decode(z)
        return x_reconst,mu,log_var

model=VAE().to(device)

optimizer=torch.optim.Adam(model.parameters(),lr=lr)

# Start training
for epoch in range(num_epochs):
    for i,(x,_) in enumerate(data_loader):
        x=x.to(device).view(-1,image_size)
        x_reconst,mu,log_var=model(x)

        # reconst_loss and KL div
        reconst_loss=F.binary_cross_entropy(x_reconst,x,size_average=False)
        kl_div=-0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp())

        loss=reconst_loss+kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))
    with torch.no_grad():
        # sample
        z=torch.randn(batch_size,z_dim).to(device)
        out=model.decode(z).view(-1,1,28,28)
        save_image(out,os.path.join(sample_dir,'sample-{}.png'.format(epoch+1)))

        out,_,_=model(x)
        x_concat=torch.cat([x.view(-1,1,28,28),out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))