import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
sequence_length=28
input_size=28
hidden_size=128
num_layers=2
num_classes=10
batch_size=128
num_epochs=5
lr=0.001


train_dataset=torchvision.datasets.MNIST(root='../../../data/',
                                         train=True,
                                         transform=transforms.ToTensor())

test_dataset=torchvision.datasets.MNIST(root='../../../data/',
                                         train=False,
                                         transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class BiRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_size*2,num_classes)# BIRNN size stack 2 layer

    def forward(self, x):
        h0=torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)

        out,_=self.lstm(x,(h0,c0))
        out=self.fc(out[:,-1,:])
        return out

model=BiRNN(input_size,hidden_size,num_layers,num_classes).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)


total_step=len(train_loader)
for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_loader):
        images=images.reshape(-1,sequence_length,input_size).to(device)
        labels=labels.to(device)

        outputs=model(images)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 ==0:
            print('Epoch[{}/{}],Step[{}/{}],Loss:{:.4f}'.format(epoch+1,num_epochs,i+1,total_step,loss.item()))

# Test model
with torch.no_grad():
    correct=0
    total=0
    for images,labels in test_loader:
        images=images.reshape(-1,sequence_length,input_size).to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,pred=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(pred==labels).sum().item()

    print("test acc::{}%".format(correct*100/total))


