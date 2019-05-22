import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 超参数定义
#===========================#
# 输入，输出，迭代次数，学习率  #
#===========================#
input_size=1
output_size=1
num_epochs=2500
lr=0.01

# 数据
x_train=np.random.rand(50,1).astype(np.float32)
print(x_train)
print(x_train.shape)
y_train=x_train*2+0.5
print(y_train)
print(y_train.shape)

# 绘图看看数据的样子
# plt.plot(x_train,y_train,'ro-')
# plt.show()

# 定义模型 dense
model=nn.Linear(input_size,output_size)

# 定义损失和优化器

criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=lr)

# 训练
for epoch in range(num_epochs):

    #numpy-->tensor
    inputs=torch.from_numpy(x_train)
    targets=torch.from_numpy(y_train)

    # 前向传播
    outputs=model(inputs)
    loss=criterion(outputs,targets)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1)%5==0:
        print('Epoch[{}/{}],loss:{:.4f}'.format(epoch,num_epochs,loss.item()))


pred=model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train,y_train,'ro',label='raw data')
plt.plot(x_train,pred,label='fit line')
plt.legend()
plt.show()

torch.save(model.state_dict(),'linear_mdoel.ckpt')
# load model
net=nn.Linear(input_size,output_size)
net.load_state_dict(torch.load('linear_mdoel.ckpt'))
pred=net(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train,y_train,'ro',label='raw data')
plt.plot(x_train,pred,label='fit line')
plt.legend()
plt.show()