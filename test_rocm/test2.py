import torch
import torch.nn as nn

net = nn.Sequential(nn.Conv2d(1,1,3)).to('cpu')
print(net)

x = torch.randn(1,1,32,32).to('cpu')
y = torch.randn(1,1,30,30).to('cpu')

net.zero_grad()
loss = (y - net(x))**2
loss = loss.mean()
loss.backward()