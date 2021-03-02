import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from model import Melanoma
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
import torch.optim as optim

dir = 'E:\\Aquib\MCA\\Python\\SIIM-ISIC Melanoma Classification\\data\\'
batch_size = 32
lr = 1e-3
epoch = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trans = tt.Compose(
    [
        tt.Resize(128),
        tt.ToTensor(),
        
    ]
)
train_set = ImageFolder(root=dir+"train",transform=trans)
test_set = ImageFolder(root=dir+"test",transform=trans)

train_dataloader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_set,batch_size,shuffle=True)

def accuracy(outputs,labels):
    _,preds = torch.max(outputs,dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))



net = Melanoma()
net = net.to(device)
optimizer = optim.Adam(net.parameters(),lr)
criterion = nn.CrossEntropyLoss()

n_epochs = 5
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        
        outputs = net(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()
        for data_t, target_t in (test_dataloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(test_dataloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'model.pt')
            print('Improvement-Detected, save-net')
    net.train()