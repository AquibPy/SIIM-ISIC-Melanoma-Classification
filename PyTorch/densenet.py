import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from torchvision.models import densenet161
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
import torch.optim as optim

epochs = 5
valid_loss_min = np.Inf
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = densenet161(pretrained=True)
# print(model.parameters)
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Linear(2208,2)
model =model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)

val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)

for epoch in range(1,epochs+1):
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'Epoch {epoch}\n')
    for batch_idx,(data_,target_) in enumerate(train_dataloader):
        data_,target_ = data_.to(device),target_.to(device)
        optimizer.zero_grad()
        outputs = model(data_)
        loss = criterion(outputs,target_)
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
        _,pred = torch.max(outputs,dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx)%20==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct/total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t = 0
    correct_t = 0
    with torch.no_grad():
        model.eval()
        for data_t,target_t in (test_dataloader):
            data_t,target_t = data_t.to(device),target_t.to(device)
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t,target_t)
            batch_loss+=loss.item()
            _,pred_t = torch.max(outputs_t,dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(test_dataloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'densenet.pt')
            print('Improvement-Detected, save-net')
    model.train()


'''
we achive accuracy of 86.5% on training set and 84.8% on test set using Densenet161 model.
'''