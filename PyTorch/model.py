import torch.nn as nn
import torch.nn.functional as F


class Melanoma(nn.Module):
    def __init__(self):
        super(Melanoma, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*32*32, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x  = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# model = Melanoma()
# print(model.parameters)