import os
import sys
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


class ConnectFourDeepNet(nn.Module):
    def __init__(self):
        super(ConnectFourDeepNet, self).__init__()
        self.fc1 = nn.Linear(42, 500)
        self.fc2 = nn.Linear(500, 20)
        self.fc3 = nn.Linear(20, 500)
        self.fc4 = nn.Linear(500, 3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.log_softmax(x, dim = 0)
        return x

model_path = resource_path("model.pth")

model = torch.load(model_path)

model.eval()

with torch.no_grad():
    input = [1., 1., 1., 2., 2., 1., 0.,
             2., 2., 2., 1., 2., 2., 0.,
             2., 1., 1., 2., 1., 1., 0.,
             1., 2., 1., 2., 2., 2., 2.,
             1., 1., 2., 2., 1., 1., 1.,
             2., 1., 2., 1., 2., 1., 2.,]
    input = torch.tensor(input)
    output = model(input)
    output = output.argmax(0)
    output = int(output.numpy())
    print(output)