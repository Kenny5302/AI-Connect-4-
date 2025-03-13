import os
import random
import sys

import numpy as np
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision, MulticlassF1Score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset



def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


class ConnectFourDeepNet(nn.Module):
    def __init__(self):
        super(ConnectFourDeepNet, self).__init__()
        self.fc1 = nn.Linear(42, 300)
        self.fc2 = nn.Linear(300, 700)
        self.fc3 = nn.Linear(700, 300)
        self.fc4 = nn.Linear(300, 3)
        
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

class ConnectFourDataset(Dataset):
    def __init__(self, rawdata):
        rawdata = np.array(rawdata)
        rawlabels = rawdata[:, 42]
        rawdatabase = rawdata[:,:42]
        self.labels = torch.from_numpy(rawlabels)
        self.database = torch.from_numpy(rawdatabase)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = self.database[idx]
        label = self.labels[idx]
        return data.float(), label.long()
    
batch_size_test = 1000

test_data_file = np.loadtxt('c4_game_database.csv', delimiter = ",", dtype = float, skiprows = 300000)

test_data = ConnectFourDataset(test_data_file)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size_test, shuffle = True)

model_path = resource_path("hard_model.pth")
model = torch.load(model_path)
model.eval()

test_loss = 0
correct = 0
size = len(test_loader.dataset)
num_batches = len(test_loader)

predictions = []
ground_truths = []
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target).item()
        correct += (output.argmax(1) == target).type(torch.float).sum().item()
        predictions.extend(output.argmax(1).numpy())
        ground_truths.extend(target.numpy())


test_loss /= num_batches
averagecorrect = correct / size
print(f"Test Error: \n NumCorrect: {(int(correct))}/{(size)}, Accuracy: {(100*averagecorrect):>0.1f}%, Avg loss: {test_loss:>8f} \n")


preds = np.array(predictions)
target = np.array(ground_truths)
preds = torch.from_numpy(preds)
target = torch.from_numpy(target)
recall = MulticlassRecall(num_classes = 3)
print(recall(preds, target).numpy())

precision = MulticlassPrecision(num_classes = 3)
print(precision(preds, target).numpy())

f1score = MulticlassF1Score(num_classes = 3)
print(f1score(preds, target).numpy())