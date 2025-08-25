import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride = 1, padding = 0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
    
        return output

class lenet_a(nn.Module):
    def __init__(self):
        super(lenet_a, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride = 1, padding = 1)
        # This size should matches the #flattened output of the last conv/pool layer
        self.fc1 = nn.Linear(1176, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print("After conv1:", x.shape) #debug prints
        x = F.max_pool2d(x, 2)
        #print("After pool1:", x.shape)
        x = torch.flatten(x, 1)
        #print("After flatten:", x.shape)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output