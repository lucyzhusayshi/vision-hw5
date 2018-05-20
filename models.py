import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        if not os.path.exists('logs'):
            os.makedirs('logs')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open('logs/' + st, 'w')

    def log(self, str):
        print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        #  TODO: try mean squared loss

        # return nn.MSELoss()
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001)

    def adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.lr 
        
        lr *= 10**(-(epoch/50))  # TODO: Implement decreasing learning rate's rules
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
       


class LazyNet(BaseModel):
    def __init__(self):
        super(LazyNet, self).__init__()
        # TODO: Define model here
        # 1 fully connected layer
        # ten categories of classification
        self.fc1 = nn.Linear(3*32*32, 10)


    def forward(self, x):
        # TODO: Implement forward pass for LazyNet
        x = x.view(-1, 3*32*32)
        x = self.fc1(x)
        return x
        

class BoringNet(BaseModel):
    def __init__(self):
        super(BoringNet, self).__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(3*32*32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)        

    def forward(self, x):
        # TODO: Implement forward pass for BoringNet
        x = x.view(-1, 3*32*32)
        
        # without activations
        # x = self.fc1(x)
        # x = self.fc2(x)

        # with RELU activations
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        x = self.fc3(x)
        return x


class CoolNet_v1(BaseModel):
    def __init__(self):
        super(CoolNet_V1, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2)
        x = x.view(-1, 16*5*5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CoolNet_v2(BaseModel):
    def __init__(self):
        super(CoolNet_v2, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Doesn't work
# class CoolNet_v2_3(BaseModel):
#     def __init__(self):
#         super(CoolNet_v2_3, self).__init__()
#         # TODO: Define model here
#         self.conv1 = nn.Conv2d(3, 6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         self.fc1 = nn.Linear(16*3*3, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # TODO: Implement forward pass for CoolNet
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, 16*3*3)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# Doesn't Work
# class CoolNet_v2_9(BaseModel):
#     def __init__(self):
#         super(CoolNet_v2_9, self).__init__()
#         # TODO: Define model here
#         self.conv1 = nn.Conv2d(3, 6, 9)
#         self.conv2 = nn.Conv2d(6, 16, 9)
#         self.fc1 = nn.Linear(16*9*9, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # TODO: Implement forward pass for CoolNet
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, 16*9*9)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class CoolNet_v3(BaseModel):
    def __init__(self):
        super(CoolNet_v3, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*5*5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CoolNet_v4(BaseModel):
    def __init__(self):
        super(CoolNet_v4, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Doesn't work
# class CoolNet_v5(BaseModel):
#     def __init__(self):
#         super(CoolNet_v5, self).__init__()
#         # TODO: Define model here
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.fc1 = nn.Linear(6*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # TODO: Implement forward pass for CoolNet
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         x = x.view(-1, 6*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# Doesn't work
# class CoolNet_v6(BaseModel):
#     def __init__(self):
#         super(CoolNet_v6, self).__init__()
#         # TODO: Define model here
#         self.conv1 = nn.Conv2d(3, 16, 5)
#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # TODO: Implement forward pass for CoolNet
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         x = x.view(-1, 16*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class CoolNet_v7(BaseModel):
    def __init__(self):
        super(CoolNet_v7, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CoolNet_v8(BaseModel):
    def __init__(self):
        super(CoolNet_v8, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

# Doesn't work
# class CoolNet_v9(BaseModel):
#     def __init__(self):
#         super(CoolNet_v9, self).__init__()
#         # TODO: Define model here
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # TODO: Implement forward pass for CoolNet
#         x = F.max_pool2d(F.relu(self.conv1(x)), 4)
#         x = F.max_pool2d(F.relu(self.conv2(x)), 4)
#         x = x.view(-1, 16*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class CoolNet_v10(BaseModel):
    def __init__(self):
        super(CoolNet_v10, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 16*5*5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CoolNet(BaseModel):
    def __init__(self):
        super(CoolNet, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 16*5*5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x