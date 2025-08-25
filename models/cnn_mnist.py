import torch.nn as nn
import torch.nn.functional as F

# class CNN_MNIST(nn.Module):
#     def __init__(self):
#         super(CNN_MNIST, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 0)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.dropout3 = nn.Dropout(0.1)
#
#         self.maxpool4 = nn.MaxPool2d((2, 2))
#         self.conv5 = nn.Conv2d(32, 64, (5, 5), 1, 0)
#         self.relu6 = nn.ReLU(inplace=True)
#         self.dropout7 = nn.Dropout(0.1)
#
#         self.maxpool5 = nn.MaxPool2d((2, 2))
#         self.flatten = nn.Flatten()
#         self.linear6 = nn.Linear(64 * 4 * 4, 512)
#         self.relu7 = nn.ReLU(inplace=True)
#         self.dropout8 = nn.Dropout(0.1)
#         self.linear9 = nn.Linear(512, 10)
#
#     def forward(self, x):
#         for module in self.children():
#             x = module(x)
#         return x

# ---------------------------- Classifiers ----------------------------#
class MNISTBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ind = None

    def forward(self, x):
        return self.conv1(F.relu(self.bn1(x)))


class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), 2, 1)  # 14
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = MNISTBlock(32, 64, 2)  # 7
        self.layer3 = MNISTBlock(64, 64, 2)  # 4
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x