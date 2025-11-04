
import torch
import torch.nn as nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        # input 32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.bn1= nn.BatchNorm2d(64)
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2) # 112x112 (64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn2= nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.bn3=nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))  # 56 → 1


        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass
        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224
        x = self.bn1(x)
        x= self.pool(x) # B x 64 x 112 x 112
        x = self.conv2(x).relu() # B x 128 x 112 x 112
        x = self.bn2(x)
        x= self.pool(x) # B x 128 x 56 x 56
        x = self.conv3(x).relu() # B x 256 x 56 x 56
        x = self.bn3(x)
        x= self.pool3(x) # B x 256 x 1 x 1
        x = torch.flatten(x, 1)
        x=self.fc1(x) # B x 200
        return x
    