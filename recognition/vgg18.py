import torch
import torch.nn as nn
import requests
import os 
import utils

class Block(torch.nn.Module):
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        """
        init function contains the structure of the block
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(planes,planes, kernel_size=3, stride=stride,padding =1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #checking for downsampling
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity #skip connection
        out = self.relu(out)

        return out

class ResNet18(torch.nn.Module):
    def __init__(self,Block,num_classes=10 ):
        """
        The full structure of our resnet18
        """
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.layers = [2,2,2,2]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #adding the four layers
        self.layer1 = self._make_layer(Block, planes=64, blocks = self.layers[0])
        self.layer2 = self._make_layer(Block, planes=128, blocks = self.layers[1],stride=2)
        self.layer3 = self._make_layer(Block, planes=256, blocks = self.layers[2],stride=2)
        self.layer4 = self._make_layer(Block, planes=512, blocks = self.layers[3],stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)
        

    def _make_layer(self,Block,planes,blocks,stride=1):

        
        downsample = None

        if stride != 1 or self.inplanes != planes :
            downsample = nn.Sequential(
                          nn.Conv2d(
                              self.inplanes,
                              planes ,
                              kernel_size=1,
                              stride=stride,
                              bias=False
                          ),
                          nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(
            Block(self.inplanes, planes,stride,downsample)
        )
        self.inplanes = planes 

        for _ in range(1, blocks):
            layers.append(Block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        

        return x

def myResNet(num_classes=10):
    return ResNet18(Block, num_classes)

# Testing VGG model  
if __name__ == '__main__':

    weights_link = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    weights_path = "recognition/assets/resnet18.pth"
    if not os.path.exists(weights_path):
        print("path not found,Downloading model weights")
        utils.download_url(weights_link,weights_path)
    model = myResNet()
    model.load_state_dict(torch.load(weights_path))
    