import torch
import torch.nn as nn
import torch.nn.functional as F
import collections



# This is to implement ResNet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) #!!!!check the kernel_size, stride and padding setting again!!
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes) # for cifar10 and GTSRB
        #self.linear = nn.Linear(8192 * block.expansion, num_classes) # for imagenet
        self.latent=False
        self.nad=False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        if self.nad:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)  #cifar10 32X32
            activation1 = out
            out = self.layer2(out)  #cifar10 16X16
            activation2 = out
            out = self.layer3(out)  #cifar10 8X8
            activation3 = out
            out = self.layer4(out)
            activation4 = out
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return activation1, activation2, activation3,activation4, out

        else:
            features = []
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)  #cifar10 32X32
            out = self.layer2(out)  #cifar10 16X16
            out = self.layer3(out)  #cifar10 8X8
            # out = self.layer4(out)
            sequential = self.layer4
            for s in sequential:
                out = s(out)
                features.append(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            if self.latent:
                return out,features
            return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes)

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# This is to add VGG to the current framework
class ClassicVGGx(nn.Module):

    #definition of commonly used VGG structures
    net_arche_cfg = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'ap', 'FC1', 'FC2', 'FC3'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'ap', 'FC1', 'FC2', 'FC3'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 'ap', 'FC1', 'FC2', 'FC3'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 'ap', 'FC1', 'FC2', 'FC3']
    }

    def __init__(self, arch_name, num_classes, num_input_channels):

        super(ClassicVGGx, self).__init__()

        self.num_classes = num_classes

        try:
            arch_structure_def = self.net_arche_cfg[arch_name.lower()]
        except KeyError:
            print(f"the specified vgg structure {arch_name} does not exist. Please define the structure in model.py before use")

        feature_layers = collections.OrderedDict() # conv layer and maxpooling layer
        classifier_layers = collections.OrderedDict() # fc layer and relu layer
        conv_layer_seq = 0
        fc_layer_seq = 0
        maxpooling_layer_seq = 0
        relu_layer_seq = 0
        dropout_layer_seq = 0
        batch_norm2d_seq = 0
        ap_layer_seq = 0

        # the var 'num_input_channels' indicates the number of input channels of the original picture
        # e.g., handwriting photo contains single channel picture, so num_input_channels = 1. RGB picture contains 3 channels, num_input_channels = 3
        in_channels = num_input_channels

        for elem in arch_structure_def:
            if elem == 'M':
                feature_layers['M'+str(maxpooling_layer_seq)] = nn.MaxPool2d(kernel_size=2, stride=2)
                maxpooling_layer_seq += 1
            elif elem == "FC1":
                classifier_layers['FC' + str(fc_layer_seq)] = nn.Linear(512 * 7 * 7, 4096) # the minimum input image size is 32*32
                fc_layer_seq += 1
                classifier_layers['ReLu'+str(relu_layer_seq)] = nn.ReLU(inplace=True)
                relu_layer_seq += 1
                classifier_layers['dp'+str(dropout_layer_seq)] = nn.Dropout()
                dropout_layer_seq += 1
            elif elem == "FC2":
                classifier_layers['FC'+str(fc_layer_seq)] = nn.Linear(4096, 4096)
                fc_layer_seq += 1
                classifier_layers['ReLu'+str(relu_layer_seq)] = nn.ReLU(inplace=True)
                relu_layer_seq += 1
                classifier_layers['dp' + str(dropout_layer_seq)] = nn.Dropout()
                dropout_layer_seq += 1
            elif elem == "FC3":
                classifier_layers['FC'+str(fc_layer_seq)] = nn.Linear(4096, self.num_classes)
                fc_layer_seq += 1
            elif elem == 'ap':
                feature_layers['ap'+str(ap_layer_seq)] = nn.AdaptiveAvgPool2d((7,7))
                ap_layer_seq += 1
            else:
                feature_layers['conv'+str(conv_layer_seq)] = nn.Conv2d(in_channels=in_channels, out_channels=elem, kernel_size=3, padding=1)
                conv_layer_seq += 1
                feature_layers['bn'+str(batch_norm2d_seq)] = nn.BatchNorm2d(elem) ################################
                batch_norm2d_seq += 1
                feature_layers['ReLu'+str(relu_layer_seq)] = nn.ReLU(inplace=True)
                relu_layer_seq += 1
                in_channels = elem
            self.feature_layers = nn.Sequential(feature_layers)
            self.classifier_layers = nn.Sequential(classifier_layers)
        return

    def forward(self, x):
        x = self.feature_layers(x)
        #x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0),-1)
        x = self.classifier_layers(x)
        return x


class ClassicCNN(nn.Module):
    def __init__(self):
        super(ClassicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = torch.nn.GroupNorm(32,32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = torch.nn.GroupNorm(32,64)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(9216, 128)  ### 9216
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2_drop(F.max_pool2d(self.conv2(x), 2)))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print(x.size())
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return x

# if __name__ == '__main__':
#     net = ResNet18()
#     print(net.children)
#     print(net.linear)
#     exit(0)
#     for name, layer in net.named_modules():
#         print('name:'+name)
#         print(layer)
#     print('finish the test')