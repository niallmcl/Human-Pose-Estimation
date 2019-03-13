import math

import torch.nn as nn
# from torchvision.models import ResNet
from se_module import SELayer_feedback
from torch.autograd import Variable
import torch
import pdb  
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, disable_SE_layer=False, feedback_size=51):
        super(SEBasicBlock, self).__init__()
        self.feedback_size = feedback_size
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer_feedback(planes, reduction, self.feedback_size)
        self.downsample = downsample
        self.stride = stride
        self.disable_SE_layer = disable_SE_layer

    def forward(self, input_tuple):

        if isinstance(input_tuple, tuple):
            x = input_tuple[0]
            p = input_tuple[1]
        else:
            x = input_tuple
            p = Variable(torch.zeros(self.feedback_size))

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if not self.disable_SE_layer:
            out = self.se(out, p)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, disable_SE_layer=False):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer_feedback(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride
        self.disable_SE_layer = disable_SE_layer

    def forward(self, input_tuple):

        if isinstance(input_tuple, tuple):
            x = input_tuple[0]
            p = input_tuple[1]
        else:
            x = input_tuple
            p = Variable(torch.zeros(51))

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if not self.disable_SE_layer:
            out = self.se(out, p)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, disable_SE_layer=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, disable_SE_layer=disable_SE_layer, feedback_size=num_classes)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, disable_SE_layer=disable_SE_layer, feedback_size=num_classes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, disable_SE_layer=disable_SE_layer, feedback_size=num_classes)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, disable_SE_layer=disable_SE_layer, feedback_size=num_classes)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)       

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, planes, blocks, stride=1, disable_SE_layer=False, feedback_size=51):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # return block(self.inplanes, planes, stride, downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, disable_SE_layer=disable_SE_layer, feedback_size=feedback_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, disable_SE_layer=disable_SE_layer, feedback_size=feedback_size))

        return nn.Sequential(*layers)

    def forward(self, x, p):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1((x, p))
        x = self.layer2((x, p))
        x = self.layer3((x, p))
        x = self.layer4((x, p))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SEFeedback(nn.Module):
    "create a senet where feedback is used to condition the seblock"

    def __init__(self, num_feedback_cycles=2, feedback_norm=False, disable_SE_layer=False, output_size = 51):
        super(SEFeedback, self).__init__()

        self.output_size = output_size
        self.base_net = ResNet(SEBasicBlock, [2, 2, 2, 2], self.output_size, disable_SE_layer)
        self.base_net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_feedback_cycles = num_feedback_cycles
        self.feedback_norm = feedback_norm

    def forward(self, x):  
        predictions = []      
        p = Variable(torch.zeros(self.output_size))
        for i in range(self.num_feedback_cycles):
            p = self.base_net(x,p)

            # take the output of the network and split into 2D and 3D predictions            
            prediction_3d = p[:,0:51].clone()
            prediction_2d = p[:,51:].clone()

            predictions.append((prediction_3d,prediction_2d))
            if self.feedback_norm:
                p = F.normalize(p) # l2 normalise the predicted pose before feedback
        return predictions

class SEFeedback_34(nn.Module):
    "create a senet where feedback is used to condition the seblock"

    def __init__(self, num_feedback_cycles=2, feedback_norm=False, disable_SE_layer=False, output_size = 51):
        super(SEFeedback_34, self).__init__()

        self.output_size = output_size
        self.base_net = ResNet(SEBasicBlock, [3, 4, 6, 3], self.output_size, disable_SE_layer)
        self.base_net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_feedback_cycles = num_feedback_cycles
        self.feedback_norm = feedback_norm

    def forward(self, x):  
        predictions = []      
        p = Variable(torch.zeros(self.output_size))
        for i in range(self.num_feedback_cycles):
            p = self.base_net(x,p)

            # take the output of the network and split into 2D and 3D predictions            
            prediction_3d = p[:,0:51].clone()
            prediction_2d = p[:,51:].clone()

            predictions.append((prediction_3d,prediction_2d))
            if self.feedback_norm:
                p = F.normalize(p) # l2 normalise the predicted pose before feedback
        return predictions


# class SEFeedback_34(nn.Module):
#     "create a senet where feedback is used to condition the seblock"

#     def __init__(self, num_feedback_cycles=2, feedback_norm=False, disable_SE_layer=False):
#         super(SEFeedback_34, self).__init__()

#         self.base_net = ResNet(SEBasicBlock, [3, 4, 6, 3], 51, disable_SE_layer)
#         self.base_net.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.num_feedback_cycles = num_feedback_cycles
#         self.feedback_norm = feedback_norm

#     def forward(self, x):  
#         predictions = []      
#         p = Variable(torch.zeros(51))
#         for i in range(self.num_feedback_cycles):
#             p = self.base_net(x,p)
#             predictions.append(p)
#             if self.feedback_norm:
#                 p = F.normalize(p) # l2 normalise the predicted pose before feedback
#         return predictions

class SEFeedback_50(nn.Module):
    "create a senet where feedback is used to condition the seblock"

    def __init__(self, num_feedback_cycles=2, feedback_norm=False, disable_SE_layer=False):
        super(SEFeedback_50, self).__init__()

        self.base_net = ResNet(SEBottleneck, [3, 4, 6, 3], 51, disable_SE_layer)
        self.base_net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_feedback_cycles = num_feedback_cycles
        self.feedback_norm = feedback_norm

    def forward(self, x):  
        predictions = []      
        p = Variable(torch.zeros(51))
        for i in range(self.num_feedback_cycles):
            p = self.base_net(x,p)
            predictions.append(p)
            if self.feedback_norm:
                p = F.normalize(p) # l2 normalise the predicted pose before feedback
        return predictions 


class SEFeedback_34_non_shared(nn.Module):
    "create a senet where feedback is used to condition the seblock"

    def __init__(self, num_feedback_cycles=2, feedback_norm=False, disable_SE_layer=False):
        super(SEFeedback_34_non_shared, self).__init__()

        self.base_net_1 = ResNet(SEBasicBlock, [3, 4, 6, 3], 51, disable_SE_layer)
        self.base_net_1.avgpool = nn.AdaptiveAvgPool2d(1)

        self.base_net_2 = ResNet(SEBasicBlock, [3, 4, 6, 3], 51, disable_SE_layer)
        self.base_net_2.avgpool = nn.AdaptiveAvgPool2d(1)

        self.base_net_3 = ResNet(SEBasicBlock, [3, 4, 6, 3], 51, disable_SE_layer)
        self.base_net_3.avgpool = nn.AdaptiveAvgPool2d(1)

        self.num_feedback_cycles = num_feedback_cycles
        self.feedback_norm = feedback_norm

    def forward(self, x):  
        predictions = []      
        p = Variable(torch.zeros(51))
        for i in range(self.num_feedback_cycles):
            if i == 0:
                p = self.base_net_1(x,p)
            elif i == 1:
                p = self.base_net_2(x,p)
            elif i == 2:
                p = self.base_net_3(x,p)
            predictions.append(p)
            if self.feedback_norm:
                p = F.normalize(p) # l2 normalise the predicted pose before feedback
        return predictions


class SEFeedback_18_non_shared(nn.Module):
    "create a senet where feedback is used to condition the seblock"

    def __init__(self, num_feedback_cycles=2, feedback_norm=False, disable_SE_layer=False):
        super(SEFeedback_18_non_shared, self).__init__()

        self.senets = nn.ModuleList()
        for i in range(num_feedback_cycles):
            tmp = ResNet(SEBasicBlock, [2, 2, 2, 2], 51, disable_SE_layer)
            tmp.avgpool = nn.AdaptiveAvgPool2d(1)
            self.senets.append(tmp)

        self.num_feedback_cycles = num_feedback_cycles
        self.feedback_norm = feedback_norm

    def forward(self, x):  
        predictions = []      
        p = Variable(torch.zeros(51))
        for i in range(self.num_feedback_cycles):            
            p = self.senets[i](x,p)

            predictions.append(p)
            if self.feedback_norm:
                p = F.normalize(p) # l2 normalise the predicted pose before feedback
        return predictions

def se_resnet18(num_classes, se_module_type=1):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, se_module_type=se_module_type)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes, se_module_type=1):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, se_module_type=se_module_type)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

