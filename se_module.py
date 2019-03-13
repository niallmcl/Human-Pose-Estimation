from torch import nn
import math
import torch
import pdb

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.num_hidden = math.floor(channel / reduction)
        self.fc = nn.Sequential(            
                nn.Linear(channel, self.num_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_hidden, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SELayer_tanh(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_tanh, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.num_hidden = math.floor(channel / reduction)
        self.fc = nn.Sequential(            
                nn.Linear(channel, self.num_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_hidden, channel),
                nn.Tanh()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SELayer_addmul(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_addmul, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.num_hidden = math.floor(channel / reduction)
        
        self.fc1 = nn.Sequential(            
                nn.Linear(channel, self.num_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_hidden, channel),
                nn.Tanh()
        )

        self.fc2 = nn.Sequential(            
                nn.Linear(channel, self.num_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_hidden, channel),
                nn.Tanh()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        a = self.avg_pool(x).view(b, c)

        y = self.fc1(a).view(b, c, 1, 1)
        z = self.fc2(a).view(b, c, 1, 1)

        return x * y + z

class SELayer_feedback(nn.Module):
    def __init__(self, channel, reduction=16, feedback_size=51):
        super(SELayer_feedback, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.num_hidden = math.floor(channel / reduction)
        self.feedback_size = feedback_size
        self.fc = nn.Sequential(            
                nn.Linear(channel + self.feedback_size, self.num_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_hidden, channel),
                nn.Sigmoid()
        )

    def forward(self, x, p):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.cat((y,p.expand(b,self.feedback_size)),1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# class SELayer_feedback(nn.Module):
#     def __init__(self, channel, reduction=16, feedback_size=51):
#         super(SELayer_feedback, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.num_hidden = math.floor(channel / reduction)
#         self.feedback_size = feedback_size

#         self.fc1 = nn.Sequential(            
#                 nn.Linear(channel + self.feedback_size, self.num_hidden),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(self.num_hidden, channel),
#                 nn.Sigmoid()
#         )

#         self.fc2 = nn.Sequential(            
#                 nn.Linear(channel + self.feedback_size, self.num_hidden),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(self.num_hidden, channel),
#                 nn.Tanh()
#         )

#     def forward(self, x, p):
#         b, c, _, _ = x.size()

#         a = self.avg_pool(x).view(b, c)
#         a = torch.cat((a,p.expand(b,self.feedback_size)),1)

#         y = self.fc1(a).view(b, c, 1, 1)
#         z = self.fc2(a).view(b, c, 1, 1)

#         return x * y + z