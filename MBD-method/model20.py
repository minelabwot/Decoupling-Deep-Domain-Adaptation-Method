import torch
import torch.nn as nn
from torch.autograd import Function
from collections import Counter
import torch.nn.functional as F
import numpy as np

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.dis1 = nn.Linear(in_features=320, out_features=128, bias=True)
        self.dis2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.dis3 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, x):
        x = self.dis1(x)
        x = F.relu(x)
        x = self.dis2(x)
        x = F.relu(x)
        x = self.dis3(x)
        y = torch.sigmoid(x)
        return y


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=16, stride=4)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.float()
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        return x


class Classifier(nn.Module):
    def __init__(self, classnum=6):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=320, out_features=128, bias=True)
        self.fc2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc3 = nn.Linear(in_features=64, out_features=classnum, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        y = self.fc3(x)
        return y


class DANN(nn.Module):

    def __init__(self, device, classnum):
        super(DANN, self).__init__()
        self.device = device
        self.feature = FeatureExtractor()
        self.classifier = Classifier(classnum)
        self.domain_classifier = Discriminator(input_dim=320, hidden_dim=128)

    def forward(self, input_data, alpha=1, source=True, only_resampled=False):
        if (only_resampled == False):           
            feature = self.feature(input_data)
            tem_feature = feature
            feature = feature.view(-1, 320)
            class_output = self.classifier(feature)
            domain_output, domain_pred = self.get_adversarial_result(feature, source, alpha)
            return class_output, domain_output, tem_feature, None, None, domain_pred
        else:
            input_data = input_data.view(-1, 320)
            class_output = self.classifier(input_data)
            domain_output, domain_pred = self.get_adversarial_result(input_data, source, alpha)
            return class_output, domain_output, None, None, None, domain_pred
        
    def get_adversarial_result(self, x, source=True, alpha=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(self.device)
        else:
            domain_label = torch.zeros(len(x)).long().to(self.device)
        x = ReverseLayerF.apply(x, alpha)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.unsqueeze(1).float())
        return loss_adv, domain_pred

class CNN(nn.Module):
    def __init__(self, device, classnum):
        super(CNN, self).__init__()
        self.device = device
        self.feature = FeatureExtractor()
        self.classifier = Classifier(classnum)

    def forward(self, input_data):
        feature = self.feature(input_data)
        tem_feature = feature
        feature = feature.view(-1, 320)
        class_output = self.classifier(feature)
        return class_output, tem_feature, None, None
