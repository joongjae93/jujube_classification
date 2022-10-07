# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=32, stride=2)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=16, stride=2)
    self.conv3 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=8, stride=2)
    self.conv4 = nn.Conv2d(in_channels=1024, out_channels=4096, kernel_size=5, stride=1)
    self.bn1 = nn.BatchNorm2d(num_features=16)
    self.bn2 = nn.BatchNorm2d(num_features=128)
    self.bn3 = nn.BatchNorm2d(num_features=1024)
    self.bn4 = nn.BatchNorm2d(num_features=4096)

  
  def forward(self, x):
    batch_size, _, _, _ = x.shape
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = x.view(batch_size, -1)

    return x
  
