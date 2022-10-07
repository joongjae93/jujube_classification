# -*- coding: utf-8 -*-

import torch
from torch import optim
# 학습된 모델의 정보를 이어받아서 추가로 학습하기를 원한다면 옵티마이저도 같이 불러와야 한다.
# 현재 코드에서는 추가 학습 파트는 포함시키지 않음. 

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import utils # 데이터 시각화에 필요한 모듈. 현재 코드에서는 시각화 파트를 포함시키지 않음.

import numpy as np
import pandas as pd
import copy
import os

from efficientnet import EfficientNet

torch.manual_seed(1)
np.random.seed(1)

# define transformation
trans = transforms.Compose([
                    transforms.Resize((224, 224), transforms.functional.InterpolationMode.BILINEAR), 
                    transforms.ToTensor()
])

# dataset loading
dataset = datasets.ImageFolder(root = './data', transform = trans)
num_data = len(dataset)
indices = list(range(num_data))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_data))
train_idx, test_idx = indices[split:], indices[:split]
train_len, test_len = len(train_idx), len(test_idx)
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

trainloader = DataLoader(dataset, batch_size=256, sampler=train_sampler, pin_memory=True)
testloader = DataLoader(dataset, batch_size=256, sampler=test_sampler, pin_memory=True)

# define model
def efficientnet_b0(num_classes=5):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.0, scale=1.0,dropout=0.2, se_scale=4)

def efficientnet_b1(num_classes=5):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.1, scale=240/224, dropout=0.2, se_scale=4)

def efficientnet_b2(num_classes=5):
    return EfficientNet(num_classes=num_classes, width_coef=1.1, depth_coef=1.2, scale=260/224., dropout=0.3, se_scale=4)

def efficientnet_b3(num_classes=5):
    return EfficientNet(num_classes=num_classes, width_coef=1.2, depth_coef=1.4, scale=300/224, dropout=0.3, se_scale=4)

def efficientnet_b4(num_classes=5):
    return EfficientNet(num_classes=num_classes, width_coef=1.4, depth_coef=1.8, scale=380/224, dropout=0.4, se_scale=4)

def efficientnet_b5(num_classes=5):
    return EfficientNet(num_classes=num_classes, width_coef=1.6, depth_coef=2.2, scale=456/224, dropout=0.4, se_scale=4)

def efficientnet_b6(num_classes=5):
    return EfficientNet(num_classes=num_classes, width_coef=1.8, depth_coef=2.6, scale=528/224, dropout=0.5, se_scale=4)

def efficientnet_b7(num_classes=5):
    return EfficientNet(num_classes=num_classes, width_coef=2.0, depth_coef=3.1, scale=600/224, dropout=0.5, se_scale=4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 단 predict 작업만 필요한 경우 device는 cpu를 쓰는 것이 빠르다.
# parser argument를 이용하여 외부에서 device 선택 및, num_workers 패러미터를 설정하여 multiprocessing이 가능하다 이 코드에는 포함되지 않음.


# initialize model
model = efficientnet_b2()
#opt = optim.Adam(model.parameters(), lr=0.01)


# loading trained model
modelpath = './models/weights2.pt'

if torch.cuda.is_available():
    model_checkpoint = torch.load(modelpath)
    model.load_state_dict(model_checkpoint['state_dict'])
#    opt.load_state_dict(model_checkpoint['optimizer'])
    model.to(device)
else:
    model_checkpoint = torch.load(modelpath, map_location='cpu')
    model.load_state_dict(model_checkpoint['state_dict'])
#    opt.load_state_dict(model_checkpoint['optimizer'])
    model.to(device)


def metric_batch_detail(output, target):
    pred = output.argmax(1, keepdim=True)
    target = target.view_as(pred)
    answer_sheet = np.zeros((5, 7))
    for i in range(0, len(pred)):
        answer_sheet[target[i].item()][pred[i]] += 1
        answer_sheet[target[i].item()][-2] += 1
    for i2 in range(0, 5):
        answer_sheet[i2][-1] = answer_sheet[i2][i2]/answer_sheet[i2][-2]
    return answer_sheet


def predict(model):
    model.eval()
    with torch.no_grad():
        num = 0
        for xb, yb in testloader:
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)
            if num == 0:
                pred = copy.deepcopy(output)
                target = copy.deepcopy(yb)
            else:
                pred = torch.cat((pred, output), dim=0)
                target = torch.cat((target, yb), dim=0)
            num += 1
    answer_sheet = metric_batch_detail(pred, target)
    return answer_sheet


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSerror:
        print('Error')


if __name__ == '__main__':
    createFolder('./models')
    createFolder('./models/results')
    answer_sheet = predict(model)
    df = pd.DataFrame(answer_sheet, index=['혹파리', '붉은점박이', '마그네슘', '정상', '노린재'],
                      columns=['혹파리', '붉은점박이', '마그네슘', '정상', '노린재', '총계', '정답률'])
    df.to_excel('./models/results/answer_sheet.xlsx')
