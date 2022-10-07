# -*- coding: utf-8 -*-

# import package

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
import pandas as pd
import time
import copy
import os

#torch.set_printoptions(threshold=np.inf) # 데이터를 출력해보기 위한 코드로 비활성화 상태로 두셔도 됩니다.

torch.manual_seed(1)
np.random.seed(1)

# define transformation
trans = transforms.Compose([
                    transforms.Resize((350, 350), transforms.functional.InterpolationMode.BILINEAR), 
                    transforms.ToTensor()
]) # 다운 샘플링(BILINEAR 옵션은 디폴트라서 따로 지정해 줄 필요는 없으나 지정하는 방법을 기록할 겸 작성함.)

dataset = datasets.ImageFolder(root = '', transform = trans) # 경로 입력 필요
num_data = len(dataset)
indices = list(range(num_data))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_data))
train_idx, test_idx = indices[split:], indices[:split]
train_len, test_len = len(train_idx), len(test_idx)
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)


trainloader = DataLoader(dataset, batch_size=128, sampler=train_sampler, pin_memory=True)
testloader = DataLoader(dataset, batch_size=128, sampler=test_sampler, pin_memory=True)

# check sample images

'''
def show(img, y=None):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg_tr)

    if y is not None:
        plt.title('labels:' + str(y))

np.random.seed(10)
torch.manual_seed(0)

grid_size=4
rnd_ind = np.random.randint(0, len(trainset), grid_size)

x_grid = [trainset[i][0] for i in rnd_ind]
y_grid = [trainset[i][1] for i in rnd_ind]

x_grid = utils.make_grid(x_grid, nrow=grid_size, padding=2)
plt.figure(figsize=(10,10))
show(x_grid, y_grid)
'''

# Swish activation function
class Swish(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
      

# SE Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels//r),
            Swish(),
            nn.Linear(in_channels//r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


class MBConv(nn.Module):
    expand = 6
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        # first MBConv is not using stochastic depth
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * MBConv.expand, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish(),
            nn.Conv2d(in_channels * MBConv.expand, in_channels * MBConv.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*MBConv.expand),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish()
        )

        self.se = SEBlock(in_channels * MBConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*MBConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)

        x_se = self.se(x_residual)
        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x


class SepConv(nn.Module):
    expand = 1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        # first SepConv is not using stochastic depth
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels * SepConv.expand, in_channels * SepConv.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*SepConv.expand),
            nn.BatchNorm2d(in_channels * SepConv.expand, momentum=0.99, eps=1e-3),
            Swish()
        )

        self.se = SEBlock(in_channels * SepConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*SepConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)

        x_se = self.se(x_residual)
        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes=5, width_coef=1., depth_coef=1., scale=1., dropout=0.2, se_scale=4, stochastic_depth=False, p=0.5):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        depth = depth_coef
        width = width_coef

        channels = [int(x*width) for x in channels]
        repeats = [int(x*depth) for x in repeats]

        # stochastic depth
        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.p = 1
            self.step = 0

        # efficient net
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels[0],3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3)
        )

        self.stage2 = self._make_Block(SepConv, repeats[0], channels[0], channels[1], kernel_size[0], strides[0], se_scale)

        self.stage3 = self._make_Block(MBConv, repeats[1], channels[1], channels[2], kernel_size[1], strides[1], se_scale)

        self.stage4 = self._make_Block(MBConv, repeats[2], channels[2], channels[3], kernel_size[2], strides[2], se_scale)

        self.stage5 = self._make_Block(MBConv, repeats[3], channels[3], channels[4], kernel_size[3], strides[3], se_scale)

        self.stage6 = self._make_Block(MBConv, repeats[4], channels[4], channels[5], kernel_size[4], strides[4], se_scale)

        self.stage7 = self._make_Block(MBConv, repeats[5], channels[5], channels[6], kernel_size[5], strides[5], se_scale)

        self.stage8 = self._make_Block(MBConv, repeats[6], channels[6], channels[7], kernel_size[6], strides[6], se_scale)

        self.stage9 = nn.Sequential(
            nn.Conv2d(channels[7], channels[8], 1, stride=1, bias=False),
            nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
            Swish()
        ) 

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(channels[8], num_classes)

    def forward(self, x):
        x = self.upsample(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def _make_Block(self, block, repeats, in_channels, out_channels, kernel_size, stride, se_scale):
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, kernel_size, stride, se_scale, self.p))
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)


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


# construct model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = efficientnet_b2().to(device)


# define loss function, optimizer, lr_scheduler
loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=0.01)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)


# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    
    return loss_b.item(), metric_b


# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    return running_loss, running_metric


# calculate the detail metric per mini-batch
def metric_batch_detail(output, target):
    pred = output.argmax(1, keepdim=True)
    target = target.view_as(pred)
    answer_sheet = np.zeros((5, 6))
    for i in range(0, len(pred)):
        answer_sheet[target[i].item()][pred[i]] += 1
        answer_sheet[target[i].item()][-1] += 1
    return answer_sheet


# function to start training
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    trainloader=params['trainloader']
    testloader=params['testloader']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights=params['path2weights']

    loss_history = {'train': [], 'test': []}
    metric_history = {'train': [], 'test': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr= {}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, trainloader, sanity_check, opt)
        loss_history['train'].append(train_loss/train_len)
        metric_history['train'].append(train_metric/train_len)

        model.eval()
        with torch.no_grad():
            test_loss, test_metric = loss_epoch(model, loss_func, testloader, sanity_check)
        loss_history['test'].append(test_loss/test_len)
        metric_history['test'].append(test_metric/test_len)

        # 추가분
        if epoch == num_epochs-1:
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

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(epoch, model, opt, path2weights)
            print('Copied best model weights!')

        lr_scheduler.step(test_loss)
        if current_lr != get_lr(opt):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        print('train loss: %.6f, test loss: %.6f, accuracy: %.4f, time: %.4f min' %(train_loss/train_len, test_loss/test_len, 100*test_metric/test_len, (time.time()-start_time)/60))
        print('-'*10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history, answer_sheet

# define the training parameters
params_train = {
    'num_epochs':100,
    'optimizer':opt,
    'loss_func':loss_func,
    'trainloader':trainloader,
    'testloader':testloader,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}

# check the directory to save weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSerror:
        print('Error')
        
        
def save_checkpoint(epoch, model, optimizer, path):
    state = {
	'state_dict': model.state_dict(),
	'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


if __name__ == '__main__':
    createFolder('./models')
    createFolder('./models/results')
    model, loss_hist, metric_hist, answer_sheet = train_val(model, params_train)
    df = pd.DataFrame(answer_sheet, index=['혹파리', '마그네슘', '정상', '노린재', '붉은점박이'],
                      columns=['혹파리', '마그네슘', '정상', '노린재', '붉은점박이', '총계'])
    df.to_excel('./models/results/answer_sheet.xlsx')
    with open('./models/results/loss.txt', "w") as f:
        f.write("Train Loss\n")
        f.write(str(loss_hist['train']))
        f.write("\n\n")
        f.write("Test Loss\n")
        f.write(str(loss_hist['test']))

    with open('./models/results/accuracy.txt', "w") as f:
        f.write("Train Loss\n")
        f.write(str(metric_hist['train']))
        f.write("\n\n")
        f.write("Test Loss\n")
        f.write(str(metric_hist['test']))
    
    num_epochs = 100
    # Plot train-val loss
    plt.figure(figsize=(12.8, 9.6))
    plt.title('Train-Test Loss', fontsize=15)
    plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
    plt.plot(range(1, num_epochs+1), loss_hist['test'], label='test')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Loss', fontsize=15)
    plt.xlabel('Training Epochs', fontsize=15)
    plt.legend()
    plt.savefig('./models/results/loss.jpg')
    
    # plot train-val accuracy
    plt.figure(figsize=(12.8, 9.6))
    plt.title('Train-Test Accuracy', fontsize=15)
    plt.plot(range(1, num_epochs+1), metric_hist['train'], label='train')
    plt.plot(range(1, num_epochs+1), metric_hist['test'], label='test')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('Training Epochs', fontsize=15)
    plt.legend()
    plt.savefig('./models/results/accuracy.jpg')
