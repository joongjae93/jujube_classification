# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import copy


class Net(nn.Module):
    def __init__(self, size, depth, w_std, b_std):
        super(Net, self).__init__()
        self.size = size
        self.depth = depth
        self.fc1 = nn.Linear(self.size*self.size*3, self.depth)
        self.fc2 = nn.Linear(self.depth, self.depth)
        self.fc3 = nn.Linear(self.depth, 5)
        self.bn = nn.BatchNorm1d(self.depth)
        
        self.w_std = w_std
        self.b_std = b_std
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=self.w_std)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=self.w_std)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=self.w_std)
        torch.nn.init.normal_(self.fc1.bias, mean=0.0, std=self.b_std)
        torch.nn.init.normal_(self.fc2.bias, mean=0.0, std=self.b_std)
        torch.nn.init.normal_(self.fc3.bias, mean=0.0, std=self.b_std)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn(x)
        x = F.relu(x)
        output = self.fc3(x)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    corrects = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += F.mse_loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            answer = target.argmax(dim=1, keepdim=True) 
            corrects += pred.eq(answer).sum().item()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader.sampler), loss.item()))

    train_loss /= len(train_loader.sampler)
    train_acc = 100*corrects/len(train_loader.sampler)
    
    print('Train Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, train_loss, corrects, len(train_loader.sampler), train_acc))
    
    return train_loss, train_acc


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    corrects = 0
    loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target).item()
            test_loss += F.mse_loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            answer = target.argmax(dim=1, keepdim=True) 
            corrects += pred.eq(answer).sum().item()

    loss /= len(test_loader.sampler)
    test_loss /= len(test_loader.sampler)
    test_acc = 100*corrects/len(test_loader.sampler)

    print('Test Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, corrects, len(test_loader.sampler), test_acc))

    return test_loss, test_acc, loss


def create_answer_sheet(model, device, test_loader):
    with torch.no_grad():
        num = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if num == 0:
                pred = copy.deepcopy(output)
                answer = copy.deepcopy(target)
            else:
                pred = torch.cat((pred, output), dim=0)
                answer = torch.cat((answer, target), dim=0)
            num += 1
    answer_sheet = metric_batch_detail(pred, answer)

    return answer_sheet


def metric_batch_detail(output, target):
    pred = output.argmax(1, keepdim=True)
    answer = target.argmax(dim=1, keepdim=True) 
    answer_sheet = np.zeros((5, 6))
    for i in range(0, len(pred)):
        answer_sheet[answer[i].item()][pred[i]] += 1
        answer_sheet[answer[i].item()][-1] += 1        
    return answer_sheet


def one_hot(target):
    NUM_CLASS = 5
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_checkpoint(model, optimizer, path):
    state = {
	'state_dict': model.state_dict(),
	'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)
    
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--size', type=int, default=88, metavar='N',
                        help='input image height (default: 88)')
    parser.add_argument('--depth', type=int, default=5000, metavar='N',
                        help='depth of hidden layers (default: 5000)')
    parser.add_argument('--numdata', type=float, default=1, metavar='N',
                        help='rate of data to use in training (default: 1)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--l2', type=float, default=0.01, metavar='L2',
                        help='L2 penalty (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    trans = transforms.Compose([
                        transforms.Resize((args.size, args.size), interpolation=2), 
                        transforms.ToTensor()])
  
    dataset = datasets.ImageFolder(root = '/home/kook/classification/data', transform = trans, target_transform = one_hot)
  
    idx_0 = list(range(0, 1015))
    idx_1 = list(range(1016, 2014))
    idx_2 = list(range(2015, 6846))
    idx_3 = list(range(6847, 14699))
    idx_4 = list(range(14700, 18035))

    np.random.shuffle(idx_0)
    idx_0_1, idx_0_2 = idx_0[200:], idx_0[:200]
    np.random.shuffle(idx_1)
    idx_1_1, idx_1_2 = idx_1[200:], idx_1[:200]
    np.random.shuffle(idx_2)
    idx_2_1, idx_2_2 = idx_2[:800], idx_2[800:1000]
    np.random.shuffle(idx_3)
    idx_3_1, idx_3_2 = idx_3[:800], idx_3[800:1000]
    np.random.shuffle(idx_4)
    idx_4_1, idx_4_2 = idx_4[:800], idx_4[800:1000]

    train_idx = idx_0_1 + idx_1_1 + idx_2_1 + idx_3_1 + idx_4_1
    test_idx = idx_0_2 + idx_1_2 + idx_2_2 + idx_3_2 + idx_4_2
   
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
  
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, pin_memory=True)

    model = Net(args.size, args.depth, 0, 0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.l2)
    #lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)

    trainlosslist = []
    trainacclist = []
    testlosslist = []
    testacclist = []
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        current_lr = get_lr(optimizer)
        print('Epoch {}/{}, current lr= {}'.format(epoch, args.epochs, current_lr))
        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_acc, loss = test(model, device, test_loader, epoch)
        trainlosslist.append(np.log10(train_loss))
        trainacclist.append(train_acc)
        testlosslist.append(np.log10(test_loss))
        testacclist.append(test_acc)
        
        if loss < best_loss:
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, './weights{}{}'.format(args.size, args.depth))
            print('Copied best model weights!')
    '''
        lr_scheduler.step(loss)
        if current_lr != get_lr(optimizer):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)
    '''    
    answer_sheet = create_answer_sheet(model, device, test_loader)
    
    createFolder('./DNN')
    createFolder('./DNN/results{}{}'.format(args.depth, args.size))
    df = pd.DataFrame(answer_sheet, index=['혹파리', '붉은점박이', '마그네슘', '정상', '노린재'],
                      columns=['혹파리', '붉은점박이', '마그네슘', '정상', '노린재', '총계'])
    df.to_excel('./DNN/results{}{}/answer_sheet.xlsx'.format(args.depth, args.size))
    with open('./DNN/results{}{}/loss.txt'.format(args.depth, args.size), "w") as f:
        f.write("Train Loss\n")
        f.write(str(trainlosslist))
        f.write("\n\n")
        f.write("Test Loss\n")
        f.write(str(testlosslist))

    with open('./DNN/results{}{}/accuracy.txt'.format(args.depth, args.size), "w") as f:
        f.write("Train Loss\n")
        f.write(str(trainacclist))
        f.write("\n\n")
        f.write("Test Loss\n")
        f.write(str(testacclist))
    
    num_epochs = args.epochs
    # Plot train-val loss
    plt.figure(figsize=(12.8, 9.6))
    plt.title('Train-Test Loss\n', fontsize=21)
    plt.plot(range(1, num_epochs+1), trainlosslist, label='train')
    plt.plot(range(1, num_epochs+1), testlosslist, label='test')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Loss\n', fontsize=21)
    plt.xlabel('\nTraining Epochs', fontsize=21)
    plt.legend(fontsize=15)
    plt.savefig('./DNN/results{}{}/loss.jpg'.format(args.depth, args.size))
    
    # plot train-val accuracy
    plt.figure(figsize=(12.8, 9.6))
    plt.title('Train-Test Accuracy\n', fontsize=21)
    plt.plot(range(1, num_epochs+1), trainacclist, label='train')
    plt.plot(range(1, num_epochs+1), testacclist, label='test')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Accuracy\n', fontsize=21)
    plt.xlabel('\nTraining Epochs', fontsize=21)
    plt.legend(fontsize=15)
    plt.savefig('./DNN/results{}{}/accuracy.jpg'.format(args.depth, args.size))

    if args.save_model:
        save_checkpoint(model, optimizer, './DNN/weights.pt')


if __name__ == '__main__':
    main()
