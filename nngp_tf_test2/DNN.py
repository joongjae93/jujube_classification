# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import copy
import time


class Net(nn.Module):
    def __init__(self, depth, w_std, b_std):
        super(Net, self).__init__()
        self.depth = depth
        self.fc1 = nn.Linear(784, self.depth)
        self.fc2 = nn.Linear(self.depth, self.depth)
        self.fc3 = nn.Linear(self.depth, 10)
        
        '''self.w_std = w_std
        self.b_std = b_std
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=self.w_std)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=self.w_std)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=self.w_std)
        torch.nn.init.normal_(self.fc1.bias, mean=0.0, std=self.b_std)
        torch.nn.init.normal_(self.fc2.bias, mean=0.0, std=self.b_std)
        torch.nn.init.normal_(self.fc3.bias, mean=0.0, std=self.b_std)'''

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)

        return output


def train(args, model, device, train_loader, optimizer, epoch, start_time):
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
    
    print('Train Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n, Time: {:.2f}\n'.format(
        epoch, train_loss, corrects, len(train_loader.sampler), train_acc, (time.time()-start_time)/60))
    
    return train_loss, train_acc


def test(model, device, test_loader, epoch, start_time):
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

    print('Test Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n, Time: {:.2f}\n'.format(
        epoch, test_loss, corrects, len(test_loader.sampler), test_acc, (time.time()-start_time)/60))

    return test_loss, test_acc, loss


def create_answer_sheet(model, device, test_loader):
    model.eval()
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
    answer_sheet = np.zeros((10, 11))
    for i in range(0, len(pred)):
        answer_sheet[answer[i].item()][pred[i]] += 1
        answer_sheet[answer[i].item()][-1] += 1
    return answer_sheet


def one_hot(target):
    NUM_CLASS = 10
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
    start_time = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--depth', type=int, default=5000, metavar='N',
                        help='depth of hidden layers (default: 2000)')
    parser.add_argument('--numdata', type=int, default=60000, metavar='N',
                        help='number of data to train (default: 60000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--l2', type=float, default=0.01, metavar='L2',
                        help='L2 penalty (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    trainlist = range(0, args.numdata)

    dataset1 = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]), target_transform=one_hot)
    dataset1 = Subset(dataset1, trainlist)
    dataset2 = datasets.MNIST(root='./MNIST', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]), target_transform=one_hot)
    train_loader = DataLoader(dataset1, batch_size=256, pin_memory=True)
    test_loader = DataLoader(dataset2, batch_size=256, pin_memory=True)
    model = Net(args.depth, 0, 0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)

    trainlosslist = []
    trainacclist = []
    testlosslist = []
    testacclist = []
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        current_lr = get_lr(optimizer)
        print('Epoch {}/{}, current lr= {}'.format(epoch, args.epochs, current_lr))
        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, start_time)
        test_loss, test_acc, loss = test(model, device, test_loader, epoch, start_time)
        trainlosslist.append(np.log10(train_loss))
        trainacclist.append(train_acc)
        testlosslist.append(np.log10(test_loss))
        testacclist.append(test_acc)
        
        if loss < best_loss:
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, './weights{}{}'.format(args.numdata, args.depth))
            print('Copied best model weights!')

        lr_scheduler.step(loss)
        if current_lr != get_lr(optimizer):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)
            
    answer_sheet = create_answer_sheet(model, device, test_loader)
    
    createFolder('./DNN')
    createFolder('./DNN/results{}{}'.format(args.depth, args.numdata))
    df = pd.DataFrame(answer_sheet, index=['0','1','2','3','4','5','6','7','8','9'],
                      columns=['0','1','2','3','4','5','6','7','8','9','총계'])
    df.to_excel('./DNN/results{}{}/answer_sheet.xlsx'.format(args.depth, args.numdata))
    with open('./DNN/results{}{}/loss.txt'.format(args.depth, args.numdata), "w") as f:
        f.write("Train Loss\n")
        f.write(str(trainlosslist))
        f.write("\n\n")
        f.write("Test Loss\n")
        f.write(str(testlosslist))

    with open('./DNN/results{}{}/accuracy.txt'.format(args.depth, args.numdata), "w") as f:
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
    plt.savefig('./DNN/results{}{}/loss.jpg'.format(args.depth, args.numdata))
    
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
    plt.savefig('./DNN/results{}{}/accuracy.jpg'.format(args.depth, args.numdata))

    if args.save_model:
        save_checkpoint(model, optimizer, './DNN/weights.pt')


if __name__ == '__main__':
    main()
