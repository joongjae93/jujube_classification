# -*- coding: utf-8 -*-

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import copy


def one_hot(target):
  NUM_CLASS = 5
  label = np.eye(NUM_CLASS)*10-1
  one_hot = label[target]
  return one_hot

def load_mnist(use_float=False,
               mean_subtraction=False,
               random_roated_labels=False):
  
  torch.manual_seed(1)
  np.random.seed(1)

    
  # define transformation
  trans = transforms.Compose([
                      transforms.Resize((350, 350), interpolation=2), 
                      transforms.ToTensor()])
  
  dataset = datasets.ImageFolder(root = '/home/kook/classification/data', transform = trans, target_transform = one_hot)
  '''
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
  train_len, test_len = len(train_idx), len(test_idx)
   
  train_sampler = SubsetRandomSampler(train_idx)
  test_sampler = SubsetRandomSampler(test_idx)
  
  trainloader = DataLoader(dataset, batch_size=train_len, sampler=train_sampler)
  testloader = DataLoader(dataset, batch_size=test_len, sampler=test_sampler)
  '''
  num_data = len(dataset)
  indices = list(range(num_data))
  np.random.shuffle(indices)
  split = int(np.floor(0.2 * num_data))
  split_train = int(np.floor((num_data-split)))
  train_idx, test_idx = indices[split:], indices[:split]
  train_idx = train_idx[:split_train]
  train_len, test_len = len(train_idx), len(test_idx)
  
  train_sampler = SubsetRandomSampler(train_idx)
  test_sampler = SubsetRandomSampler(test_idx)
  
  trainloader = DataLoader(dataset, batch_size=train_len, sampler=train_sampler)
  testloader = DataLoader(dataset, batch_size=test_len, sampler=test_sampler)
  
  trainimage, trainlabel = next(iter(trainloader))
  testimage, testlabel = next(iter(testloader))
  trainimage = torch.flatten(trainimage, start_dim=1).numpy()
  testimage = torch.flatten(testimage, start_dim=1).numpy()
  trainlabel = trainlabel.numpy()
  testlabel = testlabel.numpy()
  print(trainimage.shape, trainlabel.shape, testimage.shape, testlabel.shape)
  
  if mean_subtraction:
    train_image_mean = np.mean(trainimage)
    train_label_mean = np.mean(trainlabel)
    trainimage -= train_image_mean
    trainlabel -= train_label_mean
    testimage -= train_image_mean
    testlabel -= train_label_mean
  
  return trainimage, trainlabel, testimage, testlabel, train_len, test_len
