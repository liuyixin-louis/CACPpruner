from lib.utils import  AverageMeter,accuracy
import time 
import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import os

class Cifar10_Valider:
    def __init__(self):
        self.val_loader = self.get_split_dataset()
        
    def validate(self, model, verbose=True):
        '''
        Validate the performance on validation set
        :param val_loader:
        :param model:
        :param verbose:
        :return:
        '''
        val_loader=self.val_loader
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        criterion = nn.CrossEntropyLoss().cuda()
        # switch to evaluate mode
        model.eval()
        end = time.time()

        t1 = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target).cuda()

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.3f' %
                  (losses.avg, top1.avg, top5.avg, t2 - t1))
    def get_split_dataset(self, batch_size=128, n_worker=0, val_size=5000, data_root='C:\\Users\\lenovo\\dataset\\cifar',\
                      shuffle=True):
        '''
            split the train set into train / val for rl search
        '''
        if shuffle:
            index_sampler = SubsetRandomSampler
        else:  # every time we use the same order for the split subset
            class SubsetSequentialSampler(SubsetRandomSampler):
                def __iter__(self):
                    return (self.indices[i] for i in torch.arange(len(self.indices)).int())
            index_sampler = SubsetSequentialSampler

#         print('=> Preparing data: {}...'.format(dset_name))
        
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
#         trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)

        valset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        n_val = len(valset)
        assert val_size < n_val
        indices = list(range(n_val))
        np.random.shuffle(indices)
        _, val_idx = indices[val_size:], indices[:val_size]
#         train_idx = list(range(len(trainset)))  # all train set for train

#         train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

#         train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
#                                                    num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 10

        return val_loader

if __name__ == "__main__":
    cv = Cifar10_Valider()
    # cv.validate()