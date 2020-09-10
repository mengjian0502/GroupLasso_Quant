"""
model training
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import models
import logging
from torchsummary import summary

from utils import *
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/ImageNet Training')
parser.add_argument('--model', type=str, help='model type')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')

parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--lr_decay', type=str, default='step', help='mode for learning rate decay')
parser.add_argument('--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None,
                    help='path to log file')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

# Acceleration
parser.add_argument('--ngpu', type=int, default=3, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

# Fine-tuning
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')

# quantization
parser.add_argument('--wbit', type=int, default=4, help='weight precision')
parser.add_argument('--abit', type=int, default=4, help='activation precision')
parser.add_argument('--alpha_init', type=int, default=10., help='initial activation clipping')
parser.add_argument('--q_mode', type=str, default="mean", help='weight quantization mode')
parser.add_argument('--k', type=int, default=2, help='coefficient of quantization boundary')

# activation clipping(PACT)
parser.add_argument('--clp', dest='clp', action='store_true', help='using clipped relu in each stage')
parser.add_argument('--a_lambda', type=float, default=0.01, help='The parameter of alpha L2 regularization')

# structured pruning
parser.add_argument('--swp', dest='swp',
                    action='store_true', help='using structured pruning')
parser.add_argument('--lamda', type=float,
                    default=0.001, help='The parameter for swp.')
parser.add_argument('--lamda_lin', type=float,
                    default=0.001, help='The parameter for swp of the fc layer.')
parser.add_argument('--ratio', type=float,
                    default=0.25, help='pruned ratio')
parser.add_argument('--group_ch', type=int,
                    default=16, help='group size for group lasso')
parser.add_argument('--targeted_group', type=int,
                    default=8, help='targeted group size for mapping')
parser.add_argument('--g_multi', type=float, nargs='+', default=[2, 2],
                    help='group size is multiplied by g_multi on schedule, number of g_multi should be equal to schedule')
parser.add_argument('--col_size', type=int,
                    default=16, help='column size')
parser.add_argument('--push', type=bool, help='forward push')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)

    logger.info(args)

    # log = open(os.path.join(args.save_path, 'log.txt'), 'w')
    
    # Preparing data
    if args.dataset == 'cifar10':
        data_path = args.data_path + 'cifar'

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes = 10
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset

        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val_')

        train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        num_classes = 1000
    else:
        raise ValueError("Dataset must be either cifar10 or imagenet_1k")  

    # Prepare the model
    logger.info('==> Building model..\n')
    model_cfg = getattr(models, args.model)
    model_cfg.kwargs.update({"num_classes": num_classes, "wbit": args.wbit, "abit":args.abit, "alpha_init": args.alpha_init, "mode": args.q_mode, "k": args.k, "ch_group":args.group_ch, "push":False})
    # model_cfg.kwargs.update({"num_classes": num_classes})
    net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs) 
    logger.info(net)

    start_epoch = 0

    for name, value in net.named_parameters():
        logger.info(f'name: {name} | shape: {list(value.size())}')
    
    if args.fine_tune:
        new_state_dict = OrderedDict()
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]
            new_state_dict[name] = v
        
        state_tmp = net.state_dict()

        if 'state_dict' in checkpoint.keys():
            state_tmp.update(new_state_dict)
        
            net.load_state_dict(state_tmp)
            logger.info("=> loaded checkpoint '{} | Acc={}'".format(args.resume, checkpoint['acc']))
        else:
            raise ValueError('no state_dict found')  

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net)

    # Loss function
    net = net.cuda()
    # summary(net, (3,32,32))
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Evaluate
    if args.evaluate:
        # net = inference_prune(net, args)
        test_acc, val_loss = test(testloader, net, criterion, 0)
        group_sparsity, overall_sparsity, sparse_groups = get_weight_sparsity(net, args=args)
        logger.info(f'Test accuracy: {test_acc}')
        logger.info(f'Group sparsity: {group_sparsity}')
        logger.info(f'Overall sparsity: {overall_sparsity}')

        exit()
    
    # Training
    start_time = time.time()
    epoch_time = AverageMeter()
    best_acc = 0.
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_time', 'te_loss', 'te_acc', 'best_acc', 'grp_spar', 'ovall_spar', 'spar_groups']

    if args.swp:
        # columns += ['lambda', 'penalty_groups', 'current_group']
        columns += ['lambda', 'penalty_groups']

    quantization_bound = []
    clipped_bound = []
    layer_thre = []
    for epoch in range(start_epoch, start_epoch+args.epochs):
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)
        
        current_lr, current_momentum = adjust_learning_rate_schedule(
            optimizer, epoch, args.gammas, args.schedule, args.lr, args.momentum)

        # currnet_group = adjust_group_schedule(epoch, args.g_multi, args.schedule, args.group_ch)
        # args.group_ch = currnet_group

        # Training phase
        train_results = train(trainloader, net, criterion, optimizer, epoch, args)
        if not args.abit == 32:
            clipped_bound.append(train_results['clp_alpha'])
        
        if args.swp:
            layer_thre.append(train_results['thre_list'])
        # Test phase
        test_acc, val_loss = test(testloader, net, criterion, epoch)        
        is_best = test_acc > best_acc

        if is_best:
            # print(f'=> Epoch:{epoch} - Bset acc obtained: {best_acc}, Saving..')
            best_acc = test_acc

        state = {
            'state_dict': net.state_dict(),
            'acc': best_acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        
        if epoch in [m for m in range(0, 200)]:
            filename=f'checkpoint_{epoch}.pth.tar'
        else:
            filename='checkpoint.pth.tar'
        
        save_checkpoint(state, is_best, args.save_path, filename=filename)

        e_time = time.time() - start_time
        epoch_time.update(e_time)
        start_time = time.time()
        
        group_sparsity, overall_sparsity, sparse_groups = get_weight_sparsity(net, args=args)
        if not args.wbit == 32:
            alpha_w = get_alpha_w(net)
            quantization_bound.append(alpha_w)
        
        values = [epoch + 1, optimizer.param_groups[0]['lr'], train_results['loss'], train_results['acc'], 
            e_time, val_loss, test_acc, best_acc, group_sparsity, overall_sparsity, sparse_groups]
        
        if args.swp:
            # values += [args.lamda, train_results['penalty_groups'], currnet_group]
            values += [args.lamda, train_results['penalty_groups']]

        print_table(values, columns, epoch, logger)

    filename = os.path.join(args.save_path, 'model_alpha_w.npy')
    np.save(filename, np.array(quantization_bound))

    filename = os.path.join(args.save_path, 'model_clp_bound.npy')
    np.save(filename, np.array(clipped_bound))

    if args.swp:
        filename = os.path.join(args.save_path, 'model_prune_thre.npy')
        np.save(filename, np.array(layer_thre)) 

if __name__ == '__main__':
    main()
