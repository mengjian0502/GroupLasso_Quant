"""
Utilities of MobileNet training
"""
import os
import sys
import time
import math
import shutil
import tabulate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from models.quant import *
from models.mpdr_score import *
from torch.autograd import Variable


def glasso_thre(var, dim=0, ratio=0.7):
    if len(var.size()) == 4:
        var = var.contiguous().view((var.size(0), var.size(1) * var.size(2) * var.size(3)))

    a = var.pow(2).sum(dim=dim).pow(1/2)
    
    mean_a  = a.mean()
    thre = ratio*mean_a
    penalty_group = a[a<thre]                                   # number of groups that penalized
    # print(f'threshold = {thre:.4f}')
    a = torch.min(a, thre)

    return a.sum(), thre, penalty_group.numel()

def glasso_global(var, dim=0, thre=0.0):
    if len(var.size()) == 4:
        var = var.contiguous().view((var.size(0), var.size(1) * var.size(2) * var.size(3)))

    a = var.pow(2).sum(dim=dim).pow(1/2)
    penalty_group = a[a<thre]
    a = torch.min(a, thre)

    return a.sum(), penalty_group.numel()

def glasso_global_mp(var, dim=0, thre=0.0):
    if len(var.size()) == 4:
        var = var.contiguous().view((var.size(0), var.size(1) * var.size(2) * var.size(3)))

    a = var.pow(2).sum(dim=dim).pow(1/2)
    b = var.abs().mean(dim=1)

    penalty_groups = a[b<thre]
    return penalty_groups.sum(), penalty_groups.numel()

def glasso_global_mpdr(var, var_score, dim=0, thre=0.0):
    # weight reshape
    if len(var.size()) == 4:
        var = var.contiguous().view((var.size(0), var.size(1) * var.size(2) * var.size(3)))

    # group level score
    if len(var_score.size()) == 4:
        var_score = var_score.contiguous().view((var_score.size(0), var_score.size(1) * var_score.size(2) * var_score.size(3)))

    a = var.pow(2).sum(dim=dim).pow(1/2)
    b = var_score.mean(dim=1)

    penalty_groups = a[b<thre]
    return penalty_groups.sum(), penalty_groups.numel()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def prepare_data_loaders(data_path, train_batch_size, eval_batch_size):

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

def train(trainloader, net, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure the data loading time
        data_time.update(time.time() - end)
        
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda()
    
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if args.swp:
            # compute the global threshold
            thre1 = net.get_global_mp_thre(args.ratio)

            lamda = torch.tensor(args.lamda).cuda()
            reg_g1 = torch.tensor(0.).cuda()

            reg_linear = torch.tensor(0.).cuda()
            
            group_ch = args.group_ch
            thre_list = []
            penalty_groups = 0
            count = 0
            lin_count = 0
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    if not count in [0]:
                        w_l = m.weight
                        kw = m.weight.size(2)
                        if kw != 1:
                            num_group = w_l.size(0) * w_l.size(1) // group_ch
                            w_l = w_l.view(w_l.size(0), w_l.size(1) // group_ch, group_ch, kw, kw)
                            w_l = w_l.contiguous().view((num_group, group_ch, kw, kw))
                            
                            # reg1, thre1, penalty_group1 = glasso_thre(w_l, 1, args.ratio)
                            reg1, penalty_group1 = glasso_global_mp(w_l, dim=1, thre=thre1)
                            reg_g1 += reg1
                            thre = thre1
                            penalty_group = penalty_group1
                        
                        # pruning statistics
                        if batch_idx == len(trainloader) - 1:
                            thre_list.append(thre)                      
                            penalty_groups += penalty_group
                    count += 1

            loss += lamda * (reg_g1)
        else:
            penalty_groups = 0
            thre_list = []

        if args.clp:
            reg_alpha = torch.tensor(0.).cuda()
            a_lambda = torch.tensor(args.a_lambda).cuda()

            alpha = []
            for name, param in net.named_parameters():
                if 'alpha' in name:
                    alpha.append(param.item())
                    reg_alpha += param.item() ** 2
            loss += a_lambda * (reg_alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        train_loss += loss.item()
        if args.clp:
            res = {
                'acc':top1.avg,
                'loss':losses.avg,
                'clp_alpha':np.array(alpha),
                'thre_list':thre_list,
                'penalty_groups':penalty_groups
                }
        else:
            res = {
                'acc':top1.avg,
                'loss':losses.avg,
                } 
    return res


def test(testloader, net, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    end = time.time()
    outputPartial_list = torch.Tensor([])
    outputDummyPartial_list = torch.Tensor([])
    
    # k = 7.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            test_loss += loss.item()

            batch_time.update(time.time() - end)
            end = time.time()
    return top1.avg, losses.avg


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)

def adjust_learning_rate_schedule(optimizer, epoch, gammas, schedule, lr, mu):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, save_path+filename)
    if is_best:
        shutil.copyfile(save_path+filename, save_path+'model_best.pth.tar')

def get_weight_sparsity(model, args):
    all_group = 0
    all_nonsparse = 0
    group_ch = args.targeted_group
    count = 0
    all_num = 0.0
    count_num_one = 0.0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if not count in [0] and not m.weight.size(2)==1:
                w_mean = m.weight.mean()
                w_l = m.weight

                with torch.no_grad():
                    if args.wbit == 2:
                        w_l = w2_quant(w_l, mode=args.q_mode, k=args.k)
                    else:
                        w_l,_,_ = odd_symm_quant(w_l, nbit=args.wbit, mode=args.q_mode, k=args.k)
                
                kw = m.weight.size(2)
                
                count_num_layer = w_l.size(0) * w_l.size(1) * kw * kw
                all_num += count_num_layer
                count_one_layer = len(torch.nonzero(w_l.view(-1)))
                count_num_one += count_one_layer

                num_group = w_l.size(0) * w_l.size(1) // group_ch
                w_l = w_l.view(w_l.size(0), w_l.size(1) // group_ch, group_ch, kw, kw)
                w_l = w_l.contiguous().view((num_group, group_ch * kw * kw))
        

                grp_values = w_l.norm(p=2, dim=1)
                non_zero_idx = torch.nonzero(grp_values) 
                num_nonzeros = len(non_zero_idx)

                all_group += num_group
                all_nonsparse += num_nonzeros

            count += 1
    overall_sparsity = 1 - count_num_one / all_num
    group_sparsity = 1 - all_nonsparse / all_group
    sparse_group = all_group - all_nonsparse
    return group_sparsity, overall_sparsity, sparse_group

            
def get_alpha_w(model):
    alpha = []
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if not count in [0] and not m.weight.size(2)==1:
                alpha.append(m.alpha_w)
            count += 1
    return alpha


def log2df(log_file_name):
    '''
    return a pandas dataframe from a log file
    '''
    with open(log_file_name, 'r') as f:
        lines = f.readlines() 
    # search backward to find table header
    num_lines = len(lines)
    for i in range(num_lines):
        if lines[num_lines-1-i].startswith('---'):
            break
    header_line = lines[num_lines-2-i]
    num_epochs = i
    columns = header_line.split()
    df = pd.DataFrame(columns=columns)
    for i in range(num_epochs):
        df.loc[i] = [float(x) for x in lines[num_lines-num_epochs+i].split()]
    return df 


if __name__ == "__main__":
    # clp_alpha = np.load('./save/resnet20/resnet20_quant_w4_a4_modemean_k_lambda_wd0.0001_swpFalse/model_clp_bound.npy')
    log = log2df('./save/sparsity_analysis/resnet20_quant_grp8/resnet20_quant_w4_a4_modemean_k2_lambda0.0020_ratio0.5_wd0.0005_lr0.005_swpFalse_groupch8_pushFalse_lr0.005/resnet20_quant_w4_a4_modemean_k2_lambda0.0020_ratio0.5_wd0.0005_lr0.005_swpFalse_groupch8_pushFalse_lr0.005.log')
    epoch = log['ep']
    grp_spar = log['grp_spar']
    ovall_spar = log['ovall_spar']
    spar_groups = log['spar_groups']
    penalty_groups = log['penalty_groups']

    table = {
        'epoch': epoch,
        'grp_spar': grp_spar,
        'ovall_spar': ovall_spar,
        'spar_groups':spar_groups,
        'penalty_groups':penalty_groups,
    }

    variable = pd.DataFrame(table, columns=['epoch','grp_spar','ovall_spar', 'spar_groups', 'penalty_groups'])
    variable.to_csv('resnet20_quant_w4_a4_modemean_k2_lambda0.0020_ratio0.5_wd0.0005_lr0.005_swpFalse_groupch8_pushFalse_lr0.005_baseline.csv', index=False)

    # diff_partial_sum = torch.load("./save/resnet20_quant/resnet20_quant_w4_a4_modemean_k2_lambda0.002_ratio0.7_wd0.0005_lr0.01_swpTrue_groupch16/diff_partial_sum.pt")
    # partial_sum = torch.load("save/resnet20_quant_eval/resnet20_quant_eval_w4_a4_modemean_k2_groupch16_colsize16_cellBit2_adc_prec5/partial_sum.pt")
    # dummy_partial_sum = torch.load("save/resnet20_quant_eval/resnet20_quant_eval_w4_a4_modemean_k2_groupch16_colsize16_cellBit2_adc_prec5/dummy_partial_sum.pt")
    
    # print(diff_partial_sum_mean.mean())

    # percentile = 99.0
    # partial_sum_percentile = np.percentile(partial_sum.numpy(), percentile)
    # dummy_partial_sum_percentile = np.percentile(dummy_partial_sum.numpy(), percentile)
    # diff_partial_sum_ub = np.percentile(diff_partial_sum.numpy(), percentile)
    # diff_partial_sum_lb = np.percentile(diff_partial_sum.numpy(), 100-percentile)

    # print(f'{percentile} percentile of partial sum = {partial_sum_percentile}')
    # print(f'{percentile} percentile of dummy partial sum = {dummy_partial_sum_percentile}')
    # print(f'{percentile} percentile of partial sum = {diff_partial_sum_ub}')
    # print(f'{100-percentile} percentile of partial sum = {diff_partial_sum_lb}')

    # style = 'whitegrid'
    
    # sns.set_style(style)
    # plt.figure(figsize=(10,6))
    # # sns.distplot(partial_sum.numpy() - dummy_partial_sum.numpy())
    # sns.distplot(diff_partial_sum.numpy())
    # plt.savefig("./save/resnet20_quant/resnet20_quant_w4_a4_modemean_k2_lambda0.002_ratio0.7_wd0.0005_lr0.01_swpTrue_groupch16/diff_partial_sum.png", dpi=300)



