"""
ResNet on CIFAR10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .quant import ClippedReLU, int_conv2d, int_linear
import math

class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, ch_group=16, push=False):
    super(ResNetBasicblock, self).__init__()   
    # self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  # quantization
    self.conv_a = int_conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=push)  # quantization
    self.bn_a = nn.BatchNorm2d(planes)
    self.relu1 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)    # Clipped ReLU function 4 - bits
    # self.relu1 = nn.ReLU(inplace=True)

    self.conv_b = int_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=push)  # quantization
    # self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # quantization
    self.bn_b = nn.BatchNorm2d(planes)
    self.relu2 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)    # Clipped ReLU function 4 - bits
    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.bn_a(basicblock)
    basicblock = self.relu1(basicblock)

    basicblock = self.conv_b(basicblock)
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return self.relu2(residual + basicblock)


class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, depth, num_classes, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, ch_group=16, push=False):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarResNet, self).__init__()

    block = ResNetBasicblock
    

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
    self.num_classes = num_classes
    self.ch_group = ch_group
    # self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv_1_3x3 = int_conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=False)   # skip the push process for the first conv layer
    self.relu0 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)
    self.bn_1 = nn.BatchNorm2d(16)

    self.inplanes = 16
    self.stage_1 = self._make_layer(block, 16, layer_blocks, 1, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push)
    self.stage_2 = self._make_layer(block, 32, layer_blocks, 2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push)
    self.stage_3 = self._make_layer(block, 64, layer_blocks, 2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push)
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = int_linear(64*block.expansion, num_classes, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=False)                         # skip the push process for the last fc layer

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, ch_group=16, push=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        int_conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=False),
        nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv_1_3x3(x)
    x = self.relu0(self.bn_1(x))
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)

  def get_group_val(self):
    val = torch.Tensor()

    if torch.cuda.is_available():
        val = val.cuda()

    count = 0
    for m in self.modules():
      if isinstance(m, int_conv2d):
        kw = m.weight.size(2)
        if kw != 1:
          if not count in [0]:
            w_l = m.weight
            num_group = w_l.size(0) * w_l.size(1) // self.ch_group
            w_l = w_l.view(w_l.size(0), w_l.size(1) // self.ch_group, self.ch_group, kw, kw)
            w_l = w_l.contiguous().view((num_group, self.ch_group*kw*kw))
            
            g = w_l.pow(2).sum(dim=1).pow(1/2)
            val = torch.cat((val.view(-1), g.view(-1)))
          count += 1
    return val

  def get_global_thre(self, ratio):
    grp_val = self.get_group_val()
    grp_mean = grp_val.mean()

    threshold = ratio * grp_mean
    # sorted_block_values, indices = torch.sort(grp_val.contiguous().view(-1))
    # thre_index = int(grp_val.data.numel() * ratio)
    # threshold = sorted_block_values[thre_index]
    return threshold


class resnet20_quant:
  base=CifarResNet
  args = list()
  kwargs = {'depth': 20}

class resnet32_quant:
  base=CifarResNet
  args = list()
  kwargs = {'depth': 32}
