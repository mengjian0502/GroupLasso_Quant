"""
quant_layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# from torch._jit_internal import weak_script_method
# from .utee import wage_initializer,wage_quantizer
from .quantizer import *


def odd_symm_quant(input, nbit, mode='mean', k=2, dequantize=True, posQ=False):
    
    if mode == 'mean':
        alpha_w = k * input.abs().mean()
    elif mode == 'sawb':
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
        alpha_w = get_scale(input, z_typical[f'{int(nbit)}bit']).item()
    
    output = input.clamp(-alpha_w, alpha_w)

    if posQ:
        output = output + alpha_w

    scale, zero_point = symmetric_linear_quantization_params(nbit, abs(alpha_w), restrict_qrange=True)

    output = linear_quantize(output, scale, zero_point)
    
    if dequantize:
        output = linear_dequantize(output, scale, zero_point)

    return output, alpha_w, scale

def activation_quant(input, nbit, sat_val, dequantize=True):
    with torch.no_grad():
        scale, zero_point = quantizer(nbit, 0, sat_val)
    
    output = linear_quantize(input, scale, zero_point)

    if dequantize:
        output = linear_dequantize(output, scale, zero_point)

    return output, scale

class ClippedReLU(nn.Module):
    def __init__(self, num_bits, alpha=8.0, inplace=False, dequantize=True):
        super(ClippedReLU, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))     
        self.num_bits = num_bits
        self.inplace = inplace
        self.dequantize = dequantize
        
    def forward(self, input):
        # print(f'ClippedRELU: input mean: {input.mean()} | input std: {input.std()}')
        input = F.relu(input)
        input = torch.where(input < self.alpha, input, self.alpha)
        
        with torch.no_grad():
            scale, zero_point = quantizer(self.num_bits, 0, self.alpha)
        input = STEQuantizer.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return input

class int_quant_func(torch.autograd.Function):
    def __init__(self, nbit, alpha_w, restrictRange=True, ch_group=16, push=False):
        super(int_quant_func, self).__init__()
        self.nbit = nbit
        self.restrictRange = restrictRange
        self.alpha_w = alpha_w
        self.ch_group = ch_group
        self.push = push

    def forward(self, input):
        self.save_for_backward(input)
        output = input.clamp(-self.alpha_w.item(), self.alpha_w.item())
        scale, zero_point = symmetric_linear_quantization_params(self.nbit, self.alpha_w, restrict_qrange=self.restrictRange)
        output = STEQuantizer_weight.apply(output, scale, zero_point, True, False, self.nbit, self.restrictRange)   

        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class int_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, nbit=4, mode='mean', k=2, ch_group=16, push=False):
        super(int_conv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.nbit = nbit
        self.mode = mode
        self.k = k
        self.ch_group = ch_group
        self.push = push
        self.iter = 0
        self.mask = torch.ones_like(self.weight).cuda()
        self.register_buffer('alpha_w', torch.Tensor([1.]))

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(w_l, z_typical[f'{int(self.nbit)}bit'])
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")                         
            
        weight_q = int_quant_func(nbit=self.nbit, alpha_w=self.alpha_w, restrictRange=True, ch_group=self.ch_group, push=self.push)(w_l)
        w_p = weight_q.clone()
        num_group = w_p.size(0) * w_p.size(1) // self.ch_group
        # if self.push and self.iter is 0:
        if self.push and (self.iter+1) % 4000 == 0:
            print("Inference Prune!")
            kw = weight_q.size(2)
            num_group = w_p.size(0) * w_p.size(1) // self.ch_group
            w_p = w_p.contiguous().view((num_group, self.ch_group, kw, kw))
            
            self.mask = torch.ones_like(w_p)

            for j in range(num_group):
                idx = torch.nonzero(w_p[j, :, :, :])
                r = len(idx) / (self.ch_group * kw * kw)
                internal_sparse = 1 - r

                if internal_sparse >= 0.85 and internal_sparse != 1.0:
                    # print(internal_sparse)
                    self.mask[j, :, :, :] = 0.0

            w_p = w_p * self.mask
            w_p = w_p.contiguous().view((num_group, self.ch_group * kw * kw))
            grp_values = w_p.norm(p=2, dim=1)
            non_zero_idx = torch.nonzero(grp_values) 
            num_nonzeros = len(non_zero_idx)
            zero_groups = num_group - num_nonzeros 
            # print(f'zero groups = {zero_groups}')

            self.mask = self.mask.clone().resize_as_(weight_q)
        
        if not self.push:
            self.mask = torch.ones_like(self.weight)
        
        weight_q = self.mask * weight_q

        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        # if weight_q.size(2) != 1:
        #     batch = output.size(0)
        #     d = output.size(1)
        #     h = output.size(2)
        #     w = output.size(3)

        #     out_tmp = output.view(batch, d, h*w)
        #     out_tmp = out_tmp.squeeze(0)

        #     out_tmp = out_tmp.norm(p=2, dim=1)
        #     non_zero_output = len(torch.nonzero(out_tmp.contiguous().view(-1)))

        #     print('=========================================')
        #     print(f'size of the input feature map: {list(input.size())}')
        #     print(f'size of the weight: {list(weight_q.size())}, number of groups/col:{num_group}| stride={self.stride} | padding={self.padding}')
        #     print(f'number of nonzero output channel: {non_zero_output}')
        #     print(f'size of the output feature map: {list(output.size())}')
        #     print('=========================================\n')
        self.iter += 1
        return output
    
    def extra_repr(self):
        return super(int_conv2d, self).extra_repr() + ', nbit={}, mode={}, k={}, ch_group={}, push={}'.format(self.nbit, self.mode, self.k, self.ch_group, self.push)


class int_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, nbit=8, mode='mean', k=2, ch_group=16, push=False):
        super(int_linear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.nbit=nbit
        self.mode = mode
        self.k = k
        self.ch_group = ch_group
        self.push = push

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(w_l, z_typical[f'{int(self.nbit)}bit'])
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")                         
            
        weight_q = int_quant_func(nbit=self.nbit, alpha_w=self.alpha_w, restrictRange=True, ch_group=self.ch_group, push=self.push)(w_l)
        output = F.linear(input, weight_q, self.bias)

        w_tmp = weight_q.clone()
        grp_val = w_tmp.norm(p=2, dim=1)
        num_non_zero = len(torch.nonzero(grp_val.contiguous().view(-1)))
        num_zero_grp = w_tmp.size(0) - num_non_zero

        # output vector
        out_tmp = F.linear(input, weight_q, bias=torch.Tensor([0]).cuda())
        non_zero_output = len(torch.nonzero(out_tmp.contiguous().view(-1)))

        # print('=========================================')
        # print(f'size of the input feature map: {list(input.size())}')
        # print(f'size of the group val:{grp_val.size()}')
        # print(f'size of the weight: {list(weight_q.size())} ')
        # print(f'number of zero groups: {num_zero_grp}')
        # print(f'number of nonzero output channel: {non_zero_output}')
        # print(f'size of the output feature map: {list(output.size())}')
        # print('=========================================\n')

        return output

    def extra_repr(self):
        return super(int_linear, self).extra_repr() + ', nbit={}, mode={}, k={}, ch_group={}, push={}'.format(self.nbit, self.mode, self.k, self.ch_group, self.push)

"""
2-bit quantization
"""

def w2_quant(input, mode='mean', k=2):
    if mode == 'mean':
            alpha_w = k * input.abs().mean()
    elif mode == 'sawb':
        alpha_w = get_scale_2bit(w_l)
    else:
        raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")
    
    output = input.clone()
    output[input.ge(alpha_w - alpha_w/3)] = alpha_w
    output[input.lt(-alpha_w + alpha_w/3)] = -alpha_w

    output[input.lt(alpha_w - alpha_w/3)*input.ge(0)] = alpha_w/3
    output[input.ge(-alpha_w + alpha_w/3)*input.lt(0)] = -alpha_w/3

    return output

class sawb_w2_Func(torch.autograd.Function):
    def __init__(self, alpha_w):
        super(sawb_w2_Func, self).__init__()
        self.alpha_w = alpha_w 

    def forward(self, input):
        self.save_for_backward(input)
        
        output = input.clone()
        output[input.ge(self.alpha_w - self.alpha_w/3)] = self.alpha_w
        output[input.lt(-self.alpha_w + self.alpha_w/3)] = -self.alpha_w

        output[input.lt(self.alpha_w - self.alpha_w/3)*input.ge(0)] = self.alpha_w/3
        output[input.ge(-self.alpha_w + self.alpha_w/3)*input.lt(0)] = -self.alpha_w/3

        return output
    
    def backward(self, grad_output):
    
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0

        return grad_input

class Conv2d_2bit(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, mode='mean', k=2):
        super(Conv2d_2bit, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.mode = mode
        self.k = k
        self.alpha_w = 1.

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            self.alpha_w = get_scale_2bit(w_l)
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ") 

        
        weight = sawb_w2_Func(alpha_w=self.alpha_w)(w_l)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output

class Linear2bit(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, mode='mean', k=2):
        super(Linear2bit, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.mode = mode
        self.k = k

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            self.alpha_w = get_scale_2bit(w_l)
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ") 
        
        weight_q = sawb_w2_Func(alpha_w=self.alpha_w)(w_l)

        output = F.linear(input, weight_q, self.bias)
        return output

    def extra_repr(self):
        return super(Linear2bit, self).extra_repr() + ', mode={}, k={}'.format(self.mode, self.k)

"""
zero skipping quantization
"""

class zero_skp_quant(torch.autograd.Function):
    def __init__(self, nbit, coef, group_ch, alpha_w):
        super(zero_skp_quant, self).__init__()
        self.nbit = nbit
        self.coef = coef
        self.group_ch = group_ch
        self.alpha_w = alpha_w
    
    def forward(self, input):
        self.save_for_backward(input)
        interval = 2*self.alpha_w / (2**self.nbit - 1) / 2
        self.th = self.coef * interval

        cout = input.size(0)
        cin = input.size(1)
        kh = input.size(2)
        kw = input.size(3)
        num_group = (cout * cin) // self.group_ch

        w_t = input.view(num_group, self.group_ch*kh*kw)

        grp_values = w_t.norm(p=2, dim=1)                                               # L2 norm
        mask_1d = grp_values.gt(self.th*self.group_ch*kh*kw).float()
        mask_2d = mask_1d.view(w_t.size(0),1).expand(w_t.size()) 

        w_t = w_t * mask_2d

        non_zero_idx = torch.nonzero(mask_1d).squeeze(1)                             # get the indexes of the nonzero groups
        non_zero_grp = w_t[non_zero_idx]                                             # what about the distribution of non_zero_group?
        
        weight_q = non_zero_grp.clone()
        alpha_w = get_scale_2bit(weight_q)

        weight_q[non_zero_grp.ge(self.alpha_w - self.alpha_w/3)] = self.alpha_w
        weight_q[non_zero_grp.lt(-self.alpha_w + self.alpha_w/3)] = -self.alpha_w
        
        weight_q[non_zero_grp.lt(self.alpha_w - self.alpha_w/3)*non_zero_grp.ge(0)] = self.alpha_w/3
        weight_q[non_zero_grp.ge(-self.alpha_w + self.alpha_w/3)*non_zero_grp.lt(0)] = -self.alpha_w/3

        # print(f'INT levels:{weight_q.unique()}')
        w_t[non_zero_idx] = weight_q
        
        output = w_t.clone().resize_as_(input)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input    

class Conv2d_W2_IP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, nbit=2, mode='mean', k=2, skp_group=16, gamma=0.3):
        super(Conv2d_W2_IP, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.nbit = nbit
        self.coef = gamma
        self.skp_group = skp_group
        self.mode = mode
        self.k=k

    def forward(self, input):
        weight = self.weight

        if self.mode == 'mean':
            self.alpha_w = self.k * weight.abs().mean()
        elif self.mode == 'sawb':
            self.alpha_w = get_scale_2bit(weight)
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")

        weight_q = zero_skp_quant(nbit=self.nbit, coef=self.coef, group_ch=self.skp_group, alpha_w=self.alpha_w)(weight)
        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output
    
    def extra_repr(self):
        return super(zero_grp_skp_quant, self).extra_repr() + ', nbit={}, coef={}, skp_group={}'.format(
                self.nbit, self.coef, self.skp_group)

# """
# ADC quantizatoin
# """

# class Qconv2d_adc(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
#                 col_size=16, group_size=8, wl_input=8, wl_weight=8,inference=0, mode='mean', k=2, cellBit=1, ADCprecision=5):
#         super(Qconv2d_adc, self).__init__(in_channels, out_channels, kernel_size,
#                                       stride, padding, dilation, groups, bias)

#         self.col_size = col_size
#         self.group_size = group_size
#         self.wl_input = wl_input
#         self.inference = inference
#         self.wl_weight = wl_weight
#         self.cellBit = cellBit
#         self.ADCprecision = ADCprecision
#         self.act_alpha = 1.
#         self.layer_idx = 0
#         self.iter = 0
#         self.mode = mode
#         self.k = k
#         self.outputPartial_tmp = 0
#         self.outputDummyPartial_tmp = 0
#         self.diffpartial_tmp = 0

#     @weak_script_method
#     def forward(self, input):

#         bitWeight = int(self.wl_weight)
#         bitActivation = int(self.wl_input)

#         weight_c = self.weight
#         weight_q, alpha_w, w_scale = odd_symm_quant(weight_c, nbit=bitWeight, mode=self.mode, k=self.k, dequantize=True, posQ=False)                                               # quantize the input weights
#         weight_int, _, _ = odd_symm_quant(weight_c, nbit=bitWeight, mode=self.mode, k=self.k, dequantize=False, posQ=True)                                                         # quantize the input weights to positive integer
#         output_original = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

#         num_group = weight_q.size(0) * weight_q.size(1) // self.group_size
#         num_col = weight_q.size(0) * weight_q.size(1) // self.col_size
        
#         total_active = 0
#         counter = 0

#         total_active_weight = 0
#         weight_counter = 0
#         if self.inference == 1:
#             # print(f'Layer: {self.layer_idx} | loaded alpha: {self.act_alpha}')
#             num_sub_groups = self.col_size // self.group_size

#             output = torch.zeros_like(output_original)
#             output_final = torch.zeros_like(output_original)
#             # del output_original
#             cellRange = 2**self.cellBit   # cell precision is 2

#             # loop over the group of rows
#             output_partial_pack = []
#             for ii in range(weight_q.size(1)//self.group_size):
#                 mask = torch.zeros_like(weight_q) 
#                 mask[:, (ii)*self.group_size:(ii+1)*self.group_size,:,:] = 1                                                                  # turn on the corresponding rows.

#                 inputQ, act_scale = activation_quant(input, nbit=bitActivation, sat_val=self.act_alpha, dequantize=False)
#                 outputIN = torch.zeros_like(output)

#                 input_mask = torch.zeros_like(inputQ)
#                 input_mask[:, (ii)*self.group_size:(ii+1)*self.group_size, :, :] = 1
                
#                 for z in range(bitActivation):
#                     inputB = torch.fmod(inputQ, 2)
#                     inputQ = torch.round((inputQ-inputB)/2)
#                     outputP = torch.zeros_like(output)

#                     X_decimal = weight_int*mask                                                                                                              # multiply the quantized integer weight with the corresponding mask
#                     outputD = torch.zeros_like(output)
#                     outputDiff = torch.zeros_like(output)
#                     dummyP = torch.zeros_like(weight_q)

#                     # # compute the active inputs
#                     # features = inputB * input_mask 
#                     # outh = (inputB.size(2) - weight_int.size(2)) // self.stride[0] + 1
#                     # outw = (inputB.size(3) - weight_int.size(3)) // self.stride[0] + 1
#                     # layer_active = 0

#                     # for g in range(outh):
#                     #     for t in range(outw):
#                     #         input_act = features[:, :, g:(g+weight_q.size(2)), t:(t+weight_q.size(3))]
#                     #         non_zero = len(torch.nonzero(input_act.contiguous().view(-1)))
#                     #         activity = non_zero / (self.col_size * weight_q.size(2)*weight_q.size(3))         
#                     #         layer_active += activity
#                     # layer_active = layer_active / (outh*outw)
#                     # # print(f'Activation bit: {ii} | average activity: {layer_active}')
#                     # total_active += layer_active
#                     # counter += 1
                    
#                     for k in range (int(bitWeight/self.cellBit)):
#                         if k == 0:
#                             dummyP[:,:,:,:] = 1.4
#                         elif k == 1:
#                             dummyP[:,:,:,:] = 1.4

#                         remainder = torch.fmod(X_decimal, cellRange)*mask
#                         X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                        
#                         # # print(remainder.size())
#                         # kw = remainder.size(2)
#                         # layer_weight_active = 0
#                         # remainder_temp = remainder.contiguous().view(num_group, self.group_size*kw*kw)
#                         # for c in range(num_group):
#                         #     non_zero_weight = torch.nonzero(remainder_temp[c, :]).size(0)
#                         #     weight_sparse = non_zero_weight / (self.group_size*kw*kw)                                                                         # sparsity of a single column
#                         #     layer_weight_active += weight_sparse

#                         # layer_weight_active = layer_weight_active / num_group
#                         # total_active_weight += layer_weight_active
#                         # weight_counter += 1

#                         outputPartial= F.conv2d(inputB, remainder, self.bias, self.stride, self.padding, self.dilation, self.groups)                      # Binarized convolution
#                         outputDummyPartial = F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups) 

#                         # ADC quantization effect:
#                         # outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision, lb=outputPartial.min(), ub=outputPartial.max())
#                         # outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision, lb=outputDummyPartial.min(), ub=outputDummyPartial.max())                        
#                         scaler = cellRange**k
#                         # outputP = outputP + outputPartialQ*scaler
#                         # outputD = outputD + outputDummyPartialQ*scaler

#                         self.outputPartial_tmp = outputPartial
#                         self.outputDummyPartial_tmp = outputDummyPartial

#                         output_diff = outputPartial - outputDummyPartial        # subtraction of each single column
#                         self.diffpartial_tmp = output_diff
#                         # k = 8.0
#                         output_diff_quant = wage_quantizer.LinearQuantizeOut(output_diff, self.ADCprecision, lb=output_diff.min(), ub=output_diff.max())
#                         # output_diff_quant = wage_quantizer.LinearQuantizeOut(output_diff, self.ADCprecision, lb=torch.Tensor([-21.9762]), ub=torch.Tensor([21.9762]))
#                         # output_diff_quant = wage_quantizer.LinearQuantizeOut(output_diff, self.ADCprecision, lb=-k*output_diff.abs().mean(), ub=k*output_diff.abs().mean())
#                         outputDiff = outputDiff + (output_diff_quant)*scaler
#                         # print(f'Diff after multiply with scalar: {torch.unique((output_diff_quant))}')
#                     scalerIN = 2**z
#                     # outputIN = outputIN + (outputP-outputD) * scalerIN
#                     outputIN = outputIN + outputDiff * scalerIN
#                 output = output + outputIN/act_scale                                                                                                       # dequantize it back                    
#             output = output/w_scale
#         # print('=========================================')
#         # print(f'size of the input feature map: {list(input.size())}')
#         # print(f'size of the weight: {list(weight_q.size())}, number of groups: {num_group}')
#         # print(f'size of the output feature map: {list(output.size())}')
#         # print(f"Layer {self.layer_idx}| average active input element: {total_active / counter}")
#         # # print(f"Layer {self.layer_idx}| average active weight element: {total_active_weight / weight_counter}")
#         # print('=========================================\n')
#         return output         