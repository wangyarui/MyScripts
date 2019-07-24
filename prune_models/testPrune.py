import torch
import argparse

from torch import nn
from tools import utils

from tools.pruneNet import PruningNet


parser = argparse.ArgumentParser()
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
args = parser.parse_args()


model = PruningNet()#.cuda()
utils.print_model_parameters(model)

print('=======================================================================')


################################################
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
# print(y)
thre_index = int(total * args.percent)
# print(thre_index)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    # print(m)
    # test = m
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        # weight_copy.cuda()
        mask = weight_copy.gt(thre).float()#.cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        # print('test')
        print(m)
        cfg.append('M')

pruned_ratio = pruned/total

print('Pre-processing Successful!')
print(pruned_ratio)
print(cfg)
# utils.print_model_parameters(model)
##############################################
