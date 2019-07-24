import os
import numpy as np
import torch
from collections import defaultdict, namedtuple


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':30} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:30} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


# def quantiModel(model):
#     # Infer dtype
#     # dtype = str(arr.dtype)
#     # Calculate frequency in arr
#     freq_map = defaultdict(int)
#     convert_map = {'float32': float, 'int32': int}
#     for name, param in model.named_parameters():
#
#
#         # value = convert_map['torch.float32'](value)
#         # freq_map[value] += 1
#
#         # if str
#         # print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
