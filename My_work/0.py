# !/usr/bin/env python
# _*_coding:utf-8_*_

import torch
from torch import nn

print("This torch version: ", torch.__version__)

a = torch.tensor([[2, 1]])
b = torch.tensor([[-1, 2]])
c = a@b.t()
print("[[2, 1]] @ [[-1, 2]] = ", c.item())