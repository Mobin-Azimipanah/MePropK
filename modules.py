import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from functions import linearUnified, linear



class Linear(nn.Module):
    '''
    A linear module (layer without activation) with meprop
    The initialization of w and b is the same with the default linear module.
    '''

    def __init__(self, in_, out_, k, unified=False):
        super(Linear, self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.k = k
        self.unified = unified

        self.w = Parameter(torch.Tensor(self.in_, self.out_))
        self.b = Parameter(torch.Tensor(self.out_))
        print(f'(from modules), Linear module initialized with input dimension {in_} and output dimension {out_}')
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_)
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
        print('(modules)w shape:', self.w.shape)
        print('(modules)b shape:', self.b.shape)
        print("(modules)this is the k: ", self.k)


    def forward(self, x):
        if self.unified:
            my_fun = linearUnified(self.k)
            return my_fun.apply(x, self.w, self.b)
        else:
            my_fun = linear(self.k)
            return my_fun.apply(x, self.w, self.b)

    def __repr__(self):
        return '{} ({} -> {} <- {}{})'.format(self.__class__.__name__,
                                              self.in_, self.out_, 'unified'
                                              if self.unified else '', self.k)

# Assume linear_module is a Linear module instance

'''
linear_module = Linear(100, 50, 10)
linear_module.reset_parameters() # initialize the parameters
x = torch.randn(20, 100) # assume batch size is 32
print("###############1111111111")
print(x.shape[0])
print(linear_module.w.shape)
print(linear_module.b.shape)
print(linear_module.k)

print("###############")
output = linear_module.forward(x) # compute the output
print(output.shape)
'''