import torch as t 
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm
from linear_var_hypernet import LinearApprNet
from kernel_hypernet import KernelNet
import torch.nn.init as init
import math


class BaseLayer(nn.Module): #однослойная сеть
    def __init__(self, in_,  out_, device,  act=F.relu, prior_sigma=1.0):         
        nn.Module.__init__(self)                    
        self.mean = nn.Parameter(t.randn(in_, out_, device=device)) # параметры средних
        t.nn.init.xavier_uniform(self.mean) 
        self.mean_b = nn.Parameter(t.randn(out_, device=device)) # то же самое для свободного коэффициента
                
        self.in_ = in_
        self.out_ = out_
        self.act = act
        
        self.prior_sigma = prior_sigma
        
        
    def forward(self,x):    
        w = self.mean 
        b = self.mean_b
            
        # функция активации 
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l = None, prior_scale=1.0):        
        # подсчет hyperloss
        return ((self.mean**2).sum() + (self.mean_b**2).sum())/self.prior_sigma/prior_scale
    
class BaseLayerLinear(nn.Module): #однослойная сеть
    def __init__(self, in_,  out_,   act=F.relu, prior_sigma=1.0):         
        nn.Module.__init__(self)                    
        self.mean = LinearApprNet((in_, out_)) # параметры средних 
        self.mean_b = LinearApprNet( out_) # то же самое для свободного коэффициента
                    
        self.in_ = in_
        self.out_ = out_
        self.act = act
        
        self.prior_sigma = prior_sigma
        
        
    def forward(self,x, l):    
        w = self.mean(l) 
        b = self.mean_b(l)
            
        # функция активации 
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l, prior_scale=1.0):
        
        # подсчет hyperloss
        return  ((self.mean(l)**2).sum() + (self.mean_b(l)**2).sum())/self.prior_sigma/prior_scale
    
class BaseLinearConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding,  act=F.relu, H = 3, W = 3, stride=(1, 1), dilation = (1, 1), groups = 1, prior_sigma = 1.0):
        super(BaseLinearConv, self).__init__()
        
        self.mean = LinearApprNet((out_channels, in_channels, H, W)) # параметры средних            
        self.mean_b = LinearApprNet((out_channels)) # то же самое для свободного коэффициента
        
        init.kaiming_uniform_(self.mean.const.data, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.const.data)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.mean_b.const.data, -bound, bound)        
        
        init.kaiming_uniform_(self.mean.const2.data, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.const2.data)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.mean_b.const2.data, -bound, bound)        
        
            
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
                
        self.size_m = (out_channels, in_channels, H, W)
        self.out_ = out_channels
        self.prior_sigma = prior_sigma
        self.act = act
        
    def forward(self, x, l):        
        w = self.mean(l)
        b = self.mean_b(l)
            
        return self.act(F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups))

    def KLD(self, l=None,  prior_scale=1.0):
        return  ((self.mean(l)**2).sum() + (self.mean_b(l)**2).sum())/self.prior_sigma/prior_scale
    
    

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding,  act=F.relu, H = 3, W = 3, stride=(1, 1), dilation = (1, 1), groups = 1, prior_sigma = 1.0):
        super(BaseConv, self).__init__()
        
        self.mean = nn.Parameter(t.randn(out_channels, in_channels, H, W)) # параметры средних            
        self.mean_b = nn.Parameter(t.randn(out_channels)) # то же самое для свободного коэффициента
        
        init.kaiming_uniform_(self.mean.data, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.data)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.mean_b.data, -bound, bound)        
            
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
                
        self.size_m = (out_channels, in_channels, H, W)
        self.out_ = out_channels
        self.prior_sigma = prior_sigma
        self.act = act
        
    def forward(self, x, l=None):        
        w = self.mean
        b = self.mean_b
            
        return self.act(F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups))

    def KLD(self, l=None,  prior_scale=1.0):
        return  ((self.mean**2).sum() + (self.mean_b**2).sum())/self.prior_sigma/prior_scale
    
    
class BaseKernelLayer(nn.Module):  # вариационная однослойная сеть
    def __init__(self, in_,  out_,  kernel_num,  prior_sigma = 1.0,  act=F.relu):
        nn.Module.__init__(self)
        self.mean = KernelNet((in_, out_), kernel_num)  # параметры средних
        self.mean_b = KernelNet(out_, kernel_num)
      
        
        self.in_ = in_
        self.out_ = out_
        self.act = act
        
        self.prior_sigma = prior_sigma

    def forward(self, x, l):
        
        w = self.mean(l)
        b = self.mean_b(l)
       
        # функция активации
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l, prior_scale=1.0):
        # подсчет hyperloss
        return  ((self.mean(l)**2).sum() + (self.mean_b(l)**2).sum())/self.prior_sigma/prior_scale
    
    
class BaseKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding,  kernel_num,  act=F.relu, H = 3, W = 3, stride=(1, 1), dilation = (1, 1), groups = 1, prior_sigma = 1.0):
        super(BaseKernelConv, self).__init__()
        
        self.mean = KernelNet((out_channels, in_channels, H, W), kernel_num) # параметры средних            
        self.mean_b = KernelNet((out_channels), kernel_num) # то же самое для свободного коэффициента
        for i in range(kernel_num):
            init.kaiming_uniform_(self.mean.const[i].data, a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.const[i].data)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.mean_b.const[i].data, -bound, bound)        
            
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
                
        self.size_m = (out_channels, in_channels, H, W)
        self.out_ = out_channels
        self.prior_sigma = prior_sigma
        self.act = act
        
    def forward(self, x, l):        
        w = self.mean(l)
        b = self.mean_b(l)
            
        return self.act(F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups))

    def KLD(self, l,  prior_scale=1.0):
        return  ((self.mean(l)**2).sum() + (self.mean_b(l)**2).sum())/self.prior_sigma/prior_scale
        