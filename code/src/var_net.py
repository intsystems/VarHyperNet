import torch as t 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

MIN_SIGMA = 1e-5

class VarReshape(nn.Module): # просто для сохранения порядка
    def __init__(self, shape):
        nn.Module.__init__(self) 
        self.shape = shape
        
    def forward(self, x, l =None):
        return x.view(self.shape)

    def KLD(self, *args, **kwargs):
        return 0.0
    
class VarPool2d(nn.MaxPool2d):
    def KLD(self, *args, **kwargs):
        return 0.0
    def forward(self, x, l =None):
        return nn.MaxPool2d.forward(self, x)

    
class VarLayer(nn.Module): # вариационная однослойная сеть
    def __init__(self, in_,  out_,   prior_sigma = 1.0, init_log_sigma=-3.0, act=F.relu):         
        nn.Module.__init__(self)                    
        self.mean = nn.Parameter(t.randn(in_, out_)) # параметры средних
        t.nn.init.xavier_uniform(self.mean) 
        self.log_sigma = nn.Parameter(t.ones(in_, out_)*init_log_sigma) # логарифм дисперсии
        self.mean_b = nn.Parameter(t.randn(out_)) # то же самое для свободного коэффициента
        self.log_sigma_b = nn.Parameter(t.ones(out_) * init_log_sigma)
                
        self.in_ = in_
        self.out_ = out_
        self.act = act
        self.prior_sigma = prior_sigma
        
    def forward(self,x, l=None):
        if self.training: # во время обучения - сэмплируем из нормального распределения
            self.eps_w = t.distributions.Normal(self.mean, MIN_SIGMA+t.exp(self.log_sigma))
            self.eps_b = t.distributions.Normal(self.mean_b, MIN_SIGMA+t.exp(self.log_sigma_b))
        
            w = self.eps_w.rsample()
            b = self.eps_b.rsample()
             
        else:  # во время контроля - смотрим средние значения параметра        
            w = self.mean 
            b = self.mean_b
            
        # функция активации         
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l=None,  prior_scale=1.0):
      
        # подсчет дивергенции
        size = self.in_, self.out_
        out = self.out_
        device = self.mean.device
        self.eps_w = t.distributions.Normal(self.mean, MIN_SIGMA+t.exp(self.log_sigma))
        self.eps_b = t.distributions.Normal(self.mean_b,  MIN_SIGMA+t.exp(self.log_sigma_b))
     
        self.h_w = t.distributions.Normal(t.zeros(size, device=device), prior_scale*self.prior_sigma * t.ones(size, device=device))
        self.h_b = t.distributions.Normal(t.zeros(out, device=device), prior_scale*self.prior_sigma * t.ones(out, device=device))  
        k1 = t.distributions.kl_divergence(self.eps_w,self.h_w).sum()        
        k2 = t.distributions.kl_divergence(self.eps_b,self.h_b).sum()        
        return k1+k2

class VarConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding, init_log_sigma, act=F.relu, H = 3, W = 3, stride=(1, 1), dilation = (1, 1), groups = 1, prior_sigma = 1.0):
        super(VarConv, self).__init__()
        
        self.mean = nn.Parameter(t.randn(out_channels, in_channels, H, W)) # параметры средних            
        self.log_sigma = nn.Parameter(t.randn(out_channels, in_channels, H, W)) # логарифм дисперсии
        self.mean_b = nn.Parameter(t.randn(out_channels)) # то же самое для свободного коэффициента
        self.log_sigma_b = nn.Parameter(t.randn(out_channels))
     
        init.kaiming_uniform_(self.mean.data, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.data)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.mean_b.data, -bound, bound)        
            
        self.log_sigma.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma.data+= init_log_sigma
     
        self.log_sigma_b.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma_b.data+= init_log_sigma    
        
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
                
        self.size_m = (out_channels, in_channels, H, W)
        self.out_ = out_channels
        self.prior_sigma = prior_sigma
        self.act = act
        
    def forward(self, x, l=None):
        if self.training: # во время обучения - сэмплируем из нормального распределения
            self.eps_w = t.distributions.Normal(self.mean, MIN_SIGMA+t.exp(self.log_sigma))
            self.eps_b = t.distributions.Normal(self.mean_b, MIN_SIGMA+t.exp(self.log_sigma_b))
        
            w = self.eps_w.rsample()
            b = self.eps_b.rsample()
             
        else:  # во время контроля - смотрим средние значения параметра        
            w = self.mean
            b = self.mean_b
            
        return self.act(F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups))

    def KLD(self, l=None,  prior_scale=1.0):
        size = self.mean.shape
        out = self.mean_b.shape
        # подсчет дивергенции        
        device = self.mean.device
        self.eps_w = t.distributions.Normal(self.mean, MIN_SIGMA+t.exp(self.log_sigma))
        self.eps_b = t.distributions.Normal(self.mean_b,  MIN_SIGMA+t.exp(self.log_sigma_b))     
        self.h_w = t.distributions.Normal(t.zeros(size, device=device), prior_scale*self.prior_sigma * t.ones(size, device=device))
        self.h_b = t.distributions.Normal(t.zeros(out, device=device), prior_scale*self.prior_sigma * t.ones(out, device=device))  
        k1 = t.distributions.kl_divergence(self.eps_w,self.h_w).sum()        
        k2 = t.distributions.kl_divergence(self.eps_b,self.h_b).sum()        
        return k1+k2
    

class VarNet(nn.Sequential):    
    # класс-обертка на случай, если у нас многослойная нейронная сеть
    def KLD(self, lam = None, prior_scale = 1.0):
        k = 0
        for l in self: 
            if lam is None:
                k+=l.KLD(prior_scale = prior_scale)
            else:
                k+=l.KLD(lam, prior_scale = prior_scale)
                
        return k
    
    def forward(self, x, lam = None):
        if lam is None:
            for l in self:
                x = l(x)
            return x
        else:
            for l in self:
                x = l(x, lam)
            return x
    
