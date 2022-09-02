import torch as t 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

MIN_SIGMA = 1e-5


class VarLayerLinearAppr(nn.Module): # вариационная однослойная сеть
    def __init__(self, in_,  out_, prior_sigma = 1.0, init_log_sigma=-3.0, 
                 act=F.relu):         
        nn.Module.__init__(self)                    
        self.mean = LinearApprNet((in_, out_)) # параметры средних            
        self.log_sigma = LinearApprNet((in_, out_)) # логарифм дисперсии
        self.mean_b = LinearApprNet( out_) # то же самое для свободного коэффициента
        self.log_sigma_b = LinearApprNet( out_)
     
        self.log_sigma.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma.const.data+= init_log_sigma
     
        self.log_sigma_b.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma_b.const.data+= init_log_sigma    
        
       
        self.log_sigma.const2.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma.const2.data+= init_log_sigma

        self.log_sigma_b.const2.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma_b.const2.data+= init_log_sigma

        self.in_ = in_
        self.out_ = out_
        self.act = act
        self.prior_sigma = prior_sigma
        
    def forward(self, x, l):
        if self.training: # во время обучения - сэмплируем из нормального распределения
            self.eps_w = t.distributions.Normal(self.mean(l), MIN_SIGMA+t.exp(self.log_sigma(l)))
            self.eps_b = t.distributions.Normal(self.mean_b(l), MIN_SIGMA+t.exp(self.log_sigma_b(l)))
        
            w = self.eps_w.rsample()
            b = self.eps_b.rsample()
             
        else:  # во время контроля - смотрим средние значения параметра        
            w = self.mean(l) 
            b = self.mean_b(l)
            
        # функция активации 
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l, prior_scale=1.0):   

        # подсчет дивергенции
        size = self.in_, self.out_
        out = self.out_
        device = self.mean.const.device
        self.eps_w = t.distributions.Normal(self.mean(l), MIN_SIGMA+t.exp(self.log_sigma(l)))
        self.eps_b = t.distributions.Normal(self.mean_b(l),  MIN_SIGMA+t.exp(self.log_sigma_b(l)))
        self.h_w = t.distributions.Normal(t.zeros(size, device=device), prior_scale* t.ones(size, device=device)*self.prior_sigma)
        self.h_b = t.distributions.Normal(t.zeros(out, device=device), prior_scale*t.ones(out, device=device)*self.prior_sigma)                
        k1 = t.distributions.kl_divergence(self.eps_w,self.h_w).sum()        
        k2 = t.distributions.kl_divergence(self.eps_b,self.h_b).sum()        
        return k1+k2            
    


class LinearVarConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding, init_log_sigma, act=F.relu, H = 3, W = 3, stride=(1, 1), dilation = (1, 1), groups = 1, prior_sigma = 1.0):
        super(LinearVarConv, self).__init__()
        
        self.mean = LinearApprNet((out_channels, in_channels, H, W))# параметры средних            
        self.log_sigma = LinearApprNet((out_channels, in_channels, H, W)) # логарифм дисперсии
        self.mean_b = LinearApprNet((out_channels)) # то же самое для свободного коэффициента
        self.log_sigma_b = LinearApprNet((out_channels))
        
        
     
        init.kaiming_uniform_(self.mean.const.data, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.const.data)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.mean_b.const.data, -bound, bound)        

        init.kaiming_uniform_(self.mean.const2.data, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.const2.data)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.mean_b.const2.data, -bound, bound)        
            
            
        self.log_sigma.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma.const.data+= init_log_sigma
     
        self.log_sigma_b.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma_b.const.data+= init_log_sigma    
        
        
        self.log_sigma.const2.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma.const2.data+= init_log_sigma

        self.log_sigma_b.const2.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma_b.const2.data+= init_log_sigma


        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
                
        self.size_m = (out_channels, in_channels, H, W)
        self.out_ = out_channels
        self.prior_sigma = prior_sigma
        self.act = act
        
    def forward(self, x, l):
        if self.training: # во время обучения - сэмплируем из нормального распределения
            self.eps_w = t.distributions.Normal(self.mean(l), MIN_SIGMA+t.exp(self.log_sigma(l)))
            self.eps_b = t.distributions.Normal(self.mean_b(l), MIN_SIGMA+t.exp(self.log_sigma_b(l)))
        
            w = self.eps_w.rsample()
            b = self.eps_b.rsample()
             
        else:  # во время контроля - смотрим средние значения параметра        
            w = self.mean(l)
            b = self.mean_b(l)
            
        return self.act(F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups))

    def KLD(self, l,  prior_scale=1.0):
        size = self.mean.const.shape
        out = self.mean_b.const.shape
        # подсчет дивергенции        
        device = self.mean.const.device
        self.eps_w = t.distributions.Normal(self.mean(l), MIN_SIGMA+t.exp(self.log_sigma(l)))
        self.eps_b = t.distributions.Normal(self.mean_b(l),  MIN_SIGMA+t.exp(self.log_sigma_b(l)))     
        self.h_w = t.distributions.Normal(t.zeros(size, device=device), prior_scale*self.prior_sigma * t.ones(size, device=device))
        self.h_b = t.distributions.Normal(t.zeros(out, device=device), prior_scale*self.prior_sigma * t.ones(out, device=device))  
        k1 = t.distributions.kl_divergence(self.eps_w,self.h_w).sum()        
        k2 = t.distributions.kl_divergence(self.eps_b,self.h_b).sum()        
        return k1+k2
    
        
class LinearApprNet(nn.Module):
    def __init__(self, size,  init_const = 1.0, init_const2 = 1.0):    
        
        nn.Module.__init__(self)        
        if isinstance(size, tuple) and len(size) == 2:
            self.in_, self.out_ = size
            self.diagonal = False
        else:
            self.out_ = size
            self.diagonal = True
                           
        
        if self.diagonal:
            # независимая от параметра lambda часть
            self.const = nn.Parameter(t.randn(self.out_) * init_const) 
            self.const2 = nn.Parameter(t.ones(self.out_) * init_const2) 
            
            
        else:
            self.const = nn.Parameter(t.randn(self.in_, self.out_)) 
            t.nn.init.xavier_uniform(self.const,  init_const)
            self.const2 = nn.Parameter(t.randn(self.in_, self.out_)) 
            t.nn.init.xavier_uniform(self.const2,  init_const2)
            
            
    def forward(self, lam):        
        if self.diagonal:
            return self.const * lam + self.const2 * (1.0-lam)
        else:
            return self.const * lam + self.const2 * (1.0 - lam )
