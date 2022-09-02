import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import math

        
MIN_SIGMA = 1e-5
class KernelNet(nn.Module):
    def __init__(self, size, kernel_num,  init_ = 'random', init_r = 10.1):    
        nn.Module.__init__(self)
        
        if not isinstance(size, tuple): # check if size is 1d
            size = (size,)
            
        self.size = size
        
        
        full_param_size = np.prod(self.size)        
        total_size = [kernel_num]+list(self.size)
        self.kernel_num = kernel_num  
        
        #self.const = nn.Parameter(t.randn(size))
        
        
        self.const = nn.Parameter(t.randn(total_size))
        if init_ == 'random':
            
            for i in range(kernel_num):
                if len(self.size)>1:
                    init.kaiming_uniform_(self.const.data[i], a= np.sqrt(5))
                else:
                
                    self.const.data[i]*=0
                    self.const.data[i]+=t.randn(size)
                
        else:
            self.const.data *=0
            self.const.data += init_
        """
        if init_ == 'random':
            if len(self.size)>1:
                init.kaiming_uniform_(self.const.data, a= np.sqrt(5))
            else:

                self.const.data*=0
                self.const.data+=t.randn(size)                
        else:
            self.const.data *=0
            self.const.data += init_
        """
        self.r = nn.Parameter(t.log(t.ones(kernel_num)*init_r))        
        self.pivots = nn.Parameter(t.tensor(np.linspace(0, 1,kernel_num)), requires_grad=False)
        
            
    def forward(self, lam):   
        lam_ = lam * 0.99999
        left = t.floor(lam_*(self.kernel_num-1)).long() 
        right = left + 1
        dist = (self.pivots[right]-lam_)/(self.pivots[right]-self.pivots[left])
        return self.const[left] * (dist) + (1.0-dist) * self.const[right]
        
        """
        dist = 1.0-abs(lam - self.pivots)
        dist = dist / dist.sum()
        result = self.const[0] * dist[0]
        for i in range(1, self.kernel_num):
            result = result +  dist[i] * self.const[i]
        return result
        """
        
       

class KernelVarConv(nn.Module):
        def __init__(self, in_channels, out_channels, padding, init_log_sigma, kernel_num, act=F.relu, H = 3, W = 3, stride=(1, 1), dilation = (1, 1), groups = 1, prior_sigma = 1.0 ):
            super(KernelVarConv, self).__init__()

            self.mean = KernelNet((out_channels, in_channels, H, W), kernel_num)# параметры средних            
            self.log_sigma = KernelNet((out_channels, in_channels, H, W), kernel_num, init_=init_log_sigma) # логарифм дисперсии
            self.mean_b = KernelNet((out_channels), kernel_num) # то же самое для свободного коэффициента
            self.log_sigma_b = KernelNet((out_channels), kernel_num, init_=init_log_sigma)
            for i in range(kernel_num):            
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
        
        def sample(self, l):
            self.eps_w = t.distributions.Normal(self.mean(l), MIN_SIGMA+t.exp(self.log_sigma(l)))
            self.eps_b = t.distributions.Normal(self.mean_b(l), MIN_SIGMA+t.exp(self.log_sigma_b(l)))
            w = self.eps_w.sample()
            b = self.eps_b.sample()
            return w,b
        
        def forward(self, x, l):
            if   self.training: # во время обучения - сэмплируем из нормального распределения
                self.eps_w = t.distributions.Normal(self.mean(l), MIN_SIGMA+t.exp(self.log_sigma(l)))
                self.eps_b = t.distributions.Normal(self.mean_b(l), MIN_SIGMA+t.exp(self.log_sigma_b(l)))

                w = self.eps_w.rsample()
                b = self.eps_b.rsample()

            else:  # во время контроля - смотрим средние значения параметра        
                w = self.mean(l)
                b = self.mean_b(l)

            return self.act(F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups))

        def KLD(self, l,  prior_scale=1.0):  
            
            size = self.mean.size
            out = self.mean_b.size
            # подсчет дивергенции 
            try:
                device = self.mean.const.device
            except:
                device = self.mean.const1.device
                
            self.eps_w = t.distributions.Normal(self.mean(l), MIN_SIGMA+t.exp(self.log_sigma(l)))
            self.eps_b = t.distributions.Normal(self.mean_b(l),  MIN_SIGMA+t.exp(self.log_sigma_b(l)))     
            self.h_w = t.distributions.Normal(t.zeros(size, device=device), prior_scale*self.prior_sigma * t.ones(size, device=device))
            self.h_b = t.distributions.Normal(t.zeros(out, device=device), prior_scale*self.prior_sigma * t.ones(out, device=device))  
            k1 = t.distributions.kl_divergence(self.eps_w,self.h_w).sum()        
            k2 = t.distributions.kl_divergence(self.eps_b,self.h_b).sum()        
            return k1+k2

class VarKernelLayer(nn.Module):  # вариационная однослойная сеть
    def __init__(self, in_,  out_,  kernel_num, prior_sigma=1.0, init_log_sigma=-3.0,  act=F.relu):
        nn.Module.__init__(self)
        self.mean = KernelNet((in_, out_), kernel_num)  # параметры средних
        self.log_sigma = KernelNet(
            (in_, out_), kernel_num, init_=init_log_sigma)  # логарифм дисперсии
        # то же самое для свободного коэффициента
        self.mean_b = KernelNet(out_, kernel_num)
        self.log_sigma_b = KernelNet(out_, kernel_num, init_=init_log_sigma)
        for i in range(kernel_num):
            t.nn.init.xavier_uniform_(self.mean.const.data[i]) 
        
        
        
        self.in_ = in_
        self.out_ = out_
        self.act = act
        self.prior_sigma = prior_sigma

    def forward(self, x, l):
        if self.training:  # во время обучения - сэмплируем из нормального распределения
            self.eps_w = t.distributions.Normal(
                self.mean(l), MIN_SIGMA+t.exp(self.log_sigma(l)))
            self.eps_b = t.distributions.Normal(
                self.mean_b(l), MIN_SIGMA+t.exp(self.log_sigma_b(l)))

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
                    
        # подсчет дивергенции 
        
        device = self.mean.const.device
       
                
        self.eps_w = t.distributions.Normal(
            self.mean(l), MIN_SIGMA+t.exp(self.log_sigma(l)))
        self.eps_b = t.distributions.Normal(
            self.mean_b(l), MIN_SIGMA+t.exp(self.log_sigma_b(l)))
        self.h_w = t.distributions.Normal(
            t.zeros(size, device=device), t.ones(size, device=device)*self.prior_sigma * prior_scale)
        self.h_b = t.distributions.Normal(
            t.zeros(out, device=device), t.ones(out, device=device)*self.prior_sigma * prior_scale)
        k1 = t.distributions.kl_divergence(self.eps_w, self.h_w).sum()
        k2 = t.distributions.kl_divergence(self.eps_b, self.h_b).sum()
        return k1+k2

    
