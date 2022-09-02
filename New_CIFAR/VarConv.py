import torch as t 
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
        self.s = []

    def forward(self, x):
        x = F.relu(self.conv1(x)) #32*32*48
        x = F.relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = F.relu(self.conv3(x)) #16*16*192
        x = F.relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = x.view(-1, 8*8*256) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VarConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding, init_log_sigma, H = 3, W = 3, stride=(1, 1), dilation = (1, 1), groups = 1, prior_sigma = 0.1):
        super(VarConv, self).__init__()        
        self.mean = LinearApprNet((out_channels, in_channels, H, W)) # параметры средних            
        self.log_sigma = LinearApprNet((out_channels, in_channels, H, W)) # логарифм дисперсии
        self.mean_b = LinearApprNet(out_channels) # то же самое для свободного коэффициента
        self.log_sigma_b = LinearApprNet( out_channels)
     
        init.kaiming_uniform_(self.mean.const, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.const)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.mean_b.const, -bound, bound)
        
        
        init.kaiming_uniform_(self.mean.const2, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.const2)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.mean_b.const2, -bound, bound)
        
        
            
        self.log_sigma.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma.const.data+= init_log_sigma
     
        self.log_sigma_b.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma_b.const.data+= init_log_sigma    
        
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
                
        self.size_m = (out_channels, in_channels, H, W)
        self.out_ = out_channels
        self.prior_sigma = prior_sigma
        
    def forward(self, x, l):
        if self.training: # во время обучения - сэмплируем из нормального распределения
            self.eps_w = t.distributions.Normal(self.mean(l), t.exp(self.log_sigma(l)))
            self.eps_b = t.distributions.Normal(self.mean_b(l), t.exp(self.log_sigma_b(l)))
        
            w = self.eps_w.rsample()
            b = self.eps_b.rsample()
             
        else:  # во время контроля - смотрим средние значения параметра        
            w = self.mean(l) 
            b = self.mean_b(l)
        
        #w = self.mean(l)
        #b = self.mean_b(l)
        
        # функция активации 
        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

    def KLD(self, l):        
        # подсчет дивергенции
        device = self.mean.const.device
     
        self.eps_w = t.distributions.Normal(self.mean(l), t.exp(self.log_sigma(l)))
        self.eps_b = t.distributions.Normal(self.mean_b(l),  t.exp(self.log_sigma_b(l)))
        self.h_w = t.distributions.Normal(t.zeros(self.size_m, device=device), t.ones(self.size_m, device=device)*self.prior_sigma)
        self.h_b = t.distributions.Normal(t.zeros(self.out_, device=device), t.ones(self.out_, device=device)*self.prior_sigma)                
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
            t.nn.init.xavier_uniform_(self.const,  init_const)
            self.const2 = nn.Parameter(t.randn(self.in_, self.out_)) 
            t.nn.init.xavier_uniform_(self.const2,  init_const2)
            
            
    def forward(self, lam):        
        if self.diagonal:
            return self.const + self.const2 * lam
        else:
            return self.const + self.const2 * lam 


class VarFC(nn.Module):
    def __init__(self, in_features, out_features, init_log_sigma, prior_sigma = 0.1):
        super(VarFC, self).__init__()     
        self.in_features = in_features
        self.out_features = out_features
        
        self.mean = LinearApprNet((out_features, in_features))
        self.log_sigma = LinearApprNet((out_features, in_features))
        self.mean_b = LinearApprNet(out_features) # то же самое для свободного коэффициента
        self.log_sigma_b = LinearApprNet(out_features)
        
     
        init.kaiming_uniform_(self.mean.const, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.const)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.mean_b.const, -bound, bound)
 
        init.kaiming_uniform_(self.mean.const2, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean.const2)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.mean_b.const2, -bound, bound)
        del bound, fan_in
           
        self.log_sigma.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma.const.data+= init_log_sigma
     
        self.log_sigma_b.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma_b.const.data+= init_log_sigma    
        
        self.prior_sigma = prior_sigma
        
    def forward(self, x, l):
        if self.training: # во время обучения - сэмплируем из нормального распределения
            self.eps_w = t.distributions.Normal(self.mean(l), t.exp(self.log_sigma(l)))
            self.eps_b = t.distributions.Normal(self.mean_b(l), t.exp(self.log_sigma_b(l)))
        
            w = self.eps_w.rsample()
            b = self.eps_b.rsample()
             
        else:  # во время контроля - смотрим средние значения параметра        
            w = self.mean(l) 
            b = self.mean_b(l)
        
        return F.linear(x, w, b)

    def KLD(self, l):        
        # подсчет дивергенции
        device = self.mean.const.device
        
        self.eps_w = t.distributions.Normal(self.mean(l), t.exp(self.log_sigma(l)))
        self.eps_b = t.distributions.Normal(self.mean_b(l),  t.exp(self.log_sigma_b(l)))
        self.h_w = t.distributions.Normal(t.zeros((self.out_features, self.in_features), device=device), t.ones((self.out_features, self.in_features), device=device)*self.prior_sigma)
        self.h_b = t.distributions.Normal(t.zeros(self.out_features, device=device), t.ones(self.out_features, device=device)*self.prior_sigma)                
        k1 = t.distributions.kl_divergence(self.eps_w,self.h_w).sum()        
        k2 = t.distributions.kl_divergence(self.eps_b,self.h_b).sum()        
        return k1+k2            



class VarConvNet(nn.Module):
    def __init__(self, init_log_sigma = -5.0, prior_sigma = 1.0):
        super(VarConvNet, self).__init__()
        self.conv1 = VarConv(in_channels=3, out_channels=48, H = 3, W = 3, padding=(1,1),
                            init_log_sigma = init_log_sigma, prior_sigma = prior_sigma)
        self.conv2 = VarConv(in_channels=48, out_channels=96, H = 3, W = 3, padding=(1,1),
                            init_log_sigma = init_log_sigma, prior_sigma = prior_sigma)
        self.conv3 = VarConv(in_channels=96, out_channels=192, H = 3, W = 3, padding=(1,1), 
                             init_log_sigma = init_log_sigma, prior_sigma = prior_sigma)
        self.conv4 = VarConv(in_channels=192, out_channels=256, H = 3, W = 3, padding=(1,1),
                             init_log_sigma = init_log_sigma, prior_sigma = prior_sigma)
        
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = VarFC(in_features=8*8*256, out_features=512, init_log_sigma = init_log_sigma)
        self.fc2 = VarFC(in_features=512, out_features=64, init_log_sigma = init_log_sigma)
        self.fc3 = VarFC(in_features=64, out_features=10, init_log_sigma = init_log_sigma)

    def forward(self, x, l):
        x = F.relu(self.conv1(x, l)) #32*32*48
        x = F.relu(self.conv2(x, l)) #32*32*96
        x = self.pool(x) #16*16*96
        x = F.relu(self.conv3(x, l)) #16*16*192
        x = F.relu(self.conv4(x, l)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = x.view(-1, 8*8*256) # reshape x
        x = F.relu(self.fc1(x, l))
        x = F.relu(self.fc2(x, l))
        x = self.fc3(x, l)
        return x
    
    def KLD(self, lam = None):
        k = 0
        k += self.conv1.KLD(lam)
        k += self.conv2.KLD(lam)
        k += self.conv3.KLD(lam)
        k += self.conv4.KLD(lam)
        k += self.fc1.KLD(lam)
        k += self.fc2.KLD(lam)
        k += self.fc3.KLD(lam)
        return k
    
