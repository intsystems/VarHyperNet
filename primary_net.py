import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from hypernetwork_modules import HyperNetwork
from resnet_blocks import ResNetBlock


class Embedding(nn.Module):

    def __init__(self, z_num, z_dim, device = 'cpu', mode = 'mean', init_log_sigma = -4.0):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim
        self.init_log_sigma = init_log_sigma

        h,k = self.z_num

        if mode == 'mean':
            for i in range(h):
                for j in range(k):
                    self.z_list.append(Parameter(t.fmod(t.randn(self.z_dim).to(device), 4)))
        else:
            for i in range(h):
                for j in range(k):
                    self.z_list.append(Parameter(t.ones(self.z_dim).to(device)*self.init_log_sigma))

    def forward(self, hyper_net, lam = None):
        ww = []
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j], lam))
            ww.append(t.cat(w, dim=1))
        return t.cat(ww, dim=0)
           
    


class PrimaryNetwork(nn.Module):

    def __init__(self, z_dim=64, device = 'cpu', prior_sigma = 1.0):
        super(PrimaryNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.z_dim = z_dim
        self.hope1 = HyperNetwork(z_dim=self.z_dim)
        self.hope2 = HyperNetwork(z_dim=self.z_dim)
        self.prior_sigma = prior_sigma
        self.device = device

        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                        [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size = [[16,16], [16,16], [16,16], [16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,32],
                            [32,32], [32,32], [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]]

        self.res_net = nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample))

        self.zs_mean = nn.ModuleList()
        self.zs_sigma = nn.ModuleList()

        for i in range(36):
            self.zs_mean.append(Embedding(self.zs_size[i], self.z_dim, self.device, 'mean'))
            self.zs_sigma.append(Embedding(self.zs_size[i], self.z_dim, self.device, 'sigma'))

        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(64,10)
        self.h1_eps = None  # lazy initialization. we check only h1. h2 is initialized at the same time
        
        
    def forward(self, x, lam = None):
        x = F.relu(self.bn1(self.conv1(x)))
        if  self.training:
            # flattening all w1 and w2 into large 1d-tensors (for 1 normal sampling step)
            w1_mean = []
            w1_sigma = []
            w2_mean = []
            w2_sigma = []
            
            # saving start-end and real shape for each tensor in large 1d-tensor
            starts1 = []
            ends1 = []
            sizes1 = []
            
            starts2 = []
            ends2 = []
            sizes2 = []
            
            for i in range(18):
                w = self.zs_mean[2*i](self.hope1, lam)
                if len(ends1)==0:
                    starts1.append(0)
                else:
                    starts1.append(ends1[-1])
                sizes1.append(w.shape)
                w1_mean += [w.flatten()]
                ends1.append(starts1[-1] + w1_mean[-1].shape[0])
                
                w = self.zs_mean[2*i+1](self.hope2, lam)
                if len(ends2)==0:
                    starts2.append(0)
                else:
                    starts2.append(ends2[-1])
                sizes2.append(w.shape)
                w2_mean += [w.flatten()]
                ends2.append(starts2[-1] + w2_mean[-1].shape[0])
                
                w1_sigma += [self.zs_sigma[2*i](self.hope1, lam).flatten()]
                w2_sigma += [self.zs_sigma[2*i+1](self.hope2, lam).flatten()]
            w1_mean_all = t.cat(w1_mean)
            w2_mean_all = t.cat(w2_mean)
            w1_sigma_all = t.cat(w1_sigma)
            w2_sigma_all = t.cat(w2_sigma)
            self.w1_eps = t.distributions.Normal(w1_mean_all, t.exp(w1_sigma_all+0.2))
            self.w2_eps = t.distributions.Normal(w2_mean_all, t.exp(w2_sigma_all+0.2))
            
            if self.h1_eps is None: # lazy h1/h2 implementation. We need it only once
                self.h1_eps = t.distributions.Normal(t.zeros_like((w1_mean_all), device=self.device),
                                               t.ones_like(w1_sigma_all, device=self.device)*self.prior_sigma)
                self.h2_eps = t.distributions.Normal(t.zeros_like((w2_mean_all), device=self.device),
                                               t.ones_like(w2_sigma_all, device=self.device)*self.prior_sigma)
            
            w1_sample = self.w1_eps.rsample()
            w2_sample = self.w2_eps.rsample()
            
            for i in range(18):
                w1 = w1_sample[starts1[i]:ends1[i]].view(sizes1[i])
                w2 = w2_sample[starts2[i]:ends2[i]].view(sizes2[i])
                x = self.res_net[i](x, w1, w2)
        else:   
            for i in range(18):
                # if i != 15 and i != 17:
                w1_mean = self.zs_mean[2*i](self.hope1, lam)
                w2_mean = self.zs_mean[2*i+1](self.hope1, lam)

                w1 = w1_mean
                w2 = w2_mean
                x = self.res_net[i](x, w1, w2)

        x = self.global_avg(x)
        x = self.final(x.view(-1,64))
        return x
    
    def KLD(self, l):
        # подсчет дивергенции
        
        k = t.distributions.kl_divergence(self.w1_eps, self.h1_eps).sum()
        k += t.distributions.kl_divergence(self.w2_eps, self.h2_eps).sum()
        return k