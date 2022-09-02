import torch as t 
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
import numpy as np

def net_copy(net, net2,  lam):  
    with t.no_grad():
        net2.conv1.weight*=0
        net2.conv1.weight+=net.conv1.mean(lam)
        net2.conv2.weight*=0
        net2.conv2.weight+=net.conv2.mean(lam)
        net2.conv3.weight*=0
        net2.conv3.weight+=net.conv3.mean(lam)
        net2.conv4.weight*=0
        net2.conv4.weight+=net.conv4.mean(lam)
        
        net2.conv1.bias*=0
        net2.conv1.bias+=net.conv1.mean_b(lam)
        net2.conv2.bias*=0
        net2.conv2.bias+=net.conv2.mean_b(lam)
        net2.conv3.bias*=0
        net2.conv3.bias+=net.conv3.mean_b(lam)
        net2.conv4.bias*=0
        net2.conv4.bias+=net.conv4.mean_b(lam)
        
        net2.fc1.weight*=0
        net2.fc1.weight+=net.fc1.mean(lam)
        net2.fc2.weight*=0
        net2.fc2.weight+=net.fc2.mean(lam)
        net2.fc3.weight*=0
        net2.fc3.weight+=net.fc3.mean(lam)
        
        net2.fc1.bias*=0
        net2.fc1.bias+=net.fc1.mean_b(lam)
        net2.fc2.bias*=0
        net2.fc2.bias+=net.fc2.mean_b(lam)
        net2.fc3.bias*=0
        net2.fc3.bias+=net.fc3.mean_b(lam)
        
        net2.conv1.bias*=0
        net2.conv1.bias+=net.conv1.mean_b(lam)
        net2.conv2.bias*=0
        net2.conv2.bias+=net.conv2.mean_b(lam)
        net2.conv3.bias*=0
        net2.conv3.bias+=net.conv3.mean_b(lam)
        net2.conv4.bias*=0
        net2.conv4.bias+=net.conv4.mean_b(lam)
        
        for i in [net.conv1, net.conv2, net.conv3, net.conv4, net.fc1, net.fc2, net.fc3]:
            net2.s.append(i.log_sigma(lam))
            net2.s.append(i.log_sigma_b(lam))
            
            
# будем удалять по 10% от модели и смотреть качество
def delete_10(net, device, callback):
    acc_delete = []
    prune_coefs = []
    for n, i in enumerate([net.conv1.weight, net.conv1.bias, net.conv2.weight, net.conv2.bias,
                           net.conv3.weight, net.conv3.bias, net.conv4.weight, net.conv4.bias, 
                           net.fc1.weight, net.fc1.bias, net.fc2.weight, net.fc2.bias,
                           net.fc3.weight, net.fc3.bias]):
        mu = i
        sigma = t.exp(2*net.s[n]) 
        prune_coefs.append((mu**2/sigma).cpu().detach().numpy())
    
    
    for j in range(10):
        for n, i in enumerate([net.conv1.weight, net.conv1.bias, net.conv2.weight, net.conv2.bias,
                           net.conv3.weight, net.conv3.bias, net.conv4.weight, net.conv4.bias, 
                           net.fc1.weight, net.fc1.bias, net.fc2.weight, net.fc2.bias,
                           net.fc3.weight, net.fc3.bias]):
            prune_coef = prune_coefs[n]
            sorted_coefs = np.sort(prune_coef.flatten())
            ids = (prune_coef <= sorted_coefs[round(j/10*len(sorted_coefs))]) 
            i.data*=(1-t.tensor(ids*1.0, device=device, dtype=t.float))                
        
        acc_delete.append(callback())
    return acc_delete 

