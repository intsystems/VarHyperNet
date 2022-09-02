import numpy as np 
import torch as t 

# будем удалять по 10% от модели и смотреть качество
def delete_10(net, device, callback, layer_num=2, mode='var'):
    acc_delete = []
    prune_coefs = []
    for l in range(layer_num):
        mu = net[l].mean
        if mode == 'var':
            sigma = t.exp(2*net[l].log_sigma)
        else:
            sigma = 1.0
            
        prune_coef = (mu**2/sigma).cpu().detach().numpy()    
        prune_coefs.append(prune_coef)
    
    
    for j in range(10):
        for l in range(layer_num):
            prune_coef = prune_coefs[l]
            sorted_coefs = np.sort(prune_coef.flatten())
            ids = (prune_coef <= sorted_coefs[round(j/10*len(sorted_coefs))]) 
            net[l].mean.data*=(1-t.tensor(ids*1.0, device=device, dtype=t.float))                
        
        acc_delete.append(callback())
    return acc_delete    


def net_copy(net, new_net,  lam, layer_num = 2, mode='var'):    
    for j in range(layer_num): # бежим по слоям        
        new_net[j].mean.data*=0
        new_net[j].mean.data+=net[j].mean(lam)
        new_net[j].mean_b.data*=0
        new_net[j].mean_b.data+=net[j].mean_b(lam)
        if mode == 'var':
            new_net[j].log_sigma.data*=0
            new_net[j].log_sigma.data+=net[j].log_sigma(lam)
            new_net[j].log_sigma_b.data*=0
            new_net[j].log_sigma_b.data+=net[j].log_sigma_b(lam)
    