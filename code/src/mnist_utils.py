import numpy as np 
import torch as t 
from torch.nn.utils import clip_grad_value_
import tqdm

        
def train_batches_net(train_loader,  train_data_num, net, device, loss_fn, optimizer, lam, label, rep = False):
    tq = tqdm.tqdm(train_loader)
    losses = []
    for x,y in tq:            
        x = x.to(device)
        y = y.to(device)          
        optimizer.zero_grad()  
        loss = 0
        
        out = net(x)
        loss = loss + loss_fn(out, y)
        if rep:
            loss += net.KLD(prior_scale = t.sqrt(1.0/lam))/train_data_num
        else:
            loss += net.KLD()*lam/train_data_num
        losses+=[loss.cpu().detach().numpy()]
        tq.set_description(label+str(np.mean(losses)))
        loss.backward()       
        clip_grad_value_(net.parameters(), 1.0) # для стабильности градиента. С этим можно играться
        optimizer.step()
        
def train_batches_hypernet(train_loader,  train_data_num, lambda_sample_num,
                  lambda_encode, net, device, loss_fn, optimizer, label, rep = False, p_gen = lambda p_:p_*4-2, pretrain=False):
    tq = tqdm.tqdm(train_loader)
    losses = []
    for x,y in tq:            
        x = x.to(device)
        y = y.to(device)          
        optimizer.zero_grad()  
        loss = 0
       
        for _ in range(lambda_sample_num):  
            p = p_gen(t.rand(1).to(device))            
            lam_param = 10**p[0]           

            out = net(x, lambda_encode(lam_param))
            loss = loss + loss_fn(out, y)/lambda_sample_num
            fn_loss = loss.cpu().detach().numpy()
            if not pretrain:
                if rep:
                    loss += net.KLD(lambda_encode(lam_param), prior_scale=t.sqrt(1.0/lam_param))/train_data_num/lambda_sample_num
                else:
                    loss += net.KLD(lambda_encode(lam_param))*lam_param/train_data_num/lambda_sample_num
            losses+=[(fn_loss, loss.cpu().detach().numpy())]
        # правдоподобие должно суммироваться по всей обучающей выборке
        # в случае батчей - она приводится к тому же порядку 
        
        tq.set_description(label+str(np.mean(losses, 0)))
        loss.backward()       
        clip_grad_value_(net.parameters(), 1.0) # для стабильности градиента. С этим можно играться
        optimizer.step()
        
        
def test_acc_net(net, device, test_loader, use_eval = True): # точность классификации
    correct = 0
    if use_eval:
        net.eval()
    cnt = 0
    with t.no_grad():
        for x,y in test_loader: 
            x = x.to(device)
            y = y.to(device)  
            out = net(x)    
            cnt += len(x)
            correct += out.argmax(1).eq(y).sum().cpu().numpy()
    acc = (correct / cnt)
    if use_eval:
        net.train()
    return acc


def test_acc_hyper(net, device, test_loader, lambda_encode, lambdas, use_eval = True): # точность классификации
    if use_eval:
        net.eval()
    cnt = 0
    accs = {}
    with t.no_grad():
        for l in lambdas:
            correct = 0
            cnt = 0
            for x,y in test_loader: 
                x = x.to(device)
                y = y.to(device)  
                out = net(x, lambda_encode(l))    
                cnt += len(x)
                correct += out.argmax(1).eq(y).sum().cpu().numpy()
            acc = (correct / cnt)
            accs[l] = acc
    if use_eval:
        net.train()
    return accs
        

# будем удалять по 10% от модели и смотреть качество
def delete_10(net, device, callback, layer_num=2, mode='var'):
    acc_delete = []
    prune_coefs = []
    for l in range(layer_num):
        try:
            mu = net[l].mean
            if mode == 'var':
                sigma = t.exp(2*net[l].log_sigma)
            else:
                sigma = 1.0

            prune_coef = (mu**2/sigma).cpu().detach().numpy()    
            prune_coefs.append(prune_coef)
        except:
            print ('layer', net[l], 'does not have mean/sigma')
            prune_coefs.append([])

    
    for j in range(10):
        for l in range(layer_num):
            prune_coef = prune_coefs[l]
            if len(prune_coef)>0:
                sorted_coefs = np.sort(prune_coef.flatten())
                ids = (prune_coef <= sorted_coefs[round(j/10*len(sorted_coefs))]) 
                net[l].mean.data*=(1-t.tensor(ids*1.0, device=device, dtype=t.float))                
        
        acc_delete.append(callback())
    return acc_delete    


def net_copy(net, new_net,  lam, layer_num = 2, mode='var'):   
    
    for j in range(layer_num): # бежим по слоям   
        try:
            new_net[j].mean.data*=0
            new_net[j].mean.data+=net[j].mean(lam)
            new_net[j].mean_b.data*=0
            new_net[j].mean_b.data+=net[j].mean_b(lam)
            if mode == 'var':
                new_net[j].log_sigma.data*=0
                new_net[j].log_sigma.data+=net[j].log_sigma(lam)
                new_net[j].log_sigma_b.data*=0
                new_net[j].log_sigma_b.data+=net[j].log_sigma_b(lam)
        except:
            print ('layer', net[j], 'does not have mean/sigma')            