from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.nn import Parameter
from gmm import *

def gmm_update_center(feature, device, n_cluster):
    print('feature',feature)
    Y=feature.numpy()
    matY = np.matrix(Y, copy=True)

 # 模型个数，即聚类的类别个数
    K = n_cluster

 # 计算 GMM 模型参数
    mu, cov, alpha = GMM_EM(matY, K, 100)

 # 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
    N = Y.shape[0]
 # 求当前模型参数下，各模型对样本的响应度矩阵
    gamma = getExpectation(matY, mu, cov, alpha)
 # 对每个样本，求响应度最大的模型下标，作为其类别标识
    category = gamma.argmax(axis=1).flatten().tolist()[0]

    return category
def gmm_initial_center(feature, device, n_cluster):
    print('feature',feature)
    Y=feature.numpy()
    matY = np.matrix(Y, copy=True)

 # 模型个数，即聚类的类别个数
    K = n_cluster

 # 计算 GMM 模型参数
    mu, cov, alpha = GMM_EM(matY, K, 100)

 # 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
    N = Y.shape[0]
 # 求当前模型参数下，各模型对样本的响应度矩阵
    gamma = getExpectation(matY, mu, cov, alpha)
 # 对每个样本，求响应度最大的模型下标，作为其类别标识
    category = gamma.argmax(axis=1).flatten().tolist()[0]

 # 将每个样本放入对应类别的列表中
    class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
    class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
    class3 = np.array([Y[i] for i in range(N) if category[i] == 2])
     
    center = torch.tensor(mu, 
                                    device=device, 
                                    dtype=torch.float)
    return center

def get_initial_center(feature, device, n_cluster):
    # fit
    print('\nbegin fit kmeans++ to get initial cluster centroids ...')
    
    
    
   
            
    kmeans = KMeans(n_cluster).fit(feature.numpy())
    center = torch.tensor(kmeans.cluster_centers_, 
                                    device=device, 
                                    dtype=torch.float)
    
    return center

def pretrain(model, opt, ds, device, epochs, save_dir):     
    print('begin train AutoEncoder ...')
    
    loss_fn = nn.MSELoss()
    n_sample = ds.shape[0]
    model.train() 
    loss_h = History('min')
    
    # fine-tune
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        print('-' * 2)
        loss = 0.
        for i, x in enumerate(ds):
            opt.zero_grad()
            x = x.to(device)
            _, gen = model(x)
            batch_loss = loss_fn(x, gen)
            batch_loss.backward()
            opt.step()
            loss += batch_loss 
            print(f'{i}/{n_batch}', end='\r')

        loss /= n_sample
        loss_h.add(loss)
        if loss_h.better:
            torch.save(model, f'{save_dir}/fine_tune_AE.pt')
        print(f'loss : {loss.item():.4f}  min loss : {loss_h.best.item():.4f}')
        print(f'lr: {opt.param_groups[0]["lr"]}')


class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = [] 
        self._check(target)
        
    def add(self, value):
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False
            
        self.value = value
        self.history.append(value.item())
        
    def _check(self, target):
        if target not in {'min', 'max'}:
            raise ValueError('target only allow "max" or "min" !')