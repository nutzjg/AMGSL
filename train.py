import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
from sklearn.cluster import KMeans
from input_data import load_data
from preprocessing import *
import args
import model
from model import SDEC
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from munkres import Munkres
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import torch.nn as nn
from for_train import pretrain,get_initial_center
from scipy import stats
import seaborn as sns
from torch.utils.data import Subset,DataLoader
def load_graph_data(dataset_name, show_details=False):
    
    load_path = "../vgae_pytorch-master/data/" + dataset_name 
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_ALTlabel.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    print('feat',feat.shape)
    print('label',label.shape)
    print('adj',adj.shape)


    # X pre-processing
    # nn_input=100
    # pca = PCA(n_components=nn_input)
    # feat = pca.fit_transform(feat)
    return feat, label, adj

def load_Sx_data(dataset_name, show_details=False):

    load_path = "data/" + dataset_name 
    Sx = np.load(load_path+"_ALTsub110_Sx.npy", allow_pickle=True)


    # X pre-processing
    # nn_input=100
    # pca = PCA(n_components=nn_input)
    # feat = pca.fit_transform(feat)
    return Sx

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""








# init model and optimizer
# print('adj_norm',adj_norm)

def plot(Z,y,cluster_id,aemodel):
   
    print('plotting ...')
    
    
    
    feature = Z
    true_label = y
    pred =cluster_id
    print('pred',pred)
    print('true_label',true_label)
    # yyy1=true_label.cpu().numpy()+pred
    # count1=0
    
    # for i in range(len(yyy1)):
    #     if yyy1[i]==2:
    #         count1=count1+1
    save_dir='visual'
    epoch='2000'
    # print('count1',count1)
    feature_2D = TSNE(2).fit_transform(feature)
    plt.scatter(feature_2D[:, 0], feature_2D[:, 1], 16, pred, cmap='Paired')
    
    plt.title('Epoch:')
    plt.savefig(f'{save_dir}/epoch_{aemodel}.png')
    plt.close()

    plt.scatter(feature_2D[:, 0], feature_2D[:, 1], 16, true_label, cmap='Paired')
    
    plt.title(f'Epoch: {epoch}')
    plt.savefig(f'{save_dir}/epoch_true{aemodel}.png')
    plt.close()
# def clustering(Z, y):
   
#     model = KMeans(n_clusters=2, n_init=20)
#     # print('Z',Z)
#     # print('Znump',Z.data.cpu().numpy().shape)
#     cluster_id = model.fit_predict(Z)
#     # print('cluster_id',cluster_id)
#     # print('y',y)
#     plot(Z,y,cluster_id)
#     acc, nmi, ari, f1 = eva(y, cluster_id, show_details=True)
#     return acc, nmi, ari, f1

def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    true_label=y_true.sum()
    false_label=len(y_true)-y_true.sum()
    true_rate=true_label/len(y_true)
    true_predlabel=y_pred.sum()
    false_predlabel=len(y_pred)-y_pred.sum()
    truepred_rate=true_predlabel/len(y_pred)
    yyy=y_true+y_pred
    count=0
    for i in range(len(yyy)):
        if yyy[i]==2:
            count=count+1
    print('count',count)
    print('y_true',y_true)
    print('y_pred',y_pred)
    print('true_label',true_label)
    print('true_predlabel',true_predlabel)
    print('true_rate',true_rate)
    print('truepred_rate',truepred_rate)

    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1

def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro

def get_scores(edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy
def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a
# train model
# print('adj_label',adj_label.shape)
def normalize_adj(adj, self_loop=True, symmetry=False):
  
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj

def distributionfuc(self,maxdistanc, latent_X,total_X,latent_total_X):
    list_count=[]
    for idm, data in enumerate(total_X):

        distanc_vec=latent_X-latent_total_X[idm]
        sam_distances = torch.matmul(distanc_vec.view(1, -1),distanc_vec.view(-1, 1))
        if sam_distances<=maxdistanc:
            list_count.append(idm)
                
    maxintradistance=0
    for ids in list_count:
        for idp in list_count:
            distanc1=latent_total_X[ids]-latent_total_X[idp]
            sam_distanc1 = torch.matmul(distanc1.view(1, -1),distanc1.view(-1, 1))
            if sam_distanc1>maxintradistance:
                maxintradistance=sam_distanc1
    
    distribution_i=maxintradistance/(len(list_count)+1)

    return distribution_i

def distributionfuction(X):
    
    # c=torch.Tensor([])
    # a = torch.randn((2, 3))
    # b = torch.randn((2, 3))
    # print('a',a)
    # print('a',a.numpy())
    # print('b',b)
    # d=torch.cat((a,b),1)
    # print('d',d)
    
    

    X=torch.from_numpy(X)
    print('X',X.shape[1])
    
    min_distance=torch.ones(X.shape[1])*100000
    max_distance=X[0]-X[0]
    zero=torch.zeros(X.shape[1])
    print(zero)
    for ids,data in enumerate(X):
        min_distanidp=torch.ones(X.shape[1])*100000
        for idp,data in enumerate(X):
            
            distanc1=X[ids]-X[idp]
  
            sam_distanc1 = torch.mul(distanc1,distanc1)
            for i,datax in  enumerate(sam_distanc1):
                if sam_distanc1[i]<min_distanidp[i] and sam_distanc1[i]>zero[i]:
                    min_distanidp[i]=sam_distanc1[i]

        for i,datax in  enumerate(min_distanidp):
                if min_distanidp[i]>max_distance[i]:
                    max_distance[i]=min_distanidp[i]
                if min_distanidp[i]<min_distance[i] and min_distanidp[i]>zero[i]:
                    min_distance[i]=min_distanidp[i]
    a=torch.div(max_distance,min_distance)
    # print('max_diatance',max_distance)
    # print('min_distance',min_distance)
    # print('a',a)
    
    sub_Sx=torch.Tensor([])
    for pp in range(1):
        pp=5
        Sx=torch.Tensor([])
        for ids,data in enumerate(X):
            list_count=torch.zeros(1,X.shape[1])
            for idp,data in enumerate(X):
                
                distanc1=X[ids]-X[idp]
                # print('distanc1',distanc1)
                sam_distanc1 = torch.mul(distanc1,distanc1)
            
                # print('sam_distanc1',sam_distanc1)
                for i,datax in  enumerate(sam_distanc1):
                    if sam_distanc1[i]<min_distance[i]*(pp+1.1)*(pp+1.1):
                        list_count[0,i]+=1
            # print('list_count',list_count)
            Sx=torch.cat((Sx,list_count),0)
        print('Sx',Sx.shape)
        sub_Sx=torch.cat((sub_Sx,Sx),1)

    np.save('data/certong_2_ASTsub5_Sx.npy',sub_Sx.numpy())
    return sub_Sx       
                
                
                # print('sam_distanc1[i]',sam_distanc1[i])
                # print('min_distance[i]',min_distance[i])

                # if sam_distanc1[i]<min_distance[i] and sam_distanc1[i]>zero[i]:
                #     min_distance[i]=sam_distanc1[i]    
    
def evaluate(model, Sx,features,adj,y):
    y_test = []
    y_pred = []
    latent_xx=[]
    ys_pred = []
    latent_gaex=[]
    
    latent_L = model.aeencoder(Sx, latent=True)
    latent_L = latent_L.detach().cpu().numpy()
    cluster_id = model.clustering.clusters
    

   
    latent_xx=latent_L
   

    _,latent_gX = model.gaeencoder(features)
    latent_gX = latent_gX.detach().cpu().numpy()
    model = KMeans(n_clusters=2, n_init=20)
    
    cluster_idg = model.fit_predict(latent_gX)
    y_pred=cluster_idg
    
    
    latent_gaex=latent_gX
        
  
    
    y_test=y
    ys_pred=cluster_id
    print('y_test',y_test)
    print('y_pred',y_pred)
    print('ys_pred',ys_pred)
    print('latent_gaex',latent_gaex)
    print('latent_xx',latent_xx)
    
    
    acc, f1 = cluster_acc(y_test, y_pred)
    accae, f1ae = cluster_acc(y_test, ys_pred)
    print('accae',accae,'nmi',normalized_mutual_info_score(y_test, ys_pred),'ari',adjusted_rand_score(y_test, ys_pred),'f1ae',f1ae)
    # plot(latent_xx,y_test,ys_pred,'ae')
    # plot(latent_gaex,y_test,y_pred,'gae')
    return (normalized_mutual_info_score(y_test, y_pred),
            adjusted_rand_score(y_test, y_pred),acc,f1)
def plotevaluate(model, Sx,features,adj,y):
    y_test = []
    y_pred = []
    latent_xx=[]
    ys_pred = []
    latent_gaex=[]
    
    latent_L = model.aeencoder(Sx, latent=True)
    _,Latent_gaeX=model.gaeencoder(features)
        # latent_X = latent_X.cpu().numpy()
    latent_cat=torch.cat((latent_L,Latent_gaeX),1)


    cluster_id = model.clustering(latent_cat)
    for u,sx_labels in enumerate(cluster_id):
        if sx_labels[0]>sx_labels[1]:
            ys_pred.append(0)
        else:
            ys_pred.append(1)
    latent_L = latent_cat.detach().cpu().numpy()
    
    

   
    latent_xx=latent_L
   

    _,latent_gX = model.gaeencoder(features)
    latent_gX = latent_gX.detach().cpu().numpy()
    model = KMeans(n_clusters=2, n_init=20)
    
    cluster_idg = model.fit_predict(latent_xx)
    y_pred=cluster_idg
    
    
    latent_gaex=latent_gX
        
  
    
    y_test=y
    # ys_pred=cluster_id
    print('y_test',y_test)
    print('y_pred',y_pred)
    print('ys_pred',ys_pred)
    print('latent_gaex',latent_gaex)
    print('latent_xx',latent_xx)
    
    
    acc, f1 = cluster_acc(y_test, y_pred)
    accae, f1ae = cluster_acc(y_test, ys_pred)
    print('accae',accae,'nmi',normalized_mutual_info_score(y_test, ys_pred),'ari',adjusted_rand_score(y_test, ys_pred),'f1ae',f1ae)
    plot(latent_xx,y_test,ys_pred,'ae')
    plot(latent_xx,y_test,y_pred,'aegae')
    plot(latent_gaex,y_test,y_pred,'gae')
    return (normalized_mutual_info_score(y_test, y_pred),
            adjusted_rand_score(y_test, y_pred),acc,f1)   

def draw_distribution_histogram(nums, path, is_hist=True, is_kde=True, is_rug=False, \
  is_vertical=False, is_norm_hist=False):
 
  sns.set()  #切换到sns的默认运行配�?
  sns.distplot(nums, bins=20, hist=is_hist, kde=is_kde, rug=is_rug, \
    hist_kws={"color":"steelblue"}, kde_kws={"color":"purple"}, \
    vertical=is_vertical, norm_hist=is_norm_hist)
  #添加x轴和y轴标�?
  plt.xlabel("ALT(U/L)")
  plt.ylabel("Density")

  #添加标题
  plt.title("Distribution")
  plt.tight_layout()  # 处理显示不完整的问题
  plt.savefig(path, dpi=300)
  plt.close()
def update_center(cluster_id,features):
    
        
   
    class1=[]
    class2=[]
    class3=[]
    for i in range(len(y)):
        if cluster_id[i]==0:
            class1.append(features[i].numpy())
        elif cluster_id[i]==1:
            class2.append(features[i].numpy())
        elif cluster_id[i]==2:
            class3.append(features[i].numpy())
 
  
    class1 = numpy_to_torch(np.array(class1))
    class2 = numpy_to_torch(np.array(class2))
    class3 = numpy_to_torch(np.array(class3))
    center1=class1.sum(0)/class1.shape[0]
    center2=class2.sum(0)/class2.shape[0]
    center3=class3.sum(0)/class3.shape[0]
    center1=torch.unsqueeze(center1,0)
    center2=torch.unsqueeze(center2,0)
    center3=torch.unsqueeze(center3,0)
    new_center=torch.cat((center1,center2,center3),0)
    

    # print('up_center',up_center.shape)
    return new_center
def clustering(cluster_id,weights):
   
    #np.save('ASTlclass3weights500_SDEC_13_2.npy',weights.detach().numpy())
    
    features, y, adj = load_graph_data('lertong_2', show_details=False)
    class1=[]
    class2=[]
    class3=[]
    for i in range(len(y)):
        if cluster_id[i]==0:
            class1.append(features[i,5])
        elif cluster_id[i]==1:
            class2.append(features[i,5])
        elif cluster_id[i]==2:
            class3.append(features[i,5])
    
    print('class1',len(class1))
    print('class2',len(class2))
    print('class3',len(class3))



    



    class1_skew=stats.skew(class1)
    class2_skew=stats.skew(class2)
    class3_skew=stats.skew(class3)
    class1_kurtosis=stats.kurtosis(class1)
    class2_kurtosis=stats.kurtosis(class2)
    class3_kurtosis=stats.kurtosis(class3)

    class1_mean=np.mean(class1)
    class1_std=np.std(class1)
    class2_mean=np.mean(class2)
    class2_std=np.std(class2)
    class3_mean=np.mean(class3)
    class3_std=np.std(class3)

    class1_sort=class1



    class1_sort.sort()
    class1_lengh=len(class1_sort)

    class2_sort=class2

    class2_sort.sort()
    class2_lengh=len(class2_sort)
    class3_sort=class3

    class3_sort.sort()
    class3_lengh=len(class3_sort)

    #path = "ASTl500class3_SDEC_13_2.png"
    #if class1_lengh>class2_lengh and class1_lengh>class3_lengh:
    #    print('1')
    #    draw_distribution_histogram(class1, path, True, True)
    #    np.save('ASTl500class3_SDEC_13_2.npy',class1)
    #elif class2_lengh>class1_lengh and class2_lengh>class3_lengh:
    #    print('2')
    #    draw_distribution_histogram(class2, path, True, True)
    #    np.save('ASTl500class3_SDEC_13_2.npy',class2)
    #elif class3_lengh>class1_lengh and class3_lengh>class2_lengh:
    #    print('3')
    #    draw_distribution_histogram(class3, path, True, True)
    #    np.save('ASTl500class3_SDEC_13_2.npy',class3)
    # else:
    #     print('2')
    #     draw_distribution_histogram(class3, path, True, True)
    #     np.save('class2_gae_13_2.npy',class3)


    # print('class1',class1_sort)
    # print('class2',class2_sort)
    # print('class3',class3_sort)

    print('class1_low',class1_sort[0])

    print('class1_2.5',class1_sort[round(class1_lengh/100*2.5)-1])
    print('class1_50',class1_sort[round(class1_lengh/2)-1])
    print('class1_97.5',class1_sort[round(class1_lengh/100*97.5)-1])
    print('class1_lengh',class1_lengh)
    print('class1_up',class1_sort[class1_lengh-1])
    print('class1_skew',class1_skew)
    print('class1_kurtosis',class1_kurtosis)
    print('class1_mean',class1_mean)
    print('class1_std',class1_std)



    print('class2_low',class2_sort[0])

    print('class2_2.5',class2_sort[round(class2_lengh/100*2.5)-1])
    print('class2_50',class2_sort[round(class2_lengh/2)-1])
    print('class2_97.5',class2_sort[round(class2_lengh/100*97.5)-1])
    print('class2_lengh',class2_lengh)
    print('class2_up',class2_sort[class2_lengh-1])
    print('class2_skew',class2_skew)
    print('class2_kurtosis',class2_kurtosis)
    print('class2_mean',class2_mean)
    print('class2_std',class2_std)

    print('class3_low',class3_sort[0])

    print('class3_2.5',class3_sort[round(class3_lengh/100*2.5)-1])
    print('class3_50',class3_sort[round(class3_lengh/2)-1])
    print('class3_97.5',class3_sort[round(class3_lengh/100*97.5)-1])
    print('class3_lengh',class3_lengh)
    print('class3_up',class3_sort[class3_lengh-1])
    print('class3_skew',class3_skew)
    print('class3_kurtosis',class3_kurtosis)
    print('class3_mean',class3_mean)
    print('class3_std',class3_std)
def solver(initial_center,args, model,features,device):
    
    # ae_rec_loss = model.aepretrain(Sx,device, epoch=args.aepre_epoch)
    # gae_rec_loss=model.gaepretrain(features,adj, device,epoch=args.gaepre_epoch)
    nmi_list = []
    ari_list = []
    acc_list = []
    f1_list = []
    kl_loss=1
    for e in range(args.epoch):
        if kl_loss>0.1:
            model.train()
            cluster_id,kl_loss,weights=model.fit(features,initial_center)
            new_center=update_center(cluster_id,features)
        # print('initial_center',initial_center)
        # print('new_center',new_center)
        
        
        
            initial_center=new_center
            model.eval()
        
            clustering(cluster_id,weights)
        
    

   
# c=torch.Tensor([])
# a = torch.randn((2, 3))
# b = torch.randn((2, 3))
# d=torch.mul(a,b)
# print('a',a)
# print('b',b)
# print('d',d)
# dfd
# devices=torch.device("cuda")



# def load_g_data(dataset_name, show_details=False):

#     load_path = "../vgae_pytorch-master/data/" + dataset_name 
#     feat = np.load(load_path+"_feat.npy", allow_pickle=True)
#     label = np.load(load_path+"_ALTlabel.npy", allow_pickle=True)
#     adj = np.load(load_path+"_adj.npy", allow_pickle=True)
#     print('feat',feat.shape)
#     print('label',label.shape)
#     print('adj',adj.shape)



#     return feat,label


# def load_ertong():


#     # Sx=load_Sx_data('ertong_1', show_details=False)
#     # print('Sx',Sx.shape)
  
#     # Sx=Sx[:,[4,17,30,43]]
#     # print('Sx1',Sx.shape)
    
#     features,y= load_g_data('ertong_3_f', show_details=False)
#     # scaler = MinMaxScaler()
#     # features = scaler.fit_transform(features)
#     x = features.reshape((features.shape[0], -1)).astype(np.float32)
#     # x = Sx.reshape((Sx.shape[0], -1)).astype(np.float32)
    
   
    
    
#     return x, y

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   
   torch.backends.cudnn.deterministic = True

# 设置随机数种�?
setup_seed(20)

# class ErtongDataset():

#     def __init__(self):
#         self.x, self.y = load_ertong()

#     def __len__(self):
#         return self.x.shape[0]

#     def __getitem__(self, idx):
#         return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(self.y[idx]))

# dataset = ErtongDataset()
# train_loader = DataLoader(
#         dataset, batch_size=args.batch_size, shuffle=True)
# for batch_idx, (data, t) in enumerate(train_loader):
#     print('batch_idx',batch_idx)
#     print('data',data.shape)
#     print('t',t.shape)


features, y, adj = load_graph_data('lertong_2', show_details=False)
#preprocessing
print('features',features)
print('features',features.shape)
print('adj',adj.shape)
# scaler = MinMaxScaler()

# features = scaler.fit_transform(features)
# print('features',features)
# Sx=distributionfuction(features[:,[4]])

# Sx=load_Sx_data('ertong_0', show_details=False)

# print('Sx',Sx.shape)
Sx=features
# Sx=Sx[:,[4,17,30,43]]

features = numpy_to_torch(features)
# adj = numpy_to_torch(adj)
Sx = numpy_to_torch(Sx)


n_cluster=args.n_cluster
# FSx=torch.cat((features,Sx),1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

initial_center=get_initial_center(features, device, n_cluster)
print('initial_center',initial_center)
alpha=1
sdec = SDEC(alpha,n_cluster,device,args).to(device) 

solver(initial_center,args,sdec,features,device)
# model = DCN(args,device,adj).to(device)  
# ae_rec_loss, gae_rec_loss = solver(args, model,features ,adj, Sx,y,device)





 

# scaler1 = MinMaxScaler()
# Z = scaler1.fit_transform(Z.detach().numpy())
# acc, nmi, ari, f1= clustering(Z, y)

# test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("End of training!")