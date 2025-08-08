import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from sklearn.cluster import KMeans
import args
from sklearn.preprocessing import MinMaxScaler
from for_train import pretrain,get_initial_center,gmm_initial_center,gmm_update_center
class VGAE(nn.Module):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		# self.gcn_lostddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
       
		self.gcn_lostddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)


	def encode(self, X):
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		# print('hidden',hidden)
		# print("self.mean 111",self.mean )
		# print("sssssssssssss")
		self.std = self.gcn_lostddev(hidden)
		gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
		sampled_z = gaussian_noise*self.std + self.mean

		# print('gaussian_noise',gaussian_noise)
		# print('self.std',self.std)

		
		# print('self.mean',self.mean)
		# print('sampled_z',sampled_z)
		return sampled_z

	def forward(self, X):
		# print('XXXX',X)
		Z = self.encode(X)
		# print('Z',Z)
		A_pred = dot_product_decode(Z)
		return A_pred,Z

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		# self.adj = adj
		self.activation = activation
        
	def forward(self, inputs,adj):
		x = inputs
		# print('x',x)
		# print('self.weight',self.weight)
		x = torch.mm(x,self.weight)
		# print('x',x)
		x = torch.mm(adj, x)
		# print('x11111',x)
		outputs = self.activation(x)
		# print('outputs111',outputs)
		
		return outputs


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	# print('torch.matmul(Z,Z.t())',Z)
	# print('A_pred111',A_pred)
	return A_pred

def glorot_init(input_dim, output_dim):
	# print('input_dim',input_dim)
	# print('output_dim',output_dim)
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	# print('init_range',init_range)
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	# print('initial',initial)
	return nn.Parameter(initial)


class GAE(nn.Module):
	def __init__(self):
		super().__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim)
		
		
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, activation=lambda x:x)

	def encode(self, X,adj):
		hidden = self.base_gcn(X,adj)
		z = self.mean = self.gcn_mean(hidden,adj)
		return z

	def forward(self, X,adj):
		# print('X',X.shape)
		
		Z = self.encode(X,adj)
		A_pred = dot_product_decode(Z)
		
		
		return A_pred,Z
		

# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Encoder(5, 500, True),   #13=784
                                     Encoder(500, 500, True),
                                     Encoder(500, 2000, True),
                                     Encoder(2000, 3, False))
        self.decoder = nn.Sequential(Decoder(3, 2000, True),
                                     Decoder(2000, 500, True),
                                     Decoder(500, 500, True),
                                     Decoder(500, 5, False))
            
    def forward(self, x, latent=False):
        x  = self.encoder(x)
        if latent:
            return x
        # gen = self.decoder(x)
        return x, self.decoder(x)

class SubclusEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
class CLusterEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(SubclusEncoder(9, 500, True),   #13=784
                                     SubclusEncoder(500, 500, True),
                                     nn.Linear(500, 2),
                                     nn.Softmax(dim=1))
    
            
    def forward(self, x):
        x  = self.encoder(x)
        
        return x
# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(SubclusEncoder(3, 500, True),   #13=784
#                                      SubclusEncoder(500, 500, True),
#                                      nn.Linear(500, 2),
#                                      nn.Softmax(dim=1))
    
            
#     def forward(self, x):
#         x  = self.encoder(x)
        
#         return x
class batch_KMeans(object):
    
    def __init__(self, args,numb):
        self.args = args
      
        self.n_cluster = args.n_cluster
        self.clusters = np.zeros((numb))
        self.count = 100 * np.ones((self.n_cluster))  # serve as learning rate

    
    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)
        
        return dis_mat
    
    def init_cluster(self, X, indices=None):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_cluster,n_init=20)

        cluster_id = model.fit_predict(X)
        self.clusters = cluster_id # copy clusters
    


    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] + 
                               eta * X[i])
            self.clusters[cluster_idx] = updated_cluster
    
    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)
        
        return np.argmin(dis_mat, axis=1)

#图结构生成部分
def get_p(q):
    
    f = q.sum(dim=0, keepdim=True)
    nom = q ** 2 / f
    denom = nom.sum(dim=1, keepdim=True)
    return nom / denom
def get_cos_similar_matrix1(v1, v2):
    num=torch.mm(v1, v1.transpose(0,1))        # 向量点乘[3243,3243]
    denom = torch.norm(v1,p=2,dim=1)
    denom=torch.unsqueeze(denom,1)
    denom=torch.mm(denom, denom.transpose(0,1)) # 求模长的乘积
    res = num / denom
    res[torch.isinf(res)] = 0
    return 0.5 + 0.5 * res

def get_guss_similar_matrix(v1,weights):
    weights=weights.sum(0)/weights.shape[0]
    weights=torch.unsqueeze(weights,0)
    
    weights=enhancemet1(weights,4)
    print('weights111',weights)
    
    weights=weights.repeat(v1.shape[0],1)

    basize, dimensio = weights.size()
    weights=torch.ones(basize , 13)
    print('weights11',weights.shape)


    v1 = torch.mul(v1, weights)
    print('v1',v1.shape)
    
    # res=torch.zeros((v1.shape[0],v1.shape[0]))
    
    # weights=weights.repeat(v1.shape[0],1)

    a12 = torch.mul(v1, v1)
    a13=a12.sum(1)

    a14=torch.unsqueeze(a13,1)
    a14=a14.repeat(1,a12.shape[0])

    a15=torch.unsqueeze(a13,0)
    a15=a15.repeat(a12.shape[0],1)

    num=torch.mm(v1, v1.transpose(0,1))  
    res=a14+a15-2*num
    
    print('res',res)
    denom =a12.shape[0]/500
    
    print('denom',denom)
    # res =torch.exp(-res*denom)
    res =torch.exp(-res)
    print('res',res)
    
    # res[np.isneginf(res)] = 0
    return res
def get_cos_similar_matrix(v1,weights):        #[3243,13], [3243,13],[13,3243]
    
    
    
    weights=weights.sum(0)/weights.shape[0]
    weights=torch.unsqueeze(weights,0)
    
    weights=enhancemet1(weights,2)
    print('weights111',weights)
    
    weights=weights.repeat(v1.shape[0],1)


    v1 = torch.mul(v1, weights)
 

    num=torch.mm(v1, v1.transpose(0,1))        # 向量点乘[3243,3243]
    denom = torch.norm(v1,p=2,dim=1)
    denom=torch.unsqueeze(denom,1)
    denom=torch.mm(denom, denom.transpose(0,1)) # 求模长的乘积
    res = num / denom
    res[torch.isinf(res)] = 0
    return 0.5 + 0.5 * res
class Attention(nn.Module):
  

    def __init__(self, dimensions, attention_type='general'):
        super().__init__()

        if attention_type not in ['dot', 'general','add']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=True)
            self.linear_inq = nn.Linear(dimensions, dimensions, bias=True)
   
        if self.attention_type == 'add':
            self.linear_in = nn.Linear(2* dimensions, dimensions//2, bias=True)
            self.v = nn.Parameter(torch.Tensor(dimensions//2, 1))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.uniform_(-stdv, stdv)
        self.sigmas = nn.Parameter(torch.Tensor(dimensions,1))
        self.linear_out = nn.Linear(dimensions, dimensions, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
      
        query=query.transpose(1,2)
        context=context.transpose(1,2)
        # print('query',query)
        # print('context',context)
        batch_size, output_len, dimensions = query.size()
        # print( batch_size, output_len, dimensions)
        query_len = context.size(1)

        
            
        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)  #[3243,12,3]
            context= context.reshape(batch_size * query_len, dimensions)
            # print('query',query)
            query = self.linear_in(query)
            
            query = query.reshape(batch_size, output_len, dimensions)
            context = context.reshape(batch_size, query_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?
        # print('query',query)
        print('query',query,query.shape)
        print('context.transpose(1, 2).contiguous()',context.transpose(1, 2),context.transpose(1, 2).shape)
        
        attention_scores = torch.bmm(query, context.transpose(1, 2))
        # print('attention_scores',attention_scores)
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        
        print('attention_scores',attention_scores)
        attention_weights = self.softmax(attention_scores)
        
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        print('attention_weights',attention_weights)
        # attention_weights=enhancemet(attention_weights,6)
        print('attention_weights',attention_weights)
        mix = torch.bmm(attention_weights, context)
        output = mix.view(batch_size, output_len, dimensions)
        
        # output = self.linear_out(mix).view(batch_size, output_len, dimensions)
        # output = self.tanh(output)

        return output, attention_weights
class Distribution_q(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        
        self.alpha = alpha
        
    def forward(self, x,center):
        # print('x',x,x.shape)
       
        
        # print('torch.pow(x[:, None, :] - self.center, 2)',torch.pow(x[:, None, :] - self.center, 2).shape)
        square_dist = torch.pow(x[:, None, :] - center, 2)

        # print('square_dist',square_dist.shape)
        nom = torch.pow(0.000001+square_dist / self.alpha, -(self.alpha + 1) / 2)
        denom = nom.sum(dim=1, keepdim=True)
        
        return nom / denom

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
def enhancemet1(weight,bg):
    print('weight',weight)
    print('p_weight',weight[0,0:5])
    # q_4=torch.unsqueeze(q_x[:,:,4],2)
    p_weight=torch.cat((weight[0,0:5], weight[0,6:13]), dim=0)
    
    
    p_weight=torch.pow(p_weight,bg)
    totoel=p_weight.sum(0)
    # totoel=torch.unsqueeze(totoel,0)
    new_weight=p_weight/totoel
    print('new_weight',new_weight.shape)
    # batch_size, output_len, dimensions = weights.size()
    new_weights=torch.cat((new_weight[0:5],1*torch.ones(1), new_weight[5:12]), dim=0)#[3243,1,12]
    new_weights=torch.unsqueeze(new_weights,0)

    print('new_weights',new_weights)
    
    return new_weights
def enhancemet(weight,bg):
    enhance_weights=torch.pow(weight,bg)
    totoel=enhance_weights.sum(2)
    totoel=torch.unsqueeze(totoel,1)
    # print('enhance_weights',enhance_weights)
    # print('totoel',totoel)
    
    return enhance_weights/totoel
class SDEC(nn.Module):

    def __init__(self, alpha,h_dim,device,args):
        super(SDEC, self).__init__()
        
        
        self.n_cluster=args.n_cluster
        self.device = device
        self.distribution_q = Distribution_q(alpha)
        self.attn = Attention(h_dim, 'general') 
        # self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_mse=nn.MSELoss(reduction='mean')

        self.beta=args.beta
        self.gaeencoder = GAE().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=args.learning_rate,
                                          weight_decay=args.wd)
        # self.gaeencoder = GAE(adj).to(device)
    # def forward(self, x):
    #     print('center',self.center)
        
    #     q_x= self.distribution_q(x)
    #     p_context=torch.cat((q_x[:,:,0:3], q_x[:,:,5:12]), dim=2)
    #     print('p_context',p_context.shape)
    #     output, weights= self.attn(q_x[:,:,4], p_context)
    #     sdfds
        
    #     return x
    
    def fit(self,x,center):

        q_x= self.distribution_q(x,center)
        
        # print('q_x',q_x[:,:,0:6])
        # print('q_x',q_x[:,:,6:13])
        q_x=get_p(q_x)
        q_4=torch.unsqueeze(q_x[:,:,5],2)
        p_context=torch.cat((q_x[:,:,0:5], q_x[:,:,6:13]), dim=2)
        print('p_context',p_context)
        # print('p_context',p_context,p_context.shape)        #[3243,3,12]
        # print('q_4',q_4,q_4.shape)                    #[3243,3,1]
        
        output, weights= self.attn(q_4, p_context)  
        # print('weights',weights)
        # print('weights',weights.shape)            #[3243,1,12]
        # print('output',output.shape)              #[3243,1,3]
        # print('output',output)
        # print('q_4.transpose(1,2)',q_4.transpose(1,2))
        kl_loss=self.loss_mse(q_4.transpose(1,2), output)
        # kl_loss=F.binary_cross_entropy(q_4.transpose(1,2), output.view(-1))
        # weights=enhancemet(weights,6)
        
        # print('weights',weights)
        
        batch_size, output_len, dimensions = weights.size()
        weights=torch.cat((weights[:,:,0:5],1*torch.ones(batch_size,1,1), weights[:,:,5:12]), dim=2)#[3243,1,12]
        weights=weights.view(batch_size * output_len, 13)

        print('weight1',weights[:,0:6])
        print('weight2',weights[:,6:12])
        scaler = MinMaxScaler()
        
        x = scaler.fit_transform(x)
        x = numpy_to_torch(x)
        # print('x',x.shape)
        # x=x[:,[0,4,5,6]]
        adj_0=get_guss_similar_matrix(x,weights)
        # x_0 = torch.mul(x, weights)
        print('adj_0',adj_0)
        # print('x_0',x_0)
        
        gaerec,latent_X = self.gaeencoder(x,adj_0)
        print('latent_X',latent_X)
        # print('gaerec',gaerec)
        # print('adj_0',adj_0)
        gaerec_loss = self.beta*0.5*F.binary_cross_entropy(gaerec.view(-1), adj_0.view(-1).detach())
        
        # kmeans = KMeans(self.n_cluster).fit(latent_X.detach().numpy())
        
        # new_center = torch.tensor(kmeans.cluster_centers_, 
        #                             device=self.device, 
        #                             dtype=torch.float)
        
        final_kmeans = KMeans(n_clusters=self.n_cluster, n_init=20)
        cluster_id = final_kmeans.fit_predict(latent_X.detach().numpy())
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # new_center=gmm_update_center(latent_X.detach(), device, 3)
        loss = kl_loss+gaerec_loss
        print('kl_loss',kl_loss)
        print('gaerec_loss',gaerec_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return  cluster_id,kl_loss,weights
####################################################




class DCN(nn.Module):
    
    def __init__(self, args,device,adj):
        super(DCN, self).__init__()
        self.args = args
        self.beta = args.beta  # coefficient of the clustering term 
        self.lamda = args.lamda  # coefficient of the reconstruction term
        self.gama = args.gama
        self.deta = args.deta
        self.device = device
        self.nada=args.nada
        # self.clustering = batch_KMeans(args,adj.shape[0])
        self.clustering = CLusterEncoder().to(device)

        self.aeencoder = AutoEncoder().to(device)
		# self.aeencoder= AutoEncoder().to(device)
        self.gaeencoder = GAE(adj).to(device)
        self.mse_loss=nn.MSELoss()
        self.criterion  = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=args.learning_rate,
                                          weight_decay=args.wd)
    
    """ Compute the Equation (5) in the original paper on a data batch """

        
    def _loss(self, Sx,features,adj):
        
        latent_Sx, aerec = self.aeencoder(Sx)
        gaerec,latent_X = self.gaeencoder(features)

      
        # _,Latent_gaeX=self.gaeencoder(features)
        # print('Latent_gaeX',Latent_gaeX.shape)

        latent_cat=torch.cat((latent_Sx,latent_X),1)
    
        
    
        cluster_id = self.clustering(latent_cat)
        print('cluster_id',cluster_id)
        # latent_X = self.autoencoder(X, latent=True)

        
        # Reconstruction error

        aerec_loss = self.lamda * self.mse_loss(Sx, aerec)
        gaerec_loss = self.beta*0.5*F.binary_cross_entropy(gaerec.view(-1), adj.view(-1))
        
        
        dist_loss = torch.tensor(0.).to(self.device)
        distSx_loss = torch.tensor(0.).to(self.device)
        clst_loss= torch.tensor(0.).to(self.device)
        # clusters = torch.FloatTensor(self.clustering.clusters).to(self.device)
        clusters=cluster_id
        n_list=torch.zeros(self.args.n_cluster)
        # clust_center=torch.zeros(self.args.n_cluster,latent_X.shape[1])
        clust_center=torch.mm(torch.transpose(clusters,0,1),latent_X)
        S_clust_center=torch.mm(torch.transpose(clusters,0,1),latent_Sx)
        # for clusterid in range(self.args.n_cluster):
        #     for ks,id_latex in enumerate(clusters):
        #         if id_latex==clusterid:
        #             n_list[clusterid]+=1
        #             clust_center[clusterid]+=latent_X[ks]
        # for dmesion in range(self.args.n_cluster):
        #     clust_center[dmesion]=clust_center[dmesion]/n_list[dmesion]
        
        # clust_center=torch.div(clust_center,n_list)
        pcluster=torch.sum(clusters, dim=0)/latent_Sx.shape[0]
        print('pcluster',pcluster)
        print('f',torch.log(pcluster+1e-10))
        clst_loss=self.nada*torch.sum(torch.mul(pcluster,torch.log(pcluster+1e-10)))
        print('clst_loss',clst_loss)
        for i in range(latent_Sx.shape[0]):
            sample_distSx_loss=torch.tensor(0.).to(self.device)
            for j in range(self.args.n_cluster):
                diff_vec = latent_Sx[i] - S_clust_center[j]
                
                sample_distSx_loss+= cluster_id[i][j]*torch.matmul(diff_vec.view(1, -1),diff_vec.view(-1, 1))[0][0]

            # vecik=clusters[cluster_id[i]]
            distSx_loss += 0.5 * self.deta * torch.squeeze(sample_distSx_loss)#/clus_count[cluster_id[i]]
        distSx_loss=distSx_loss/latent_Sx.shape[0]
        for i in range(latent_X.shape[0]):
            sample_dist_loss=torch.tensor(0.).to(self.device)
            for j in range(self.args.n_cluster):
                diff_vec = latent_X[i] - clust_center[j]
                
                sample_dist_loss+= cluster_id[i][j]*torch.matmul(diff_vec.view(1, -1),diff_vec.view(-1, 1))[0][0]

            # vecik=clusters[cluster_id[i]]
            dist_loss += 0.5 * self.gama * torch.squeeze(sample_dist_loss)#/clus_count[cluster_id[i]]
        dist_loss=dist_loss/latent_X.shape[0]
        print('aerec_loss',aerec_loss)
        print('dist_loss',dist_loss)
        print('gaerec_loss',gaerec_loss)
        print('distSx_loss',distSx_loss)
        return aerec_loss + dist_loss+gaerec_loss+distSx_loss+clst_loss
    
    def aepretrain(self, Sx, device,epoch=50, verbose=True):
        
        
        
        print('========== Start pretraining ==========')
        
        ae_loss_list  =[]
        
        self.train()
        for e in range(epoch):
            self.optimizer.zero_grad()
            Sx = Sx.to(device)
            _,gen = self.aeencoder(Sx)
            ae_loss = self.mse_loss(Sx,gen)
            ae_loss_list.append(ae_loss.detach().cpu().numpy())
            ae_loss.backward()
            self.optimizer.step()
        
        
        if verbose:
            print('========== End pretraining ==========\n')
        
        # Initialize clusters in self.clustering after pre-training
        
        return ae_loss_list
	
    def gaepretrain(self,features,adj,device,epoch=50, verbose=True):
        
        
        
        print('========== Start pretraining ==========')
        
        gae_loss_list  =[]
        
        self.train()
        for e in range(epoch):

            # for batch_idx, data in enumerate(Sx):
            self.optimizer.zero_grad()
            features = features.to(device)
            A_pred,Z= self.gaeencoder(features)
            gae_loss = 0.5*F.binary_cross_entropy(A_pred.view(-1), adj.view(-1))
            # loss = self.criterion(data, rec_X)
            gae_loss_list.append(gae_loss.detach().cpu().numpy())
            gae_loss.backward()
            self.optimizer.step()
        
        
        if verbose:
            print('========== End pretraining ==========\n')
        
        # Initialize clusters in self.clustering after pre-training
        
        return gae_loss_list
    
    
    def fit(self, epoch, Sx,features,adj, verbose=True):

        #gai
        # total_data=torch.tensor([])
      
       
        # for batch_idx, (data, _) in enumerate(train_loader):
        
        
            # Get the latent features
        # with torch.no_grad():
        
        
        # latent_L = self.aeencoder(Sx, latent=True)
        # _,Latent_gaeX=self.gaeencoder(features)
        # print('Latent_gaeX',Latent_gaeX.shape)
        # # latent_X = latent_X.cpu().numpy()
        # # latent_cat=torch.cat((latent_L,Latent_gaeX),1)
        #     # [Step-1] Update the assignment results
        
    
        # cluster_id = self.clustering(latent_cat)
        # print('cluster_id',cluster_id)
        # print('self.clustering.clusters',self.clustering.clusters)
            # print("cluster_id",cluster_id)
            # [Step-2] Update clusters in batch Clustering
        
     
            # print('cluster_id',cluster_id)
            
            # [Step-3] Update the network parameters         
        loss = self._loss(Sx,features,adj)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        