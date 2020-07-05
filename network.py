
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler
from os import makedirs
from os.path import exists, join, isfile, dirname, abspath
from sklearn.cluster import KMeans

import helper_torch_util
from laplacian import Laplacian


class LapCluster(nn.Module):
    def __init__(self, cfg):
        
        super(LapCluster,self).__init__()
        self.cfg            = cfg
        self.compute_lap    = Laplacian(cot=True)

        f_conv2d  = helper_torch_util.conv2d(True, 1, 256, kernel_size=[1,28])
        setattr(self, 'mlp_in', f_conv2d)

        for i in range(self.cfg.num_blocks):
            f_conv2d  = helper_torch_util.conv2d(True, 256, 64)
            setattr(self, 'mlp_'+str(i)+'_1', f_conv2d)
            
            f_conv2d  = helper_torch_util.conv2d(True, 64, 128)
            setattr(self, 'mlp_'+str(i)+'_2', f_conv2d)
            
            f_conv2d  = helper_torch_util.conv2d(True, 128, 128)
            setattr(self, 'mlp_'+str(i)+'_3', f_conv2d)


        f_conv2d = helper_torch_util.conv2d(True, 256, 128)
        setattr(self, 'mlp_out1', f_conv2d)

        f_conv2d = helper_torch_util.conv2d(True, 128, 128)
        setattr(self, 'mlp_out2', f_conv2d)

        f_dense = nn.Linear(128, 256)
        setattr(self, 'dense_1', f_dense)
        self.dense_bn1 = nn.BatchNorm1d(256, eps=1e-6, momentum=0.99)

        f_dense = nn.Linear(256, 256)
        setattr(self, 'dense_2', f_dense)
        self.dense_bn2 = nn.BatchNorm1d(256, eps=1e-6, momentum=0.99)

        f_dense = nn.Linear(256, self.cfg.cat_num)
        setattr(self, 'dense_3', f_dense)

    def compute_cluster(self, feature, num_cluster):
        n_blocks = len(num_cluster)
        cluster_idx = []
        for k in num_cluster:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(feature)
            idx = kmeans.labels_
            cluster_idx.append(np.expand_dims(idx, axis=0))

        return np.concatenate(cluster_idx, axis=0)

    def pooling_block(self, feature, group_idx, num_cluster):
        # feature   B x d x nv x 1
        # group_idx B x 
        B, d, N, k = feature.size()
        _inf            = -100
        padding_inf     = _inf * torch.ones([B, d, N, k])
        cluster_feature = []

        m_pool2d    = nn.MaxPool2d([N, 1])

        for i in range(num_cluster):
            cluster = torch.eq(group_idx, i).unsqueeze(-1).unsqueeze(1)
            cluster = cluster.expand(B, d, N, 1)
            cluster_pooling = torch.where(cluster, feature, padding_inf)
            cluster_pooling = m_pool2d(cluster_pooling)

            cluster_feature.append(cluster_pooling.squeeze(-1).squeeze(-1))
            
        cluster_feature = torch.cat([x.unsqueeze(-1) for x in cluster_feature], dim=-1)

        pooling_sum = torch.zeros([B, d, N, 1])

        for i in range(num_cluster):
            net = cluster_feature[:,:,i].unsqueeze(-1).unsqueeze(-1)
            net = net.expand(B, d, N, 1)
            cluster = torch.eq(group_idx, i).unsqueeze(-1).unsqueeze(1)
            cluster = cluster.expand(B, d, N, 1)
            net = torch.where(cluster, net, 0*net)
            pooling_sum = pooling_sum + net
        return pooling_sum

    def forward(self, inputs):
        # B x nv x 3
        verts       = inputs['verts']
        # B x nf x 3
        faces       = inputs['faces']

        batch_size  = verts.size()[0]

        clusters    = []
        eigs        = []

        for i in range(batch_size):
            #features    = vertices
            face        = faces[i]
            vert        = verts[i]

            lap_matrix  = self.compute_lap(face, vert)

            eig_val, eig_vec = torch.symeig(lap_matrix, eigenvectors=True)
            eig_vec = eig_vec.narrow(1, 1, self.cfg.num_eig)
            feature = torch.cat([vert, eig_vec], 1)
            eigs.append(feature.unsqueeze(0))

            cluster_idx = self.compute_cluster(eig_vec, self.cfg.num_cluster)

            clusters.append(np.expand_dims(cluster_idx, axis=1))

        # layer x B x nv
        clusters    = torch.from_numpy(np.concatenate(clusters, axis=1))
   
        # B x nv x d
        features    = torch.cat(eigs, 0).unsqueeze(1)
    
        K           = features.size()[-1]

        m_conv2d    = getattr(self, 'mlp_in')
        features    = m_conv2d(features)

        for i in range(self.cfg.num_blocks):
            m_conv2d    = getattr(self, 'mlp_'+str(i)+'_1')
            features    = m_conv2d(features)

            m_conv2d    = getattr(self, 'mlp_'+str(i)+'_2')
            features    = m_conv2d(features)

            out_pool    = self.pooling_block(features, clusters[i], self.cfg.num_cluster[i])

            m_conv2d    = getattr(self, 'mlp_'+str(i)+'_3')
            features    = m_conv2d(features)

            features    = torch.cat([features, out_pool], 1)

        m_conv2d    = getattr(self, 'mlp_out1')
        out1        = m_conv2d(features)

        m_conv2d    = getattr(self, 'mlp_out2')
        out2        = m_conv2d(out1)

        nv          = features.size()[2]
        m_pool2d    = nn.MaxPool2d([nv, 1])
        out_max     = m_pool2d(out2)

        net         = torch.reshape(out_max, (batch_size, -1))


        m_dense     = getattr(self, 'dense_1')
        m_leakyrelu = nn.LeakyReLU(0.2)
        m_dropout   = nn.Dropout(0.7)
        #net         = m_dropout(m_leakyrelu(self.dense_bn1(m_dense(net).unsqueeze(1)))).squeeze(1)
        net         = m_dropout(m_leakyrelu(m_dense(net)))

        m_dense     = getattr(self, 'dense_2')
        m_leakyrelu = nn.LeakyReLU(0.2)
        m_dropout   = nn.Dropout(0.7)
        #net         = m_dropout(m_leakyrelu(self.dense_bn1(m_dense(net).unsqueeze(1)))).squeeze(1)
        net         = m_dropout(m_leakyrelu(m_dense(net)))

        m_dense     = getattr(self, 'dense_3')
        net         = m_dense(net)

        return out_max
