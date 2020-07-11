import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from sklearn.cluster import KMeans

import helper_torch_util
from laplacian import Laplacian
from normal import compute_normal 


class LapCluster(nn.Module):
    def __init__(self, cfg):
        
        super(LapCluster,self).__init__()
        self.cfg            = cfg
        self.compute_lap    = Laplacian(cot=True)

        f_conv2d  = helper_torch_util.conv2d(True, 1, 256, kernel_size=[1,22])
        setattr(self, 'mlp_in', f_conv2d)

        for i in range(self.cfg.num_blocks):
            f_conv2d  = helper_torch_util.conv2d(True, 256, 64)
            setattr(self, 'mlp_'+str(i)+'_1', f_conv2d)
            
            f_conv2d  = helper_torch_util.conv2d(True, 64, 128)
            setattr(self, 'mlp_'+str(i)+'_2', f_conv2d)
            
            f_conv2d  = helper_torch_util.conv2d(True, 128, 128)
            setattr(self, 'mlp_'+str(i)+'_3', f_conv2d)


            f_conv2d  = helper_torch_util.conv2d(True, 128, 128)
            setattr(self, 'mlp_corr_'+str(i)+'_1', f_conv2d)
            f_conv2d  = helper_torch_util.conv2d(True, 128, 256)
            setattr(self, 'mlp_corr_'+str(i)+'_2', f_conv2d)
            f_conv2d  = helper_torch_util.conv2d(True, 256, 32)
            setattr(self, 'mlp_corr_'+str(i)+'_3', f_conv2d)


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

    def pooling_block(self, feature, group_idx, num_block):
        # feature   B x d x nv x 1
        # group_idx B x 
        B, d, N, k  = feature.size()
        num_cluster = self.cfg.num_cluster[num_block]
        _inf            = -100
        padding_inf     = _inf * torch.ones([B, d, N, k], device=self.device)
        cluster_feature = []
        cluster_corr    = []

        m_pool2d    = nn.MaxPool2d([N, 1])
        m_pool2d_c  = nn.MaxPool2d([N, 1])


        m_conv2d    = getattr(self, 'mlp_corr_'+str(num_block)+'_1')
        corr_feat   = m_conv2d(feature)
        m_conv2d    = getattr(self, 'mlp_corr_'+str(num_block)+'_2')
        corr_feat   = m_conv2d(corr_feat)
        m_conv2d    = getattr(self, 'mlp_corr_'+str(num_block)+'_3')
        corr_feat   = m_conv2d(corr_feat)
        pad_corr    = _inf * torch.ones(corr_feat.size(), device=self.device)
        d2 = corr_feat.size()[1]

        for i in range(num_cluster):
            cluster_o       = torch.eq(group_idx, i).unsqueeze(-1).unsqueeze(1)
            cluster         = cluster_o.expand(B, d, N, 1)
            cluster_pooling = torch.where(cluster, feature, padding_inf)
            cluster_pooling = m_pool2d(cluster_pooling)
            cluster_feature.append(cluster_pooling.squeeze(-1).squeeze(-1))

            cluster         = cluster_o.expand(B, d2, N, 1)
            corr_pooling    = torch.where(cluster, corr_feat, pad_corr)
            corr_pooling    = m_pool2d_c(corr_pooling)
            cluster_corr.append(corr_pooling.squeeze(-1).squeeze(-1))  

        cluster_feature = torch.cat([x.unsqueeze(-1) 
                                for x in cluster_feature], dim=-1)
        cluster_corr    = torch.cat([x.unsqueeze(-1) 
                                for x in cluster_corr], dim=-1)
        corr_matrix     = cluster_corr.transpose(-2,-1)
        corr_matrix     = F.normalize(corr_matrix, dim=-1, p=2)
        corr_matrix     = corr_matrix.matmul(corr_matrix.transpose(-2,-1))
        
        cluster_feature = cluster_feature.matmul(corr_matrix)
       
        pooling_sum = torch.zeros([B, d, N, 1], device=self.device)

        for i in range(num_cluster):
            net = cluster_feature[:,:,i].unsqueeze(-1).unsqueeze(-1)
            net = net.expand(B, d, N, 1)
            cluster = torch.eq(group_idx, i).unsqueeze(-1).unsqueeze(1)
            cluster = cluster.expand(B, d, N, 1)
            net = torch.where(cluster, net, 0*net)
            pooling_sum = pooling_sum + net

        print(pooling_sum.size())
        exit()

        return pooling_sum

    def forward(self, inputs, device):
        features, clusters = self.preprocess(inputs)

        self.device = device
        self.to(device)
        features    = features.to(device)
        clusters    = clusters.to(device)

        K           = features.size()[-1]
        batch_size  = features.size()[0]

        m_conv2d    = getattr(self, 'mlp_in')
        features    = m_conv2d(features)

        for i in range(self.cfg.num_blocks):
            m_conv2d    = getattr(self, 'mlp_'+str(i)+'_1')
            features    = m_conv2d(features)

            m_conv2d    = getattr(self, 'mlp_'+str(i)+'_2')
            features    = m_conv2d(features)

            out_pool    = self.pooling_block(features, clusters[i], i)

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

        return net

    def preprocess(self, inputs):
        # B x nv x 3
        verts       = inputs['verts']
        # B x nf x 3
        faces       = inputs['faces']

        batch_size  = verts.size()[0]

        clusters    = []
        new_feature = []

        for i in range(batch_size):
            #features    = vertices
            face        = faces[i]
            vert        = verts[i]

            normal      = compute_normal(vert, face)
            lap_matrix  = self.compute_lap(face, vert)

            eig_val, eig_vec = torch.symeig(lap_matrix, eigenvectors=True)
            eig_vec = eig_vec.narrow(1, 1, self.cfg.num_eig)

            if 'features' not in inputs.keys():
                feature = torch.cat([vert, eig_vec, 
                                    torch.from_numpy(normal)], 1)
                new_feature.append(feature.unsqueeze(0))

            cluster_idx = self.compute_cluster(eig_vec, self.cfg.num_cluster)
            clusters.append(np.expand_dims(cluster_idx, axis=1))

        # B x nv x d
        if 'features' not in inputs.keys():
            features    = torch.cat(new_feature, 0).unsqueeze(1)
        else:
            features    = inputs['features']

        # layer x B x nv
        clusters        = torch.from_numpy(np.concatenate(clusters, axis=1))
      
        return features, clusters