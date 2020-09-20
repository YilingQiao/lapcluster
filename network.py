import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from sklearn.cluster import KMeans

import helper_torch_util
from laplacian import Laplacian
from normal import compute_normal 
from utils import trans_augment



class LapCluster(nn.Module):
    def __init__(self, cfg):
        
        super(LapCluster,self).__init__()
        self.cfg            = cfg
        self.compute_lap    = Laplacian(cot=True)

        self.criterion = torch.nn.CrossEntropyLoss()

        ld = 1

        concat_feature = []

        for i, d in enumerate(cfg.d_in):
            if i == 0:
                f_conv2d  = helper_torch_util.conv2d(True, ld, d, kernel_size=[1,cfg.d_feature])
                setattr(self, 'mlp_in_'+str(i), f_conv2d)
                ld = d
            else:
                f_conv2d  = helper_torch_util.conv2d(True, ld, d, kernel_size=[1,1])
                setattr(self, 'mlp_in_'+str(i), f_conv2d)
                ld = d

        for i in range(len(cfg.d_b1)):
            ld1 = ld
            for j, d in enumerate(cfg.d_b1[i]):
                f_conv2d  = helper_torch_util.conv2d(True, ld1, d)
                setattr(self, "mlp_{}_{}".format(i, j), f_conv2d)
                ld1 = d

            ld2 = ld1
            for j, d in enumerate(cfg.d_b2[i]):
                f_conv2d  = helper_torch_util.conv2d(True, ld2, d)
                setattr(self, "mlp_corr_{}_{}".format(i, j), f_conv2d)
                ld2 = d

            ld = ld1
            if cfg.pooling:
                ld += ld2


        ld = cfg.d_in[-1]
        for i in range(len(cfg.d_b1)):
            ld += cfg.d_b1[i][-1] 
            if cfg.pooling:
                ld += cfg.d_b2[i][-1]


        for i, d in enumerate(cfg.d_outmlp):
            f_conv2d  = helper_torch_util.conv2d(True, ld, d, kernel_size=[1,1])
            setattr(self, 'mlp_out_'+str(i), f_conv2d)
            ld = d

        for i, d in enumerate(cfg.d_finalmlp):
            f_conv2d  = helper_torch_util.conv2d(True, ld, d, kernel_size=[1,1])
            setattr(self, 'final_'+str(i), f_conv2d)
            ld = d

            if i != len(cfg.d_finalmlp) - 1:
                bn = nn.BatchNorm2d(d)
                setattr(self, 'final_bn_'+str(i), bn)
                f_dropout = nn.Dropout(0.5)
                setattr(self, 'final_dropout_'+str(i), f_dropout)



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
        _inf            = -1e7
        padding_inf     = _inf * torch.ones([B, d, N, k], device=self.device)
        cluster_feature = []
        cluster_corr    = []

        m_pool2d    = nn.MaxPool2d([N, 1])
        m_pool2d_c  = nn.MaxPool2d([N, 1])


        corr_feat = feature
        for j, d in enumerate(self.cfg.d_b2[num_block]):
            m_conv2d  = getattr(self, "mlp_corr_{}_{}".format(num_block, j))
            corr_feat = m_conv2d(corr_feat)

        pad_corr    = _inf * torch.ones(corr_feat.size(), device=self.device)
        d2 = corr_feat.size()[1]

        for i in range(num_cluster):
            cluster_o       = torch.eq(group_idx, i).unsqueeze(-1)
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
        cluster_feature = cluster_feature.matmul(corr_matrix.transpose(-2,-1))
       
        pooling_sum = torch.zeros([B, d, N, 1], device=self.device)

        for i in range(num_cluster):
            net = cluster_feature[:,:,i].unsqueeze(-1).unsqueeze(-1)
            net = net.expand(B, d, N, 1)
            cluster = torch.eq(group_idx, i).unsqueeze(-1)
            cluster = cluster.expand(B, d, N, 1)
            net = torch.where(cluster, net, 0*net)
            pooling_sum = pooling_sum + net


        return pooling_sum


    def forward(self, inputs):
        device = self.device
        cfg = self.cfg

        feature = inputs['data']['feature'].to(device)
        cluster = inputs['data']['cluster'].to(device)
        feature = feature.unsqueeze(1)

        concat_features = []
        for i, d in enumerate(cfg.d_in):
            m_conv2d  = getattr(self, 'mlp_in_'+str(i))
            feature    = m_conv2d(feature)
        concat_features.append(feature)

        for i in range(len(cfg.d_b1)):
            for j, d in enumerate(cfg.d_b1[i]):
                m_conv2d  = getattr(self, "mlp_{}_{}".format(i, j))
                feature = m_conv2d(feature)

            if cfg.pooling:
                out_pool = self.pooling_block(feature, cluster.narrow(1,i,1), i)
                feature = torch.cat([feature, out_pool], 1)
            concat_features.append(feature)

        feature = torch.cat(concat_features, axis=1)
        for i, d in enumerate(cfg.d_outmlp):
            m_conv2d = getattr(self, 'mlp_out_'+str(i))
            feature = m_conv2d(feature)

        for i, d in enumerate(cfg.d_finalmlp):
            m_conv2d = getattr(self, 'final_'+str(i))
            feature = m_conv2d(feature)

            if i != len(cfg.d_finalmlp) - 1:
                m_bn = getattr(self, 'final_bn_'+str(i))
                m_dropout = getattr(self, 'final_dropout_'+str(i))
                feature = m_bn(feature)
                feature = m_dropout(feature)

        feature = F.log_softmax(feature, dim=1)
        feature = feature.squeeze(-1)

        return feature


    def get_loss(self, outputs, inputs):
        label = inputs['data']['label'].to(self.device)
        
        loss = F.nll_loss(outputs, label)
        return loss

    def get_metric(self, outputs, inputs):
        label = inputs['data']['label'].to(self.device)
        predictions = torch.max(outputs, dim=-2).indices
        result = (label==predictions).float().mean()
        return result

    def transform(self, data, attr):
        cfg = self.cfg

        normal = data['normal'] 
        vert = data['vert'] 
        eig_vec = data['eig_vec'] 
        cluster = data['cluster'] 
        label = data['label']

        if self.epoch > 10:
            vert, normal = trans_augment(cfg.t_augment, vert, normals=normal)

        feature = np.hstack([vert, normal, eig_vec])

        n = feature.shape[0]
        if n > cfg.num_points:
            idx = np.random.choice(n,
                size=cfg.num_points,
                replace=False)
            if cfg.task == 'segmentation':
                label = label[idx]
            else:
                label = label.expand_dims(0)
            feature = feature[idx, :]
            cluster = cluster[:, idx]


        inputs = dict()
        inputs['feature'] = torch.from_numpy(feature)
        inputs['cluster'] = torch.from_numpy(cluster)
        inputs['label'] = torch.from_numpy(label)

        return inputs

    def preprocess(self, data, attr=None):
        # nv x 3
        vert       = data['verts']
        # nf x 3
        face       = data['faces']
        label = np.array(data['label']) 


        normal = compute_normal(vert, face)
        lap_matrix  = self.compute_lap(torch.from_numpy(vert), torch.from_numpy(face))

        eig_val, eig_vec = np.linalg.eigh(lap_matrix)
        eig_vec = eig_vec[:,1:1+self.cfg.num_eig]
    

        features = np.hstack([vert, normal,  eig_vec])

        cluster = self.compute_cluster(eig_vec, self.cfg.num_cluster)

        
        data = dict()
        data['normal'] = normal
        data['vert'] = vert
        data['face'] = face
        data['eig_vec'] = eig_vec
        data['cluster'] = cluster
        data['label'] = label
        return data