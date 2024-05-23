# Graph with 2 hops (style -> content; content->content)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pyinn
import numpy as np
import dgl
import dgl.nn.pytorch as dglnn
import pdb
import networkx as nx
import matplotlib.pyplot as plt
from torchsummary import summary
from dpt_models.box_coder import *
from dpt_models.depatch_embed import Simple_DePatch

#############################################################################
# Graph Block                                                               #
#############################################################################

class GraphBlock(nn.Module):
    r"""
    Graph Construction
    """
    def __init__(self, n_c=256, k=5, patch_size=5, patch_stride=1, stride=1, dist='normcrosscorrelation', conv='gatconv', graph='heter', padding=None):
        r"""
        :param n_c: number of feature channels
        :param n_ec: number of embedding channels
        :param k(optional): number of neighbors to sample
        :param patch_size(optional): size of patches that are matched
        :param stride(optional): stride with which patches are extracted
        """
        super(GraphBlock, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.depatch_embed = build_depatch_embed(in_size=64, in_chans=n_c, 
                                                embed_dim=n_c*patch_size*patch_size, 
                                                patch_size=patch_size, patch_stride=1)
        self.gnn_block = GNN(feat_c=n_c, k=k, patch_size=patch_size, stride=stride, dist_type=dist, conv_type=conv, graph_type=graph)
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(n_c*2, n_c, 3), 
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(n_c, n_c, 3)
        )

    def forward(self, ys, yc, args=None):
        r"""
        :param xs: style image
        :param xc: content image
        :param ys: style features
        :param yc: content features
        :return y_agg: aggregated features
        """ 
        # AdaIN 
        # ys = adaptive_instance_normalization(ys, yc)
        
        b,c,h,w = ys.shape
        # Convert images to patches 
        ys_patch, (H,W) = self.depatch_embed((ys, yc))
        ys_patch = ys_patch.view(b,H,W,-1)
        
        #### use PyINN
        # ys_patch = im2patch(ys, self.patch_size, self.stride, self.padding)
        yc_patch, y_padding = im2patch(yc, self.patch_size, self.stride, self.padding, returnpadding=True)
        #print(yc_patch.shape)
       #### use torch.nn.Fold
        #ys_patch = image_to_patch(ys, self.patch_size, self.stride)
        #yc_patch = image_to_patch(yc, self.patch_size, self.stride)

        # generate aggregated patches
        agg_patch = self.gnn_block(ys_patch, yc_patch)
        
        # Convert patches to image 
        #print(agg_patch.shape)
        #### use PyINN
        y_agg = patch2im(agg_patch, self.patch_size, self.stride, y_padding)
        #print(y_agg.shape)
        #### use torch.nn.Fold
        #y_agg = patch_to_image(agg_patch, [h,w], self.patch_size, self.stride)
        
        # Fuse 
        # y_fuse = self.fuse(torch.cat((yc,y_agg), dim=1))
        return y_agg
        

#############################################################################
# GNN (Graph Neural Network)                                                #
#############################################################################

class GNN(nn.Module):
    def __init__(self, feat_c, k, patch_size, stride, dist_type='normcrosscorrelation', conv_type='graphconv', graph_type='heter'):
        r"""
        :param k: number of neighbors
        :param patch_size: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param dist(optional): distance metric
        :param conv(optional): graph convolutional layer type
        """
        super(GNN, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.k = k
        self.dist_type = dist_type
        self.feat_c = feat_c
        self.patch_size = patch_size
        self.feat_pc = feat_c*patch_size*patch_size
        self.conv_type=conv_type
        self.graph_type=graph_type
        

        if graph_type == 'heter':
            self.conv_model = HeterConv(self.feat_c, self.patch_size, conv_type) 
        else:
            raise AttributeError('No such conv layer provide!')

    def knn_graph(self, x, y, k):
        r'''
        for each node in y, search for k-nearest-neighbor in x
        :param x: BxNxE tensor, B batches, N nodes, E embeddings
        :param y: BxMxE tensor, B batches, M nodes, E embeddings
        :param k: k nearest neighbors
        :return idx_k: BxMxK tensor, B batches, M node indices in y, K neighboring node indices in x
        '''
        I = indexer(x, y)
        
        # compute distance
        if self.dist_type == 'euclidean':
            D = euclidean_distance(y, x.permute(0,2,1)).gather(dim=2, index=I) + 1e-5
        elif self.dist_type ==  'normcrosscorrelation':
            D = normalized_cross_correlation(y, x.permute(0,2,1)).gather(dim=2, index=I) + 1e-5
        else:
            raise AttributeError('No such distance matric!')
        
        # print(D.shape) torch.Size([8, 784, 784])
        # get k nearest neighbors' indices
        _, idx_k = torch.topk(D, k, dim=2, largest=True, sorted=True)
        idx_k = I.gather(dim=2, index=idx_k)
        # print(idx_k.shape) torch.Size([8, 784, 5])
        # print(idx_k[0,0,:])tensor([ 0,  1, 28, 29,  2], device='cuda:0')
        return idx_k
        
    def forward(self, ys, yc):
        r"""
        :param ys: patches of features of style image
        :param yc: patches of features of content image
        :return agg: patches of aggregated image
        """
        
        # k(neighbors); b(batch_size); ce(embed_channel); cf(feature_channel); p1,p2(patch_size); n1,n2,m1,m2(patch_num)
        # b,c,p,p,n1,n2 = ys.shape
        b,n1,n2,_ = ys.shape
        # print(ys.shape) torch.Size([8, 512, 5, 5, 28, 28])
        b,c,p,p,m1,m2 = yc.shape

        k=self.k; n = n1*n2; m=m1*m2; f=c*p*p; assert f == self.feat_pc, 'Feature dim does not match the setting!'
        # feat_s = ys.permute(0,4,5,1,2,3).contiguous().view(b,n,f)
        feat_s = ys.contiguous().view(b,n,f)
        
        feat_c = yc.permute(0,4,5,1,2,3).contiguous().view(b,m,f)
        # print(feat_s.shape) torch.Size([8, 784, 12800]) 12800=512*5*5 784=28*28

        # KNN
        idx_k1 = self.knn_graph(feat_s, feat_c, k) # [b,m,k]
        # print(idx_k1.shape) torch.Size([8, 784, 5]) 784=28*28
        idx_k2 = self.knn_graph(feat_c, feat_c, k)
        #idx_k2 = self.knn_graph(feat_c, feat_c, k+1)[:,:,1:] # [b,m,k]
        
        # Aggregation
        agg = self.conv_model(feat_c, feat_s, idx_k1, idx_k2)
            
        # b,m,f -> b,cf,p1,p2,m1,m2
        agg = agg.permute(0,2,1).contiguous().view(b,c,p,p,m1,m2)
        return agg
        
    
    
class HeterConv(nn.Module):
    def __init__(self, feat_c, patch_size, conv):
        super().__init__()
        self.feat_c = feat_c
        self.patch_size = patch_size
        self.feat_pc = feat_c*patch_size*patch_size
        if conv == 'gatconv':
            self.conv1 = dglnn.GATConv(self.feat_pc, self.feat_pc, num_heads=1, allow_zero_in_degree=True)
            self.conv2 = dglnn.GATConv(self.feat_pc, self.feat_pc, num_heads=1, allow_zero_in_degree=True)
        elif conv == 'graphconv':
            self.conv1 = dglnn.GraphConv(self.feat_pc, self.feat_pc, allow_zero_in_degree=True)
            self.conv2 = dglnn.GraphConv(self.feat_pc, self.feat_pc, allow_zero_in_degree=True)            
        elif conv == 'edgeconv':
            self.conv1 = dglnn.EdgeConv(self.feat_pc, self.feat_pc, allow_zero_in_degree=True)
            self.conv2 = dglnn.EdgeConv(self.feat_pc, self.feat_pc, allow_zero_in_degree=True)

    def forward(self, feat_c, feat_s, idx_k1, idx_k2):
        b,n,f = feat_s.shape
        # print(feat_s.shape) torch.Size([8, 784, 12800]) 12800=512*5*5 784=28*28
        b,m,f = feat_c.shape
        _,_,k = idx_k1.shape
        c = self.feat_c; p = self.patch_size
        dev=feat_c.get_device()
        
        torch.cuda.empty_cache()
        # print(feat_s.shape) torch.Size([8, 16, 32768])
        # print(idx_k1)
        # print(idx_k1.shape) torch.Size([8, 784, 5]) 784=28*28
        # print(idx_k2.shape) torch.Size([8, 784, 5])
        
        # indices
        idx_v = torch.tensor(range(m), device=dev, dtype=torch.int64).view(1,m,1).expand(2,m,k).contiguous().view(-1)
        idx_u = torch.cat((idx_k2, idx_k1+m), dim=1).contiguous().view(b,-1)
        # print(idx_v.shape) torch.Size([7840])
        # print(idx_u.shape) torch.Size([8, 7840])

        # a = torch.cat((idx_k2, idx_k1+m), dim=1)
        # print(a.shape) torch.Size([8, 1568, 5])
        # b = torch.tensor(range(m), device=dev, dtype=torch.int64)
        # print(b)
        # print(b.shape)
        # print(idx_v[:100])

        
        # features
        feat = torch.cat((feat_c, feat_s), dim=1)
        
        agg = torch.tensor([]).to(dev)
        for i in range(b):
            graph = dgl.graph((idx_u[i], idx_v), num_nodes=m+n)
            # v_graph = dgl.to_networkx(graph.cpu())
            # pos = nx.kamada_kawai_layout(v_graph)
            # nx.draw(v_graph, with_labels=True)
            # plt.savefig('fig.jpg')
            # assert(False)
            feat_out = self.conv(graph, feat[i])[:m] # m,f
            agg = torch.cat((agg, feat_out.contiguous().view(1,m,f)), dim=0)
            
        return agg
        
    def conv(self, graph, x):
        h = self.conv1(graph, x)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
        
        

#############################################################################
# Operations                                                                #
#############################################################################

def gather_nodes(x, idx_k):
    r'''
    for x[b,n,d], select k in n
    '''
    b,n,d = x.shape
    b,m,k = idx_k.shape
    x_ext = x.permute(0,2,1).contiguous().view(b,1,d,n).expand(b,m,d,n)
    idx_k_ext = idx_k.view(b,m,1,k).expand(b,m,d,k)
    # b,m,k,d
    return torch.gather(x_ext, dim=3, index=idx_k_ext).permute(0,1,3,2)

index_cache = {}
def indexer(x, y):
    r'''
    for each of the m nodes in y, construct an indexer of n node indices in x (save in index_cache{})
    '''
    b,n,e = x.shape
    b,m,e = y.shape
    dev = x.get_device()
    key = "{}_{}_{}_{}_{}".format(b,n,m,e,dev)
    if not key in index_cache:
        index_cache[key] = torch.tensor(range(n), device=x.get_device(), dtype=torch.int64).view(1,1,-1).repeat(b,m,1)
    return Variable(index_cache[key], requires_grad=False)

def adaptive_patch_normalization(x, y, eps=1e-5):
    r'''
    AdaPN(x,y) = y.std() * ( (x - x.mean()) / x.std() ) + y.mean()
    '''
    b,m,c,_,k = x.shape
    x_std = (x.var(dim=3) + eps).sqrt().view(b, m, c, 1, k)
    x_mean = x.mean(dim=3).view(b, m, c, 1, k)
    y_std = (y.var(dim=3) + eps).sqrt().view(b, m, c, 1, 1)
    y_mean = y.mean(dim=3).view(b, m, c, 1, 1)
    return y_std*((x-x_mean)/x_std)+y_mean
    
def adaptive_instance_normalization(x, y, eps=1e-5):
    r'''
    AdaIN(x,y) = y.std() * ( (x - x.mean()) / x.std() ) + y.mean()
    '''
    b,c,_,_ = x.shape
    x_std = (x.view(b,c,-1).var(dim=2) + eps).sqrt().view(b,c,1,1)
    x_mean = x.view(b,c,-1).mean(dim=2).view(b,c,1,1)
    y_std = (x.view(b,c,-1).var(dim=2) + eps).sqrt().view(b,c,1,1)
    y_mean = y.view(b,c,-1).mean(dim=2).view(b,c,1,1)
    return y_std*((x-x_mean)/x_std)+y_mean

def calc_padding(x_shape, patchsize, stride, padding=None):
    if padding is None:
        xdim = x_shape
        padvert = -(xdim[0] - patchsize) % stride
        padhorz = -(xdim[1] - patchsize) % stride
        padtop = int(np.floor(padvert / 2.0))
        padbottom = int(np.ceil(padvert / 2.0))
        padleft = int(np.floor(padhorz / 2.0))
        padright = int(np.ceil(padhorz / 2.0))
    else:
        padtop = padbottom = padleft = padright = padding
    return padtop, padbottom, padleft, padright

def im2patch(x, patchsize, stride, padding=None, returnpadding=False):
    padtop, padbottom, padleft, padright = calc_padding(x.shape[2:], patchsize, stride, padding)
    xpad = F.pad(x, pad=(padleft, padright, padtop, padbottom))
    x2col = pyinn.im2col(xpad, [patchsize]*2, [stride]*2, [0,0])
    if returnpadding:
        return x2col, (padtop, padbottom, padleft, padright)
    else:
        return x2col

def patch2im(x_patch, patchsize, stride, padding):
    padtop, padbottom, padleft, padright = padding
    counts = pyinn.col2im(torch.ones_like(x_patch), [patchsize]*2, [stride]*2, [0,0])
    x = pyinn.col2im(x_patch.contiguous(), [patchsize]*2, [stride]*2, [0,0])
    x = x/counts
    x = x[:,:,padtop:x.shape[2]-padbottom, padleft:x.shape[3]-padright]
    return x
    
def euclidean_distance(x,y):
    out = -2*torch.matmul(x, y)
    out += (x**2).sum(dim=-1, keepdim=True)
    out += (y**2).sum(dim=-2, keepdim=True)
    return out
    
def normalized_cross_correlation(x, y, eps=1e-8):
    dev_xy = torch.matmul(x,y)
    dev_xx = torch.mul(x,x)
    dev_yy = torch.mul(y,y)

    dev_xx_sum = torch.sum(dev_xx, dim=-1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=-2, keepdim=True)

    ncc = torch.div(dev_xy + eps,
                    torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    return ncc
    
def build_depatch_embed(in_size, patch_size=4, in_chans=256, embed_dim=256, patch_stride=1):
    padtop, padbottom, padleft, padright = calc_padding((in_size, in_size), patch_size, patch_stride)
    patch_count = math.floor((in_size + padtop + padbottom - (patch_size-1) - 1) / patch_stride + 1)
    # print("padding and patch count: ", padtop, padbottom , padleft, padright, patch_count, patch_stride)
    box_coder = pointwhCoder(input_size=in_size, 
                            patch_count=patch_count, weights=(1.,1.,1.,1.), pts=3, 
                            tanh=True, wh_bias=torch.tensor(5./3.).sqrt().log())
    patch_embed = Simple_DePatch(box_coder, img_size=in_size, 
                            patch_size=patch_size, patch_pixel=3, patch_count=patch_count, patch_stride=patch_stride,
                            in_chans=in_chans, embed_dim=embed_dim, another_linear=True, use_GE=True, with_norm=True)
    return patch_embed


if __name__ == '__main__':
    g_layer = GraphBlock(dist='euclidean', conv='gatconv', k=5, patch_size=5, patch_stride=3, stride=2).cuda()

    b, c, h, w = 1, 256, 64, 64
    ys = torch.ones((b,c,h,w)).cuda()
    yc = torch.ones((b,c,h,w)).cuda()
    y_gg = g_layer(ys, yc)
    print(y_gg.shape)
    #summary(g_layer, [(1,256,64,64), (1,256,64,64)])

    b, c, h, w = 1, 256, 128, 128
    ys = torch.ones((b,c,h,w)).cuda()
    yc = torch.ones((b,c,h,w)).cuda()
    y_gg = g_layer(ys, yc)
    print(y_gg.shape)

    b, c, h, w = 1, 256, 254, 254
    ys = torch.ones((b,c,h,w)).cuda()
    yc = torch.ones((b,c,h,w)).cuda()
    y_gg = g_layer(ys, yc)
    print(y_gg.shape)
