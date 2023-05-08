import torch
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
import torch.utils.checkpoint as checkpoint
from lib.utils import square_distance


def get_graph_feature(coords, feats, k=10):
    """
    Apply KNN search based on coordinates, then concatenate the features to the centroid features
    Input:
        X:          [B, 3, N]
        feats:      [B, C, N]
    Return:
        feats_cat:  [B, 2C, N, k]
    """
    # apply KNN search to build neighborhood
    B, C, N = feats.size()
    dist = square_distance(coords.transpose(1,2), coords.transpose(1,2))

    idx = dist.topk(k=k+1, dim=-1, largest=False, sorted=True)[1]  #[B, N, K+1], here we ignore the smallest element as it's the query itself  
    idx = idx[:,:,1:]  #[B, N, K]

    idx = idx.unsqueeze(1).repeat(1,C,1,1) #[B, C, N, K]
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)  # [B, C, N, N]

    neighbor_feats = torch.gather(all_feats, dim=-1,index=idx) #[B, C, N, K]

    # concatenate the features with centroid
    feats = feats.unsqueeze(-1).repeat(1,1,1,k)

    feats_cat = torch.cat((feats, neighbor_feats-feats),dim=1)

    return feats_cat



class SelfAttention(nn.Module):
    def __init__(self,feature_dim,k=10):
        super(SelfAttention, self).__init__() 
        self.conv1 = nn.Conv2d(feature_dim*2, feature_dim, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)
        
        self.conv2 = nn.Conv2d(feature_dim*2, feature_dim * 2, kernel_size=1, bias=False)
        self.in2 = nn.InstanceNorm2d(feature_dim * 2)

        self.conv3 = nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)

        self.k = k

    def forward(self, coords, features):
        """
        Here we take coordinats and features, feature aggregation are guided by coordinates
        Input: 
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """
        B, C, N = features.size()

        x0 = features.unsqueeze(-1)  #[B, C, N, 1]

        x1 = get_graph_feature(coords, x0.squeeze(-1), self.k)
        x1 = F.leaky_relu(self.in1(self.conv1(x1)), negative_slope=0.2)
        x1 = x1.max(dim=-1,keepdim=True)[0]

        x2 = get_graph_feature(coords, x1.squeeze(-1), self.k)
        x2 = F.leaky_relu(self.in2(self.conv2(x2)), negative_slope=0.2)
        x2 = x2.max(dim=-1, keepdim=True)[0]

        x3 = torch.cat((x0,x1,x2),dim=1)
        x3 = F.leaky_relu(self.in3(self.conv3(x3)), negative_slope=0.2).view(B, -1, N)

        return x3


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class MultiHeadedAttentionCat(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.distribute = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([self.distribute, deepcopy(self.distribute), deepcopy(self.distribute)])
        self.merge = nn.Conv1d(d_model+7*4, d_model+7*4, kernel_size=1)

    def forward(self, query, key, value, coords0, coords1):
        batch_dim = query.size(0)
        num_q_points = coords0.size(2)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        # print(f"value shape: {value.shape}; coords shape: {coords1.shape}", flush=True)
        value = torch.cat([value, coords1.repeat(1, 4, 1).view(1, 4, 3, -1).transpose(1,2)], dim=1)
        x, _ = attention(query, key, value)
        augment1 = x[:, self.dim:self.dim+3, :, :] - coords0.repeat(1, 4, 1).view(1, 4, 3, -1).transpose(1,2)
        augment2 = torch.norm(augment1, dim=1, keepdim=True)
        y = torch.zeros((batch_dim, self.dim+7, self.num_heads, num_q_points), dtype=augment2.dtype).to(query.device)
        for i in range(batch_dim):
            y[i] = torch.cat([x[i], augment1[i], augment2[i]], dim=0)
        del x
        return self.merge(y.contiguous().view(batch_dim, (self.dim+7)*self.num_heads, -1))


class AttentionalPropagationCat(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttentionCat(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2+7*4, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, feats0, feats1, coords0, coords1):
        message = self.attn(feats0, feats1, feats1, coords0, coords1)
        return self.mlp(torch.cat([feats0, message], dim=1))


class GCN(nn.Module):
    """
        Alternate between self-attention and cross-attention
        Input:
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """
    def __init__(self, num_head: int, feature_dim: int, k: int, layer_names: list):
        super().__init__()
        self.layers=[]
        for atten_type in layer_names:
            if atten_type == 'cross':
                self.layers.append(AttentionalPropagation(feature_dim,num_head))
            elif atten_type == 'cross_cat':
                self.layers.append(AttentionalPropagationCat(feature_dim,num_head))
            elif atten_type == 'self':
                self.layers.append(SelfAttention(feature_dim, k))
        self.layers = nn.ModuleList(self.layers)
        self.names = layer_names

    def forward(self, coords0, coords1, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                # desc0 = desc0 + checkpoint.checkpoint(layer, desc0, desc1)
                # desc1 = desc1 + checkpoint.checkpoint(layer, desc1, desc0)
                desc0 = desc0 + layer(desc0, desc1)
                desc1 = desc1 + layer(desc1, desc0)
            elif name == 'cross_cat':
                desc0 = desc0 + layer(desc0, desc1, coords0, coords1)
                desc1 = desc1 + layer(desc1, desc0, coords1, coords0)
            elif name == 'self':
                desc0 = layer(coords0, desc0)
                desc1 = layer(coords1, desc1)
        return desc0, desc1