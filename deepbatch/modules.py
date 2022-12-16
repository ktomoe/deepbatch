import torch
import torch.nn as nn
import dgl
from dgl.nn import GATv2Conv
import dgl.function as fn

class DeepGraphConvLayer(nn.Module):
    def __init__(self, inputs, nodes, num_heads):
        super(DeepGraphConvLayer, self).__init__()

        gat_nodes = nodes // num_heads

        self.nodes = nodes
        self.num_heads = num_heads

        self.conv = GATv2Conv(inputs, 
                              gat_nodes, 
                              num_heads=num_heads,
                              residual=True,
                              share_weights=False,
                              allow_zero_in_degree=True)

        self.bn = nn.BatchNorm1d(nodes)
        self.relu = nn.ReLU()

    def forward(self, g):
        in_feat = g.ndata['features']

        node_size = in_feat.size(0)

        in_feat, attn = self.conv(g, in_feat, get_attention=True)

        in_feat = in_feat.view(node_size, self.nodes)
        in_feat = self.bn(in_feat)
        in_feat = self.relu(in_feat)

        g.ndata['features'] = in_feat

        return g, attn

class DeepGraphConvModel(nn.Module):
    def __init__(self, inputs, layers, nodes, num_heads):
        super(DeepGraphConvModel, self).__init__()

        convs = [DeepGraphConvLayer(inputs, nodes, num_heads)]

        for ii in range(layers-1):
            convs.append(DeepGraphConvLayer(nodes, nodes, num_heads))

        self.convs = nn.Sequential(*convs)  

    def forward(self, g):
        feats = []
        attns = []
        for conv in self.convs:
            g, attn = conv(g)
            feats.append(dgl.mean_nodes(g, 'features'))
            attns.append(attn)

        return feats, attns

class DeepBatchModel(nn.Module):
    def __init__(self, feature_layers=4, nodes=16, num_heads=2):
        super(DeepBatchModel, self).__init__()
      
        self.features = DeepGraphConvModel(21, feature_layers, nodes, num_heads)
        self.fc1 = nn.Linear(nodes*feature_layers, nodes)
        self.bn = nn.BatchNorm1d(nodes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(nodes, 5)

    def forward(self, g):
        batch_size = g.batch_size
        node_size = g.ndata['features'].size(0) // batch_size
        edge_size = node_size * node_size

        feats, attns = self.features(g)
        feat = torch.cat(feats, 1)

        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.dropout(feat)
        feat = self.fc2(feat)

        attns = torch.stack(attns)
        attns = torch.squeeze(attns)
        attns = attns.permute(1, 0)

        return feat, attns

class MLPModel(nn.Module):
    def __init__(self, inputs=21*621, layers=4, nodes=16):
        super(MLPModel, self).__init__()

        convs = [
            nn.Linear(inputs, nodes),
            nn.BatchNorm1d(nodes),
            nn.ReLU(),
            nn.Dropout(0.2)
        ]

        for ii in range(layers-2):
            convs.append(nn.Linear(nodes, nodes))
            convs.append(nn.BatchNorm1d(nodes))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout(0.2))
        convs.append(nn.Linear(nodes, 5))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)

        return x
