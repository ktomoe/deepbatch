import torch
import dgl
import numpy as np
import copy

def random_sampling(data, targets):
    rtn = []
    for idata in data:
        num_samples = len(idata-1)
        rand_samples = int(num_samples*0.1)
        delete_index = np.random.choice(np.arange(1, num_samples), rand_samples, replace=False)
        delete_index = np.sort(delete_index)

        idata = np.delete(idata, delete_index, 0)
        rtn.append(idata)

    rtn = np.array(rtn, dtype=object)

    return rtn, targets

def get_dgl(batch):
    data, targets = batch
    graphs = []

    for idata in data:
        num_nodes = len(idata)
        edges_src = np.array([0] * num_nodes)
        edges_dst = np.array(list(range(num_nodes)))
       
        graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
        graph = dgl.to_bidirected(graph)

        graph.ndata['features'] = torch.tensor(idata, dtype=torch.float32)
        graphs.append(graph)

    graphs = dgl.batch(graphs)

    return graphs, targets
