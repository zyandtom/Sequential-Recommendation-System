import torch
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes

def add_self_loops_partial(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index
    mask = row == col
    masked_weight = edge_weight[mask]
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    loop_weight = torch.full((num_nodes,), fill_value).cuda()
    loop_weight[row[mask]] = masked_weight
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    assert edge_index.shape[-1] == edge_weight.shape[0]
    return edge_index, edge_weight