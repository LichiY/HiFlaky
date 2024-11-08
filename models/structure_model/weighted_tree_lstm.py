# --coding:utf-8--
#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class WeightedHierarchicalTreeLSTMEndtoEnd(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_matrix, out_matrix,
                 in_dim,
                 dropout=0.0,
                 device=torch.device('cpu'),
                 root=None,
                 hierarchical_label_dict=None,
                 label_trees=None):
        super(WeightedHierarchicalTreeLSTMEndtoEnd, self).__init__()
        self.root = root
        mem_dim = in_dim // 2
        self.hierarchical_label_dict = hierarchical_label_dict
        self.label_trees = label_trees
        self.bottom_up_lstm = WeightedChildSumTreeLSTMEndtoEnd(in_dim, mem_dim, num_nodes, in_matrix, device)
        self.top_down_lstm = WeightedTopDownTreeLSTMEndtoEnd(in_dim, mem_dim, num_nodes, out_matrix, device)
        self.tree_projection_layer = torch.nn.Linear(2 * mem_dim, mem_dim)
        self.node_dropout = torch.nn.Dropout(dropout)
        self.num_nodes = num_nodes
        self.mem_dim = mem_dim

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)
        for i in self.hierarchical_label_dict[self.root.idx]:
            self.bottom_up_lstm(self.label_trees[i + 1], inputs)
            self.top_down_lstm(self.label_trees[i + 1], inputs)

        tree_label_feature = []
        nodes_keys = list(self.label_trees.keys())
        nodes_keys.sort()
        for i in nodes_keys:
            if i == 0:
                continue
            tree_label_feature.append(
                torch.cat((self.node_dropout(self.label_trees[i].bottom_up_state[1].view(inputs.shape[1], 1, self.mem_dim)),
                           self.node_dropout(self.label_trees[i].top_down_state[1].view(inputs.shape[1], 1, self.mem_dim))),
                          2))
        label_feature = torch.cat(tree_label_feature, 1)

        return label_feature


class WeightedChildSumTreeLSTMEndtoEnd(nn.Module):
    def __init__(self, in_dim, mem_dim,
                 num_nodes=-1, prob=None,
                 device=torch.device('cpu')):
        super(WeightedChildSumTreeLSTMEndtoEnd, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.node_transformation = torch.nn.ModuleList()
        self.node_transformation_decompostion = torch.nn.ModuleList()
        self.prob = torch.Tensor(prob).to(device)
        self.prob = Parameter(self.prob)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(2) // 3, dim=2)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(self.fh(child_h) + self.fx(inputs).repeat(len(child_h), 1, 1))
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                   1)
            child_h = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                   1)
        else:
            child_c, child_h = zip(
                *map(lambda x: (self.prob[tree.idx][x.idx] * y for y in x.bottom_up_state), tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        tree.bottom_up_state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.bottom_up_state


class WeightedTopDownTreeLSTMEndtoEnd(nn.Module):
    def __init__(self, in_dim, mem_dim,
                 num_nodes=-1, prob=None,
                 device=torch.device('cpu')):
        super(WeightedTopDownTreeLSTMEndtoEnd, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.node_transformation = torch.nn.ModuleList()
        self.node_transformation_decompostion = torch.nn.ModuleList()
        self.prob = torch.Tensor(prob).to(device)
        self.prob = Parameter(self.prob)

    def node_forward(self, inputs, parent_c, parent_h):
        iou = self.ioux(inputs) + self.iouh(parent_h)
        i, o, u = torch.split(iou, iou.size(2) // 3, dim=2)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(self.fh(parent_h) + self.fx(inputs).repeat(len(parent_h), 1, 1))
        fc = torch.mul(f, parent_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, inputs, state=None, parent=None):
        if state is None:
            parent_c = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                    1)
            parent_h = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                    1)
        else:
            parent_c = self.prob[parent.idx][tree.idx] * state[0]
            parent_h = self.prob[parent.idx][tree.idx] * state[1]

        tree.top_down_state = self.node_forward(inputs[tree.idx], parent_c, parent_h)
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs, tree.top_down_state, tree)
        return tree.top_down_state
