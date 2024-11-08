# --coding:utf-8--


import codecs
import torch
from models.structure_model.tree import Tree


def load_checkpoint(model_file, model, config, optimizer=None):

    checkpoint_model = torch.load(model_file)
    config.train.start_epoch = checkpoint_model['epoch'] + 1
    best_performance = checkpoint_model['best_performance']
    model.load_state_dict(checkpoint_model['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_model['optimizer'])
    return best_performance, config


def save_checkpoint(state, model_file):
    torch.save(state, model_file)


def get_hierarchy_relations(hierar_taxonomy, label_map, root=None, fortree=False):

    label_tree = dict()
    label_tree[0] = root
    hierar_relations = {}
    with codecs.open(hierar_taxonomy, "r", "utf8") as f:
        for line in f:
            line_split = line.rstrip().split('\t')
            parent_label, children_label = line_split[0], line_split[1:]

            if parent_label not in label_map:
                if fortree and parent_label == 'Root':
                    parent_label_id = -1
                else:
                    continue
            else:
                parent_label_id = label_map[parent_label]
            children_label_ids = [label_map[child_label] \
                                  for child_label in children_label if child_label in label_map]
            hierar_relations[parent_label_id] = children_label_ids
            if fortree:
                assert (parent_label_id + 1) in label_tree
                parent_tree = label_tree[parent_label_id + 1]

                for child in children_label_ids:
                    assert (child + 1) not in label_tree
                    child_tree = Tree(child)
                    parent_tree.add_child(child_tree)
                    label_tree[child + 1] = child_tree
    if fortree:
        return hierar_relations, label_tree
    else:
        return hierar_relations
