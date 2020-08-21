import multiprocessing as mp
from tqdm import tqdm
import torch
import pandas as pd

"""
The code below is inspired and extracted from IGMC paper by Zhang et al 2020.
"""


class Links2subgraphs():
    def __init__(self):
        self.A = pd.DataFrame()  # or also called as A
        self.graph_labels = pd.DataFrame()
        self.links = None
        self.is_parallel = False
        self.max_node_label = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.train_labels = None
        self.val_labels = None
        self.test_labels = None
        self.h = 1
        self.sample_ratio = 1.0
        self.max_nodes_per_hop = None
        self.u_features = None
        self.v_features = None
        self.max_node_label = None
        self.class_values = None
        self.testing = False
        self.parallel = True

    def nx_to_PyGGraph(g, graph_label, node_labels, node_features, max_node_label, class_values):
        # convert networkx graph to pytorch_geometric data format
        y = torch.FloatTensor([class_values[graph_label]])
        if len(g.edges()) == 0:
            i, j = [], []
        else:
            i, j = zip(*g.edges())
        edge_index = torch.LongTensor([i+j, j+i])
        edge_type_dict = nx.get_edge_attributes(g, 'type')
        edge_type = torch.LongTensor(
            [edge_type_dict[(ii, jj)] for ii, jj in zip(i, j)])
        edge_type = torch.cat([edge_type, edge_type], 0)
        edge_attr = torch.FloatTensor(
            class_values[edge_type]
        ).unsqueeze(1)  # continuous ratings, num_edges * 1
        x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
        if node_features is not None:
            if type(node_features) == list:
                # node features are only provided for target user and item
                u_feature, v_feature = node_features
            else:
                # node features are provided for all nodes
                x2 = torch.FloatTensor(node_features)
                x = torch.cat([x, x2], 1)

        data = Data(x, edge_index, edge_attr=edge_attr, y=y)
        data.edge_type = edge_type
        if type(node_features) == list:
            data.u_feature = torch.FloatTensor(u_feature).unsqueeze(0)
            data.v_feature = torch.FloatTensor(v_feature).unsqueeze(0)
        return data

    def PyGGraph_to_nx(data):
        edges = list(
            zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
        g = nx.from_edgelist(edges)
        g.add_nodes_from(range(len(data.x)))  # in case some nodes are isolated
        # transform r back to rating label
        edge_types = {(u, v): data.edge_type[i].item()
                      for i, (u, v) in enumerate(edges)}
        nx.set_edge_attributes(g, name='type', values=edge_types)
        node_types = dict(
            zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
        nx.set_node_attributes(g, name='type', values=node_types)
        g.graph['rating'] = data.y.item()
        return g

    def helper(self, A, links, g_labels):
        g_list = []
        if not self.is_parallel or self.max_node_label is None:
            with tqdm(total=len(links[0])) as pbar:
                # for node-labelling process, need to change hard-coded 0 and 1 to be dynamic depending on the size of matrices
                for i, j, g_label in zip(links[0], links[1], g_labels):
                    g, n_labels, n_features = subgraph_extraction_labeling(
                        (i, j), A, h, sample_ratio, max_nodes_per_hop, u_features,
                        v_features, class_values
                    )
                    if max_node_label is None:
                        max_n_label['max_node_label'] = max(
                            max(n_labels), max_n_label['max_node_label']
                        )
                        g_list.append((g, g_label, n_labels, n_features))
                    else:
                        g_list.append(nx_to_PyGGraph(
                            g, g_label, n_labels, n_features, max_node_label, class_values
                        ))
                    pbar.update(1)
        else:
            # == timing - start
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            # asynchronous pooling for training
            results = pool.starmap_async(
                parallel_worker,
                [
                    (g_label, (i, j), A, h, sample_ratio, max_nodes_per_hop, u_features,
                     v_features, class_values)
                    for i, j, g_label in zip(links[0], links[1], g_labels)
                ]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready():
                    break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            # == timing - end

            # == time loader for transformation
            print("Transforming to pytorch_geometric graphs...".format(end-start))
            g_list += [
                nx_to_PyGGraph(g, g_label, n_labels, n_features,
                               max_node_label, class_values)
                for g_label, g, n_labels, n_features in tqdm(results)
            ]
            del results
            end2 = time.time()
            print("Time eplased for transforming to pytorch_geometric graphs: {}s".format(
                end2-end))
            # timing - end
        return g_list
