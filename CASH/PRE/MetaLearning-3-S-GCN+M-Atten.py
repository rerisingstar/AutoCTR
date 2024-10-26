import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch_geometric.nn import APPNP
import torch.nn.functional as F
import random

import sys
import os
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))))),"CTRAPI"))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))))))
#print(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))))))

from CTRAPI.pruned_search_space import cal_SearchSpace_size
from CTRAPI.pruned_search_space import get_node_num_graph
from CASH.autocash.converter import SolutionConverter


class MetaDataSet(Dataset):
    def __init__(self, data):
        super(MetaDataSet, self).__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        dataset_id = int(self.data[index, 0])
        solution_id = int(self.data[index, 1])
        auc_loss = torch.tensor(self.data[index, 2:])
        return dataset_id, solution_id, auc_loss

class MetaPredictor(nn.Module):
    def __init__(self, meta_path, solution_converter: SolutionConverter, em_size=32, device='cpu', use_attn=True, meta_ratio=1):
        super(MetaPredictor, self).__init__()
        self.device = device
        self.em_size = em_size

        self.mms = MinMaxScaler(feature_range=(0, 1))
        self.read_meta_info(meta_path, meta_ratio=meta_ratio)

        ds_num = 10
        st_num = cal_SearchSpace_size()
        self.st_num = st_num
        self.Demb = nn.Embedding(ds_num, em_size).to(device)
        #self.Semb = nn.Embedding(st_num, em_size).to(device)

        self.gcn_flag = use_attn
        if self.gcn_flag:
            edge_index, node_num, edge_index_node = get_node_num_graph(reindex=False)
            #self.edge_index = torch.IntTensor(edge_index).permute(1, 0).type(torch.LongTensor).to(device)
            self.edge_index_node = edge_index_node
            self.Semb = nn.Embedding(node_num, em_size).to(device)
            self.lin1 = Linear(em_size, em_size).to(device)
            self.lin2 = Linear(em_size, em_size).to(device)
            self.gcn = APPNP(K=10, alpha=0.1)#.to(device)

            self.solution_converter = solution_converter
            mo_num = solution_converter.get_model_num()
            self.mo_num = mo_num
            self.Memb = nn.Embedding(mo_num, em_size).to(device)
            self.multihead_atom_type = nn.MultiheadAttention(em_size, 4,
                                                         dropout=0.5, batch_first=True).to(device)
            self.merge_all = nn.MultiheadAttention(em_size, 4, dropout=0.5, batch_first=True).to(device)
            #self.nodes = torch.LongTensor([i for i in range(node_num)]).to(device)
        else:
            self.Semb = nn.Embedding(st_num, em_size).to(device)

        hidden_size = em_size
        self.dnnD = nn.Sequential(
            # nn.Linear(em_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(device)
        self.dnnS = nn.Sequential(
            # nn.Linear(em_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(device)
        self.output = nn.Sequential(
            nn.Linear(hidden_size*2, 2),
            nn.Sigmoid(),
        ).to(device)

        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, dataset_id, solution_emb):
        '''
        :param dataset_id: tensor(B)
        :param solution_emb: tensor(B, E)
        :return: predict_auc_loss: (B, 2)
        '''
        # dataset_id, solution_id = torch.chunk(input, 2, dim=1)

        dt_emb = self.Demb(dataset_id) #(B, E)
        # sl_emb = self.Semb(solution_id).squeeze()

        dt_ten = self.dnnD(dt_emb) #(B, H)
        sl_ten = self.dnnS(solution_emb)
        #print(dt_ten.shape)
        #print(sl_ten.shape)

        cat_ten = torch.cat((dt_ten, sl_ten), dim=1) #(B, H*2)

        predict_auc_loss = self.output(cat_ten).to(torch.float32) #(B, 2)
        return predict_auc_loss

    def _load_json(self, meta_path, meta_ratio):
        if meta_ratio == 1:
            meta_filename = meta_path
            with open(meta_filename, 'r') as f:
                meta_info = json.load(f)
            return meta_info
        else:
            meta_filename = meta_path[:-5] + '_' + str(meta_ratio) + '.json'
            try:
                f = open(meta_filename, 'r')
            except FileNotFoundError:
                with open(meta_path, 'r') as fa:
                    meta_info_full = json.load(fa)
                meta_info_s = random.sample(meta_info_full,
                                            int(len(meta_info_full) * meta_ratio))
                json.dump(meta_info_s, open(meta_filename, 'w'))
                f = open(meta_filename, 'r')
            meta_info = json.load(f)
            f.close()
            return meta_info

    def read_meta_info(self, meta_path, batch_size=32, meta_ratio=1):
        # feature_names = ['dataset_id', 'solution_id', 'test_auc', 'test_logloss']

        # with open(meta_path, 'r') as f:
        #     meta_info = json.load(f)
        meta_info = self._load_json(meta_path, meta_ratio)
        meta_info = [t for t in meta_info if t[3] < 2]
        data = np.array(meta_info)

        # mms = MinMaxScaler(feature_range=(0, 1))
        data[:, 3] = self.mms.fit_transform(data[:, 3].reshape(-1, 1)).reshape(1, -1)

        self.train_data, self.test_data = train_test_split(data, test_size=0.2, random_state=2022)

        self.train_loader = DataLoader(MetaDataSet(self.train_data), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(MetaDataSet(self.test_data), batch_size=batch_size, shuffle=True)
        self.batch_size = batch_size

    def sample_neighbors_edges(self, target_nodes, device, sample_num=100):
        neighbor_nodes, neighbor_edge_index = target_nodes.tolist(), []
        added_solution_info = []
        target_nodes_num = len(neighbor_nodes)
        sample_num = sample_num*target_nodes_num

        neighbor_nodes_mapping = {}
        for i in range(len(neighbor_nodes)):
            neighbor_nodes_mapping[neighbor_nodes[i]] = i
        
        left_nodes = list(neighbor_nodes)
        max_num, min_num = int(sample_num/len(left_nodes)), max(int(sample_num/len(left_nodes)*0.3),1)
        while len(left_nodes) > 0 and sample_num > 0:
            node = left_nodes[0]
            node_index = neighbor_nodes_mapping[node]

            selected_num = min(random.randint(min_num, max_num), len(self.edge_index_node[node]))
            selected_neighbors = random.sample(self.edge_index_node[node], selected_num)

            for neighbor in selected_neighbors:
                left_nodes.append(neighbor)
                if neighbor not in neighbor_nodes:
                    neighbor_nodes_mapping[neighbor] = len(neighbor_nodes)
                    neighbor_nodes.append(neighbor)
                    #if neighbor < self.st_num:
                    #    added_solution_info.append([neighbor, neighbor_nodes_mapping[neighbor]])
                neighbor_index = neighbor_nodes_mapping[neighbor]
                edge = [node_index, neighbor_index]
                if edge not in neighbor_edge_index:
                    neighbor_edge_index.append([node_index, neighbor_index])
                    neighbor_edge_index.append([neighbor_index, node_index])
                    sample_num -= 2

            left_nodes.remove(node)

        neighbor_nodes = torch.LongTensor(neighbor_nodes).to(device)
        neighbor_edge_index = torch.IntTensor(neighbor_edge_index).permute(1, 0).type(torch.LongTensor).to(device)
        return neighbor_nodes, neighbor_edge_index, target_nodes_num, added_solution_info

    def construct_input(self, x, device):
        model_type_q = []
        model_type_k = []
        for key in x:
            model_components, model_type = self.solution_converter.get_all_idx(key)
            model_type_q.append(model_type)
            model_types = set(range(self.mo_num))
            model_types.remove(model_type)
            model_type_k.append(list(model_types))

        def convert(list_):
            return torch.tensor(list_).to(device).long()
        model_type_q = convert(model_type_q)
        model_type_k = convert(model_type_k)

        model_type_q = self.Memb(model_type_q)
        model_type_k = self.Memb(model_type_k)
        return model_type_q, model_type_k

    def get_solution_embedding(self, gcn_flag, target_nodes, device, sample_num=100):
        if gcn_flag:
            neighbor_nodes, neighbor_edge_indx, target_nodes_num, added_solution_info = self.sample_neighbors_edges(target_nodes, device, sample_num=sample_num)
            x = self.Semb(neighbor_nodes)
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)
            x = self.gcn(x, neighbor_edge_indx)
            solution_emb = x[:target_nodes_num]
            added_solution_info = solution_emb

            model_type_q, model_type_k = self.construct_input(target_nodes, device)
            model_type_emb, _ = self.multihead_atom_type(model_type_q.unsqueeze(1), model_type_k, model_type_k)
            solution_emb, _ = self.merge_all(solution_emb.unsqueeze(1), model_type_emb, model_type_emb)
            solution_emb = solution_emb.squeeze(1)
            #for i in range(len(added_solution_info)):
            #    added_solution_info[i][1] = x[added_solution_info[i][1]]
        else:
            solution_emb = self.Semb(target_nodes)
            added_solution_info = None
        return solution_emb, added_solution_info

    def get_solution_embedding_light(self, gcn_flag, target_nodes, device, sample_num=100):
        if gcn_flag:
            with torch.no_grad():
                solution_emb = self.Semb(target_nodes)
                print(solution_emb.shape)

                model_type_q, model_type_k = self.construct_input(target_nodes, device)
                model_type_emb, _ = self.multihead_atom_type(model_type_q.unsqueeze(1), model_type_k, model_type_k)
                solution_emb, _ = self.merge_all(solution_emb, model_type_emb, model_type_emb)
                solution_emb = solution_emb.squeeze(1)
        else:
            solution_emb = self.Semb(target_nodes)
        return solution_emb

    def evaluate(self, test_loader):
        with torch.no_grad():
            pred_ans = []
            test_loss = 0
            for _, (di, si, y) in enumerate(test_loader):
                di = di.to(self.device)
                si = si.to(self.device)
                solution_emb, _ = self.get_solution_embedding(self.gcn_flag, si, self.device, sample_num=100)
                y = y.to(self.device).to(torch.float32)

                y_pred = self(di, solution_emb)

                test_loss += self.loss_func(y_pred, y)

            test_loss /= len(test_loader)
            return test_loss

    def update_embeddings(self, gcn_flag, target_nodes, device, sample_num=100):
        if gcn_flag:
            with torch.no_grad():
                new_solution_emb, added_solution_info = self.get_solution_embedding(self.gcn_flag, target_nodes, device, sample_num=sample_num)
                target_nodes = target_nodes.tolist()
                for i in range(len(target_nodes)):
                    target_node = target_nodes[i]
                    self.Semb.weight[target_node] = added_solution_info[i]
                #for i in range(len(added_solution_info)):
                #    added_node = added_solution_info[i][0]
                #    self.Semb.weight[added_node] = added_solution_info[i][1]
        return

    def construct(self, epochs=10, sample_num=100):
        '''
        :param data_sets: list, ['str']
        :param solutions: list, [int] index of every solutions
        :param aucs: list, test_auc of every (dataset, solution)
        :return:
        '''
        batch_size = 32

        steps_per_epoch = len(self.train_data) // batch_size + 1
        print(f"{len(self.train_data)} samples in total, {steps_per_epoch} steps per epoch")
        for epoch in range(epochs):
            totalloss = 0
            with tqdm(enumerate(self.train_loader)) as t:
                for _, (di, si, y) in t:
                    di = di.to(self.device)
                    si = si.to(self.device)
                    solution_emb, _ = self.get_solution_embedding(self.gcn_flag, si, self.device, sample_num=sample_num)
                    y = y.to(self.device).to(torch.float32)

                    y_pred = self(di, solution_emb)

                    loss = self.loss_func(y_pred, y)
                    totalloss += loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.update_embeddings(self.gcn_flag, si, self.device, sample_num=sample_num)


            totalloss /= steps_per_epoch
            test_loss = self.evaluate(self.test_loader)
            print(f"epoch-{epoch+1}, loss={totalloss}, test_loss={test_loss}")


if __name__ == '__main__':
    # model_code = [('linear', 2, 'sigmoid'), 'cat', ('linear', 1, 'sigmoid')]
    meta_predicter = MetaPredictor(device='cpu', meta_path='./meta_full.json', gcn_flag=True)
    meta_predicter.construct(epochs=20)

    torch.save(meta_predicter, 'Meta.model')







