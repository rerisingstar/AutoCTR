import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.nn import Linear
#from torch_geometric.nn import APPNP
import torch.nn.functional as F
import random
import math

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
    def __init__(self, meta_path, solution_converter: SolutionConverter, em_size=32, device='cpu', use_attn=True, dropout=1e-2, meta_ratio=1):
        super(MetaPredictor, self).__init__()
        self.device = device
        self.device = device
        self.read_meta_info(meta_path, meta_ratio=meta_ratio)

        ds_num = 10
        self.Demb = nn.Embedding(ds_num, em_size).to(device)

        self.solution_converter = solution_converter
        self.use_attn = use_attn
        st_num = solution_converter.get_solution_num()
        co_num = solution_converter.get_components_num()
        mo_num = solution_converter.get_model_num()
        self.st_num = st_num
        self.mo_num = mo_num
        self.co_num = co_num
        self.Semb = nn.Embedding(st_num, em_size).to(device)
        self.Cemb = nn.Embedding(co_num, em_size).to(device)
        self.Pemb = nn.Embedding(co_num, em_size).to(device)
        self.Memb = nn.Embedding(mo_num, em_size).to(device)

        self.multihead_component = nn.MultiheadAttention(em_size, 4, dropout=dropout).to(device)#, batch_first=True).to(device)
        self.multihead_atom_type = nn.MultiheadAttention(em_size, 4,
                                                         dropout=dropout).to(device)#, batch_first=True).to(device)
        self.merge_all = nn.MultiheadAttention(em_size, 4, dropout=dropout).to(device)#, batch_first=True).to(device)
        self.merge_output = nn.MultiheadAttention(em_size, 4, dropout=dropout).to(device)#, batch_first=True).to(device)

        hidden_size = em_size

        self.output = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 2),
            nn.Sigmoid(),
        ).to(device)

        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())

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
        #data[:, 3] = self.mms.fit_transform(data[:, 3].reshape(-1, 1)).reshape(1, -1)

        self.train_data, self.test_data = train_test_split(data, test_size=0.2, random_state=2022)

        self.train_loader = DataLoader(MetaDataSet(self.train_data), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(MetaDataSet(self.test_data), batch_size=batch_size, shuffle=True)
        self.batch_size = batch_size

    def construct_input(self, x, device, component_sample_num=300):
        model_type_q = []
        model_type_k = []
        model_component_q = []
        model_component_k = []
        max_component_num = 0
        for key in x:
            model_components, model_type = self.solution_converter.get_all_idx(key)
            model_type_q.append(model_type)
            model_types = set(range(self.mo_num))
            model_types.remove(model_type)

            model_type_k.append(list(model_types))
            if len(model_components) > max_component_num:
                max_component_num = len(model_components)
            model_component_q.append(torch.tensor(model_components))
        component_sample_num = min((component_sample_num/max_component_num)*max_component_num, int(self.co_num/max_component_num)*max_component_num)
        for key in x:
            component_k_ = []
            while len(component_k_) != component_sample_num:
                c = random.randint(0, self.co_num - 1)
                if c in model_components:
                    continue
                else:
                    component_k_.append(c)
            model_component_k.append(torch.tensor(component_k_))
        def convert(list_):
            return torch.tensor(list_).to(device).long()
        model_type_q = convert(model_type_q)
        model_type_k = convert(model_type_k)
        model_component_q = nn.utils.rnn.pad_sequence(model_component_q, batch_first=True).to(device).long()
        model_component_k = nn.utils.rnn.pad_sequence(model_component_k, batch_first=True).to(device).long()

        model_component_q_pos = torch.tensor(list(range(max_component_num))).to(device).long().unsqueeze(0)
        model_component_k_pos = torch.tensor(list(range(component_sample_num))).to(device).long().unsqueeze(0)
        model_type_q = self.Memb(model_type_q)
        model_type_k = self.Memb(model_type_k)
        model_component_q = self.Cemb(model_component_q)
        model_component_k = self.Cemb(model_component_k)
        model_component_q_pos = self.Pemb(model_component_q_pos)
        model_component_k_pos = self.Pemb(model_component_k_pos)

        model_component_q = model_component_q + model_component_q_pos * 0.1
        model_component_k = model_component_k + model_component_k_pos * 0.1
        return model_type_q, model_type_k, model_component_q, model_component_k

    def forward(self, dataset_id, solution_emb):
        dt_emb = self.Demb(dataset_id) 
        solution_dt_emb = torch.cat([dt_emb, solution_emb], dim=1)
        #solution_dt_emb, _ = self.merge_output(solution_emb.unsqueeze(1), dt_emb.unsqueeze(1), dt_emb.unsqueeze(1))
        #solution_dt_emb = solution_dt_emb.squeeze(1)
        predict_auc_loss = self.output(solution_dt_emb).to(torch.float32) #(B, 2)
        return predict_auc_loss

    def get_solution_embedding(self, use_attn, target_nodes, device):
        '''
        :param x: (B, 1)
        :return: predict_auc: (B, 2)
        '''
        if use_attn:
            model_type_q, model_type_k, model_component_q, model_component_k = self.construct_input(target_nodes, device)
            model_type_emb, _ = self.multihead_atom_type(model_type_q.unsqueeze(1), model_type_k, model_type_k)
            #component_emb, _ = self.multihead_component(model_component_q, model_component_k, model_component_k)
            model_emb = self.Semb(target_nodes)
            try:
                model_component_emb, _ = self.merge_all(model_emb.unsqueeze(1), model_component_q, model_component_q)
                real_model_emb = model_emb.unsqueeze(1)
            except:
                model_component_emb, _ = self.merge_all(model_emb, model_component_q, model_component_q)
                real_model_emb = model_emb
            model_emb, _ = self.merge_all(model_component_emb, real_model_emb, real_model_emb)
            solution_emb = model_emb.squeeze(1)
            added_solution_info = None
        else:
            solution_emb = self.Semb(target_nodes)
            added_solution_info = None
        return solution_emb, added_solution_info

    def evaluate(self, test_loader):
        with torch.no_grad():
            pred_ans = []
            test_loss = 0
            for _, (di, si, y) in enumerate(test_loader):
                di = di.to(self.device)
                si = si.to(self.device)
                solution_emb, _ = self.get_solution_embedding(self.use_attn, si, self.device)
                y = y.to(self.device).to(torch.float32)

                y_pred = self(di, solution_emb)

                test_loss += self.loss_func(y_pred, y)

            test_loss /= len(test_loader)
            return test_loss

    def construct(self, epochs=10):
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
                    solution_emb, _ = self.get_solution_embedding(self.use_attn, si, self.device)
                    y = y.to(self.device).to(torch.float32)

                    y_pred = self(di, solution_emb)

                    loss = self.loss_func(y_pred, y)
                    totalloss += loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    #self.update_embeddings(self.gcn_flag, si, self.device, sample_num=sample_num)

            totalloss /= steps_per_epoch
            test_loss = self.evaluate(self.test_loader)
            print(f"epoch-{epoch+1}, loss={totalloss}, test_loss={test_loss}")
    


if __name__ == '__main__':
    # model_code = [('linear', 2, 'sigmoid'), 'cat', ('linear', 1, 'sigmoid')]
    meta_predicter = MetaPredictor(device='cpu', solution_converter=SolutionConverter(), meta_path='./meta_full.json', use_attn=True)
    meta_predicter.construct(epochs=50)

    torch.save(meta_predicter, 'Meta.model')







