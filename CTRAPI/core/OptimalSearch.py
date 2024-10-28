import pickle
import sys

import numpy as np
import torch
import math
import json
import torch.nn as nn
from tqdm import tqdm
import random
import heapq

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from MetaLearning import MetaPredictor, MetaDataSet
from RandomSearch import data_name_index
from log import set_logger
from pruned_search_space import cal_SearchSpace_size



class AutoCTR:
    def __init__(self, data_name_index, Meta=None, logger=None, device='cpu'):
        self.epochs = 100
        self.cluster_epoch = 10
        self.cluster_num = 10
        self.finetune_epoch = 20
        self.device = device

        self.data_name_index = data_name_index
        self.Meta = Meta

        self.logger = logger

        # with open('../meta.json', 'r') as f:
        #     self.meta_info = json.load(f)

    def prepare_newinfo_dataloader(self, NewInfo, dataset, val, batch_size):
        if len(NewInfo) == 0:
            return None, None
        dataset_id = self.data_name_index[dataset]
        ds_newinfo = [([dataset_id] + t) for t in NewInfo]
        new_data = np.array(ds_newinfo)
        mms = MinMaxScaler(feature_range=(0, 1))
        new_data[:, 3] = mms.fit_transform(new_data[:, 3].reshape(-1, 1)).reshape(1, -1)
        train_data, test_data = train_test_split(new_data, test_size=0.2, random_state=2022) \
            if val else (new_data, None)
        train_loader = DataLoader(MetaDataSet(train_data), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(MetaDataSet(test_data), batch_size=batch_size, shuffle=True) if val else None

        return train_loader, test_loader

    def train_new_dataset(self, NewInfo, dataset, epochs=1, batch_size=32, val=False):
        # construct new input data with NewInfo, add the datset_id into it
        train_loader, test_loader = self.prepare_newinfo_dataloader(NewInfo, dataset, val, batch_size)
        if train_loader is None:
            return

        self.logger.info('training new data embedding...')
        newds_optimizer = torch.optim.Adam(self.Meta.Demb)

        # steps_per_epoch = len(train_data) // batch_size + 1
        # self.logger.info(f"{len(train_data)} samples in total, {steps_per_epoch} steps per epoch")
        for epoch in range(epochs):
            totalloss = 0
            with tqdm(enumerate(train_loader)) as t:
                for _, (x, y) in t:
                    x = x.long().to(self.device)
                    y = y.to(self.device).to(torch.float32)

                    y_pred = self.Meta(x)

                    loss = self.Meta.loss_func(y_pred, y)
                    totalloss += loss

                    newds_optimizer.zero_grad()
                    loss.backward()
                    newds_optimizer.step()
            totalloss /= len(train_loader)
            self.logger.info(f"epoch-{epoch}, loss={totalloss}")
            if val:
                test_loss = self.Meta.evaluate(test_loader)
                self.logger.info(f"test_loss={test_loss}")

        self.logger.info('dataset training finished')

    def _cal_km_dis(self, solution_emb, center_info, dataset_id):
        '''
        :param solution_id: tensor(E)
        :param center_info: tuple(center_emb: tensor(E), center_Mw: float)
        :param dataset_id: int
        :return:
        '''
        center_emb, center_Mw = center_info

        dis = torch.norm(((solution_emb - center_emb)/center_emb.size()[0]).unsqueeze(0),
                         p=2, dim=1).item()

        dataset_id = torch.tensor([dataset_id]).to(self.device)
        solution_emb = solution_emb.unsqueeze(0)
        center_emb = center_emb.unsqueeze(0)
        dis += math.fabs(self.Meta(dataset_id, solution_emb).squeeze()[0].item() -
                         center_Mw)
        return dis

    def _init_center_info(self, center_id, dataset_id):
        '''
        get the embedding, and cal the Mw in the Meta for a center_id
        :param center_id: int
        :param dataset_id: int
        :return: (center_emb: tensor(E), center_Mw: float)
        '''
        center_emb = self.Meta.Semb(torch.tensor(center_id).to(self.device))
        dataset_id = torch.tensor([dataset_id]).to(self.device)
        center_Mw = self.Meta(dataset_id, center_emb.unsqueeze(0)).squeeze()[0].item()
        return (center_emb, center_Mw)

    def _recal_center_info(self, cluster):
        '''
        cal the centre emb and Mw of the whole cluster
        :param cluster: set(int)
        :return: (center_emb: tensor(E), center_Mw: float)
        '''
        cluster_size = len(cluster)
        # semb_sum = torch.tensor([0]).to(self.device)  zeros


    def k_means(self, dataset, ks_times=1):
        '''
        :return: list(set(int))
        '''
        st_num = cal_SearchSpace_size()
        center_ids = random.sample(range(st_num), self.cluster_num)
        dataset_id = self.data_name_index[dataset]

        # center_embs = [self.Meta.Semb(torch.tensor(i).to(self.device))
        #                for i in center_ids]
        center_infos = [self._init_center_info(i, dataset_id) for i in center_ids]
        clusters = [{c} for c in center_ids]
        for ks_time in range(ks_times):
            # for st_id in tqdm(range(st_num),
            #                   desc=f'{ks_time}th cluster'):  # meta_record
            #     min_dis = math.inf
            #     min_cluster_id = -1
            #     st_emb = self.Meta.Semb(torch.tensor(st_id).to(self.device))
            #     for cluster_id, center_info in enumerate(center_infos):
            #         dis = self._cal_km_dis(st_emb, center_info, dataset_id)
            #         if dis < min_dis:
            #             min_dis = dis
            #             min_cluster_id = cluster_id
            #     clusters[min_cluster_id].add(st_id)
            # pickle.dump(clusters, open(f'clusters_{dataset}_{ks_time}.pkl', 'wb'))
            # TODO: 重新计算聚类中心
            clusters = pickle.load(open(f'clusters_{dataset}_{ks_time}.pkl', 'rb'))
            for c_id, cluster in enumerate(clusters):
                pass

        return clusters

    def cal_UCB(self, clusters, epoch):
        '''
        :param clusters: list(list(int))
        :return: list(val)
        '''
        ucb_scores = []
        for cluster in clusters: #t ()
            sample_num = len(cluster)
            score = sum([t[2] for t in cluster]) / sample_num #t[2]: test_auc
            score += math.sqrt(2 * math.log2(epoch) / sample_num)
            ucb_scores.append(score)
        return torch.tensor(ucb_scores).to(self.device)

    def get_best_sl_incluster(self, cluster, dataset_id):
        '''
            calculate the predict score using self.Meta for every record in cluster
        and choose the best
        '''
        best_sl = -1
        best_auc = 0
        for record in cluster:
            with torch.no_grad():
                predict_auc = self.Meta((dataset_id, record[1]))[0]
            if predict_auc > best_auc:
                best_sl = record[1]
                best_auc = predict_auc
        return best_sl, best_auc

    def fine_tune(self, NewInfo, dataset, epochs=1, batch_size=32):
        train_loader, test_loader = self.prepare_newinfo_dataloader(NewInfo, dataset,
                                                                    val=False, batch_size=batch_size)
        self.logger.info('fine tuning Meta...')
        optimizer = torch.optim.Adam(self.Meta.parameters())
        for epoch in range(epochs):
            totalloss = 0
            with tqdm(enumerate(train_loader)) as t:
                for _, (x, y) in t:
                    x = x.long().to(self.device)
                    y = y.to(self.device).to(torch.float32)

                    y_pred = self.Meta(x)

                    loss = self.Meta.loss_func(y_pred, y)
                    totalloss += loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            totalloss /= len(train_loader)
            self.logger.info(f"epoch-{epoch}, loss={totalloss}")

        self.logger.info('Meta fine tuning finished')

    def search(self, dataset):
        NewInfo = [] #[(solution_id, test_auc, test_logloss), ...]
        clusters = [[]] * self.cluster_num
        ucb_scores = [1 / self.cluster_num] * self.cluster_num

        for epoch in range(1, self.epochs+1):
            self.train_new_dataset(NewInfo, dataset)

            if epoch-1 % self.cluster_epoch == 0:
                with torch.no_grad():
                    clusters = self.k_means(dataset)
                    ucb_scores = self.cal_UCB(clusters, epoch)

            cluster_best = random.choices(clusters, weights=ucb_scores, k=1)[0]
            solution_best, best_auc = self.get_best_sl_incluster(cluster_best,
                                                       self.data_name_index(dataset))

            NewInfo.append((solution_best, best_auc))

            if epoch % self.finetune_epoch == 0:
                self.fine_tune(NewInfo, dataset)

        best_solution_id = heapq.nlargest(1, NewInfo, key=lambda x: x[1])[0]
        return best_solution_id



if __name__ == "__main__":
    # device = sys.argv[1]
    device = 'cpu'
    Meta = torch.load('Meta.model', map_location=device)
    logger = set_logger('AutoCTR', '', DIR='../')
    autoctr = AutoCTR(data_name_index, Meta=Meta, logger=logger, device=device)
    autoctr.search('MovieLens')
