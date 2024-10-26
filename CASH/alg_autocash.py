import sys
import logging
import shutil
import glob
import time
import math
import torch
import copy
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import random
import json
from cash_evaluation import CTREvaluation
from final_evaluation import FinalCTREvaluation
from alg_random import Random
from autocash.converter import SolutionConverter

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))), "CTRAPI"))
from pruned_search_space import BaseHPs, SearchSpace, ModelUse, index_solutions_prn_init, init_hp_dict_prn, cal_SearchSpace_size, index_solutions_prn
from log import *
from autocash.MetaLearning import MetaPredictor, MetaDataSet


class AutoCASH(Random):
    def __init__(self, args):
        if args.cluster_num < 1:
            raise ValueError('illegal cluster_num')
        if args.retain_metapredictorepochs < -1:
            raise ValueError('illegal retain_metapredictorepochs')

        self.space = [BaseHPs, SearchSpace, ModelUse]
        self.model_visited_num = {}
        for model_name in ModelUse.keys():
            self.model_visited_num[model_name] = 0
        self.model2id, self.id2model, self.allmodelnum = index_solutions_prn_init(reindex=args.reindex)

        self.environment_setting(args, 'AutoCASH')
        if args.device == -1:
            device_name = "cpu"
        else:
            device_name = "cuda:" + str(args.device)
        self.evaluator = CTREvaluation(args.search_epochs, args.full_data_name, args.epochs, device_name,
                                       sample_ratio=args.sample_ratio, logger=self.logger)

        self.search_epochs = args.search_epochs
        self.solution_epochs = args.epochs
        self.sample_num = args.sample_num
        self.args = args

        self.runcount = int(self.search_epochs / self.solution_epochs)

        # self.ks_times = args.ks_times
        self.meta_ratio = args.meta_ratio
        # self.cluster_num = args.cluster_num
        # self.cluster_epoch = args.cluster_epoch
        self.finetune_epoch = args.finetune_epoch
        base_dir = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))), "CASH")
        if self.meta_ratio == 1.0:
            meta_ratio_name = ""
        else:
            meta_ratio_name = "_"+str(self.meta_ratio)
        self.gcn_flag = False if args.gcn_flag=='False' else True 
        if self.gcn_flag:
            gcn_name = ""
        else:
            gcn_name = "_nogcn"
        if args.retain_metapredictorepochs > 0:
            meta_predicter = MetaPredictor(device=device_name, meta_path=base_dir+'/autocash/meta_full.json',
                                           gcn_flag=self.gcn_flag, meta_ratio=self.meta_ratio)
            meta_predicter.construct(epochs=args.retain_metapredictorepochs, sample_num=args.sample_num)
            torch.save(meta_predicter, base_dir+'/autocash/Meta'+meta_ratio_name+gcn_name+'.model')
        if args.retain_metapredictorepochs == 0:
            self.Meta = MetaPredictor(device=device_name, meta_path=base_dir + '/autocash/meta_full.json',
                                           gcn_flag=self.gcn_flag, meta_ratio=self.meta_ratio)
        else:
            self.Meta = torch.load(base_dir+'/autocash/Meta'+meta_ratio_name+gcn_name+'.model', map_location=device_name)
        self.data_name_index = {
            'AmazonElectronics': 0,
            'AmazonClothing_Shoes_and_Jewelry': 1,
            'AmazonPet_Supplies': 2,
            'AmazonMovies_and_TV': 3,
            'AmazonSports_and_Outdoors': 4,
            'AmazonToys_and_Games': 5,
            'AmazonVideo_Games': 6,
            #'MovieLens': 7,
        }
        self.data_name_index[args.full_data_name] = 7
        self.finetune_infonum = args.finetune_infonum
        #self.gcn_flag = False if args.gcn_flag=='False' else True 
        self.logger.info(f'--- self.gcn_flag: {self.gcn_flag}')
        self.logger.info(f'--- self.gcn_flag==False: {self.gcn_flag==False}')
        self.solution_randomelpsion = args.solution_randomelpsion

        self.device = device_name
        return

    def get_batch_NewInfo(self, NewInfo, batch_size):
        latest_NewInfo = NewInfo[-2:]
        NewInfo = sorted(list(NewInfo), key=lambda item: item[1], reverse=False)
        if batch_size == -1:
            return list(NewInfo)
        if len(NewInfo) > batch_size:
            batch_NewInfo = latest_NewInfo
            batch_NewInfo += NewInfo[int(batch_size*0.5)*(-1)+2:]
            batch_NewInfo += random.sample(NewInfo[:int(batch_size*0.5)*(-1)], batch_size-len(batch_NewInfo))
        else:
            batch_NewInfo = list(NewInfo)
        return batch_NewInfo

    def prepare_newinfo_dataloader(self, NewInfo, dataset, batch_size):
        if len(NewInfo) == 0:
            return None, None
        dataset_id = self.data_name_index[dataset]
        ds_newinfo = [([dataset_id] + t) for t in NewInfo]
        new_data = np.array(ds_newinfo)
        new_data[:, 3] = new_data[:, 3].reshape(1, -1)
        #new_data[:, 3] = self.Meta.mms.transform(new_data[:, 3].reshape(-1, 1)).reshape(1, -1)
        # train_data, test_data = train_test_split(new_data, test_size=0.2, random_state=2022) \
        #    if val else (new_data, None)
        train_loader = DataLoader(MetaDataSet(new_data), batch_size=batch_size, shuffle=True)
        #test_loader = DataLoader(MetaDataSet(test_data), batch_size=batch_size, shuffle=True) if val else None

        return train_loader

    def train_new_dataset(self, NewInfo, dataset, epochs=1, batch_size=32):
        # construct new input data with NewInfo, add the datset_id into it
        batch_NewInfo = self.get_batch_NewInfo(NewInfo, self.finetune_infonum)
        train_loader = self.prepare_newinfo_dataloader(batch_NewInfo, dataset, batch_size=batch_size)
        if None in train_loader or len(NewInfo) == 0:
            return None
        #print(train_loader)

        self.logger.info('--- training new data embedding...')
        all_params = [
            {
                "params":[p for n, p in self.Meta.named_parameters() if "Demb" not in n ],
                'lr': 5e-4
            },
            {
                "params":[p for n, p in self.Meta.named_parameters() if "Demb" in n],
                'lr': 1e-3
            }
        ]
        newds_optimizer = torch.optim.Adam(all_params)

        # steps_per_epoch = len(train_data) // batch_size + 1
        # self.logger.info(f"{len(train_data)} samples in total, {steps_per_epoch} steps per epoch")
        for epoch in range(epochs):
            totalloss = 0
            # with tqdm(enumerate(train_loader)) as t:
            for _, (di, si, y) in enumerate(train_loader):
                di = di.to(self.device)
                si = si.to(self.device)
                #solution_emb = self.Meta.Semb(si)
                solution_emb, _ = self.Meta.get_solution_embedding(self.gcn_flag, si, self.device, sample_num=self.sample_num)
                y = y.to(self.device).to(torch.float32)
                y_pred = self.Meta(di, solution_emb)
                loss = self.Meta.loss_func(y_pred, y)
                #totalloss += loss
                newds_optimizer.zero_grad()
                loss.backward()
                newds_optimizer.step()
                self.Meta.update_embeddings(self.gcn_flag, si, self.device, sample_num=self.sample_num)

                solution_emb, _ = self.Meta.get_solution_embedding(False, si, self.device, sample_num=self.sample_num)
                y = y.to(self.device).to(torch.float32)
                y_pred = self.Meta(di, solution_emb)
                loss = self.Meta.loss_func(y_pred, y)
                totalloss += loss
                newds_optimizer.zero_grad()
                loss.backward()
                newds_optimizer.step()

            totalloss /= len(train_loader)
            self.logger.info(f"--- epoch-{epoch}, loss={totalloss}")
            # if val:
            #     test_loss = self.Meta.evaluate(train_loader)
            #     self.logger.info(f"--- test_loss={test_loss}")

        self.logger.info('dataset training finished')

        with torch.no_grad():
            all_pred_ans, all_solution_indexs = [], []
            for _, (di, si, y) in enumerate(train_loader):
                di = di.to(self.device)
                si = si.to(self.device)
                solution_emb, all_solution_info = self.Meta.get_solution_embedding(self.gcn_flag, si, self.device, sample_num=self.sample_num)
                all_solution_emb, all_solution_index = all_solution_info

                #print(di.shape)
                di = di[0].repeat(all_solution_emb.shape[0])
                #print(all_solution_emb.shape)
                #print(di.shape)
                all_y_pred = self.Meta(di, all_solution_emb)
                all_pred_ans.append(all_y_pred)
                all_solution_indexs += all_solution_index

            all_pred_ans = torch.cat(all_pred_ans, dim=0)
            values, indices = all_pred_ans.topk(10, largest=True, sorted=True, dim=0)
            top10_indices = indices[:,0]
            #print(indices)
            #print(top10_indices)
            top10_solutions = [all_solution_indexs[item.item()] for item in top10_indices]
            #print(top10_solutions)
            self.logger.info(f'all_pred_ans[:10]: {all_pred_ans[:10]}')
            self.logger.info(f'all_solution_indexs[:10]: {all_solution_indexs[:10]}')
            self.logger.info(f'top10_pred_ans: {values}')
            self.logger.info(f'top10_indices: {top10_indices}')
            self.logger.info(f'top10_solutions: {top10_solutions}')

        return top10_solutions


    # def _cal_km_dis(self, solution_embs, center_embs, center_Mws, dataset_id):
    #     '''
    #     :param solution_embs: tensor(B, E)
    #     :param center_embs: tensor(cluster_num, E)
    #     :param center_Mws: tensor(cluster_num, 1)
    #     :param dataset_id: int
    #     :return: solution_dis: tensor(B, cluster_num)
    #     '''
    #     center_embs_chunks = torch.chunk(center_embs, self.cluster_num, dim=0) #(1, E)
    #     center_Mws_chunks = torch.chunk(center_Mws, self.cluster_num, dim=0) #(1, 1)

    #     solutions_dis = []
    #     # tensor(B, cluster_num)

    #     for i, center_embs_slide in enumerate(center_embs_chunks):
    #         dis_former = (solution_embs - center_embs_slide) / center_embs.size()[1] #(B, E)
    #         dis_former = torch.norm(dis_former, p=2, dim=1).unsqueeze(1) #(B, 1)

    #         dataset_ids = torch.tensor([dataset_id] * solution_embs.size()[0]).to(self.device) #(B)
    #         dis_latter = self.Meta(dataset_ids, solution_embs)[:, 0].view(-1, 1) #(B, 1)
    #         dis_latter -= center_Mws_chunks[i] #(B, 1)

    #         dis = dis_former + torch.abs(dis_latter) #(B, 1)
    #         solutions_dis.append(dis)

    #     solutions_dis = torch.cat(solutions_dis, dim=1)

    #     return solutions_dis

    # def _init_center_info(self, center_ids, dataset_id):
    #     '''
    #     get the embedding, and cal the Mw in the Meta for a center_id
    #     :param center_ids: list(int)
    #     :param dataset_id: int
    #     :return: (center_embs: tensor(cluster_num, E),
    #                 center_Mws: tensor(cluster_num, 1))
    #     '''
    #     center_ids = torch.tensor(center_ids).view(-1).to(self.device)
    #     # tensor(cluster_num)
    #     center_embs = self.Meta.Semb(center_ids)
    #     #with torch.no_grad():
    #     #    center_embs, _ = self.Meta.get_solution_embedding(self.gcn_flag, center_ids, self.device)
    #     #    # tensor(cluster_num, E)

    #     #    dataset_id = torch.tensor([dataset_id]*self.cluster_num).to(self.device)
    #     center_Mws = self.Meta(dataset_id, center_embs)[:, 0].view(-1, 1)
    #     # tensor(cluster_num, 1)
    #     return (center_embs, center_Mws)

    # def k_means(self, dataset, st_batch_size=512):
    #     '''
    #     :return: list(set(int))
    #     '''
    #     st_num = cal_SearchSpace_size()
    #     if self.cluster_num == 1:
    #         clusters = [set(range(st_num))]
    #         return clusters

    #     ks_times = self.ks_times
    #     center_ids = random.sample(range(st_num), self.cluster_num)
    #     dataset_id = self.data_name_index[dataset]

    #     solutions = torch.tensor(range(st_num)).view(-1, 1).to(self.device)
    #     # tensor(st_num, 1)
    #     center_embs, center_Mws = self._init_center_info(center_ids, dataset_id)
    #     # center_embs: tensor(cluster_num, E)
    #     # center_Mws: tensor(cluster_num, 1)

    #     clusters = [{c} for c in center_ids]

    #     st_batch_num = (st_num - 1) // st_batch_size + 1
    #     batch_begin_ends = [(i*st_batch_size, (i+1)*st_batch_size)
    #                         for i in range(st_batch_num-1)]
    #     batch_begin_ends.append(((st_batch_num-1)*st_batch_size, st_num))

    #     total_batches = st_batch_num * ks_times
    #     pbar = tqdm(desc=f'Being clustered, {ks_times} times, {total_batches}its in total')

    #     for ks_time in range(ks_times):
    #         for _, (s, t) in enumerate(batch_begin_ends):  # meta_record
    #             st_batch = solutions[s:t, :].squeeze()
    #             # tensor(B)
    #             st_embs = self.Meta.Semb(st_batch)
    #             #with torch.no_grad():
    #             #    st_embs, _ = self.Meta.get_solution_embedding(self.gcn_flag, st_batch, self.device)
    #             # tensor(B, E)

    #             dis_clusters = self._cal_km_dis(st_embs, center_embs, center_Mws, dataset_id)
    #             # tensor(B, cluster_num)

    #             min_cluster_ids = torch.argmin(dis_clusters, dim=1) #(B)
    #             for i, clu_id in enumerate(min_cluster_ids.tolist()):
    #                 st_id = st_batch[i].item()
    #                 clusters[clu_id].add(st_id)

    #             pbar.update(1)

    #         #    ?          
    #         center_embs = []
    #         center_Mws = []
    #         for c_id, cluster in enumerate(clusters):
    #             #cluster: set(st_ids)
    #             cluster_solutions = torch.tensor(list(cluster)).view(-1, 1).to(self.device)
    #             # tensor(cluster_size, 1)
    #             #print("ok")
    #             cluster_embs = self.Meta.Semb(cluster_solutions).squeeze(1)
    #             cluster_Mws = self.Meta(dataset_ids, cluster_embs)[:, 0].view(-1, 1)
    #             #print(cluster_embs.shape)
    #             #print(self.Meta.Semb(cluster_solutions).shape)

    #             #batch_size = 256
    #             #cluster_embs, cluster_Mws = [], []
    #             #for idx in range((len(cluster_solutions) - 1) // batch_size + 1):
    #             #    cluster_solutions_tmp = cluster_solutions[idx * batch_size: min(idx * batch_size + batch_size, len(cluster_solutions))]
    #             #    with torch.no_grad():
    #             #        cluster_embs_tmp, _ = self.Meta.get_solution_embedding(self.gcn_flag, cluster_solutions_tmp, self.device)
    #             #        cluster_embs_tmp = cluster_embs_tmp.squeeze()
    #             #       dataset_ids = torch.tensor([dataset_id] * len(cluster)).to(self.device)
    #             #        cluster_Mws_tmp = self.Meta(dataset_ids, cluster_embs_tmp)[:, 0].view(-1, 1)
    #             #        cluster_embs.append(cluster_embs_tmp.cpu())
    #             #        cluster_Mws.append(cluster_Mws_tmp.cpu())
    #             #cluster_embs = torch.cat(cluster_embs, dim = 0)
    #             #cluster_Mws = torch.cat(cluster_Mws, dim = 0)
    #             # tensor(cluster_size, 1)

    #             center_embs.append(torch.mean(cluster_embs, dim=0).view(1, -1))
    #             # tensor(1, E)
    #             center_Mws.append(torch.mean(cluster_Mws, dim=0).view(1, -1))
    #             # tensor(1, 1)
    #         center_embs = torch.cat(center_embs, dim=0)
    #         center_Mws = torch.cat(center_Mws, dim=0)

    #     return clusters

    # def cal_UCB(self, NewInfo, clusters, epoch, dataset):
    #     '''
    #     :param clusters: list(list(int))
    #     :return: tensor(cluster_num)
    #     '''
    #     if len(NewInfo) == 0 or self.cluster_num == 1:
    #         ucb_scores = [1 / self.cluster_num] * self.cluster_num
    #         ucb_scores = torch.tensor(ucb_scores).to(self.device)
    #         return ucb_scores

    #     dataset_id = self.data_name_index[dataset]

    #     ucb_scores = []
    #     for cluster in clusters: #t ()
    #         # sample_num = len(cluster)
    #         # solution_ids = torch.tensor(list(cluster)).view(-1, 1).to(self.device)
    #         # # tensor(sample_num, 1)
    #         # solution_embs = self.Meta.Semb(solution_ids).squeeze()
    #         # # tensor(sample_num, E)
    #         # dataset_ids = torch.tensor([dataset_id] * sample_num).to(self.device) #tensor(sn)
    #         # solution_Mws = self.Meta(dataset_ids, solution_embs)[:, 0].view(-1, 1)
    #         # # tensor(sample_num, 1)
    #         # aver_P = torch.mean(solution_Mws, dim=0).squeeze().item() #float

    #         evluated_solutions = []
    #         for info in NewInfo:
    #             if info[0] in list(cluster):
    #                 evluated_solutions.append(info)
    #         sample_num = len(evluated_solutions)

    #         if sample_num == 0:
    #             ucb_scores.append(0.5)
    #         else:
    #             aver_P = np.mean([st[1] for st in evluated_solutions])
    #             score = math.sqrt(2 * math.log2(epoch) / sample_num) + aver_P
    #             ucb_scores.append(score)
    #     ucb_scores = torch.tensor(ucb_scores).to(self.device)
    #     softmax_func = nn.Softmax(dim=0).to(self.device)
    #     return softmax_func(ucb_scores)

    # def get_best_sl_incluster(self, cluster, dataset_id):
    #     '''
    #         calculate the predict score using self.Meta for every record in cluster
    #     and choose the best
    #     :param cluster: set(int)
    #     :return best_solution_id: int
    #     :return best_solution_auc: float
    #     :return best_solution_logloss: float
    #     '''
    #     solution_ids = torch.tensor(list(cluster)).view(-1, 1).to(self.device)
    #     # tensor(C, 1)
    #     solution_embs = self.Meta.Semb(solution_ids).squeeze()
    #     dataset_ids = torch.tensor([dataset_id] * solution_embs.size()[0]).to(self.device) #(C)
    #     predict_auclosses = self.Meta(dataset_ids, solution_embs).view(-1, 2)

    #     #batch_size = 256
    #     #predict_auclosses = []
    #     #for idx in range((len(solution_ids) - 1) // batch_size + 1):
    #     #    solution_ids_tmp = solution_ids[idx * batch_size: min(idx * batch_size + batch_size, len(solution_ids))]
    #     #    with torch.no_grad():
    #     #        solution_embs, _ = self.Meta.get_solution_embedding(self.gcn_flag, solution_ids_tmp.squeeze(), self.device)
    #     #        # tensor(C, E)
    #     #        dataset_ids = torch.tensor([dataset_id] * solution_embs.size()[0]).to(self.device) #(C)
    #     #        predict_auclosses_tmp = self.Meta(dataset_ids, solution_embs).view(-1, 2)
    #     #        predict_auclosses.append(predict_auclosses_tmp.cpu())
    #     #predict_auclosses = torch.cat(predict_auclosses, dim=0)

    #     # tensor(C, 2)
    #     best_index = torch.argmax(predict_auclosses, dim=0)[0].item() #int
    #     best_solution_id = solution_ids[best_index].item()
    #     best_solution_auc, best_solution_logloss = predict_auclosses[best_index].tolist()

    #     return best_solution_id, best_solution_auc, best_solution_logloss

    # def get_sample_sl_incluster(self, cluster, dataset_id):
    #     '''
    #         calculate the predict score using self.Meta for every record in cluster
    #     and choose the best
    #     :param cluster: set(int)
    #     :return best_solution_id: int
    #     :return best_solution_auc: float
    #     :return best_solution_logloss: float
    #     '''
    #     solution_ids = torch.tensor(list(cluster)).view(-1, 1).to(self.device)
    #     # tensor(C, 1)
    #     solution_embs = self.Meta.Semb(solution_ids).squeeze()
    #     dataset_ids = torch.tensor([dataset_id] * solution_embs.size()[0]).to(self.device) #(C)
    #     predict_auclosses = self.Meta(dataset_ids, solution_embs).view(-1, 2)

    #     #batch_size = 256
    #     #predict_auclosses = []
    #     #for idx in range((len(solution_ids) - 1) // batch_size + 1):
    #     #    solution_ids_tmp = solution_ids[idx * batch_size: min(idx * batch_size + batch_size, len(solution_ids))]
    #     #    with torch.no_grad():
    #     #        solution_embs, _ = self.Meta.get_solution_embedding(self.gcn_flag, solution_ids_tmp, self.device)
    #     #        solution_embs = solution_embs.squeeze()
    #     #        # tensor(C, E)
    #     #        dataset_ids = torch.tensor([dataset_id] * solution_embs.size()[0]).to(self.device) #(C)
    #     #        predict_auclosses_tmp = self.Meta(dataset_ids, solution_embs).view(-1, 2)
    #     #        predict_auclosses.append(predict_auclosses_tmp.cpu())
    #     #        # tensor(C, 2)
    #     #predict_auclosses = torch.cat(predict_auclosses, dim=0)

    #     selected_index = torch.multinomial(predict_auclosses[:,0].view(-1),1)[0].item()
    #     selected_solution_id = solution_ids[selected_index].item()
    #     selected_solution_auc, selected_solution_logloss = predict_auclosses[selected_index].tolist()
    #     return selected_solution_id, selected_solution_auc, selected_solution_logloss

    #def get_random_solution(self):
    #    BaseHPs, SearchSpace, ModelUse = self.space
    #    model_name = random.choice(list(ModelUse.keys()))
    #    hp_dict = init_hp_dict_prn(model_name)
    #
    #    solution_best = index_solutions_prn(model_name, hp_dict)
    #    return solution_best, model_name, hp_dict

    def get_best_solution_(self):
        dataset_id = self.data_name_index[self.args.full_data_name]
        solution_ids = torch.tensor([i for i in range(self.Meta.st_num)]).view(-1, 1).to(self.device)
        # tensor(C, 1)
        solution_embs = self.Meta.Semb(solution_ids).squeeze()
        dataset_ids = torch.tensor([dataset_id] * solution_embs.size()[0]).to(self.device) #(C)
        predict_auclosses = self.Meta(dataset_ids, solution_embs).view(-1, 2)

        values, indices = predict_auclosses.topk(10, largest=True, sorted=True, dim=0)
        top10_indices = indices[:,0]
        #print(top10_indices)
        top10_solutions = [solution_ids[item.item()].item() for item in top10_indices]
        #print(top10_solutions)
        return top10_solutions

    def get_best_solution(self, NewInfo, evaluated_solutions):
        NewInfo = sorted(list(NewInfo), key=lambda item: item[1], reverse=True)
        NewInfo = random.sample(NewInfo[:10], min(5, len(NewInfo)))
        num = len(NewInfo)
        si = [item[0] for item in NewInfo]
        model_names = list(self.model_visited_num.keys())
        for i in range(num, 32):
            model_name = random.sample(model_names, 1)[0]
            hp_dict = init_hp_dict_prn(model_name)
            solution_best = index_solutions_prn(model_name, hp_dict)
            si.append(solution_best)
        si = torch.tensor(si)

        with torch.no_grad():
            all_pred_ans, all_solution_indexs = [], []
            if True:
                di = torch.tensor([self.data_name_index[self.args.full_data_name]]).to(self.device)
                si = si.to(self.device)
                solution_emb, all_solution_info = self.Meta.get_solution_embedding(self.gcn_flag, si, self.device, sample_num=self.sample_num)
                all_solution_emb, all_solution_index = all_solution_info

                #print(di.shape)
                di = di[0].repeat(all_solution_emb.shape[0])
                #print(all_solution_emb.shape)
                #print(di.shape)
                all_y_pred = self.Meta(di, all_solution_emb)
                all_pred_ans.append(all_y_pred)
                all_solution_indexs += all_solution_index

            all_pred_ans = torch.cat(all_pred_ans, dim=0)

        select = random.random()
        if self.fail_num <= 10 or select >= 0.5:
            self.logger.info(f'--- top 20, fail_num: {self.fail_num}, select: {select}')
            values, indices = all_pred_ans.topk(20, largest=True, sorted=True, dim=0)
        elif self.fail_num <= 30 or select >= 0.2:
            self.logger.info(f'--- top 60, fail_num: {self.fail_num}, select: {select}')
            values, indices = all_pred_ans.topk(60, largest=True, sorted=True, dim=0)
        else:
            self.logger.info(f'--- top 200, fail_num: {self.fail_num}, select: {select}')
            values, indices = all_pred_ans.topk(100, largest=True, sorted=True, dim=0)

        top10_indices = indices[:,0]
        top10_solutions = [all_solution_indexs[item.item()] for item in top10_indices]
        solution_best = random.sample(top10_solutions, 1)[0]
        try_num = 1
        while solution_best in evaluated_solutions:
            solution_best = random.sample(top10_solutions, 1)[0]
            try_num += 1
            if try_num >= 20:
                break
        if try_num >= 20:
            solution_best = random.sample(all_solution_indexs, 1)[0]
            self.logger.info(f'--- ... Replacing random solution_best-{solution_best}')
        elif try_num > 1:
            self.logger.info(f'--- ... Replacing top solution_best-{solution_best}')
        return solution_best, None, None

    def get_random_solution_(self, NewInfo):
        #BaseHPs, SearchSpace, ModelUse = self.space
        #model_name = random.choice(list(ModelUse.keys()))
        good_models = ['PNN','DIEN']
        model_names = list(self.model_visited_num.keys())
        model_visited_num_list = [self.model_visited_num[model_names[i]] for i in range(len(model_names))]
        self.logger.info(f'--- model_visited_num_list: {model_visited_num_list}')

        if random.random() > 0.5 and len(NewInfo) > 0:
            max_visited_num = max(model_visited_num_list)
            model_visited_num_list = [max_visited_num-model_visited_num_list[i] for i in range(len(model_names))]
            model_name_index = torch.multinomial(torch.tensor(model_visited_num_list).float(),1)[0].item()
            model_name = model_names[model_name_index]
        else:
            model_name = random.sample(good_models, 1)[0]
        self.logger.info(f'--- random chosen model_name: {model_name}')

        hp_dict = init_hp_dict_prn(model_name)

        solution_best = index_solutions_prn(model_name, hp_dict)
        return solution_best, model_name, hp_dict

    def get_random_solution(self, NewInfo, evaluated_solutions):
        si = []
        model_names = list(self.model_visited_num.keys())
        for i in range(32):
            model_name = random.sample(model_names, 1)[0]
            hp_dict = init_hp_dict_prn(model_name)
            solution_best = index_solutions_prn(model_name, hp_dict)
            si.append(solution_best)
        si = torch.tensor(si)

        with torch.no_grad():
            all_pred_ans, all_solution_indexs = [], []
            if True:
                di = torch.tensor([self.data_name_index[self.args.full_data_name]]).to(self.device)
                si = si.to(self.device)
                solution_emb, all_solution_info = self.Meta.get_solution_embedding(self.gcn_flag, si, self.device, sample_num=self.sample_num)
                all_solution_emb, all_solution_index = all_solution_info

                #print(di.shape)
                di = di[0].repeat(all_solution_emb.shape[0])
                #print(all_solution_emb.shape)
                #print(di.shape)
                all_y_pred = self.Meta(di, all_solution_emb)
                all_pred_ans.append(all_y_pred)
                all_solution_indexs += all_solution_index

            all_pred_ans = torch.cat(all_pred_ans, dim=0)
            values, indices = all_pred_ans.topk(90, largest=True, sorted=True, dim=0)
            top10_indices = indices[:,0]
            top10_solutions = [all_solution_indexs[item.item()] for item in top10_indices]
        if random.random() < 0.33:
            solution_best = random.sample(top10_solutions[:30], 1)[0]
            self.logger.info(f'--- random 0.33 solution_best-{solution_best}')
        elif random.random() < 0.66:
            solution_best = random.sample(top10_solutions[30:60], 1)[0]
            self.logger.info(f'--- random 0.66 solution_best-{solution_best}')
        else:
            solution_best = random.sample(top10_solutions[60:], 1)[0]
            self.logger.info(f'--- random 1.0 solution_best-{solution_best}')
        return solution_best, None, None

    # def fine_tune(self, NewInfo, dataset, epochs=5, batch_size=32):
    #     batch_NewInfo = self.get_batch_NewInfo(NewInfo, self.finetune_infonum)
    #     train_loader = self.prepare_newinfo_dataloader(batch_NewInfo, dataset, batch_size=batch_size)
    #     self.logger.info('--- fine tuning Meta...')
    #     other_params = [
    #         {
    #             "params":[p for n, p in self.Meta.named_parameters() if "Demb" not in n ],
    #             'lr': 5e-4
    #         },
    #         {
    #             "params":[self.Meta.Demb.parameters()],
    #             'lr': 1e-3
    #         }
    #     ]
    #     optimizer = torch.optim.Adam(other_params)
    #     for epoch in range(epochs):
    #         totalloss = 0
    #         # with tqdm(enumerate(train_loader)) as t:
    #         for _, (di, si, y) in enumerate(train_loader):
    #             #print([di,si,y])
    #             di = di.to(self.device)
    #             si = si.to(self.device)
    #             #solution_emb = self.Meta.Semb(si)
    #             solution_emb, _ = self.Meta.get_solution_embedding(self.gcn_flag, si, self.device, sample_num=self.sample_num)
    #             y = y.to(self.device).to(torch.float32)

    #             y_pred = self.Meta(di, solution_emb)

    #             loss = self.Meta.loss_func(y_pred, y)
    #             totalloss += loss

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #         totalloss /= len(train_loader)
    #         self.logger.info(f"--- epoch-{epoch}, loss={totalloss}")

    #     self.logger.info('--- Meta fine tuning finished')

    def get_autoctr_solution(self, top_k=1):
        NewInfo = [] #[(solution_id, test_auc, test_logloss), ...]
        #clusters = [[]] * self.cluster_num
        dataset = self.args.full_data_name
        evaluated_solutions = []
        top10_solutions = None
        self.fail_num, self.best_auc = 0, 0

        for epoch in range(1, self.runcount+1):
            if epoch >= 10:
                top10_solutions = self.train_new_dataset(NewInfo, dataset)

            # if (epoch - 1) % self.cluster_epoch == 0:
            #     self.logger.info(f'--- epoch-{epoch}, clustering')
            #     with torch.no_grad():
            #         clusters = self.k_means(dataset)
            #         ucb_scores = self.cal_UCB(NewInfo, clusters, epoch, dataset)

            # print("ok")
            # print(ucb_scores)
            #cluster_best = random.choices(clusters, weights=ucb_scores, k=1)[0]
            if epoch < 10:
                real_solution_randomelpsion = 1.0
            # # # # #elif epoch < 20:
            # # # # #    real_solution_randomelpsion = 0.5
            else:
                real_solution_randomelpsion = self.solution_randomelpsion
            if random.random() < real_solution_randomelpsion:
                self.logger.info('--- Random Solution Selected')
                solution_best, model_name, hp_dict = self.get_random_solution(NewInfo, evaluated_solutions)
                evaluated_solutions.append(solution_best)
                self.logger.info(f'--- ... solution_best-{solution_best}')

                solution_dict = copy.deepcopy(self.id2model[solution_best])
                model_name = solution_dict['model_name']
                del solution_dict['model_name']
                hp_dict = solution_dict
            else:
                # cluster_best_index = torch.multinomial(ucb_scores.float().cpu(),1)[0].item()
                # cluster_best = clusters[cluster_best_index]
                self.logger.info('--- Best Solution Selected')
                solution_best, _, _ = self.get_best_solution(NewInfo, evaluated_solutions)
                # if random.random() < 0.3 or top10_solutions == None:
                #     top10_solutions = list(all_top10_solutions)
                #     self.logger.info('--- Best Solution Selected flag: all best')
                # else:
                #     self.logger.info('--- Best Solution Selected flag: batch best')
                self.logger.info(f'--- ... solution_best-{solution_best}')

                # if solution_best in evaluated_solutions:
                #     solution_best = None
                #     for index in range(1, len(top10_solutions)):
                #         if top10_solutions[index] not in evaluated_solutions:
                #             solution_best = top10_solutions[index]

                #     if solution_best == None:
                #         solution_best, model_name, hp_dict = self.get_random_solution(NewInfo)

                #     self.logger.info(f'--- ... Replacing solution_best-{solution_best}')
                evaluated_solutions.append(solution_best)

                solution_dict = copy.deepcopy(self.id2model[solution_best])
                model_name = solution_dict['model_name']
                del solution_dict['model_name']
                hp_dict = solution_dict
                
            auc, logloss, _, _ = self.evaluator.evaluate_single_model(model_name=model_name, hp_dict=hp_dict)
            NewInfo.append([solution_best, auc, min(logloss, 1)])
            self.model_visited_num[model_name] += 1

            if auc > self.best_auc:
                self.fail_num, self.best_auc = 0, auc
            else:
                self.fail_num += 1

            #if epoch % self.finetune_epoch == 0:
            #    self.logger.info(f'--- epoch-{epoch}, finetuning')
            #    self.fine_tune(NewInfo, dataset)

    def main(self):
        start_time = time.time()

        self.logger.info('\n\nAutoCASH Search Begin ......')
        self.get_autoctr_solution()
        self.logger.info('AutoCASH Search End ......')

        self.logger.info('\n\nTop10 Final Evaluation Begin ......')
        top10_solutions = self.evaluator.top10_solutions
        self.logger.info('+ top10_solutions: %s', str(top10_solutions))
        for i in range(len(top10_solutions)):
            self.logger.info('\n\n@@@ Top %d solution', i + 1)
            self.logger.info('@@@ model_name: %s, hp_dict: %s', str(top10_solutions[i][0]), str(top10_solutions[i][1]))
            final_auc, final_logloss, file_position = self.final_evaluation(top10_solutions[i])
            self.logger.info('@@@ model_name: %s, hp_dict: %s', str(top10_solutions[i][0]), str(top10_solutions[i][1]))
            self.logger.info('@@@ final_auc: %.4f, final_logloss: %.4f', final_auc, final_logloss)
            self.logger.info('@@@ file_position: %s', file_position)
        self.logger.info('Top10 Final Evaluation End ......')
        return
