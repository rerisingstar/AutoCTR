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
from autocash.MetaLearning import MetaPredictor, MetaDataSet

import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))), "CTRAPI"))
from pruned_search_space import BaseHPs, SearchSpace, ModelUse, index_solutions_prn_init, init_hp_dict_prn, cal_SearchSpace_size, index_solutions_prn
from log import *


class AutoCASH_NEW(Random):
    def __init__(self, args):
        if args.retain_metapredictorepochs < -1:
            raise ValueError('illegal retain_metapredictorepochs')

        self.space = [BaseHPs, SearchSpace, ModelUse]
        self.model_names = list(ModelUse.keys())
        self.model2id, self.id2model, self.allmodelnum = index_solutions_prn_init(reindex=args.reindex)

        self.environment_setting(args, 'AutoCASH_NEW')
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

        self.meta_ratio = args.meta_ratio
        self.finetune_epoch = args.finetune_epoch
        base_dir = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))), "CASH")
        if self.meta_ratio == 1.0:
            meta_ratio_name = ""
        else:
            meta_ratio_name = "_"+str(self.meta_ratio)
        self.gcn_flag = False if args.gcn_flag=='False' or args.gcn_flag==False else True 
        if self.gcn_flag:
            gcn_name = ""
        else:
            gcn_name = "_nogcn"
        if args.sample_num == 100:
            sample_name = ""
        else:
            sample_name = "_"+str(args.sample_num)+"samples"
        if args.retain_metapredictorepochs > 0:
            meta_predicter = MetaPredictor(device=device_name, meta_path=base_dir+'/autocash/meta_full.json',
                                           gcn_flag=self.gcn_flag, meta_ratio=self.meta_ratio)
            meta_predicter.construct(epochs=args.retain_metapredictorepochs, sample_num=args.sample_num)
            torch.save(meta_predicter, base_dir+'/autocash/Meta'+meta_ratio_name+gcn_name+sample_name+'.model')
        if args.retain_metapredictorepochs == 0:
            self.Meta = MetaPredictor(device=device_name, meta_path=base_dir + '/autocash/meta_full.json',
                                           gcn_flag=self.gcn_flag, meta_ratio=self.meta_ratio)
        else:
            self.Meta = torch.load(base_dir+'/autocash/Meta'+meta_ratio_name+gcn_name+sample_name+'.model', map_location=device_name)
        self.data_name_index = {
            'AmazonElectronics': 0,
            'AmazonClothing_Shoes_and_Jewelry': 1,
            'AmazonPet_Supplies': 2,
            'AmazonMovies_and_TV': 3,
            'AmazonSports_and_Outdoors': 4,
            'AmazonToys_and_Games': 5,
            'AmazonVideo_Games': 6,
        }
        self.data_name_index[args.full_data_name] = 7
        self.finetune_infonum = args.finetune_infonum
        self.logger.info(f'--- self.gcn_flag: {self.gcn_flag}, self.gcn_flag==False: {self.gcn_flag==False}')
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
            batch_NewInfo += random.sample(NewInfo, batch_size-len(batch_NewInfo))
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
        train_loader = DataLoader(MetaDataSet(new_data), batch_size=batch_size, shuffle=True)
        return train_loader

    def train_new_dataset(self, NewInfo, dataset, epochs=1, batch_size=32):
        # construct new input data with NewInfo, add the datset_id into it
        batch_NewInfo = self.get_batch_NewInfo(NewInfo, self.finetune_infonum)
        train_loader = self.prepare_newinfo_dataloader(batch_NewInfo, dataset, batch_size=batch_size)
        if None in train_loader or len(NewInfo) == 0:
            return None

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
        self.logger.info('dataset training finished')
        return

    def get_best_solution(self, NewInfo, arm_usb_scores):
        # line 175 -- 5: best solution sampling num, line 178 -- 32: other solution sampling num (parameter analysis: 3/16, 5/32, 10/64, 15/96)
        NewInfo = sorted(list(NewInfo), key=lambda item: item[1], reverse=True)
        
        total_model_nums = self.args.total_model_nums
        newinfo_model_nums = self.args.newinfo_model_nums
        
        # NewInfo = random.sample(NewInfo[:10], min(5, len(NewInfo)))
        NewInfo = random.sample(NewInfo[:10], min(newinfo_model_nums, len(NewInfo)))
        num = len(NewInfo)
        si = [item[0] for item in NewInfo]
        # for i in range(num, 32):
        for i in range(num, total_model_nums):
            model_name = random.sample(self.model_names, 1)[0]
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

                di = di[0].repeat(all_solution_emb.shape[0])
                all_y_pred = self.Meta(di, all_solution_emb)
                all_pred_ans.append(all_y_pred)
                all_solution_indexs += all_solution_index

            all_pred_ans = torch.cat(all_pred_ans, dim=0)

        values, indices = all_pred_ans.topk(50, largest=True, sorted=True, dim=0)
        top10_indices = indices[:,0]
        top10_solutions = [all_solution_indexs[item.item()] for item in top10_indices]

        top10_model_names, top10_solution_indexs = [], []
        for i in range(len(top10_solutions)):
            solution_best = top10_solutions[i]
            model_name = self.id2model[solution_best]['model_name']
            if model_name not in top10_model_names:
                top10_model_names.append(model_name)
                top10_solution_indexs.append(solution_best)
        top10_model_ucb = []
        for i in range(len(top10_model_names)):
            index = self.model_names.index(top10_model_names[i])
            model_ucb = arm_usb_scores[index]
            top10_model_ucb.append(model_ucb)

        try:
            model_best = torch.multinomial(torch.tensor(top10_model_ucb),1)[0].item()
            solution_best = top10_solution_indexs[model_best]
        except:
            solution_best = random.sample(top10_solution_indexs, 1)[0]
            model_best = top10_solution_indexs.index(solution_best)
        self.logger.info(f'--- top10_model_names-{top10_model_names}')
        self.logger.info(f'--- top10_solution_indexs-{top10_solution_indexs}')
        self.logger.info(f'--- top10_model_ucb-{top10_model_ucb}')
        self.logger.info(f'--- best top50 model_best-{top10_model_names[model_best]}')
        self.logger.info(f'--- best top50 solution_best-{solution_best}')
        return solution_best

    def get_random_solution(self, NewInfo, arm_usb_scores):
        model_name1 = self.model_names[arm_usb_scores.index(max(arm_usb_scores))]
        try:
            model_name2 = self.model_names[torch.multinomial(torch.tensor(arm_usb_scores),1)[0].item()]
        except:
            model_name2 = random.sample(self.model_names, 1)[0]
        selected_model_names = [model_name1, model_name2]

        si = []
        for i in range(32):
            model_name = selected_model_names[i%2]
            hp_dict = init_hp_dict_prn(model_name)
            solution_best = index_solutions_prn(model_name, hp_dict)
            si.append(solution_best)
        self.logger.info(f'--- selected_model_names: {selected_model_names}')
        self.logger.info(f'--- len(si): {len(si)}, si: {si}')
        si = torch.tensor(si)

        with torch.no_grad():
            all_pred_ans, all_solution_indexs = [], []
            if True:
                di = torch.tensor([self.data_name_index[self.args.full_data_name]]).to(self.device)
                si = si.to(self.device)
                solution_emb, all_solution_info = self.Meta.get_solution_embedding(self.gcn_flag, si, self.device, sample_num=self.sample_num)
                all_solution_emb, all_solution_index = all_solution_info

                di = di[0].repeat(all_solution_emb.shape[0])
                all_y_pred = self.Meta(di, all_solution_emb)
                all_pred_ans.append(all_y_pred)
                all_solution_indexs += all_solution_index

            all_pred_ans = torch.cat(all_pred_ans, dim=0)
            values, indices = all_pred_ans.topk(20, largest=True, sorted=True, dim=0)
            top10_indices = indices[:,0]
            top10_solutions = [all_solution_indexs[item.item()] for item in top10_indices]
        if random.random() < 0.5:
            solution_best = random.sample(top10_solutions[:10], 1)[0]
            self.logger.info(f'--- random top10 solution_best-{solution_best}')
        else:
            solution_best = random.sample(top10_solutions, 1)[0]
            self.logger.info(f'--- random top20 solution_best-{solution_best}')
        return solution_best

    def get_ucb_score(self, Num_ucb, Reward_ucb, epoch):
        arm_usb_scores = []
        for i in range(len(Num_ucb)):
            ucb_score = Reward_ucb[i] + math.sqrt(math.log(epoch+1)/(Num_ucb[i]+1e-8)) * (self.runcount-epoch)/self.runcount
            arm_usb_scores.append(ucb_score)
        self.logger.info(f'--- get_ucb_score ......')
        self.logger.info(f'--- model_names-{self.model_names}')
        self.logger.info(f'--- Num_ucb-{Num_ucb}')
        self.logger.info(f'--- Reward_ucb-{Reward_ucb}')
        self.logger.info(f'--- arm_usb_scores-{arm_usb_scores}')
        return arm_usb_scores

    def get_autoctr_solution(self):
        NewInfo = [] #[(solution_id, test_auc, test_logloss), ...]
        Num_ucb, Reward_ucb = [], []
        for i in range(len(self.model_names)):
            Num_ucb.append(0)
            Reward_ucb.append(0)

        evaluated_solutions, top10_solutions = [], None
        dataset = self.args.full_data_name

        for epoch in range(self.runcount):
            if epoch >= 10:
                self.train_new_dataset(NewInfo, dataset, epochs=self.finetune_epoch)

            arm_usb_scores = self.get_ucb_score(Num_ucb, Reward_ucb, epoch)
            while True:
                if random.random() < self.solution_randomelpsion:
                    self.logger.info('--- Random Solution Selected')
                    solution_best = self.get_random_solution(NewInfo, arm_usb_scores)
                else:
                    self.logger.info('--- Best Solution Selected')
                    solution_best = self.get_best_solution(NewInfo, arm_usb_scores)
                if solution_best not in evaluated_solutions:
                    break
                    
            evaluated_solutions.append(solution_best)
            self.logger.info(f'--- solution_best-{solution_best}')
            solution_dict = copy.deepcopy(self.id2model[solution_best])
            model_name = solution_dict['model_name']
            del solution_dict['model_name']
            hp_dict = solution_dict
            auc, logloss, _, _ = self.evaluator.evaluate_single_model(model_name=model_name, hp_dict=hp_dict)
            
            NewInfo.append([solution_best, auc, min(logloss, 1)])
            model_index = self.model_names.index(model_name)
            Reward_ucb[model_index] = (Reward_ucb[model_index]*Num_ucb[model_index]+auc)*1.0/(Num_ucb[model_index]+1)
            Num_ucb[model_index] += 1
        return

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
