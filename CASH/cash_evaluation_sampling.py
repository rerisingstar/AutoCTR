import sys
import json
import pickle
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from random import choice
import time
import heapq
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import ConfigSpace

import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))),"CTRAPI"))
from deepctr_torch_local.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch_local.models import *
from OurAPI_sampling import *
from pruned_search_space import SearchSpace
import numpy as np


class CTREvaluation(object):
    def __init__(self, search_epochs=1000, full_data_name='MovieLens', epochs=5, device='cpu', sample_ratio=1.0, logger=None):
        '''
        full_data_name: 'AmazonBeauty', 'AmazonBooks', 'MovieLens'
        '''
        self.step, self.combined_prediction_step =0, 0

        if full_data_name[:6] == 'Amazon':
            data_name, sub_name = full_data_name[:6], full_data_name[6:]
        else:
            data_name, sub_name = full_data_name, ''
        self.batch_size = 2048 if sub_name in ['Books', 'Movies_and_TV', 'Electronics'] else 512

        self.ctr = CTR(args=None)
        self.data_name, self.sub_name, self.sample_ratio = data_name, sub_name, sample_ratio
        self.data_dict, self.feature_columns = self.ctr.CTRData(full_data_name=(data_name, sub_name), sample_ratio=sample_ratio)
        self.epochs = epochs
        self.device = device
        self.searching_flag = True
        self.search_epochs = search_epochs
        self.logger = logger

        self.start_time = time.time()
        self.top10_solutions = []
        self.used_epochs, self.i = 0, 0
        return

    def update_top10_solutions(self, top10_solutions, new_solution):
        if new_solution not in top10_solutions:
            model_name, hp_dict, auc, logloss = new_solution
            update, best = False, False
            for i in range(len(top10_solutions)):
                if top10_solutions[i][2] < auc or (top10_solutions[i][2] == auc and top10_solutions[i][3] > logloss):
                    top10_solutions.insert(i, new_solution)
                    update = True
                    if i == 0:
                        best = True
                    break
            if len(top10_solutions) < 10 and update == False:
                top10_solutions.append(new_solution)
                update = True
                if len(top10_solutions) == 1:
                    best = True
            top10_solutions = top10_solutions[:10]
        else:
            update, best = False, False
        return top10_solutions, update, best

    ### 5 folds data sampling code ###
    def five_folds_data_sampling(self, data_dict):
        random_state = 2020
        
        train_model_input, train_label = data_dict['train_input'], data_dict['train_target']
        
        feature_names_list = list(train_model_input.keys())
        train_model_input['target'] = train_label
        five_folds = []
        # 2 random folds
        for index in range(2):
            splited = [
                train_test_split(train_model_input[feat], train_size=0.2, random_state=random_state*index)
                for feat in train_model_input.keys()
            ]
            model_input = {}
            for i, name in enumerate(feature_names_list):
                model_input[name] = splited[i][0]
            target = splited[-1][0]
            five_folds.append(list([model_input, target]))
        # 3 biased folds
        data_num = train_model_input['uid'].shape[0]
        data_num_split = [0, int(data_num*0.3), int(data_num*0.6), data_num]
        for index in range(3):
            splited_biased = [
                train_test_split(train_model_input[feat][data_num_split[index]:data_num_split[index+1]], train_size=0.5, random_state=random_state*index)
                for feat in train_model_input.keys()
            ]
            splited = [
                train_test_split(train_model_input[feat], train_size=0.05, random_state=random_state * index)
                for feat in train_model_input.keys()
            ]
            model_input = {}
            for i, name in enumerate(feature_names_list):
                model_input[name] = np.concatenate((splited_biased[i][0], splited[i][0]), axis=0)
            target = np.concatenate((splited_biased[-1][0], splited[-1][0]), axis=0)
            five_folds.append(list([model_input, target]))
            
        five_folds_data_dict = []
        for index in range(len(five_folds)):
            model_input, target = five_folds[index]
            other_folds_inputs, other_folds_targets = [], []
            for t in range(index):
                other_folds_inputs.append(five_folds[t][0])
                other_folds_targets.append(five_folds[t][1])
            for t in range(index+1, len(five_folds)):
                other_folds_inputs.append(five_folds[t][0])
                other_folds_targets.append(five_folds[t][1])
                
            train_model_input = {}
            
            for key in model_input.keys():
                inputs_to_concatenate = [fold[key] for fold in other_folds_inputs]
                train_model_input[key] = np.concatenate(inputs_to_concatenate, axis=0)
                
            train_target = np.concatenate(other_folds_targets, axis=0)
            data_dict_fold = {
                'train_input': train_model_input,
                'train_target': train_target,
                'test_input': model_input,
                'test_target': target
            }
            five_folds_data_dict.append(data_dict_fold) 
        
        return five_folds_data_dict   
    ### 5 folds data sampling code ###

    def evaluate_single_model(self, model_name='PNN',
               hp_dict={'use_inner':True, 'use_outter':False, 'kernel_type':'mat', 'dnn_dropout':0, 'dnn_activation':'relu', 'init_std': 0.0001, 'seed': 1024, 'embedding_size': 32, 'optimizer': 'adam'}):
        self.i += 1
        self.logger.info("### try model_name: %s, hp_dict: %s", str(model_name), str(hp_dict))

        # self.data_dict, self.feature_columns = self.ctr.CTRData(full_data_name=(self.data_name, self.sub_name),
        #                                                         sample_ratio=self.sample_ratio)
        Found = True
        try:
            optimizer = hp_dict['optimizer']
            model = self.ctr.CTRModel(model_name, hp_dict, self.feature_columns, device=self.device,
                                logger=self.logger)
            model.compile(optimizer, "binary_crossentropy",
                            metrics=["binary_crossentropy", "auc"], )

            ### 5 folds data sampling code ###
            test_logloss_list, test_auc_list, pred_ans_list, final_test_target_list = [], [], [], []
            five_folds_data_dict = self.five_folds_data_sampling(self.data_dict)
            for data_fold in five_folds_data_dict:
                history, test_logloss, test_auc, pred_ans, final_test_target = self.ctr.CTRtrain(model, data_fold,
                                                               epochs=self.epochs, batch_size=self.batch_size,
                                                               searching_flag=False, logger=self.logger)
                test_logloss_list.append(test_logloss)
                test_auc_list.append(test_auc)
                pred_ans_list.append(pred_ans)
                final_test_target_list.append(final_test_target)
            test_logloss = np.mean(test_logloss_list) + 0.1*np.var(test_logloss_list)
            test_auc = np.mean(test_auc_list) + 0.1*np.var(test_auc_list)
            pred_ans = pred_ans_list[-1]
            final_test_target = final_test_target_list[-1]
            ### 5 folds data sampling code ###

            #print(type(pred_ans))
            #print(pred_ans.shape)
        except Exception as e:
            self.logger.error(repr(e))
            test_auc, test_logloss, pred_ans, final_test_target = 0, 1, 0, 0
            Found = False
        if Found == True:
            self.logger.info("### Success")

        self.used_epochs += self.epochs
        self.logger.info('@ solution: %d, used_epochs: %d, search_epochs: %d', self.i, self.used_epochs, self.search_epochs)
        self.logger.info('* model_name: %s, hp_dict: %s', str(model_name), str(hp_dict))
        self.logger.info('* auc: %.4f, logloss: %.4f', test_auc, test_logloss)

        new_solution = [model_name, hp_dict, test_auc, test_logloss]
        self.top10_solutions, update, best = self.update_top10_solutions(self.top10_solutions, new_solution)
        if best:
            self.logger.info('** Updated Better Solution Found !!!')
        if update: 
            self.logger.info('** Updated top10_solutions: %s', str(self.top10_solutions))
        running_time = time.time() - self.start_time
        self.logger.info('* current running time: %f h', running_time/60.0/60.0)
        return test_auc, test_logloss, pred_ans, final_test_target

    def __call__(self, config):
        if isinstance(config, ConfigSpace.configuration_space.Configuration):
            dict_config = dict(config)
            if 'algorithm' in dict_config.keys(): # SMAC_SMBO
                model_name = dict_config['algorithm']
                hp_dict = {}
                for hp in dict_config:
                    if hp != "algorithm":
                        hp_name = hp.split(':')[1]
                        if "," in hp_name:
                            hp_name = hp_name.split(",")[1]
                        if isinstance(dict_config[hp], str) and dict_config[hp][:5] == "list@":
                            value_index = int(dict_config[hp].split("@")[1])
                            hp_value = SearchSpace[hp_name][value_index]
                        else:
                            hp_value = dict_config[hp]
                        hp_dict[hp_name] = hp_value
            elif 'estimator' in dict_config.keys(): # MONO_MAB_SMBO
                model_name = dict_config['estimator']
                hp_dict = {}
                for hp in dict_config:
                    if hp != "estimator":
                        hp_name = hp.split(':')[1]
                        if "," in hp_name:
                            hp_name = hp_name.split(",")[1]
                        if isinstance(dict_config[hp], str) and dict_config[hp][:5] == "list@":
                            value_index = int(dict_config[hp].split("@")[1])
                            hp_value = SearchSpace[hp_name][value_index]
                        else:
                            hp_value = dict_config[hp]
                        hp_dict[hp_name] = hp_value
        elif isinstance(config, dict): # TPE_SMBO
            dict_config = config['estimator']
            model_name = dict_config[0]
            hp_dict = {}
            for hp_name in dict_config[1]:
                if "," in hp_name:
                    real_hp_name = hp_name.split(",")[1]
                hp_dict[real_hp_name] = dict_config[1][hp_name]

        test_auc, test_logloss, pred_ans, final_test_target = self.evaluate_single_model(model_name=model_name, hp_dict=hp_dict)
        return test_auc*(-1)



if __name__ == '__main__':
    obj = CTREvaluation()
    obj.evaluate_single_model()
    print()
