import json 
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))), "CTRAPI"))
from pruned_search_space import index_solutions_prn_init, init_hp_dict_prn, cal_SearchSpace_size
from log import *
from autocash.MetaLearning import MetaPredictor, MetaDataSet


def read_meta_info(meta_path='./autocash/meta_full.json'):
    # feature_names = ['dataset_id', 'solution_id', 'test_auc', 'test_logloss']

    with open(meta_path, 'r') as f:
        meta_info = json.load(f)
    meta_info = [t for t in meta_info if t[3] < 2]
    data = np.array(meta_info)

    model_name_list = {}
    best_solution = {}
    model2id, id2model, allmodelnum = index_solutions_prn_init(reindex=False)
    for info in meta_info:
        dataset_id, solution_id, test_auc, test_logloss = info
        model_name = id2model[solution_id]['model_name']
        if model_name not in model_name_list.keys():
            model_name_list[model_name] = 0
        model_name_list[model_name] += 1
        if dataset_id not in best_solution.keys():
            best_solution[dataset_id] = [solution_id, test_auc, test_logloss]
        else:
            if best_solution[dataset_id][1] < test_auc:
                best_solution[dataset_id] = [solution_id, test_auc, test_logloss]
    print(model_name_list)
    data_info_all = {
        0: ['AmazonElectronics', 1.69, 7.82],
        1: ['AmazonClothing_Shoes_and_Jewelry', 0.28, 5.75],
        2: ['AmazonPet_Supplies', 0.16, 1.24],
        3: ['AmazonMovies_and_TV', 1.70, 4.61],
        4: ['AmazonSports_and_Outdoors', 0.30, 3.27],
        5: ['AmazonToys_and_Games', 0.17, 2.25],
        6: ['AmazonVideo_Games', 0.23, 1.32],
    } # name, reviews num (*10^6), rating num (10^6)
    new_meta_info = []
    model2id, id2model, allmodelnum = index_solutions_prn_init(reindex=False)
    for dataset_id in best_solution.keys():
        solution_id, test_auc, test_logloss = best_solution[dataset_id]
        model_name = id2model[solution_id]['model_name']

        data_info = data_info_all[dataset_id]
        new_meta_info.append([data_info, model_name])

    print(new_meta_info)
    return new_meta_info


read_meta_info()


