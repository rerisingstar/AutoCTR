import argparse

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"CASH"))
from final_evaluation import FinalCTREvaluation

def final_evaluation(solution, device=0, full_data_name="MovieLens10m"):
     obj = FinalCTREvaluation()
     model_name, hp_dict, auc, logloss = solution
     if device == -1:
          device_name = "cpu"
     else:
          device_name = "cuda:0"
     final_auc, final_logloss, file_position = obj.evaluate_single_model(full_data_name=full_data_name, model_name=model_name, \
                    epochs=2, device=device_name, searching_flag=False, hp_dict=hp_dict)
     return final_auc, final_logloss, file_position

# tpe
top10_solutions = ['DIFM', {'att_head_num': 2, 'att_res': False, 'dnn_activation': 'relu', 'dnn_dropout': 0.2, 'dnn_hidden_units': (8,), 'init_std': 0.0001, 'seed': 1024, 'embedding_size': 64, 'optimizer': 'adam'}, 0.9503, 0.2838] # tpe
# # random
top10_solutions = ['DIN', {'att_activation': 'linear', 'att_weight_normalization': True, 'dnn_hidden_units': [768, 768], 'dnn_activation': 'relu', 'att_hidden_size': [200, 200], 'init_std': 0.0001, 'seed': 1024, 'embedding_size': 64, 'optimizer': 'adam'}, 0.9471, 0.2895]
# # smac
# top10_solutions = ['DIEN', {'alpha': 0.6, 'att_hidden_units': (256, 128), 'gru_type': 'GRU', 'init_std': 0.0001, 'seed': 1024, 'use_bn': True, 'embedding_size': 64, 'optimizer': 'adam'}, 0.9496, 0.2862]
# # mono
# top10_solutions = ['DCN', {'cross_num': 2, 'cross_parameterization': 'vector', 'dnn_activation': 'relu', 'dnn_dropout': 0, 'dnn_hidden_units': [400, 400, 400], 'init_std': 0.0001, 'seed': 1024, 'embedding_size': 32, 'optimizer': 'adam'}, 0.9457, 0.2922]

final_auc, final_logloss, file_position = final_evaluation(top10_solutions)