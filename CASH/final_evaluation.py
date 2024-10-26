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

import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))),"CTRAPI"))
from deepctr_torch_local.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch_local.models import *
from OurAPI import *
from log import *


class FinalCTREvaluation(object):
    def __init__(self):
        self.step, self.combined_prediction_step =0, 0
        return

    def evaluate_single_model(self, full_data_name='MovieLens', model_name='PNN', epochs=5, device='cpu', searching_flag=False,
               hp_dict={'use_inner':True, 'use_outter':False, 'kernel_type':'mat', 'dnn_dropout':0, 'dnn_activation':'relu', 'init_std': 0.0001, 'seed': 1024, 'embedding_size': 32, 'optimizer': 'adam'}):
        '''
        full_data_name: 'AmazonBeauty', 'AmazonBooks', 'MovieLens'
        '''
        if full_data_name[:6] == 'Amazon':
            data_name, sub_name = full_data_name[:6], full_data_name[6:]
        else:
            data_name, sub_name = full_data_name, ''
        batch_size = 2048 if sub_name in ['Books', 'Movies_and_TV', 'Electronics', 'MovieLens10m'] else 512

        ctr = CTR(args=None)
        data_dict, feature_columns = ctr.CTRData(full_data_name=(data_name, sub_name))
        #
        logger, file_position = set_logger('SingleEvaluation', data_name + '_' + sub_name, 'Evaluation')

        logger.info(f'SingleEvaluation, dataset is {data_name + sub_name}, '
                    f'max_epochs={epochs}, batchsize={batch_size}')
        logger.info(f'SingleEvaluation, model_name is {model_name}, '
                    f'device={device}, searching_flag={searching_flag},'
                    f'hp_dict={hp_dict}')

        time_start = time.time()
        logger.info(f'\n\nSingle CTR model ......')
        optimizer = hp_dict['optimizer']
        model = ctr.CTRModel(model_name, hp_dict, feature_columns, device=device,
                            logger=logger)
        model.compile(optimizer, "binary_crossentropy",
                        metrics=["binary_crossentropy", "auc"], )

        try:
            history, test_logloss, test_auc, pred_ans, final_test_target = ctr.CTRtrain(model, data_dict,
                                                           epochs=epochs, batch_size=batch_size,
                                                           searching_flag=searching_flag, logger=logger)
            #print(type(pred_ans))
            #print(pred_ans.shape)
        except Exception as e:
            logger.error(repr(e))
            test_auc, test_logloss = 0, 1
        running_time = time.time()
        logger.info(f'\n\ntest_auc: {test_auc}, test_logloss: {test_logloss}')
        logger.info(f'used {running_time-time_start} seconds')
        return test_auc, test_logloss, file_position



if __name__ == '__main__':
    obj = FinalCTREvaluation()
    obj.evaluate_single_model()
    print()
