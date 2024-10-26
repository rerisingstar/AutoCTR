import sys
import logging
import shutil
import glob
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
import json
from cash_evaluation import CTREvaluation
from final_evaluation import FinalCTREvaluation
from alg_random import Random

from hyperopt import tpe, fmin, Trials, STATUS_OK
from mindware.utils.logging_utils import get_logger

import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))),"CTRAPI"))
from pruned_search_space import BaseHPs, SearchSpace, ModelUse, index_solutions_prn_init, init_hp_dict_prn
from log import *
from search_space.hyperopt_space import get_hyperopt_space


class TPE_SMBO(Random):
    def __init__(self, args):
        self.space = [BaseHPs, SearchSpace, ModelUse]
        self.model2id, self.id2model, self.allmodelnum = index_solutions_prn_init(reindex=args.reindex)

        self.environment_setting(args, 'TPE_SMBO')
        if args.device == -1:
            device_name = "cpu"
        else:
            device_name = "cuda:"+str(args.device) 
        self.evaluator = CTREvaluation(args.search_epochs, args.full_data_name, args.epochs, device_name, sample_ratio=args.sample_ratio, logger=self.logger)

        _, self.config_space = get_hyperopt_space(self.space)
        self.estimators = list(self.config_space.keys())
        self.trials = Trials()

        def objective(x): 
            return {
                'loss': self.evaluator(x), # loss = -auc
                'status': STATUS_OK,
                'config': x
            }
        self.objective = objective

        self.search_epochs = args.search_epochs
        self.solution_epochs = args.epochs
        self.args = args

        self.runcount = int(self.search_epochs/self.solution_epochs)
        return

    def main(self):
        start_time = time.time()
        
        self.logger.info('\n\nTPE_SMBO Search Begin ......')
        fmin(self.objective, self.config_space, tpe.suggest, self.runcount, trials=self.trials)
        self.logger.info('TPE_SMBO Search End ......')

        self.logger.info('\n\nTop10 Final Evaluation Begin ......')
        top10_solutions = self.evaluator.top10_solutions
        self.logger.info('+ top10_solutions: %s', str(top10_solutions))
        for i in range(len(top10_solutions)):
            self.logger.info('\n\n@@@ Top %d solution', i+1)
            self.logger.info('@@@ model_name: %s, hp_dict: %s', str(top10_solutions[i][0]), str(top10_solutions[i][1]))
            final_auc, final_logloss, file_position = self.final_evaluation(top10_solutions[i])
            self.logger.info('@@@ model_name: %s, hp_dict: %s', str(top10_solutions[i][0]), str(top10_solutions[i][1]))
            self.logger.info('@@@ final_auc: %.4f, final_logloss: %.4f', final_auc, final_logloss)
            self.logger.info('@@@ file_position: %s', file_position)
        self.logger.info('Top10 Final Evaluation End ......')
        return 
