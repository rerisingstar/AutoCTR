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

import pickle
from mindware.utils.constant import MAX_INT
from mindware.utils.logging_utils import get_logger
from smac.facade.smac_ac_facade import SMAC4AC as SMAC
from smac.scenario.scenario import Scenario

import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))),"CTRAPI"))
from pruned_search_space import BaseHPs, SearchSpace, ModelUse, index_solutions_prn_init, init_hp_dict_prn
from log import *
from search_space.configuration_space import get_configuration_space


class SMAC_SMBO(Random):
    def __init__(self, args):
        self.space = [BaseHPs, SearchSpace, ModelUse]
        self.model2id, self.id2model, self.allmodelnum = index_solutions_prn_init(reindex=args.reindex)

        self.environment_setting(args, 'SMAC_SMBO')
        if args.device == -1:
            device_name = "cpu"
        else:
            device_name = "cuda:"+str(args.device) 
        self.evaluator = CTREvaluation(args.search_epochs, args.full_data_name, args.epochs, device_name, sample_ratio=args.sample_ratio, logger=self.logger)

        self.cs = get_configuration_space(self.space)
        self.runcount = int(args.search_epochs/args.epochs)
        scenario_dict = {
            'abort_on_first_run_crash': False,
            "run_obj": "quality",
            "cs": self.cs,
            "deterministic": "true",
            "runcount-limit": self.runcount,
            "output_dir": None
        }
        self.scenario = Scenario(scenario_dict)
        self.smac = SMAC(scenario=self.scenario, rng=np.random.RandomState(args.seed), tae_runner=self.evaluator)
        
        self.search_epochs = args.search_epochs
        self.solution_epochs = args.epochs
        self.args = args
        return

    def main(self):
        self.logger.info('\n\nSMAC_SMBO Search Begin ......')
        self.smac.optimize()
        self.logger.info('SMAC_SMBO Search End ......')

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
