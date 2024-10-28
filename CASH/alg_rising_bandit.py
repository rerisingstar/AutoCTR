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
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from mindware.utils.constant import MAX_INT
from mindware.utils.logging_utils import get_logger
from smac.facade.smac_ac_facade import SMAC4AC as SMAC
from smac.scenario.scenario import Scenario

import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))),"CTRAPI"))
from pruned_search_space import BaseHPs, SearchSpace, ModelUse, index_solutions_prn_init, init_hp_dict_prn
from log import *
from search_space.configuration_space import get_configuration_space_module


class MONO_MAB_SMBO(Random):
    def __init__(self, args):
        self.space = [BaseHPs, SearchSpace, ModelUse]
        self.model2id, self.id2model, self.allmodelnum = index_solutions_prn_init(reindex=args.reindex)

        self.environment_setting(args, 'MONO_MAB_SMBO')
        if args.device == -1:
            device_name = "cpu"
        else:
            device_name = "cuda:"+str(args.device) 
        self.evaluator = CTREvaluation(args.search_epochs, args.full_data_name, args.epochs, device_name, sample_ratio=args.sample_ratio, logger=self.logger)

        self.estimator_arms = list(ModelUse.keys())

        self.smac_containers = dict()
        self.cnts = dict()
        self.rewards = dict()
        self.updated_rewards = dict()

        for estimator in self.estimator_arms:
            config_space = get_configuration_space_module(self.space, estimator)
            scenario_dict = {
                'abort_on_first_run_crash': False,
                "run_obj": "quality",
                "cs": config_space,
                "deterministic": "true",
                'runcount-limit': 1,
                "output_dir": None
            }

            smac = SMAC(scenario=Scenario(scenario_dict),
                        rng=args.seed, tae_runner=self.evaluator)
            self.smac_containers[estimator] = smac
            self.cnts[estimator] = 0
            self.rewards[estimator] = list()
            self.updated_rewards[estimator] = list()

        self.search_epochs = args.search_epochs
        self.solution_epochs = args.epochs
        self.args = args

        self.rising_mode, self.rising_c = args.rising_mode, args.rising_c # 2, 10
        self.runcount = int(args.search_epochs/args.epochs)
        return

    def main(self):
        self.logger.info('\n\nMONO_MAB_SMBO Search Begin ......')
        arm_set = list(self.estimator_arms)
        T = self.runcount
        iter_num, tmp_iter, duration = 0, 0, self.rising_c
        while True:
            # Pull each arm exactly once.
            tmp_iter += 1
            p, q, es_flag = list(), list(), False
            for arm in arm_set:
                self.smac_containers[arm].optimize()
                runhistory = self.smac_containers[arm].solver.runhistory
                self.logger.info("arm: %s", str(arm))
                self.logger.info("runhistory: %s", str(runhistory.data))

                # Observe the reward.
                runkeys = list(runhistory.data.keys())
                for key in runkeys[self.cnts[arm]:]:
                    reward = runhistory.data[key][0]*(-1) # reward = auc
                    self.rewards[arm].append(reward)
                    self.updated_rewards[arm].append(max(self.rewards[arm]))

                # Determine whether to stop early.
                if len(arm_set) == 1 and len(runkeys[self.cnts[arm]:]) == 0:
                    es_flag = True

                iter_num += (len(runkeys) - self.cnts[arm])
                self.cnts[arm] = len(runhistory.data.keys())

                acc_reward = self.updated_rewards[arm]
                if self.cnts[arm] > 2:
                    if len(acc_reward) >= duration:
                        estimated_slope = (acc_reward[-1] - acc_reward[-duration]) / duration
                    else:
                        estimated_slope = 1.

                    if self.rising_mode == 1:
                        F = sum(acc_reward)
                        pred = sum([min(1., acc_reward[-1] + estimated_slope * (t - tmp_iter))
                                    for t in range(tmp_iter+1, T)])
                        p.append(F + pred)
                        q.append(F + acc_reward[-1]*(T - tmp_iter))
                    elif self.rising_mode == 2:
                        p.append(min(1., acc_reward[-1] + estimated_slope * (T - tmp_iter)))
                        q.append(acc_reward[-1])
                    else:
                        raise ValueError('Invalid mode: %d.' % self.rising_mode)
                else:
                    p.append(acc_reward[-1])
                    q.append(acc_reward[-1])

                if self.evaluator.used_epochs >= self.search_epochs:
                    break

            if self.evaluator.used_epochs >= self.search_epochs or es_flag:
                break

            # Remove some arm.
            N = len(arm_set)
            flags = [False] * N
            for i in range(N):
                for j in range(N):
                    if i != j:
                        if p[i] < q[j]:
                            flags[i] = True

            self.logger.info('>>>>> Remove Models: %s' % [item for index, item in enumerate(arm_set) if flags[index]])
            arm_set = [item for index, item in enumerate(arm_set) if not flags[index]]
        self.logger.info('MONO_MAB_SMBO Search End ......')

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

    