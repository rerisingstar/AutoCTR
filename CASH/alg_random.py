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

import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__)))),"CTRAPI"))
from pruned_search_space import BaseHPs, SearchSpace, ModelUse, index_solutions_prn_init, init_hp_dict_prn
from log import *


class Random(object):
	def __init__(self, args):
		super(Random, self).__init__()
		self.space = [BaseHPs, SearchSpace, ModelUse]
		self.model2id, self.id2model, self.allmodelnum = index_solutions_prn_init(reindex=args.reindex)

		self.environment_setting(args, 'Random')
		if args.device == -1:
			device_name = "cpu"
		else:
			device_name = "cuda:"+str(args.device) 
		self.evaluator = CTREvaluation(args.search_epochs, args.full_data_name, args.epochs, device_name, sample_ratio=args.sample_ratio, logger=self.logger)

		self.search_epochs = args.search_epochs
		self.solution_epochs = args.epochs
		self.args = args
		return

	def create_exp_dir(self, path, scripts_to_save=None):
		if not os.path.exists(path):
			os.mkdir(path)

		if scripts_to_save is not None:
			os.mkdir(os.path.join(path, 'scripts'))
			for script in scripts_to_save:
				dst_file = os.path.join(path, 'scripts', os.path.basename(script))
				shutil.copyfile(script, dst_file)
		return
	def environment_setting(self, args, name):
		dir_dir_name = os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__))))
		logging_dir = dir_dir_name+'/LOG/Search/'
		if not os.path.exists(logging_dir):
			os.mkdir(logging_dir)
		logging_path = logging_dir + 'CASH_'+name+'-Run-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
		self.create_exp_dir(logging_path, scripts_to_save=glob.glob('*.py'))
		self.logging_path = logging_path
		
		log_format = '%(asctime)s %(message)s'
		self.logger = logging.getLogger()
		self.logger.setLevel(logging.INFO)
		fh = logging.FileHandler(os.path.join(logging_path, 'log.txt'))
		fh.setFormatter(logging.Formatter(log_format))
		self.logger.addHandler(fh)
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(logging.Formatter(log_format))
		self.logger.addHandler(stream_handler)

		if not torch.cuda.is_available() or args.device == -1:
			self.logger.info('+ no gpu device available')
			np.random.seed(args.seed)
			self.device = torch.device("cpu") 
			self.logger.info("+ args = %s", str(args))
			return 
			#sys.exit(1)

		np.random.seed(args.seed)
		torch.cuda.set_device(args.device)
		cudnn.benchmark = True
		torch.manual_seed(args.seed)
		cudnn.enabled = True
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
		random.seed(args.seed)
		os.environ['PYTHONHASHSEED'] = str(args.seed)
		self.device = torch.device('cuda')
		self.logger.info('+ gpu device = %d' % args.device)
		self.logger.info("+ args = %s", str(args))
		return

	def get_random_solution(self):
		BaseHPs, SearchSpace, ModelUse = self.space
		model_name = random.choice(list(ModelUse.keys()))
		hp_dict = init_hp_dict_prn(model_name)
		return model_name, hp_dict

	def final_evaluation(self, solution):
		obj = FinalCTREvaluation()
		model_name, hp_dict, auc, logloss = solution
		if self.args.device == -1:
			device_name = "cpu"
		else:
			device_name = "cuda:"+str(self.args.device) 
		final_auc, final_logloss, file_position = obj.evaluate_single_model(full_data_name=self.args.full_data_name, model_name=model_name, \
					epochs=20, device=device_name, searching_flag=False, hp_dict=hp_dict)
		return final_auc, final_logloss, file_position

	def main(self):
		start_time = time.time()
		used_epochs = 0
		
		self.logger.info('\n\nRandom Search Begin ......')
		while used_epochs < self.search_epochs:		
			model_name, hp_dict = self.get_random_solution()
			auc, logloss, _, _ = self.evaluator.evaluate_single_model(model_name=model_name, hp_dict=hp_dict)
			used_epochs += self.solution_epochs
		self.logger.info('Random Search End ......')

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

