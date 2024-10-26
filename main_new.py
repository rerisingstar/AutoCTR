import argparse

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"CASH"))
from alg_random import Random
from alg_tpe_smbo import TPE_SMBO
from alg_autocash_new import AutoCASH_NEW, MetaPredictor, MetaDataSet
# from alg_smac_smbo import SMAC_SMBO
# from alg_rising_bandit import MONO_MAB_SMBO
# from alg_rising_bandit_sampling import Bandit_Sampling

parser = argparse.ArgumentParser("CASH")

parser.add_argument('--search_method', type=str, default="AutoCASH",
                    help='search_method')
parser.add_argument('--full_data_name', type=str, default="MovieLens",
                    help='full_data_name')

parser.add_argument('--seed', type=int, default=1024,
                    help='seed')   
parser.add_argument('--device', type=int, default=-1,
                    help='device')
parser.add_argument('--reindex', type=bool, default=False,
                    help='reindex')

parser.add_argument('--epochs', type=int, default=8,
                    help='epochs')
parser.add_argument('--search_epochs', type=int, default=640,
                    help='search_epochs')

parser.add_argument('--rising_mode', type=int, default=1,
                    help='rising_mode')
parser.add_argument('--rising_c', type=int, default=10,
                    help='rising_c')

# AutoCASH
parser.add_argument('--finetune_epoch', type=int, default=1,
                    help='how many epochs to do a finetuning')
parser.add_argument('--sample_num', type=int, default=100,
                    help='how many edges to construct the sampled gcn graph')
parser.add_argument('--retain_metapredictorepochs', type=int, default=-1,
                    help='the number of epochs to retain meta predictor '
                         '(-1 means not retrain, 0 means using a new inited metapredictor, 20 means retrain)')
parser.add_argument('--meta_ratio', type=float, default=1.0,
                    help='the ratio of meta_info used for training meta predictor')
parser.add_argument('--finetune_infonum', type=int, default=-1,
                    help='how many newinfo to do a finetuning (-1 means all)')
parser.add_argument('--gcn_flag', default=False, action='store_true',
                    help='whether use gcn to lean the solution embeddings')
parser.add_argument('--solution_randomelpsion', type=float, default=0.3,
                    help='the possibility of evaluating a random solution instead of the best solution')
parser.add_argument('--sample_ratio', type=float, default=1.0,
                    help='the percentage of dataset that are used for cash search')
# new sampling params
parser.add_argument('--total_model_nums', type=int, default=32,
                    help='total model nums when return best solution models')
parser.add_argument('--newinfo_model_nums', type=int, default=5,
                    help='model nums for NewInfo, so randomly sample total_model_nums-newinfo_model_nums models')

args = parser.parse_args()


if __name__ == '__main__':
     if args.search_method == "Random":
          Random(args).main()
     elif args.search_method == "TPE_SMBO":
          TPE_SMBO(args).main()
     elif args.search_method == 'AutoCASH':
          AutoCASH_NEW(args).main()
     elif args.search_method == "SMAC_SMBO":
          SMAC_SMBO(args).main()
     elif args.search_method == "MONO_MAB_SMBO":
          MONO_MAB_SMBO(args).main()
     elif args.search_method == "Bandit_Sampling":
         Bandit_Sampling(args).main()
