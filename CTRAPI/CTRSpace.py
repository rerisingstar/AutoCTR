import copy
import random
import pickle
from tqdm import tqdm

'''
init_std, seed, task, device, gpus
'''



BaseHPs = {
    'init_std': [0.0001, ],
    'seed': [1024, ],
    'embedding_size': [32, 64, 128],
    'optimizer': ['adam', 'adagrad', 'rmsprop', 'sgd']
}

SearchSpace = {
    'l2_reg_cin': [0, 1e-1, 1e-2, 1e-3, 1e-4, 3e-5, 1e-5, 5e-6, 1e-6],
    'l2_reg_att': [0, 1e-1, 1e-2, 1e-3, 1e-4, 3e-5, 1e-5, 5e-6, 1e-6],
    'l2_reg_dnn': [0, 1e-1, 1e-2, 1e-3, 1e-4, 3e-5, 1e-5, 5e-6, 1e-6],
    'l2_reg_cross': [0, 1e-1, 1e-2, 1e-3, 1e-4, 3e-5, 1e-5, 5e-6, 1e-6],
    'l2_reg_linear': [0, 1e-1, 1e-2, 1e-3, 1e-4, 3e-5, 1e-5, 5e-6, 1e-6],
    'l2_reg_embedding': [0, 1e-1, 1e-2, 1e-3, 1e-4, 3e-5, 1e-5, 5e-6, 1e-6],

    'dnn_dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'dnn_use_bn': [False, True],
    'dnn_activation': ['relu', 'sigmoid', 'linear', 'prelu'],
    'dnn_hidden_units': [[400, 400, 400], [2000, 2000, 2000],
                         [1024, 1024, 1024], [512, 256, 64], [256, 64],
                         [32, 32, 32], (256, 128), [512, 512, 512],
                         [768, 768], [768, 768, 768],
                         (256, 256), (128, 128), [256, ], [256, 256, 256],
                         [16, 16, 16], [64, 64, 64], [128, 128, 128],
                         [512, 512], [8], [16]],  #2-5 32-1024

    'att_res': [True, False],
    'att_head_num': [1, 2, 3, 4, 8],
    'att_layer_num': [1, 2, 3, 4],
    'att_activation': ['relu', 'sigmoid', 'linear', 'prelu'],
    'att_hidden_units': [[600, 600], [300, 300], [200, 200], [80, 80], (256, 128)], #DIEN
    'att_hidden_size': [[600, 600], [300, 300], [200, 200], [80, 80], (256, 128)],  #DIN
    'attention_factor': [8, 20, 32, 40],
    'att_weight_normalization': [True, False],

    'cross_num': [1, 2, 3, 4, 5, 6],
    'cross_parameterization': ['vector', 'matrix'],

    'conv_filters': [[4, 4, 2], [3, 4, 2], (4, 4)],
    'conv_kernel_width': [[6, 5, 3], (6, 5)],

    'ltl_hidden_size': [3, 5, 10, 100, 256, 512, 1024],

    'afm_dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],

    'afn_dnn_hidden_units': [[100], [100, 100], [100, 100, 100], [100, 100, 100, 100],
                             [400], [400, 400], [400, 400, 400], [400, 400, 400, 400],
                             [1000], [1000, 1000], [1000, 1000, 1000], [1000, 1000, 1000, 1000],
                             [256, 128], [128, 256]],  #1,2,3,4  100 400 1000

    'low_rank': [16, 24, 32, 48, 64], #

    'num_experts': [3, 4, 5], #

    'use_fm': [True, False],
    'use_bn': [False, True],
    'use_inner': [True, False],
    'use_outter': [False, True],
    'use_attention': [True, False],
    # 'use_negsampling': [False, True],

    'gru_type': ['GRU', 'AIGRU', 'AGRU', 'AUGRU'],
    'kernel_type': ['mat', 'vec', 'num'],

    'alpha': [0.2, 0.4, 0.6, 0.8, 1.0],

    'bi_dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'bilinear_type': ['interaction', 'all', 'each'],

    'reduction_ratio': [1, 2, 3, 4, 5],

    'region_num': [4, 8, 16, 24, 32],

    'cin_layer_size': [[100, 100], [100, 100, 100], [100, 100, 100, 100],
                       [200, 200], [200, 200, 200], [200, 200, 200, 200],
                       (256, 128)],
    'cin_split_half': [True, False],
    'cin_activation': ['relu', 'sigmoid', 'linear', 'prelu'],
}

SearchSpace.update(BaseHPs)

ModelUse = {
    'AFM': ['use_attention', 'attention_factor', 'l2_reg_linear', 'l2_reg_embedding',
            'l2_reg_att', 'afm_dropout', ],
    'AFN': ['ltl_hidden_size', 'afn_dnn_hidden_units', 'l2_reg_linear',
            'l2_reg_embedding', 'l2_reg_dnn', 'dnn_dropout', 'dnn_activation', ],
    'AutoInt': ['att_layer_num', 'att_head_num', 'att_res', 'dnn_hidden_units',
                'dnn_activation', 'l2_reg_dnn', 'l2_reg_embedding', 'dnn_use_bn',
                'dnn_dropout'],
    'CCPM': ['conv_kernel_width', 'conv_filters', 'dnn_hidden_units', 'l2_reg_linear',
             'l2_reg_embedding', 'l2_reg_dnn', 'dnn_dropout', 'dnn_use_bn', 'dnn_activation'],
    'DCN': ['cross_num', 'cross_parameterization', 'dnn_hidden_units', 'l2_reg_linear',
            'l2_reg_embedding', 'l2_reg_cross', 'l2_reg_dnn', 'dnn_dropout',
            'dnn_activation', 'dnn_use_bn'],
    'DCNMix': ['cross_num', 'dnn_hidden_units', 'l2_reg_linear', 'l2_reg_embedding',
               'l2_reg_cross', 'l2_reg_dnn', 'dnn_dropout', 'low_rank', 'num_experts',
               'dnn_activation', 'dnn_use_bn'],
    'DeepFM': ['dnn_hidden_units', 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn',
               'dnn_dropout', 'dnn_activation'],
    'DIEN': ['gru_type', 'alpha', 'use_bn', 'dnn_hidden_units',
             'dnn_activation', 'att_hidden_units', 'att_activation', 'att_weight_normalization',
             'l2_reg_dnn', 'l2_reg_embedding', 'dnn_dropout'],
    'DIFM': ['att_head_num', 'att_res', 'dnn_hidden_units', 'l2_reg_linear', 'l2_reg_embedding',
             'l2_reg_dnn', 'dnn_dropout', 'dnn_activation', 'dnn_use_bn'],
    'DIN': ['dnn_use_bn', 'dnn_hidden_units', 'dnn_activation', 'att_hidden_size',
            'att_activation', 'att_weight_normalization', 'l2_reg_dnn',
            'l2_reg_embedding', 'dnn_dropout'],
    'FiBiNET': ['bilinear_type', 'reduction_ratio', 'dnn_hidden_units',
                'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn', 'dnn_dropout',
                'dnn_activation',],
    'IFM': ['dnn_hidden_units', 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn',
            'dnn_dropout', 'dnn_activation', 'dnn_use_bn'],
    'MLR': ['region_num', 'l2_reg_linear'],
    'NFM': ['dnn_hidden_units', 'l2_reg_embedding', 'l2_reg_linear', 'l2_reg_dnn',
            'bi_dropout', 'dnn_dropout', 'dnn_activation'],
    'ONN': ['dnn_hidden_units', 'l2_reg_embedding', 'l2_reg_linear', 'l2_reg_dnn',
            'dnn_dropout', 'dnn_use_bn', 'dnn_activation'],
    'PNN': ['l2_reg_embedding', 'l2_reg_dnn', 'dnn_dropout', 'dnn_activation',
            'use_inner', 'use_outter', 'kernel_type'],
    'WDL': ['dnn_hidden_units', 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn',
            'dnn_dropout', 'dnn_activation', 'dnn_use_bn'],
    'xDeepFM': ['dnn_hidden_units', 'cin_layer_size', 'cin_split_half',
                'cin_activation', 'l2_reg_linear', 'l2_reg_embedding',
                'l2_reg_dnn', 'l2_reg_cin', 'dnn_dropout', 'dnn_activation',
                'dnn_use_bn']
}


def _get_func(func_name='random'):
    """
    :param func_name: str, the func name to generate a hp
    :return: a function, used to get a hp
    """
    if func_name == 'random':
        return random.choice
    if func_name == 'first':
        def get_first(hp_list):
            return hp_list[0]
        return get_first

def init_hp_dict(model_name, func_name='random'):
    hp_dict = {}
    generate_func = _get_func(func_name)
    hps = ModelUse[model_name]
    for hp_name in hps:
        hp = generate_func(SearchSpace[hp_name])
        hp_dict[hp_name] = hp
    for hp_name in BaseHPs.keys():
        hp_dict[hp_name] = generate_func(BaseHPs[hp_name])
    return hp_dict

def str_hp(model_hp):
    hp_string = str()
    for key in model_hp.keys():
        hp_string += key + '='
        if isinstance(model_hp[key], str):
            hp_string += '\'' + model_hp[key] + '\'' + ', '
        else:
            hp_string += str(model_hp[key]) + ', '
    return hp_string

def cal_SearchSpace_size(models=None):
    size = 0
    models = ModelUse.keys() if models is None else models
    for model in models:
        hps = 1
        model_hps = ModelUse[model] + list(BaseHPs.keys())
        for hp_name in model_hps:
            hps *= len(SearchSpace[hp_name])
        size += hps
    return size

solutions_list = []
def dfs(model_name, hp_ind, solution_dict, pbar):
    global solutions_list

    model_hps = ModelUse[model_name] + list(BaseHPs.keys())
    hp_name = model_hps[hp_ind]

    if hp_ind + 1 >= len(model_hps):
        for hp_val in SearchSpace[hp_name]:
            solution_dict[hp_name] = hp_val
            solutions_list.append(copy.deepcopy(solution_dict))
            pbar.update(1)
        return

    for hp_val in SearchSpace[hp_name]:
        solution_dict[hp_name] = hp_val
        dfs(model_name, hp_ind+1, solution_dict, pbar)


def index_solutions(model_name: str, hp_dict: dict):
    """Input the solution(model_name, hp_dict), get the index of this solution in all the solutions.

        Input
            - model_name: a str in all model names.
            - hp_dict: a dict that can init the model.
        Output
            - int, the index of the solution.
    """
    try:
        solutions2id, id2solutions = pickle.load(open('solutions_map.pkl', 'rb'))
    except:
        size = cal_SearchSpace_size(['AFM'])
        pbar = tqdm(range(size))
        pbar.set_description('Indexing:')
        # for model in ModelUse.keys():
        for model in ['AFM']:
            solution_dict = {'model_name': model, }
            dfs(model, 0, solution_dict, pbar)
        global solutions_list
        solutions2id = {}
        id2solutions = {}
        for id, solution_dict in enumerate(solutions_list):
            solutions2id[str(solution_dict)] = id
            id2solutions[id] = solution_dict

        pickle.dump((solutions2id, id2solutions), open('solutions_map.pkl', 'wb'))






if __name__ == '__main__':
    solutions2id, id2solutions = pickle.load(open('solutions_map.pkl', 'rb'))

    hp_dict = {'use_attention': False, 'attention_factor': 40, 'l2_reg_linear': 3e-05, 'l2_reg_embedding': 0, 'l2_reg_att': 0.001, 'afm_dropout': 0.7, 'init_std': 0.0001, 'seed': 1024, 'embedding_size': 128, 'optimizer': 'rmsprop'}
    # solution = {'model_name': 'AFM'}.update(solution)
    solution_dict = {'model_name': 'AFM'}
    solution_dict.update(hp_dict)
    print(solutions2id[str(solution_dict)])

    # size = cal_SearchSpace_size()
    # print(size)
    #
    #
    # index_solutions('a', {})
    # print(solutions_list)



    # print(str_hp(SearchSpace['DIEN']))
    # a = init_hp_dict('DIEN', 'first')
    # b = str_hp(a)
    # print(b)