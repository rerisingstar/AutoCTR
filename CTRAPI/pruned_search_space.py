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
    'optimizer': ['adam', 'adagrad', 'rmsprop']
}

SearchSpace = {
    'l2_reg_cin': [0, 1e-5, 1e-6],
    'l2_reg_att': [0, 1e-5, 1e-6],
    'l2_reg_dnn': [0, 1e-5, 1e-6],
    'l2_reg_cross': [0, 1e-5, 1e-6],
    'l2_reg_linear': [0, 1e-5, 1e-6],
    'l2_reg_embedding': [0, 1e-5, 1e-6],

    'dnn_dropout': [0, 0.2, 0.5, 0.8],
    'dnn_use_bn': [False, True],
    'dnn_activation': ['relu', 'sigmoid', 'linear', 'prelu'],
    'dnn_hidden_units': [[400, 400, 400],
                         [512, 256, 64], [256, 64],
                         [32, 32, 32], (256, 128), [512, 512, 512],
                         [768, 768], (256, 256), (128, 128),
                         [256, ], [256, 256, 256], [64, 64, 64],
                         [128, 128, 128], [512, 512], [8], [16]],  #2-5 32-1024

    'att_res': [True, False],
    'att_head_num': [1, 2, 4, 8],
    'att_layer_num': [1, 2, 3, 4],
    'att_activation': ['relu', 'sigmoid', 'linear', 'prelu'],
    'att_hidden_units': [[600, 600], [300, 300], [200, 200], [80, 80], (256, 128)], #DIEN
    'att_hidden_size': [[600, 600], [300, 300], [200, 200], [80, 80], (256, 128)],  #DIN
    'attention_factor': [8, 20, 32, 40],
    'att_weight_normalization': [True, False],

    'cross_num': [2, 3, 4, 5],
    'cross_parameterization': ['vector', 'matrix'],

    'conv_filters': [[4, 4, 2], [3, 4, 2], (4, 4)],
    'conv_kernel_width': [[6, 5, 3], (6, 5)],

    'ltl_hidden_size': [5, 10, 100, 256, 512],

    'afm_dropout': [0, 0.2, 0.5, 0.8],

    'afn_dnn_hidden_units': [[400], [100, 100, 100, 100],
                             [400, 400], [400, 400, 400],
                             [256, 128]],  #1,2,3,4  100 400 1000

    'low_rank': [24, 32, 48, 64], #

    'num_experts': [3, 4, 5], #

    'use_fm': [True, False],
    'use_bn': [False, True],
    'use_inner': [True, False],
    'use_outter': [False, True],
    'use_attention': [True, False],

    'gru_type': ['GRU', 'AIGRU', 'AGRU', 'AUGRU'],
    'kernel_type': ['mat', 'vec', 'num'],

    'alpha': [0.2, 0.4, 0.6, 0.8, 1.0],

    'bi_dropout': [0, 0.2, 0.5, 0.8],
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
    'AFM': ['use_attention', 'attention_factor', 'afm_dropout',],
            # 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_att', ],
    'AFN': ['ltl_hidden_size', 'afn_dnn_hidden_units', 'dnn_dropout', 'dnn_activation', ],
            # 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn', ],
    'AutoInt': ['att_layer_num', 'att_head_num', 'att_res',
                'dnn_hidden_units', 'dnn_activation', ],
                # 'l2_reg_dnn', 'l2_reg_embedding', 'dnn_use_bn', 'dnn_dropout'],
    'CCPM': ['conv_kernel_width', 'conv_filters',
             'dnn_hidden_units', 'dnn_activation', 'dnn_dropout', ],
             # 'dnn_use_bn', 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn', ],
    'DCN': ['cross_num', 'cross_parameterization',
            'dnn_hidden_units', 'dnn_dropout', 'dnn_activation', ],
            # 'dnn_use_bn', 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_cross', 'l2_reg_dnn', ],
    'DCNMix': ['cross_num', 'low_rank', 'num_experts',
               'dnn_hidden_units', 'dnn_dropout', ],
               # 'dnn_activation', 'dnn_use_bn', 'l2_reg_linear',
               # 'l2_reg_embedding', 'l2_reg_cross', 'l2_reg_dnn',  ],
    'DeepFM': ['dnn_hidden_units', 'dnn_dropout', 'dnn_activation', ],
               # 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn',],
    'DIEN': ['gru_type', 'alpha', 'use_bn', 'att_hidden_units', ],
             # 'dnn_hidden_units', 'dnn_activation', 'att_activation', 'att_weight_normalization',
             # 'l2_reg_dnn', 'l2_reg_embedding', 'dnn_dropout'],
    'DIFM': ['att_head_num', 'att_res', 'dnn_hidden_units', 'dnn_dropout', 'dnn_activation', ],
             # 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn', 'dnn_use_bn'],
    'DIN': ['att_activation', 'att_weight_normalization', 'dnn_hidden_units',
            'dnn_activation', 'att_hidden_size', ],
            # 'l2_reg_dnn', 'dnn_use_bn', 'l2_reg_embedding', 'dnn_dropout'],
    'FiBiNET': ['bilinear_type', 'reduction_ratio', 'dnn_hidden_units',
                'dnn_dropout', 'dnn_activation', ],
                # 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn', ],
    'IFM': ['dnn_hidden_units', 'dnn_dropout', 'dnn_activation', 'dnn_use_bn', 'l2_reg_linear', ],
            # 'l2_reg_embedding', 'l2_reg_dnn',],
    'MLR': ['region_num', 'l2_reg_linear'],
    'NFM': ['bi_dropout', 'dnn_hidden_units', 'dnn_dropout', 'dnn_activation', ],
            # 'l2_reg_embedding', 'l2_reg_linear', 'l2_reg_dnn',],
    'ONN': ['dnn_hidden_units', 'dnn_dropout', 'dnn_use_bn', 'dnn_activation', ],
            # 'l2_reg_embedding', 'l2_reg_linear', 'l2_reg_dnn',],
    'PNN': ['use_inner', 'use_outter', 'kernel_type', 'dnn_dropout', 'dnn_activation',],
            # 'l2_reg_embedding', 'l2_reg_dnn', ],
    'WDL': ['dnn_hidden_units', 'dnn_dropout', 'dnn_activation', 'dnn_use_bn', ],
            # 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn',],
    'xDeepFM': ['dnn_hidden_units', 'cin_layer_size', 'cin_split_half',
                'cin_activation', 'l2_reg_cin', ],
                # 'l2_reg_linear', 'l2_reg_embedding', 'l2_reg_dnn',
                # 'dnn_dropout', 'dnn_activation', 'dnn_use_bn']
}

def get_node_num_graph(reindex=False):
    solutions2id, id2solutions, solutions_num = index_solutions_prn_init(reindex=reindex)
    node_num = solutions_num
    added_nodes, edge_index, edge_index_node = {}, [], {}
    model_name_ids = []
    #print("ok")
    #print(solutions_num)
    for id in range(solutions_num):
        #print(id)
        solution = id2solutions[id]
        #print(solution)
        for key in solution.keys():
            value = solution[key]
            node_name = str(key)+"@"+str(value)
            if node_name not in added_nodes.keys():
                added_nodes[node_name] = node_num
                node_num += 1
            if key == "model_name":
                model_name_ids.append(added_nodes[node_name])
                #print("ok")
            #print(added_nodes[node_name])
            #print([added_nodes[node_name], id])
            edge_index.append([added_nodes[node_name], id])
            edge_index.append([id, added_nodes[node_name]])
            if added_nodes[node_name] not in edge_index_node.keys():
                edge_index_node[added_nodes[node_name]] = []
            edge_index_node[added_nodes[node_name]].append(id)
            if id not in edge_index_node.keys():
                edge_index_node[id] = []
            edge_index_node[id].append(added_nodes[node_name])
    #node_name = "model_name@ROOT"
    #added_nodes[node_name] = node_num
    #node_num += 1
    #edge_index_node[added_nodes[node_name]] = list(model_name_ids)
    #for mnid in model_name_ids:
    #    edge_index_node[mnnid].append(added_nodes[node_name])
    return edge_index, node_num, edge_index_node

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

def init_hp_dict_prn(model_name, func_name='random'):
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


def index_solutions_prn(model_name: str, hp_dict: dict, reindex=False):
    """Input the solution(model_name, hp_dict), get the index of this solution in all the solutions.

        Input
            - model_name: a str in all model names.
            - hp_dict: a dict that can init the model.
        Output
            - int, the index of the solution.
    """
    if not reindex:
        solutions2id, id2solutions = pickle.load(open('solutions_map.pkl', 'rb'))
    else:
        size = cal_SearchSpace_size()
        pbar = tqdm(range(size))
        pbar.set_description('Indexing:')
        # for model in ModelUse.keys():
        for model in ModelUse.keys():
            solution_dict = {'model_name': model, }
            dfs(model, 0, solution_dict, pbar)
        global solutions_list
        solutions2id = {}
        id2solutions = {}
        for id, solution_dict in enumerate(solutions_list):
            solutions2id[str(solution_dict)] = id
            id2solutions[id] = solution_dict

        pickle.dump((solutions2id, id2solutions), open('solutions_map.pkl', 'wb'))

    solution_dict = {'model_name': model_name}
    solution_dict.update(hp_dict)
    return solutions2id[str(solution_dict)]


def index_solutions_prn_init(reindex=False):
    """Input the solution(model_name, hp_dict), get the index of this solution in all the solutions.

        Input
            - model_name: a str in all model names.
            - hp_dict: a dict that can init the model.
        Output
            - int, the index of the solution.
    """
    if not reindex:
        solutions2id, id2solutions = pickle.load(open('solutions_map.pkl', 'rb'))
    else:
        size = cal_SearchSpace_size()
        pbar = tqdm(range(size))
        pbar.set_description('Indexing:')
        # for model in ModelUse.keys():
        for model in ModelUse.keys():
            solution_dict = {'model_name': model, }
            dfs(model, 0, solution_dict, pbar)
        global solutions_list
        solutions2id = {}
        id2solutions = {}
        for id, solution_dict in enumerate(solutions_list):
            solutions2id[str(solution_dict)] = id
            id2solutions[id] = solution_dict

        pickle.dump((solutions2id, id2solutions), open('solutions_map.pkl', 'wb'))

    return solutions2id, id2solutions, len(list(id2solutions.keys()))







if __name__ == '__main__':
    #hp_dict = init_hp_dict_prn('DIEN')
    #print(hp_dict)
    #id = index_solutions_prn('DIEN', hp_dict)
    #print(id)
    get_node_num_graph(reindex=False)

    # solutions2id, id2solutions = pickle.load(open('solutions_map.pkl', 'rb'))
    #
    # solution = {'use_attention': False, 'attention_factor': 40, 'l2_reg_linear': 3e-05, 'l2_reg_embedding': 0, 'l2_reg_att': 0.001, 'afm_dropout': 0.7, 'init_std': 0.0001, 'seed': 1024, 'embedding_size': 128, 'optimizer': 'rmsprop'}
    # solution = {'model_name': 'AFM'}.update(solution)
    # print(solutions2id[str(solution)])

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