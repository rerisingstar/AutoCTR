import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__))))))), "CTRAPI"))
from pruned_search_space import BaseHPs, SearchSpace, ModelUse, index_solutions_prn_init, init_hp_dict_prn
from log import *
import random
import copy


class SolutionConverter:
    def __init__(self, edge_index_node):
        self.sk_map = self.component_to_keys()
        self.mk_map = self.model_names_to_keys()
        self.hp_dict_list, self.solution2id = self.init_solutions()
        self.choice_record = [0 for _ in range(len(self.hp_dict_list))]
        self.hpo_idx_map = dict()
        self.sampled_sidx_small_dict = self.get_solution_idx_small(edge_index_node)
        self.sampled_sidx_big_dict = self.get_solution_idx_big(edge_index_node)


    def get_components_num(self):
        return len(self.sk_map)

    def get_model_num(self):
        return len(self.mk_map)

    def get_solution_num(self):
        return len(self.hp_dict_list)

    def component_to_keys(self):
        '''
        :return dict component_name x embedding_idx
        '''
        keys = []
        def get_name(struct, name=""):
            if isinstance(struct, dict):
                for key in struct:
                    get_name(struct[key], name + key + ".")
            elif isinstance(struct, list):
                for content in struct:
                    keys.append(name + str(content))
        get_name(dict(BaseHPs,**SearchSpace))
        sk_map = {keys[i]: i for i in range(len(keys))}
        return sk_map

    def convert_solution_name(self, solution: dict):
        '''
        :return: embedding_idxes of the solution
        '''
        embeding_idxes = []
        for key in solution:
            name = key + "." + str(solution[key])
            embeding_idxes.append(self.sk_map[name])
        return embeding_idxes

    def model_names_to_keys(self):
        '''
        :return dict model_name x model_embedding_idx
        '''
        model_names = list(ModelUse.keys())
        return {model_names[i] : i for i in range(len(model_names))}

    def convert_model_name(self, model_name: str):
        return self.mk_map[model_name]

    def all_hp_dicts(self, hps):
        params = list(hps)
        if len(params) == 1:
            hp_dicts = [] 
            for value in SearchSpace[params[0]]:
                hp_dict = {}
                hp_dict[params[0]] = value
                hp_dicts.append(hp_dict)
        else:
            hp_dicts, pre_hp_dicts = [], []
            for value in SearchSpace[params[0]]:
                hp_dict = {}
                hp_dict[params[0]] = value
                pre_hp_dicts.append(hp_dict)
            new_hps = list(hps[1:])
            next_hp_dicts = self.all_hp_dicts(new_hps)
            for item1 in pre_hp_dicts:
                for item2 in next_hp_dicts:
                    hp_dict = copy.deepcopy(item1)
                    hp_dict.update(copy.deepcopy(item2))
                    hp_dicts.append(hp_dict)
        return hp_dicts

    def init_solutions(self):
        '''
        :return list(solution_dict: dict)
        '''
        hp_dict_list = []
        solution2id = {}
        for model_name in ModelUse:
            hps = copy.deepcopy(ModelUse[model_name]) + copy.deepcopy(list(BaseHPs.keys()))
            all_hp_dicts = self.all_hp_dicts(hps)
            for item in all_hp_dicts:
                solution_dict = {'model_name': model_name}
                solution_dict.update(item)
                hp_dict_list.append(solution_dict) 
                solution_name = self.get_solution_name(solution_dict)
                solution2id[solution_name] = len(hp_dict_list)-1
        return hp_dict_list, solution2id

    def get_solution_name(self, config):
        model_name = config['model_name']
        name = "model_name."+model_name
        key_names = ModelUse[model_name] + list(BaseHPs.keys())
        for key in key_names:
            value = config[key]
            name += "+"+str(key)+"."+str(value)
        return name

    def get_random_solution(self):
        model_name = random.sample(list(space.keys()), 1)[0]
        solution_dict = {'model_name': model_name}
        #print(model_name)
        #print(space[model_name].keys())
        #print(list(self.solution2id.keys())[:3])
        for key in space[model_name].keys():
            #print(key)
            solution_dict[key] = random.sample(space[model_name][key], 1)[0]
        solution_name = self.get_solution_name(solution_dict)
        sid = self.solution2id[solution_name]
        #print(solution_name)
        #print(sid)
        return sid
    def id2solution(self, sid: int):
        '''
        :return: dict model_code, string model_name
        '''
        model_code = self.hp_dict_list[sid]
        model_name = model_code["model_name"]
        model_code = model_code.copy()
        model_code.pop("model_name")
        return model_code, model_name

    def get_all_idx(self, solution_idx: int):
        '''
        :return: component_idxes: list, model_idx: int
        '''
        model_code, model_name = self.id2solution(solution_idx)
        model_type_idx = self.convert_model_name(model_name)
        component_idxes = self.hpo_idx_map.get(model_type_idx, None)
        if component_idxes is None:
            component_idxes = self.convert_solution_name(model_code)
            self.hpo_idx_map[model_type_idx] = component_idxes
        return component_idxes, model_type_idx

    def get_solution_idx_big(self, edge_index_node, max_samplenum=100):
        '''
        :return: component_idxes: list, model_idx: int
        '''
        sampled_sidx_big_dict = []
        for solution_idx in range(self.get_solution_num()):
            #print([solution_idx, self.get_solution_num()])
            sampled_sidx_big = []
            codename_neighbor_idxs = edge_index_node[solution_idx]
            while len(sampled_sidx_big) < max_samplenum:
                for idx in codename_neighbor_idxs:
                    tmp_num = 2
                    sampled_sidx_big += random.sample(edge_index_node[idx], tmp_num)
                    if len(sampled_sidx_big) == max_samplenum:
                        break
                    #print(len(sampled_sidx_big))
            sampled_sidx_big_dict.append(list(sampled_sidx_big))
        return sampled_sidx_big_dict

    def get_solution_idx_small(self, edge_index_node, max_samplenum=10):
        '''
        :return: component_idxes: list, model_idx: int
        '''
        sampled_sidx_small_dict = []
        for solution_idx in range(self.get_solution_num()):
            sampled_sidx_big = []
            codename_neighbor_idxs = edge_index_node[solution_idx]
            while len(sampled_sidx_big) < max_samplenum:
                for idx in codename_neighbor_idxs:
                    tmp_num = 1
                    sampled_sidx_big += random.sample(edge_index_node[idx], tmp_num)
                    if len(sampled_sidx_big) == max_samplenum:
                        break
            sampled_sidx_small_dict.append(list(sampled_sidx_big))
        return sampled_sidx_small_dict

    def record_choice(self, model_code):
        self.choice_record[model_code] += 1

    def get_record_choice(self):
        return self.choice_record.copy()






