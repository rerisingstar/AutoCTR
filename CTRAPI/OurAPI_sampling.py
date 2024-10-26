import copy
import pickle
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch_local.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch_local.models import *
from CTRSpace import *

from log import *
import numpy as np




class FeatureColumns:
    def __init__(self, feature_names, sparse_feat_num, max_hist_num=20):
        '''
        :param feature_names: dict, {'sparse':[], 'dense':[], 'vallen':[]}
        :param sparse_feat_num: dict, max num of items in every sparse feature
        :param max_hist_num: int, must equal to the param in load_data
        '''
        self.sparse_feat = feature_names['sparse']
        self.dense_feat = feature_names['dense']
        self.vallen_feat = feature_names['vallen']

        self.sparse_feat_num = sparse_feat_num

        self.max_hist_num = max_hist_num

    def get_fc(self, embedding_size=64):
        sparse_feature_columns = []
        for feat in self.sparse_feat:
            max_num = self.sparse_feat_num[feat]
            sparse_feature_columns.append(SparseFeat(feat, max_num, embedding_dim=embedding_size))

        dense_feature_columns = [DenseFeat(feat, 1, ) for feat in self.dense_feat]

        hist_feature_columns = []
        behavior_feature_list = []
        for feat in self.vallen_feat:
            sparse_feat = feat[5:]
            behavior_feature_list.append(sparse_feat)
            hist_feature_columns.append(
                VarLenSparseFeat(SparseFeat(feat, vocabulary_size=self.sparse_feat_num[sparse_feat],
                                            embedding_dim=embedding_size, embedding_name=sparse_feat),
                                 maxlen=self.max_hist_num, length_name='seq_length')
            )

        linear_feature_columns = sparse_feature_columns + dense_feature_columns + hist_feature_columns
        dnn_feature_columns = linear_feature_columns

        return linear_feature_columns, dnn_feature_columns, behavior_feature_list




class CTR(nn.Module):
    def __init__(self, args):
        super(CTR, self).__init__()
        self.args = args

    def split(self, x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in self.key2index:
                # Notice : input value 0 is a special "padding",
                # so we do not use 0 to encode valid feature for sequence input
                self.key2index[key] = len(self.key2index) + 1
        return list(map(lambda x: self.key2index[x], key_ans))

    def CTRModel(self, model_name, hp_dict, feature_columns,
                 device='cpu', task='binary', gpus=None,
                 logger=None, embedding_size=None):
        '''
        :param model_name: str
        :param hp_dict: dict
        :param feature_columns: class, FeatureColumns
        :param device: str
        :param task: str
        :param gpus: None
        :param optimizer: str
        :return: nn.Module
        '''

        if embedding_size is not None:
            em_size = embedding_size
        else:
            em_size = hp_dict['embedding_size']
        linear_feature_columns, dnn_feature_columns, behavior_feature_list = feature_columns.get_fc(em_size)
        optimizer = hp_dict['optimizer']
        del hp_dict['embedding_size'], hp_dict['optimizer']


        model_init_str = model_name
        if model_name in ['DIEN', 'DIN']:
            model_init_str += '(dnn_feature_columns, behavior_feature_list, '
        elif model_name in ['PNN']:
            model_init_str += '(dnn_feature_columns=dnn_feature_columns, '
        elif model_name in ['MLR']:
            model_init_str += '(region_feature_columns=dnn_feature_columns, '
        elif model_name in ['FiBiNET', 'xDeepFM', 'FiBiNET', 'AFN', 'AutoInt']:
            new_linear_columns = [feat for feat in linear_feature_columns
                                  if not isinstance(feat, VarLenSparseFeat)]
            new_dnn_columns = [feat for feat in dnn_feature_columns
                               if not isinstance(feat, VarLenSparseFeat)]
            model_init_str += '(linear_feature_columns=new_linear_columns, ' \
                              'dnn_feature_columns=new_dnn_columns, '
        elif model_name in ['CCPM', 'AFM']:
            new_linear_columns = [feat for feat in linear_feature_columns
                                  if not isinstance(feat, DenseFeat)]
            new_dnn_columns = [feat for feat in dnn_feature_columns
                               if not isinstance(feat, DenseFeat)]
            model_init_str += '(linear_feature_columns=new_linear_columns, ' \
                              'dnn_feature_columns=new_dnn_columns, '
        else:
            model_init_str += '(linear_feature_columns=linear_feature_columns, ' \
                              'dnn_feature_columns=dnn_feature_columns, '
        model_init_str += str_hp(hp_dict)
        model_init_str += 'device=device, task=task, gpus=gpus)'

        model = eval(model_init_str)

        hp_dict['embedding_size'] = em_size
        hp_dict['optimizer'] = optimizer
        logger.info(f"model name is {model_name}")
        logger.info(f'model hp dict is {hp_dict}')

        return model

    def CTRData(self, full_data_name, searching_flag=False, ratio=0.2, random_state=2020, sample_ratio=1.0):
        '''
        :param data_name: str
        :param searching_flag: bool
        :param ratio:
        :param random_state:
        :return: data_dict, feature_info
            data_dict: dict, {'train_input', ....}, data and target of train, val, and test,
                    don't has val if searching flag = false
            feature_columns: class: FeatureColumns
        '''

        data_name, sub_name = full_data_name

        print('loading ' + data_name + sub_name + '...')
        import os
        dir_name = os.path.dirname(os.path.realpath(__file__))
        full_data = pickle.load(open(dir_name+'/data/processed_' +
                                     data_name + sub_name + '.pkl', 'rb'))
        print('dataset ready!')

        feature_names = {}
        sparse_feature_nums = {}

        if data_name == 'Amazon':
            feature_names['sparse'] = ['uid', 'mid', 'cat']
            feature_names['dense'] = []
            feature_names['vallen'] = ['hist_mid', 'hist_cat']
        elif data_name in ['MovieLens', 'MovieLens1m']:
            feature_names['sparse'] = ["uid", "mid", 'cat',
                                       "gender", "occup", "u_zip", ]
            feature_names['dense'] = ['age']
            feature_names['vallen'] = ['hist_mid', 'hist_cat']
        elif data_name in ['MovieLens10m', 'MovieLens20m']:
            feature_names['sparse'] = ["uid", "mid", 'cat']
            feature_names['dense'] = []
            feature_names['vallen'] = ['hist_mid', 'hist_cat']
        for feat in feature_names['sparse']:
            sparse_feature_nums[feat] = int(max(full_data[feat]) + 1)
        feature_columns = FeatureColumns(feature_names, sparse_feature_nums)

        # labeling is in the preprocessing session

        # minmax the dense feature
        mms = MinMaxScaler(feature_range=(0, 1))
        for feat in feature_names['dense']:
            full_data[feat] = mms.fit_transform(full_data[feat])


        # select and split data
        feature_names_list = []
        for key in feature_names.keys():
            feature_names_list += feature_names[key]
        feature_names_list.append('seq_length')
        part_data = {
            feature_name: full_data[feature_name] for feature_name in feature_names_list
        }
        part_data['target'] = full_data['target']

        # splited: [[train, test] * 4]
        splited = [
            train_test_split(part_data[feat], test_size=ratio, random_state=random_state)
            for feat in part_data.keys()
        ]
        train_model_input = {}
        test_model_input = {}
        for i, name in enumerate(feature_names_list):
            train_model_input[name] = splited[i][0]
            test_model_input[name] = splited[i][1]
        train_target = splited[-1][0]
        test_target = splited[-1][1]

        if sample_ratio < 1.0:
            for name in train_model_input.keys():
                _, train_model_input[name] = train_test_split(
                    train_model_input[name], test_size=sample_ratio, shuffle=False
                )
            _, train_target = train_test_split(
                    train_target, test_size=sample_ratio, shuffle=False
                )
            
        data_dict = {
            'train_input': train_model_input,
            'train_target': train_target,
            'test_input': test_model_input,
            'test_target': test_target
        }

        # ### 5 folds data sampling code ###
        # five_folds, data_dict = [], []
        # # 2 random folds
        # for index in range(2):
        #     splited = [
        #         train_test_split(part_data[feat], train_size=0.2, random_state=random_state*index)
        #         for feat in part_data.keys()
        #     ]
        #     model_input = {}
        #     for i, name in enumerate(feature_names_list):
        #         model_input[name] = splited[i][0]
        #     target = splited[-1][0]
        #     five_folds.append(list([model_input, target]))
        # # 3 biased folds
        # data_num = part_data[feat].shape[0]
        # data_num_split = [0, int(data_num*0.3), int(data_num*0.6), data_num]
        # for index in range(3):
        #     splited_biased = [
        #         train_test_split(part_data[feat][data_num_split[index]:data_num_split[index+1]], train_size=0.5, random_state=random_state*index)
        #         for feat in part_data.keys()
        #     ]
        #     splited = [
        #         train_test_split(part_data[feat], train_size=0.05, random_state=random_state * index)
        #         for feat in part_data.keys()
        #     ]
        #     model_input = {}
        #     for i, name in enumerate(feature_names_list):
        #         model_input[name] = np.concatenate((splited_biased[i][0], splited[i][0]), axis=0)
        #     target = np.concatenate((splited_biased[-1][0], splited[-1][0]), axis=0)
        #     five_folds.append(list([model_input, target]))
        # data_dict = []
        # for index in range(len(five_folds)):
        #     model_input, target = five_folds[index]
        #     other_folds_inputs, other_folds_targets = [], []
        #     for t in range(index):
        #         other_folds_inputs.append(five_folds[t][0])
        #         other_folds_targets.append(five_folds[t][1])
        #     for t in range(index+1, len(five_folds)):
        #         other_folds_inputs.append(five_folds[t][0])
        #         other_folds_targets.append(five_folds[t][1])
                
        #     train_model_input = {}
            
        #     for key in model_input.keys():
        #         inputs_to_concatenate = [fold[key] for fold in other_folds_inputs]
        #         train_model_input[key] = np.concatenate(inputs_to_concatenate, axis=0)
                
        #     train_target = np.concatenate(other_folds_targets, axis=0)
        #     data_dict_fold = {
        #         'train_input': train_model_input,
        #         'train_target': train_target,
        #         'test_input': model_input,
        #         'test_target': target
        #     }
        #     data_dict.append(data_dict_fold)
        ### 5 folds data sampling code ###

        return data_dict, feature_columns


    def CTRtrain(self, model, data_dict,
                 batch_size=256, epochs=10, searching_flag=False, verbose=1,
                 logger=None):
        '''
        :param model: nn.Module
        :param data_dict: dict, {'train_input':.., 'train_target':...}
        :param batch_size: int
        :param epochs: int
        :param searching_flag:
        :param verbose: show the bar or not
        :return: history, test_logloss, test_auc
            history: dict
            test_logloss, test_auc: float64
        '''
        train_model_input = copy.deepcopy(data_dict['train_input'])
        train_target = copy.deepcopy(data_dict['train_target'])
        test_model_input = copy.deepcopy(data_dict['test_input'])
        test_target = copy.deepcopy(data_dict['test_target'])

        ##

        if searching_flag == True:
            val_model_input = {}
            for name in train_model_input.keys():
                train_model_input[name], val_model_input[name] = train_test_split(
                    train_model_input[name], test_size=0.2, shuffle=False
                )
            train_target, val_target = train_test_split(train_target,
                                                        test_size=0.2, shuffle=False)
            final_test_input = val_model_input
            final_test_target = val_target
        else:
            final_test_input = test_model_input
            final_test_target = test_target

        recorder = ModelRecorder()
        history = model.fit(train_model_input, train_target, batch_size=batch_size, epochs=epochs,
                            verbose=verbose, validation_split=0.2, recorder=recorder, logger=logger)

        best_model = copy.deepcopy(recorder.best_model)
        pred_ans = best_model.predict(final_test_input, 256)
        test_logloss = round(log_loss(final_test_target, pred_ans), 4)
        test_auc = round(roc_auc_score(final_test_target, pred_ans), 4)
        logger.info("test LogLoss:" + str(test_logloss))
        logger.info("test AUC:" + str(test_auc))

        return history.history, test_logloss, test_auc, pred_ans, final_test_target




class ModelRecorder:
    '''
    Func1: record the best model through epochs
    Func2: early stop
    '''
    def __init__(self, stop_ep=2):
        # Func1
        self.best_model = None
        self.best_val_auc = 0

        # Func2
        self.stop_ep = stop_ep
        self.last_auc = 0
        self.drop_times = 0

    def update(self, model, val_auc, epoch):
        '''
        :param model: nn.model
        :param val_auc: int
        :param epoch: int
        :return: True if the training can go on, False if the training should stop
        '''

        if val_auc <= self.last_auc:
            self.drop_times += 1
            if self.drop_times >= 2:
                return False
        else:
            self.drop_times = 0

        self.last_auc = val_auc

        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_model = copy.deepcopy(model)

        return True


def test_model(model_name, hp_dict,
               data_name='Amazon', sub_name='Beauty',
               device='cuda:6', epochs=5, batch_size=512):
    assert model_name != None
    assert hp_dict != None

    # batch_size = 2048 if sub_name == 'Books' else 512
    # batch_size = 512
    logger = set_logger(model_name, data_name + '_' + sub_name)

    ctr = CTR(args)
    data_dict, feature_columns = ctr.CTRData(full_data_name=(data_name, sub_name))

    optimizer = hp_dict['optimizer']
    model = ctr.CTRModel(model_name, hp_dict, feature_columns, device=device, logger=logger)
    model.compile(optimizer, "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    logger.info(f'epochs={epochs}, batchsize={batch_size}, begin training...')
    ctr.CTRtrain(model, data_dict, epochs=epochs, batch_size=batch_size, searching_flag=False, logger=logger)

'''
def train(args):
    ctr = CTR(args)
    data_name = args.data
    model_name = args.model
    sub_name = args.sub_dataname
    em_size = args.embedding_size
    batch_size = 2048 if sub_name == 'Books' else 512
    epochs = 10


    data_dict, feature_columns = ctr.CTRData(full_data_name=(data_name, sub_name))

    logger = set_logger(model_name + str(em_size), data_name + '_' + sub_name)

    # hp_dict = init_hp_dict(model_name, 'first')

    model_name = 'PNN'
    hp_dict = {
        'l2_reg_embedding': 1e-05,
        'l2_reg_dnn': 0.0001,
        'dnn_dropout': 0.6,
        'dnn_activation': 'relu',
        'use_inner': True,
        'use_outter': False,
        'kernel_type': 'vec',
        'init_std': 0.0001,
        'seed': 1024,
        'embedding_size': 64,
        'optimizer': 'rmsprop'
    }
    optimizer = hp_dict['optimizer']
    # del hp_dict['optimizer']
    model = ctr.CTRModel(model_name, hp_dict, feature_columns, device=args.device, logger=logger)
    model.compile(optimizer, "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    logger.info(f'epochs={epochs}, batchsize={batch_size}, begin training...')
    ctr.CTRtrain(model, data_dict, epochs=epochs, batch_size=batch_size, searching_flag=False, logger=logger)
'''

if __name__ == "__main__":
    model_name = 'AFN'
    # hp_dict = {'gru_type': 'AIGRU', 'alpha': 0.4, 'use_bn': True, 'dnn_hidden_units': [2000, 2000, 2000], 'dnn_activation': 'relu', 'att_hidden_units': (256, 128), 'att_activation': 'sigmoid', 'att_weight_normalization': False, 'l2_reg_dnn': 0.0001, 'l2_reg_embedding': 1e-06, 'dnn_dropout': 0.7, 'init_std': 0.0001, 'seed': 1024, 'embedding_size': 32, 'optimizer': 'adam'}
    hp_dict = {'ltl_hidden_size': 512, 'afn_dnn_hidden_units': [400, 400], 'dnn_dropout': 0.8, 'dnn_activation': 'linear', 'init_std': 0.0001, 'seed': 1024, 'embedding_size': 64, 'optimizer': 'rmsprop'}
    test_model(model_name, hp_dict, device='cuda:7', data_name='MovieLens', sub_name='', batch_size=512)

    # train(args)

    # ctr = CTR(args)
    # data_name = args.data
    # model_name = args.model
    # sub_name = args.sub_dataname
    # em_size = args.embedding_size
    # device = args.device
    #
    # data_dict, feature_columns = ctr.CTRData(full_data_name=(data_name, sub_name))
    #
    # logger = set_logger(model_name+str(em_size), data_name+'_'+sub_name)
    #
    # hp_dict = init_hp_dict(model_name, 'first')
    # # model = ctr.CTRModel(model_name, hp_dict, feature_columns, device=args.device, embedding_size=em_size)
    # linear_feature_columns, dnn_feature_columns, behavior_feature_list = feature_columns.get_fc(em_size)
    # model = AutoInt(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                 task='binary',
    #                 l2_reg_embedding=1e-5, device=device)
    # print(f'sucessfully init a {model_name} model')
    # model.compile("adagrad", "binary_crossentropy",
    #               metrics=["binary_crossentropy", "auc"], )
    # ctr.CTRtrain(model, data_dict, epochs=10, batch_size=512, searching_flag=False, logger=logger)

