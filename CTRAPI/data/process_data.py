import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

'''
# 1.Label Encoding for sparse features,and process sequence features
        for feat in feature_names['sparse']:
            lbe = LabelEncoder()
            full_data[feat] = lbe.fit_transform(full_data[feat])
            hist_feat = 'hist_' + feat
            if hist_feat in feature_names['vallen']:
                for i, hist_list in enumerate(tqdm(full_data[hist_feat],
                                                   total=len(full_data[hist_feat]),
                                                   desc='labeling '+hist_feat)):
                    try:
                        zero_index = hist_list.index(0)
                    except ValueError:
                        zero_index = len(hist_list)
                    true_hist_list = lbe.transform(hist_list[:zero_index])
                    full_data[hist_feat][i] = list(true_hist_list) + hist_list[zero_index:]
'''

def process_amazon(max_his_length=20, sub_name='Beauty'):
    DIR = 'Amazon/' + sub_name + '/'
    try:
        data = pickle.load(open('processed_Amazon' + sub_name + '.pkl', 'rb'))
    except:
        uid_voc = pickle.load(open(DIR + 'uid_voc.pkl', 'rb'))
        mid_voc = pickle.load(open(DIR + 'mid_voc.pkl', 'rb'))
        cat_voc = pickle.load(open(DIR + 'cat_voc.pkl', 'rb'))
        source_dicts = []
        for source_dict in [uid_voc, mid_voc, cat_voc]:
            source_dicts.append(source_dict)

        feature_names = ['uid', 'mid', 'cat', 'hist_mid', 'hist_cat', 'seq_length']
        # feature_names += ['item_bhvs_uid_feats', 'item_bhvs_id_feats', 'item_bhvs_cat_feats']

        data = {}
        total_num = sum([1 for _ in open(DIR + 'local_all_sample_sorted_by_time', 'r')])
        with open(DIR + 'local_all_sample_sorted_by_time', 'r') as f_samp:
            for line in tqdm(f_samp, total=total_num, desc='reading local_all_sample_sorted_by_time'):
                ss = line.strip('\n').split('\t')
                uid = source_dicts[0][ss[1]] if ss[1] in source_dicts[0] else 0
                mid = source_dicts[1][ss[2]] if ss[2] in source_dicts[1] else 0
                cat = source_dicts[2][ss[3]] if ss[3] in source_dicts[2] else 0

                tmp = []
                for fea in ss[4].split("|"):
                    m = source_dicts[1][fea] if fea in source_dicts[1] else 0
                    tmp.append(m)
                seq_length = len(tmp)
                tmp += [0 for _ in range(max_his_length - len(tmp))]
                hist_mid = tmp

                tmp1 = []
                for fea in ss[5].split("|"):
                    c = source_dicts[2][fea] if fea in source_dicts[2] else 0
                    tmp1.append(c)
                tmp1 += [0 for _ in range(max_his_length - len(tmp1))]
                hist_cat = tmp1

                for feature in feature_names:
                    if feature in data.keys():
                        data[feature].append(eval(feature))
                    else:
                        data[feature] = [eval(feature)]

                if 'target' in data.keys():
                    data['target'].append(float(ss[0]))
                else:
                    data['target'] = [float(ss[0])]

            sample_num = len(data['uid'])
            for feat in feature_names + ['target']:
                data[feat] = np.array(data[feat]).reshape(sample_num, -1)

            pickle.dump(data, open('processed_Amazon' + sub_name + '.pkl', 'wb'))

def process_movielens(max_his_length=20):
    DIR = 'MovieLens/'
    try:
        data = pickle.load(open('processed_MovieLens.pkl', 'rb'))
    except:
        # load genre ids
        genre_dict = {}
        with open(DIR + 'u.genre', 'r') as f:
            for line in f:
                name, id = line.strip().split('|')
                genre_dict[name] = id
        gender_map = {
            'M': 1,
            'F': 2,
        }
        zip_voc = pickle.load(open(DIR + 'zipcode_voc.pkl', 'rb'))

        feature_names = ['uid', 'mid', 'cat',
                         'hist_mid', 'hist_cat', 'seq_length',
                         'gender', 'age', 'occup', 'u_zip'] #uid (user), mid (item), cat (first cat id) #hist_mid (pre 20) hist_cat (pre 20 cat)
        data = {}
        total_num = sum([1 for line in open(DIR + 'all_sample_with_histinfo', 'r')])
        with open(DIR + 'all_sample_with_histinfo', 'r') as f_samp:
            for line in tqdm(f_samp, total=total_num, desc='reading all_sample_with_histinfo'):
                ss = line.strip('\n').split('\t')

                uid, mid = [int(i) for i in ss[1:3]]
                genre_name = ss[3].split('|')[0]
                cat = int(genre_dict[genre_name])
                gender = gender_map[ss[6]]
                age, occup = [int(i) for i in ss[7:-2]]
                zipcode = ss[-2]
                u_zip = zip_voc[zipcode] if zipcode in zip_voc.keys() else 0

                tmp_m = [int(i) for i in ss[4].split(';')]
                tmp_m += [0 for _ in range(max_his_length - len(tmp_m))]
                seq_length = len(tmp_m)
                hist_mid = tmp_m

                tmp_g = []
                for fea in ss[5].split(';'):
                    str_list = fea.split('|')
                    id_list = [genre_dict[i] for i in str_list]
                    tmp_g.append(int(id_list[0]))
                tmp_g += [0 for _ in range(max_his_length - len(tmp_g))]
                hist_cat = tmp_g

                for feature in feature_names:
                    if feature in data.keys():
                        data[feature].append(eval(feature))
                    else:
                        data[feature] = [eval(feature)]

                if 'target' in data.keys():
                    data['target'].append(float(ss[0]))
                else:
                    data['target'] = [float(ss[0])]

            sample_num = len(data['uid'])
            for feat in feature_names + ['target']:
                data[feat] = np.array(data[feat]).reshape(sample_num, -1)

            pickle.dump(data, open('processed_MovieLens.pkl', "wb"))

def process_movielens10m(max_his_length=20):
    DIR = 'MovieLens10m/'
    try:
        data = pickle.load(open('processed_MovieLens10m.pkl', 'rb'))
    except:
        # # 电影排名
        # rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        # ratings = pd.read_table('MovieLens10m/ratings.dat', sep='::', header=None, names=rnames,engine='python')
        # print(ratings[:10])
        # # 电影信息
        # mnames = ['movie_id', 'title', 'genres']
        # movies = pd.read_table('MovieLens10m/movies.dat', sep='::', header=None, names=mnames, engine='python')
        # print(movies[:10])
        feature_names = ['uid', 'mid', 'cat',
                         'hist_mid', 'hist_cat', 'seq_length'] #uid (user), mid (item), cat (first cat id) #hist_mid (pre 20) hist_cat (pre 20 cat)

        # item_cat_mapping, user-hisitem
        cat_mapping = {}
        item_cat_mapping = {}
        total_num = sum([1 for line in open(DIR + 'movies.dat', 'r')])
        with open(DIR + 'movies.dat', 'r') as f_samp:
            for row in tqdm(f_samp, total=total_num, desc='reading movies.dat'):
                if row[-1] == '\n':
                    row = row[:-1]
                movie_id, title, genres = row.split("::")
                cat = genres.split("|")[0]
                if cat not in cat_mapping.keys():
                    cat_mapping[cat] = len(list(cat_mapping.keys()))
                item_cat_mapping[movie_id] = cat_mapping[cat]
                # print([movie_id, item_cat_mapping[movie_id]])

        user_items_mapping = {}
        total_num = sum([1 for line in open(DIR + 'ratings.dat', 'r')])
        with open(DIR + 'ratings.dat', 'r') as f_samp:
            for row in tqdm(f_samp, total=total_num, desc='reading ratings.dat'):
                if row[-1] == '\n':
                    row = row[:-1]
                user_id, movie_id, rating, timestamp = row.split("::")
                if user_id not in user_items_mapping.keys():
                    user_items_mapping[user_id] = []
                user_items_mapping[user_id].append([int(movie_id), int(timestamp)])
        for user in user_items_mapping.keys():
            items = user_items_mapping[user]
            items = sorted(items, key=lambda x: x[1])
            user_items_mapping[user] = [item[0] for item in items]

        data = {}
        total_num = len(list(user_items_mapping.keys()))
        for user in tqdm(user_items_mapping.keys(), total=total_num, desc='reading user_items_mapping'):
            uid = int(user)
            sorted_items = user_items_mapping[user]
            for i in range(len(sorted_items)):
                # possitive sample
                mid = int(sorted_items[i])
                cat = int(item_cat_mapping[str(mid)])

                hist_mid_tmp = sorted_items[:i][-1*max_his_length:]
                hist_mid_tmp = [int(item) for item in hist_mid_tmp]
                hist_mid = hist_mid_tmp + [0 for _ in range(max_his_length - len(hist_mid_tmp))]
                seq_length = len(hist_mid)
                hist_cat = [item_cat_mapping[str(hist_mid_tmp[j])] for j in range(len(hist_mid_tmp))]
                hist_cat += [0 for _ in range(max_his_length - len(hist_cat))]

                target = 1

                for feature in feature_names:
                    if feature in data.keys():
                        data[feature].append(eval(feature))
                    else:
                        data[feature] = [eval(feature)]

                if 'target' in data.keys():
                    data['target'].append(float(target))
                else:
                    data['target'] = [float(target)]

                # negitive sample
                neg_mid = random.sample(list(item_cat_mapping.keys()), 1)[0]
                while neg_mid in sorted_items:
                    neg_mid = random.sample(list(item_cat_mapping.keys()), 1)[0]
                mid = int(neg_mid)
                cat = int(item_cat_mapping[str(mid)])
                target = 0

                for feature in feature_names:
                    if feature in data.keys():
                        data[feature].append(eval(feature))
                    else:
                        data[feature] = [eval(feature)]

                if 'target' in data.keys():
                    data['target'].append(float(target))
                else:
                    data['target'] = [float(target)]

        sample_num = len(data['uid'])
        print("sample_num", sample_num)
        for feat in feature_names + ['target']:
            data[feat] = np.array(data[feat]).reshape(sample_num, -1)

        pickle.dump(data, open('processed_MovieLens10m.pkl', "wb"))
 

if __name__ == "__main__":
    process_amazon(sub_name='Beauty')
    #process_movielens()
    # process_movielens10m()
    