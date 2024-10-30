import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import pickle

def read_item_info():
    genre_dict = {}
    with open('u.genre', 'r') as f:
        for line in f:
            name, id = line.strip().split('|')
            genre_dict[int(id)] = name

    movie_info = {}
    total_items = sum([1 for _ in open('./u.item', 'r', encoding='ISO-8859-1')])
    with open('./u.item', 'r', encoding='ISO-8859-1') as f:
        for line in tqdm(f, total=total_items, desc='reading u_items...'):
            line = line.strip().split('|')
            mid, title, genres = line[0], line[1], list(map(int, line[5:]))
            title = '\"' + title + '\"'

            genre_str = ""
            if genres[0] == 1:
                genre_str = 'unknown|'
            else:
                for genre_id in list(genre_dict.keys())[1:]:
                    if genres[genre_id] == 1:
                        genre_str += genre_dict[genre_id] + '|'

            movie_info[mid] = {'title':title, 'genre':genre_str[:-1]}
    return movie_info

def read_user_info():
    total_users = sum([1 for _ in open('u.user', 'r')])
    user_info = {}
    with open('u.user', 'r') as f:
        for line in tqdm(f, total=total_users, desc='reading u.user...'):
            line = line.strip().split('|')
            uid, age, gender, occup, zip = line
            user_info[uid] = {'age':age, 'gender':gender, 'occupation':occup, 'zip':zip}
    return user_info

def read_rating_info(user_info, movie_info):
    occup_info = {}
    with open('u.occupation', 'r') as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]
        for i, occup in enumerate(lines):
            occup_info[occup] = str(i)



    total_ratings = sum([1 for _ in open('u.data', 'r')])
    with open('u.data', 'r') as f:
        sample_list = []
        for line in tqdm(f, total=total_ratings, desc='reading u.data...'):
            line = line.strip().split('\t')
            sample_list.append((line, line[-1]))
    sample_list = sorted(sample_list, key=lambda x: int(x[1]))

    zipcodes = []
    with open('joined-time-new_mvl', 'w') as f:
        for line in tqdm(sample_list, total=len(sample_list),
                         desc='adding neg to rating data...'):
            uid, mid, rating, timestamp = line[0]
            user = user_info[uid]
            zipcodes.append(user['zip'])
            j = 0
            while True:
                movie_neg = random.randint(1, len(movie_info))
                movie_neg = str(movie_neg)
                if movie_neg == mid:
                    continue
                movie = movie_info[movie_neg]

                f.write('\t'.join(['0', uid, movie_neg, rating, timestamp,
                                   movie['genre'],
                                   user['gender'], user['age'],
                                   occup_info[user['occupation']], user['zip']])+'\n')
                j += 1
                if j == 1:
                    break

            movie = movie_info[mid]
            f.write('\t'.join(['1', uid, mid, rating, timestamp,
                               movie['genre'],
                               user['gender'], user['age'],
                               occup_info[user['occupation']], user['zip']]) + '\n')

    zip_voc = {}
    zipcodes = list(set(zipcodes))
    for id in range(len(zipcodes)):
        zip_voc[zipcodes[id]] = id + 1
    pickle.dump(zip_voc, open('zipcode_voc.pkl', 'wb'))

def get_hist_info():
    fin = open('joined-time-new_mvl', 'r')
    ftest = open('all_sample_with_histinfo', 'w')

    max_hist_len = 20
    user_his_items = {}
    user_his_genres = {}

    total_len = sum([1 for i in open('joined-time-new_mvl', 'r')])
    for line in tqdm(fin, total=total_len, desc='geting all sample with hist info...'):
        items = line.strip().split('\t')
        flag = int(items[0])
        uid, mid, rating, timestamp = items[1:5]
        genres = items[5].split('|')
        gender, age, occup, u_zip = items[6:]

        if uid in user_his_items:
            his_items = user_his_items[uid][-max_hist_len:]
            his_genres = user_his_genres[uid][-max_hist_len:]
        else:
            his_items = []
            his_genres = []

        his_items_str = ';'.join(his_items)
        his_genres_str = ';'.join(['|'.join(i) for i in his_genres])

        if len(his_items) >= 1:
            ftest.write('\t'.join([items[0], uid, mid, items[5], his_items_str,
                                   his_genres_str, gender, age, occup, u_zip,
                                   timestamp]) + '\n')

        if flag:
            if uid not in user_his_items:
                user_his_items[uid] = []
                user_his_genres[uid] = []
            user_his_items[uid].append(mid)
            user_his_genres[uid].append(genres)



if __name__ == '__main__':
    try:
        movie_info = pickle.load(open('movie_info.pkl', 'rb'))
    except :
        movie_info = read_item_info()
        pickle.dump(movie_info, open('movie_info.pkl', 'wb'))

    try:
        user_info = pickle.load(open('user_info.pkl', 'rb'))
    except:
        user_info = read_user_info()
        pickle.dump(user_info, open('user_info.pkl', 'wb'))

    read_rating_info(user_info, movie_info)

    get_hist_info()



