import os
import sys
import numpy
from tqdm import tqdm
import random
import pickle

DIR = ''

def process_meta(file):
    num_fi = sum([1 for i in open(DIR+file, "r")])
    fi = open(DIR+file, "r")
    with open(DIR+"item-info", "w") as fo:
        for line in tqdm(fi, total = num_fi, desc="item_info"):
            obj = eval(line)
            cat = obj["categories"][0][-1]
            fo.write(obj["asin"] + "\t" + cat + "\n")

def process_reviews(file):
    num_fi = sum([1 for i in open(DIR+file, "r")])
    fi = open(DIR+file, "r")
    with open(DIR+"reviews-info", "w") as fo:
        for line in tqdm(fi, total=num_fi, desc="reviews_info"):
            obj = eval(line)
            userID = obj["reviewerID"]
            itemID = obj["asin"]
            rating = obj["overall"]
            time = obj["unixReviewTime"]
            fo.write(userID + "\t" + itemID + "\t" +
                     str(rating) + "\t" + str(time) + "\n")

def manual_join_as_time():
    num_f = sum([1 for i in open(DIR+"reviews-info", "r")])
    with open(DIR+"reviews-info", "r") as f:
        item_list = []
        sample_list = []
        for line in tqdm(f, total=num_f, desc="samples"):
            line = line.strip()
            items = line.split("\t")
            sample_list.append((line, float(items[-1])))
            item_list.append(items[1])
        sample_list = sorted(sample_list, key=lambda x: x[1])

    num_f = sum([1 for i in open(DIR+"item-info", "r")])
    with open(DIR+"item-info", "r") as f:
        meta_map = {}
        for line in tqdm(f, total=num_f, desc="meta_map"):
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1]

    with open(DIR+"joined-time-new", "w") as f:
        for line in tqdm(sample_list, desc="join_time_new"):
            items = line[0].split("\t")
            asin = items[1]
            j = 0
            while True:
                asin_neg_index = random.randint(0, len(item_list) - 1)
                asin_neg = item_list[asin_neg_index]
                if asin_neg == asin:
                    continue
                items[1] = asin_neg
                f.write("0" + "\t" + "\t".join(items) +
                        "\t" + meta_map[asin_neg] + "\n")
                j += 1
                if j == 1:
                    break
            if asin in meta_map:
                f.write("1" + "\t" + line[0] +
                        "\t" + meta_map[asin] + "\n")
            else:
                f.write("1" + "\t" + line[0] +
                        "\t" + meta_map[asin] + "default_cat")

maxlen = 20
user_maxlen = 50
def get_all_samples():
    fin = open(DIR+"joined-time-new", "r")
    # ftrain = open("local_train", "w")
    ftest = open(DIR+"local_all_sample_sorted_by_time", "w")

    user_his_items = {}
    user_his_cats = {}
    item_his_users = {}
    last_user = "0"
    common_fea = ""
    line_idx = 0
    for line in fin:
        items = line.strip().split("\t")
        clk = int(items[0])
        user = items[1]
        item_id = items[2]
        dt = items[4]
        cat1 = items[5]
        if user in user_his_items:
            bhvs_items = user_his_items[user][-maxlen:]
        else:
            bhvs_items = []
        if user in user_his_cats:
            bhvs_cats = user_his_cats[user][-maxlen:]
        else:
            bhvs_cats = []

        user_history_clk_num = len(bhvs_items)
        bhvs_items_str = "|".join(bhvs_items)
        bhvs_cats_str  = "|".join(bhvs_cats)

        if item_id in item_his_users:
            item_clk_users = item_his_users[item_id][-user_maxlen:]
        else:
            item_clk_users = []
        item_history_user_num = len(item_clk_users)
        history_users_feats = ";".join(item_clk_users)
        if user_history_clk_num >= 1:    # 8 is the average length of user behavior
            ftest.write(items[0] + "\t" + user + "\t" + item_id + "\t" + cat1 +"\t" + bhvs_items_str + "\t" +
                bhvs_cats_str+ "\t" + history_users_feats+"\t" +dt + "\n")
        if clk:
            if user not in user_his_items:
                user_his_items[user] = []
                user_his_cats[user] = []
            user_his_items[user].append(item_id)
            user_his_cats[user].append(cat1)
            if item_id not in item_his_users:
                item_his_users[item_id] = []
            if user_history_clk_num >=1:
                item_bhvs_feat = user+'_'+bhvs_items_str+'_'+bhvs_cats_str
            else:
                item_bhvs_feat = user+'_'+''+'_'+''
            if user_history_clk_num >= 1:
                item_his_users[item_id].append(item_bhvs_feat)
        line_idx += 1

def get_cut_time(percent=0.85):
    time_list = []
    fin = open(DIR+"local_all_sample_sorted_by_time", "r")
    for line in fin:
        line = line.strip()
        time = float(line.split("\t")[-1])
        time_list.append(time)
    sample_size = len(time_list)
    print(sample_size)
    train_size = int(sample_size * percent)
    time_list = sorted(time_list, key=lambda x: x)
    cut_time = time_list[train_size]
    return cut_time

def split_test_by_time(cut_time):
  fin = open(DIR+"local_all_sample_sorted_by_time", "r")
  ftrain = open(DIR+"local_train_sample_sorted_by_time", "w")
  ftest = open(DIR+"local_test_sample_sorted_by_time", "w")

  for line in fin:
    line = line.strip()
    time = float(line.split("\t")[-1])

    if time <= cut_time:
      ftrain.write(line+"\n")
    else:
      ftest.write(line+'\n')

def get_voc():
    with open(DIR+"local_all_sample_sorted_by_time", 'r') as f:
        uid_dict = {}
        mid_dict = {}
        cat_dict = {}
        iddd = 0
        for line in f:
            arr = line.strip("\n").split("\t")
            clk = arr[0]
            uid = arr[1]
            mid = arr[2]
            cat = arr[3]
            mid_list = arr[4]
            cat_list = arr[5]
            if uid not in uid_dict:
                uid_dict[uid] = 0
            uid_dict[uid] += 1
            if mid not in mid_dict:
                mid_dict[mid] = 0
            mid_dict[mid] += 1
            if cat not in cat_dict:
                cat_dict[cat] = 0
            cat_dict[cat] += 1
            if len(mid_list) == 0:
                continue
            for m in mid_list.split("|"):
                if m not in mid_dict:
                    mid_dict[m] = 0
                mid_dict[m] += 1
            # print iddd
            iddd += 1
            for c in cat_list.split("|"):
                if c not in cat_dict:
                    cat_dict[c] = 0
                cat_dict[c] += 1

        sorted_uid_dict = sorted(uid_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_mid_dict = sorted(mid_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_cat_dict = sorted(cat_dict.items(), key=lambda x: x[1], reverse=True)

        uid_voc = {}
        uid_voc["default_uid"] = 0
        index = 1
        # Ranking of statistical user mentions
        for key, value in sorted_uid_dict:
            uid_voc[key] = index
            index += 1

        mid_voc = {}
        mid_voc["default_mid"] = 0
        index = 1
        for key, value in sorted_mid_dict:
            mid_voc[key] = index
            index += 1

        cat_voc = {}
        cat_voc["default_cat"] = 0
        index = 1
        for key, value in sorted_cat_dict:
            cat_voc[key] = index
            index += 1

        pickle.dump(uid_voc, open(DIR+"uid_voc.pkl", "wb"))
        pickle.dump(mid_voc, open(DIR+"mid_voc.pkl", "wb"))
        pickle.dump(cat_voc, open(DIR+"cat_voc.pkl", "wb"))



if __name__ == "__main__":
    sub_dataname = sys.argv[1]
    DIR = './' + sub_dataname + '/'

    download_path = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
    reviews_file_name = 'reviews_' + sub_dataname + '_5.json.gz'
    meta_file_name = 'meta_' + sub_dataname + '.json.gz'

    try:
        os.mkdir(DIR)
        os.chdir(DIR)

        os.system('wget -c ' + download_path + reviews_file_name)
        os.system('wget -c ' + download_path + meta_file_name)

        os.system('gzip -d ' + reviews_file_name)
        os.system('gzip -d ' + meta_file_name)

        os.chdir('../')
    except FileExistsError:
        pass

    process_meta(meta_file_name[:-3])
    process_reviews(reviews_file_name[:-3])
    manual_join_as_time()
    get_all_samples()

    get_voc()