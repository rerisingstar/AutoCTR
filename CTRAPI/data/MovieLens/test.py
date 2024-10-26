

if __name__ == '__main__':
    d = {'a': 1, 'b': 2}
    e = {'c': 3, 'd': 4}
    d.update(e)
    print(d)
#     l = [1, 2, 3, 4, 5]
#     a = random.choice(l)
#     print(a)

    # a = ','.join(['a', 'b'])
    # print(a)

    # a = [{'A':1, 'B':0}, {'A':0, 'B':1}]
    # a = pd.DataFrame(a)
    #
    # b = pd.DataFrame(columns=['A', 'B', 'C'])
    # for i, row in a.iterrows():
    #     if row['A'] == 1:
    #         tmp = pd.Series([row['A'], row['B'], 1], index=list(b.columns))
    #         b = b.append(tmp, ignore_index=True)
    #
    # item_data_raw = pd.read_csv('u.item', sep='|', encoding='ISO-8859-1',
    #                             names=['movie id', 'movie title', 'release date', 'video release date',
    #                                    'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
    #                                    'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    #                                    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    #                                    'Thriller', 'War', 'Western', ])
    # item_data = item_data_raw.drop(columns=['movie title', 'release date', 'video release date', 'IMDb URL'])
    # item_data_new = pd.DataFrame(columns=['movie id', 'genre'])
    # for i, row in item_data.iterrows():
    #     if row['unknown'] == 1:
    #         tmp = pd.Series([row['movie id'], 'unknown'], index=list(item_data_new.columns))
    #         item_data_new.append(tmp, ignore_index=True)
    #     else:
    #         genre_str = ""
    #         for genre in list(item_data.columns[-18:]):
    #             if row[genre] == 1:
    #                 genre_str += genre + '|'
    #         tmp = pd.Series([row['movie id'], genre_str[:-1]], index=list(item_data_new.columns))
    #         item_data_new.append(tmp, ignore_index=True)
    # print(item_data.head())
    #
    # print(b)
