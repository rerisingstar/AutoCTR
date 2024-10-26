# python main.py --search_method AutoCASH --full_data_name MovieLens --epochs 5 --device 0 --gcn_flag --retain_metapredictorepochs 100
# python main_new.py --search_method Bandit_Sampling --full_data_name MovieLens --epochs 5 --device 0
# python main_new.py --search_method Bandit_Sampling --full_data_name AmazonBeauty --epochs 5 --device 0
# python autodl_notice.py Beauty
python main_new.py --search_method Bandit_Sampling --full_data_name AmazonElectronics --device 0
# python autodl_notice.py Electronics
# python main_new.py --search_method Bandit_Sampling --full_data_name AmazonMovies_and_TV --device 0
# python autodl_notice.py AmazonMovies_and_TV