# nohup python main_new.py --device 0 --search_method AutoCASH --full_data_name AmazonBeauty --epochs 5 --retain_metapredictorepochs 100 > LOG/Nohup/AutoCASH_Beauty_test.txt & 

# 5/32
# python main_new.py --device 0 --search_method AutoCASH --epochs 5 --newinfo_model_nums 5 --total_model_nums 32 --full_data_name AmazonElectronics > LOG/AmazonElectronics+AutoCASH_NEW+5+32+nohup_5.txt & 
# sleep 20
# python main_new.py --device 0 --search_method AutoCASH --epochs 5 --newinfo_model_nums 5 --total_model_nums 32 --full_data_name MovieLens > LOG/Movielens100k+AutoCASH_NEW+5+32+nohup_5.txt &
# sleep 20

# 3/16
# nohup python main_new.py --device 0 --search_method AutoCASH --gcn_flag --newinfo_model_nums 3 --total_model_nums 16 --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+AutoCASH_NEW+3+16+nohup_5.txt &
# wait
# python autodl_notice.py AmazonElectronics+AutoCASH_NEW+3+16
# nohup python main_new.py --device 0 --search_method AutoCASH --gcn_flag --newinfo_model_nums 3 --total_model_nums 16 --full_data_name MovieLens > LOG/Nohup/Movielens100k+AutoCASH_NEW+3+16+nohup_5.txt &
# wait
# python autodl_notice.py Movielens100k+AutoCASH_NEW+3+16

# # 10/64
# nohup python main_new.py --device 0 --search_method AutoCASH --gcn_flag --newinfo_model_nums 10 --total_model_nums 64 --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+AutoCASH_NEW+10+64+nohup_5.txt &
# wait
# python autodl_notice.py AmazonElectronics+AutoCASH_NEW+10+64
# nohup python main_new.py --device 0 --search_method AutoCASH --gcn_flag --newinfo_model_nums 10 --total_model_nums 64 --full_data_name MovieLens > LOG/Nohup/Movielens100k+AutoCASH_NEW+10+64+nohup_5.txt &
# wait
# python autodl_notice.py Movielens100k+AutoCASH_NEW+10+64

# 15/96
nohup python main_new.py --device 0 --search_method AutoCASH --gcn_flag --newinfo_model_nums 15 --total_model_nums 96 --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+AutoCASH_NEW+15+96+nohup_5.txt &
wait
python autodl_notice.py AmazonElectronics+AutoCASH_NEW+15+96
nohup python main_new.py --device 0 --search_method AutoCASH --gcn_flag --newinfo_model_nums 15 --total_model_nums 96 --full_data_name MovieLens > LOG/Nohup/Movielens100k+AutoCASH_NEW+15+96+nohup_5.txt &
wait
python autodl_notice.py Movielens100k+AutoCASH_NEW+15+96