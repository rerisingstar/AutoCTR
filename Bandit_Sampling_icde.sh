# nohup python main_new.py --device 0 --search_method Bandit_Sampling --search_epochs 128 --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+Bandit_Sampling+nohup.txt &
# wait
# python autodl_notice.py AmazonElectronics+Bandit_Sampling

# nohup python main_new.py --device 0 --search_method Bandit_Sampling --search_epochs 128 --full_data_name AmazonBeauty > LOG/Nohup/AmazonBeauty+Bandit_Sampling+nohup.txt &
# wait
# python autodl_notice.py AmazonBeauty+Bandit_Sampling

# nohup python main_new.py --device 0 --search_method Bandit_Sampling --search_epochs 128 --full_data_name AmazonMovies_and_TV > LOG/Nohup/AmazonMovies_and_TV+Bandit_Sampling+nohup.txt &
# wait
# python autodl_notice.py AmazonMovies_and_TV+Bandit_Sampling

# nohup python main_new.py --device 0 --search_method Bandit_Sampling --search_epochs 128 --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+Bandit_Sampling+nohup.txt &
# wait
# python autodl_notice.py AmazonElectronics+Bandit_Sampling

nohup python main_new.py --device 0 --search_method Bandit_Sampling --search_epochs 128 --full_data_name MovieLens1m > LOG/Nohup/MovieLens1m+Bandit_Sampling+nohup.txt &
wait
python autodl_notice.py MovieLens1m+Bandit_Sampling

nohup python main_new.py --device 0 --search_method Bandit_Sampling --search_epochs 128 --full_data_name MovieLens > LOG/Nohup/MovieLens+Bandit_Sampling+nohup.txt &
wait
python autodl_notice.py MovieLens+Bandit_Sampling


