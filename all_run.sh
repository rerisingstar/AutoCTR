
# Beauty
nohup python main.py --device 5 --search_method Random --full_data_name AmazonBeauty > LOG/Nohup/AmazonBeauty+0_Random+nohup.txt & 
sleep 15
nohup python main.py --device 5 --search_method TPE_SMBO --full_data_name AmazonBeauty > LOG/Nohup/AmazonBeauty+0+TPE_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 5 --search_method SMAC_SMBO --full_data_name AmazonBeauty > LOG/Nohup/AmazonBeauty+0+SMAC_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 5 --search_method MONO_MAB_SMBO --full_data_name AmazonBeauty > LOG/Nohup/AmazonBeauty+0+MONO_MAB_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 6 --search_method AutoCASH --full_data_name AmazonBeauty --use_attn > LOG/Nohup/AmazonBeauty+1+AutoCASH+nohup.txt & 
sleep 15

# Movies and TV
nohup python main.py --device 6 --search_method Random --full_data_name AmazonMovies_and_TV --sample_ratio 0.15 > LOG/Nohup/AmazonMovies_and_TV+1+Random+nohup.txt & 
sleep 15
nohup python main.py --device 6 --search_method TPE_SMBO --full_data_name AmazonMovies_and_TV --sample_ratio 0.15 > LOG/Nohup/AmazonMovies_and_TV+1+TPE_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 6 --search_method SMAC_SMBO --full_data_name AmazonMovies_and_TV --sample_ratio 0.15 > LOG/Nohup/AmazonMovies_and_TV+1+SMAC_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 7 --search_method MONO_MAB_SMBO --full_data_name AmazonMovies_and_TV --sample_ratio 0.15 > LOG/Nohup/AmazonMovies_and_TV+6+MONO_MAB_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonMovies_and_TV --use_attn --sample_ratio 0.15 > LOG/Nohup/AmazonMovies_and_TV+6+AutoCASH+nohup.txt & 
sleep 15

# Electronics
nohup python main.py --device 7 --search_method Random --full_data_name AmazonElectronics --sample_ratio 0.15 > LOG/Nohup/AmazonElectronics+6+Random+nohup.txt & 
sleep 15
nohup python main.py --device 7 --search_method TPE_SMBO --full_data_name AmazonElectronics --sample_ratio 0.15 > LOG/Nohup/AmazonElectronics+6+TPE_SMBO+nohup.txt & 
sleep 15

---------------------

nohup python main.py --device 5 --search_method SMAC_SMBO --full_data_name AmazonElectronics --sample_ratio 0.15 > LOG/Nohup/AmazonElectronics+7+SMAC_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 5 --search_method MONO_MAB_SMBO --full_data_name AmazonElectronics --sample_ratio 0.15 > LOG/Nohup/AmazonElectronics+7+MONO_MAB_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics --use_attn --sample_ratio 0.15 > LOG/Nohup/AmazonElectronics+7+AutoCASH+nohup.txt & 
sleep 15

# MovieLens
nohup python main.py --device 5 --search_method Random --full_data_name MovieLens > LOG/Nohup/MovieLens+7+Random+nohup.txt & 
sleep 15
nohup python main.py --device 6 --search_method TPE_SMBO --full_data_name MovieLens > LOG/Nohup/MovieLens+0+TPE_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 6 --search_method SMAC_SMBO --full_data_name MovieLens > LOG/Nohup/MovieLens+0+SMAC_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 6 --search_method MONO_MAB_SMBO --full_data_name MovieLens > LOG/Nohup/MovieLens+0+MONO_MAB_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 6 --search_method AutoCASH --full_data_name MovieLens > LOG/Nohup/MovieLens+0+AutoCASH+nohup.txt & 
sleep 15

# ablation
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonElectronics --use_attn --retain_metapredictorepochs 0 --sample_ratio 0.15 > LOG/Nohup/Ablation+AmazonElectronics+1+AutoCASH-Meta+nohup.txt & 
sleep 15
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonElectronics --retain_metapredictorepochs 50 --sample_ratio 0.15 > LOG/Nohup/Ablation+AmazonBeauty+1+AutoCASH-Attention+nohup.txt & 
sleep 10
nohup python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens --use_attn --retain_metapredictorepochs 0 > LOG/Nohup/Ablation+MovieLens+6+AutoCASH-Meta+nohup.txt & 
sleep 600
nohup python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens > LOG/Nohup/Ablation+MovieLens+7+AutoCASH-Attention+nohup.txt & 
sleep 10

---------------------

# meta ratio
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --use_attn --meta_ratio 0.1 --retain_metapredictorepochs 50 > LOG/Nohup/Parameter+AmazonBeauty+6+AutoCASH-m0.1+nohup.txt & 
sleep 10
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --use_attn --meta_ratio 0.3 --retain_metapredictorepochs 50 > LOG/Nohup/Parameter+AmazonBeauty+6+AutoCASH-m0.3+nohup.txt & 
sleep 10
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --use_attn --meta_ratio 0.5 --retain_metapredictorepochs 50 > LOG/Nohup/Parameter+AmazonBeauty+6+AutoCASH-m0.5+nohup.txt & 
sleep 10
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --use_attn --meta_ratio 0.7 --retain_metapredictorepochs 50 > LOG/Nohup/Parameter+AmazonBeauty+6+AutoCASH-m0.7+nohup.txt & 
sleep 10
nohup python main.py --device 6 --search_method AutoCASH --full_data_name MovieLens --use_attn --meta_ratio 0.9 --retain_metapredictorepochs 50 > LOG/Nohup/Parameter+AmazonBeauty+7+AutoCASH-m0.9+nohup.txt & 
sleep 10

# MovieLens1m
nohup python main.py --device 6 --search_method Random --full_data_name MovieLens1m --sample_ratio 0.15 > LOG/Nohup/MovieLens1m+7+Random+nohup.txt & 
sleep 10
nohup python main.py --device 6 --search_method TPE_SMBO --full_data_name MovieLens1m --sample_ratio 0.15 > LOG/Nohup/MovieLens1m+7+TPE_SMBO+nohup.txt & 
sleep 10
nohup python main.py --device 6 --search_method SMAC_SMBO --full_data_name MovieLens1m --sample_ratio 0.15 > LOG/Nohup/MovieLens1m+7+SMAC_SMBO+nohup.txt & 
sleep 10

nohup python main.py --device 7 --search_method MONO_MAB_SMBO --full_data_name MovieLens1m --sample_ratio 0.15 > LOG/Nohup/MovieLens1m+0+MONO_MAB_SMBO+nohup.txt & 
sleep 10
nohup python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens1m --use_attn --sample_ratio 0.15 > LOG/Nohup/MovieLens1m+0+AutoCASH+nohup.txt & 
sleep 10


