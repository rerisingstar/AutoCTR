#python main.py --device 6 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --retain_metapredictorepochs 20 --sample_num 100

nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --gcn_flag > LOG/Nohup/MovieLens+5+AutoCASH+nohup.txt & 
nohup python main.py --device 5 --search_method Random --full_data_name MovieLens --gcn_flag > LOG/Nohup/MovieLens+5+Random+nohup.txt & 

nohup python main.py --device 6 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag > LOG/Nohup/AmazonBeauty+6+AutoCASH+nohup.txt & 
nohup python main.py --device 6 --search_method Random --full_data_name AmazonBeauty --gcn_flag > LOG/Nohup/AmazonBeauty+6+Random+nohup.txt & 

nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag > LOG/Nohup/AmazonElectronics+7+AutoCASH+nohup.txt & 
nohup python main.py --device 7 --search_method Random --full_data_name AmazonElectronics --gcn_flag > LOG/Nohup/AmazonElectronics+7+Random+nohup.txt & 

nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonMovies_and_TV --gcn_flag > LOG/Nohup/AmazonMovies_and_TV+5+AutoCASH+nohup.txt & 
nohup python main.py --device 6 --search_method Random --full_data_name AmazonMovies_and_TV --gcn_flag > LOG/Nohup/AmazonMovies_and_TV+6+Random+nohup.txt & 

nohup python main.py --device 6 --search_method AutoCASH --full_data_name MovieLens1m --gcn_flag > LOG/Nohup/MovieLens1m+6+AutoCASH+nohup.txt & 
nohup python main.py --device 5 --search_method Random --full_data_name MovieLens1m --gcn_flag > LOG/Nohup/MovieLens1m+5+Random+nohup.txt & 

nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --retain_metapredictorepochs 0 > LOG/Nohup/AmazonBeauty+7+AutoCASH-meta+nohup.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonBeauty --retain_metapredictorepochs 20 > LOG/Nohup/AmazonBeauty+7+AutoCASH-gcn+nohup.txt & 

nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --retain_metapredictorepochs 0 > LOG/Nohup/AmazonElectronics+7+AutoCASH-meta+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+5+AutoCASH-gcn+nohup.txt & 

nohup python main.py --device 0 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --solution_randomelpsion 0.1 > LOG/Nohup/AmazonBeauty+0+AutoCASH-r0.1+nohup.txt & 
nohup python main.py --device 1 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --solution_randomelpsion 0.5 > LOG/Nohup/AmazonBeauty+1+AutoCASH-r0.5+nohup.txt & 
nohup python main.py --device 1 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --solution_randomelpsion 0.7 > LOG/Nohup/AmazonBeauty+1+AutoCASH-r0.7+nohup.txt & 

nohup python main.py --device 0 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --solution_randomelpsion 0.1 > LOG/Nohup/AmazonElectronics+0+AutoCASH-r0.1+nohup.txt & 
nohup python main.py --device 0 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --solution_randomelpsion 0.5 > LOG/Nohup/AmazonElectronics+0+AutoCASH-r0.5+nohup.txt & 
nohup python main.py --device 6 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --solution_randomelpsion 0.7 > LOG/Nohup/AmazonElectronics+6+AutoCASH-r0.7+nohup.txt & 

nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --meta_ratio 0.1 --retain_metapredictorepochs 20 > LOG/Nohup/AmazonBeauty+5+AutoCASH-m0.1+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --meta_ratio 0.3 --retain_metapredictorepochs 20 > LOG/Nohup/AmazonBeauty+5+AutoCASH-m0.3+nohup.txt & 

---------------
# performance analyze
nohup python main.py --device 1 --search_method AutoCASH --full_data_name MovieLens --gcn_flag > LOG/Nohup/MovieLens+1+AutoCASH+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag > LOG/Nohup/AmazonBeauty+5+AutoCASH+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag > LOG/Nohup/AmazonElectronics+5+AutoCASH+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonMovies_and_TV --gcn_flag > LOG/Nohup/AmazonMovies_and_TV+5+AutoCASH+nohup.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens1m --gcn_flag > LOG/Nohup/MovieLens1m+7+AutoCASH+nohup.txt & 

# performance analyze
nohup python main.py --device 0 --search_method AutoCASH --full_data_name MovieLens --gcn_flag > LOG/Nohup/MovieLens+0+AutoCASH+nohup_2.txt & 
nohup python main.py --device 1 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag > LOG/Nohup/AmazonBeauty+1+AutoCASH+nohup_2.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag > LOG/Nohup/AmazonElectronics+5+AutoCASH+nohup_2.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonMovies_and_TV --gcn_flag > LOG/Nohup/AmazonMovies_and_TV+7+AutoCASH+nohup_2.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens1m --gcn_flag > LOG/Nohup/MovieLens1m+7+AutoCASH+nohup_2.txt & 

# ablation analyze
nohup python main.py --device 0 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --retain_metapredictorepochs 0 > LOG/Nohup/MovieLens+0+AutoCASH-m0.0+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --meta_ratio 0.1 --retain_metapredictorepochs 20 > LOG/Nohup/MovieLens+5+AutoCASH-m0.1+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --meta_ratio 0.3 --retain_metapredictorepochs 20 > LOG/Nohup/MovieLens+5+AutoCASH-m0.3+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --meta_ratio 0.5 --retain_metapredictorepochs 20 > LOG/Nohup/MovieLens+5+AutoCASH-m0.5+nohup.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --meta_ratio 0.7 --retain_metapredictorepochs 20 > LOG/Nohup/MovieLens+7+AutoCASH-m0.7+nohup.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --meta_ratio 0.9 --retain_metapredictorepochs 20 > LOG/Nohup/MovieLens+7+AutoCASH-m0.9+nohup.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens --retain_metapredictorepochs 20 > LOG/Nohup/MovieLens+7+AutoCASH-nogcn+nohup.txt & 

# ablation analyze
nohup python main.py --device 0 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --retain_metapredictorepochs 0 > LOG/Nohup/AmazonElectronics+0+AutoCASH-m0.0+nohup.txt & 
nohup python main.py --device 0 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --meta_ratio 0.1 > LOG/Nohup/AmazonElectronics+0+AutoCASH-m0.1+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --meta_ratio 0.3 > LOG/Nohup/AmazonElectronics+5+AutoCASH-m0.3+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --meta_ratio 0.5 > LOG/Nohup/AmazonElectronics+5+AutoCASH-m0.5+nohup.txt & 
nohup python main.py --device 6 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --meta_ratio 0.7 > LOG/Nohup/AmazonElectronics+6+AutoCASH-m0.7+nohup.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --meta_ratio 0.9 > LOG/Nohup/AmazonElectronics+7+AutoCASH-m0.9+nohup.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+7+AutoCASH-nogcn+nohup.txt & 

# ablation analyze
nohup python main.py --device 0 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --retain_metapredictorepochs 0 > LOG/Nohup/AmazonBeauty+0+AutoCASH-m0.0+nohup.txt & 
nohup python main.py --device 0 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --meta_ratio 0.1 > LOG/Nohup/AmazonBeauty+0+AutoCASH-m0.1+nohup.txt & 
nohup python main.py --device 0 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --meta_ratio 0.3 > LOG/Nohup/AmazonBeauty+0+AutoCASH-m0.3+nohup.txt & 
nohup python main.py --device 1 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --meta_ratio 0.5 > LOG/Nohup/AmazonBeauty+1+AutoCASH-m0.5+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --meta_ratio 0.7 > LOG/Nohup/AmazonBeauty+5+AutoCASH-m0.7+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --meta_ratio 0.9 > LOG/Nohup/AmazonBeauty+5+AutoCASH-m0.9+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonBeauty > LOG/Nohup/AmazonBeauty+5+AutoCASH-nogcn+nohup.txt & 

# ablation analyze
nohup python main.py --device 0 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --sample_num 50 --retain_metapredictorepochs 20 > LOG/Nohup/MovieLens+0+AutoCASH-s50+nohup.txt & 
nohup python main.py --device 0 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --sample_num 150 --retain_metapredictorepochs 20 > LOG/Nohup/MovieLens+0+AutoCASH-s150+nohup.txt & 
nohup python main.py --device 1 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --sample_num 200 --retain_metapredictorepochs 20 > LOG/Nohup/MovieLens+1+AutoCASH-s200+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --gcn_flag --sample_num 250 --retain_metapredictorepochs 20 > LOG/Nohup/MovieLens+5+AutoCASH-s250+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --retain_metapredictorepochs 0 > LOG/Nohup/MovieLens+5+AutoCASH-m0.0-nogcn+nohup.txt & 

# ablation analyze
nohup python main.py --device 1 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --sample_num 50 > LOG/Nohup/AmazonElectronics+1+AutoCASH-s50+nohup.txt & 
nohup python main.py --device 0 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --sample_num 150 > LOG/Nohup/AmazonElectronics+0+AutoCASH-s150+nohup.txt & 
nohup python main.py --device 0 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --sample_num 200 > LOG/Nohup/AmazonElectronics+0+AutoCASH-s200+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag --sample_num 250 > LOG/Nohup/AmazonElectronics+5+AutoCASH-s250+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics --retain_metapredictorepochs 0 > LOG/Nohup/AmazonElectronics+5+AutoCASH-m0.0-nogcn+nohup.txt & 

# ablation analyze
nohup python main.py --device 1 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --sample_num 50 > LOG/Nohup/AmazonBeauty+1+AutoCASH-s50+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --sample_num 150 > LOG/Nohup/AmazonBeauty+5+AutoCASH-s150+nohup.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --sample_num 200 > LOG/Nohup/AmazonBeauty+5+AutoCASH-s200+nohup.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --sample_num 250 > LOG/Nohup/AmazonBeauty+7+AutoCASH-s250+nohup.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonBeauty --retain_metapredictorepochs 0 > LOG/Nohup/AmazonBeauty+7+AutoCASH-m0.0-nogcn+nohup.txt & 

# mate curve
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag --retain_metapredictorepochs 20 > LOG/Nohup/AmazonBeauty+7+AutoCASH+nohup.txt & 

# performance comparison
nohup python main.py --device 0 --search_method Random --full_data_name MovieLens > LOG/Nohup/MovieLens+0+Random+nohup.txt & 
nohup python main.py --device 1 --search_method TPE_SMBO --full_data_name MovieLens > LOG/Nohup/MovieLens+1+TPE_SMBO+nohup.txt & 
nohup python main.py --device 5 --search_method SMAC_SMBO --full_data_name MovieLens > LOG/Nohup/MovieLens+5+SMAC_SMBO+nohup.txt & 
nohup python main.py --device 5 --search_method MONO_MAB_SMBO --full_data_name MovieLens > LOG/Nohup/MovieLens+5+MONO_MAB_SMBO+nohup.txt & 

# performance comparison
nohup python main.py --device 0 --search_method Random --full_data_name AmazonBeauty > LOG/Nohup/AmazonBeauty+0+Random+nohup.txt & 
nohup python main.py --device 5 --search_method TPE_SMBO --full_data_name AmazonBeauty > LOG/Nohup/AmazonBeauty+5+TPE_SMBO+nohup.txt & 
nohup python main.py --device 5 --search_method SMAC_SMBO --full_data_name AmazonBeauty > LOG/Nohup/AmazonBeauty+5+SMAC_SMBO+nohup.txt & 
nohup python main.py --device 7 --search_method MONO_MAB_SMBO --full_data_name AmazonBeauty > LOG/Nohup/AmazonBeauty+7+MONO_MAB_SMBO+nohup.txt & 


nohup python main.py --device 5 --search_method Random --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+5+Random+nohup.txt & 
nohup python main.py --device 5 --search_method TPE_SMBO --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+5+TPE_SMBO+nohup.txt & 
nohup python main.py --device 0 --search_method SMAC_SMBO --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+0+SMAC_SMBO+nohup.txt & 
nohup python main.py --device 7 --search_method MONO_MAB_SMBO --full_data_name AmazonElectronics > LOG/Nohup/AmazonElectronics+7+MONO_MAB_SMBO+nohup.txt & 


nohup python main.py --device 5 --search_method Random --full_data_name AmazonMovies_and_TV > LOG/Nohup/AmazonMovies_and_TV+5+Random+nohup.txt & 
nohup python main.py --device 7 --search_method TPE_SMBO --full_data_name AmazonMovies_and_TV > LOG/Nohup/AmazonMovies_and_TV+7+TPE_SMBO+nohup.txt & 
nohup python main.py --device 0 --search_method SMAC_SMBO --full_data_name AmazonMovies_and_TV > LOG/Nohup/AmazonMovies_and_TV+0+SMAC_SMBO+nohup.txt & 
nohup python main.py --device 0 --search_method MONO_MAB_SMBO --full_data_name AmazonMovies_and_TV > LOG/Nohup/AmazonMovies_and_TV+0+MONO_MAB_SMBO+nohup.txt & 


nohup python main.py --device 5 --search_method Random --full_data_name MovieLens1m > LOG/Nohup/MovieLens1m+5+Random+nohup.txt & 
nohup python main.py --device 5 --search_method TPE_SMBO --full_data_name MovieLens1m > LOG/Nohup/MovieLens1m+5+TPE_SMBO+nohup.txt & 
nohup python main.py --device 7 --search_method SMAC_SMBO --full_data_name MovieLens1m > LOG/Nohup/MovieLens1m+7+SMAC_SMBO+nohup.txt & 
nohup python main.py --device 7 --search_method MONO_MAB_SMBO --full_data_name MovieLens1m > LOG/Nohup/MovieLens1m+7+MONO_MAB_SMBO+nohup.txt & 

