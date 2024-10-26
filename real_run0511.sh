

sleep 20
nohup python main.py --device 5 --search_method Random --full_data_name MovieLens1m --epochs 5 --sample_ratio 0.3 > LOG/Nohup/MovieLens1m+5+Random+nohup_5.txt & 
sleep 20
nohup python main.py --device 5 --search_method TPE_SMBO --full_data_name MovieLens1m --epochs 5 --sample_ratio 0.3 > LOG/Nohup/MovieLens1m+5+TPE_SMBO+nohup_5.txt & 
sleep 20
nohup python main.py --device 5 --search_method SMAC_SMBO --full_data_name MovieLens1m --epochs 5 --sample_ratio 0.3 > LOG/Nohup/MovieLens1m+5+SMAC_SMBO+nohup_5.txt & 
sleep 20
nohup python main.py --device 6 --search_method AutoCASH --full_data_name MovieLens1m --epochs 5 --gcn_flag --sample_ratio 0.3 > LOG/Nohup/MovieLens1m+6+AutoCASH+nohup_5.txt & 
sleep 20
nohup python main.py --device 6 --search_method MONO_MAB_SMBO --full_data_name MovieLens1m --epochs 5 --sample_ratio 0.3 > LOG/Nohup/MovieLens1m+6+MONO_MAB_SMBO+nohup_5.txt & 


sleep 20
nohup python main.py --device 6 --search_method Random --full_data_name AmazonElectronics --epochs 5 --sample_ratio 0.3 > LOG/Nohup/AmazonElectronics+6+Random+nohup_5.txt & 
sleep 20
nohup python main.py --device 7 --search_method TPE_SMBO --full_data_name AmazonElectronics --epochs 5 --sample_ratio 0.3 > LOG/Nohup/AmazonElectronics+7+TPE_SMBO+nohup_5.txt & 
sleep 20
nohup python main.py --device 7 --search_method SMAC_SMBO --full_data_name AmazonElectronics --epochs 5 --sample_ratio 0.3 > LOG/Nohup/AmazonElectronics+7+SMAC_SMBO+nohup_5.txt & 
sleep 20
nohup python main.py --device 6 --search_method AutoCASH --full_data_name AmazonElectronics --epochs 5 --gcn_flag --sample_ratio 0.3 > LOG/Nohup/AmazonElectronics+6+AutoCASH+nohup_5.txt & 
sleep 20
nohup python main.py --device 7 --search_method MONO_MAB_SMBO --full_data_name AmazonElectronics --epochs 5 --sample_ratio 0.3 > LOG/Nohup/AmazonElectronics+7+MONO_MAB_SMBO+nohup_5.txt & 


sleep 20
nohup python main.py --device 6 --search_method Random --full_data_name AmazonBeauty --epochs 5 > LOG/Nohup/AmazonBeauty+6+Random+nohup_5.txt & 
sleep 20
nohup python main.py --device 6 --search_method TPE_SMBO --full_data_name AmazonBeauty --epochs 5 > LOG/Nohup/AmazonBeauty+6+TPE_SMBO+nohup_5.txt & 
sleep 20
nohup python main.py --device 7 --search_method SMAC_SMBO --full_data_name AmazonBeauty --epochs 5 > LOG/Nohup/AmazonBeauty+7+SMAC_SMBO+nohup_5.txt & 
sleep 20
nohup python main.py --device 6 --search_method AutoCASH --full_data_name AmazonBeauty --epochs 5 --gcn_flag --retain_metapredictorepochs 100 > LOG/Nohup/AmazonBeauty+6+AutoCASH+nohup_5.txt & 
sleep 20
nohup python main.py --device 7 --search_method MONO_MAB_SMBO --full_data_name AmazonBeauty --epochs 5 > LOG/Nohup/AmazonBeauty+7+MONO_MAB_SMBO+nohup_5.txt & 


sleep 20
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonBeauty --epochs 5 --retain_metapredictorepochs 100 > LOG/Nohup/Ablation+AmazonBeauty+7+AutoCASH-Gcn+nohup_5.txt & 
sleep 20
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonBeauty --epochs 5 --gcn_flag --retain_metapredictorepochs 0 > LOG/Nohup/Ablation+AmazonBeauty+7+AutoCASH-Meta+nohup_5.txt & 

---------
---------
---------
---------
---------
---------

sleep 20
nohup python main.py --device 5 --search_method Random --full_data_name MovieLens1m --epochs 5 --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/MovieLens1m+5+Random+nohup_5_.txt & 
sleep 20
nohup python main.py --device 5 --search_method TPE_SMBO --full_data_name MovieLens1m --epochs 5 --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/MovieLens1m+5+TPE_SMBO+nohup_5_.txt & 
sleep 20
nohup python main.py --device 5 --search_method SMAC_SMBO --full_data_name MovieLens1m --epochs 5 --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/MovieLens1m+5+SMAC_SMBO+nohup_5_.txt & 
sleep 20
nohup python main.py --device 6 --search_method AutoCASH --full_data_name MovieLens1m --epochs 5 --gcn_flag --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/MovieLens1m+6+AutoCASH+nohup_5_.txt & 
sleep 20
nohup python main.py --device 6 --search_method MONO_MAB_SMBO --full_data_name MovieLens1m --epochs 5 --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/MovieLens1m+6+MONO_MAB_SMBO+nohup_5_.txt & 


sleep 20
nohup python main.py --device 6 --search_method Random --full_data_name MovieLens --epochs 5 --search_epochs 500 > LOG/Nohup/MovieLens+6+Random+nohup_5.txt & 
sleep 20
nohup python main.py --device 7 --search_method TPE_SMBO --full_data_name MovieLens --epochs 5 --search_epochs 500 > LOG/Nohup/MovieLens+7+TPE_SMBO+nohup_5.txt & 
sleep 20
nohup python main.py --device 7 --search_method SMAC_SMBO --full_data_name MovieLens --epochs 5 --search_epochs 500 > LOG/Nohup/MovieLens+7+SMAC_SMBO+nohup_5.txt & 
sleep 20
nohup python main.py --device 7 --search_method MONO_MAB_SMBO --full_data_name MovieLens --epochs 5 --search_epochs 500 > LOG/Nohup/MovieLens+7+MONO_MAB_SMBO+nohup_5.txt & 
------
sleep 20
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --epochs 5 --gcn_flag --search_epochs 500 > LOG/Nohup/MovieLens+5+AutoCASH+nohup_5.txt & 


sleep 20
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --epochs 5 --retain_metapredictorepochs 100 --search_epochs 500 > LOG/Nohup/Ablation+MovieLens+5+AutoCASH-Gcn+nohup_5.txt & 
sleep 20
nohup python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --epochs 5 --gcn_flag --retain_metapredictorepochs 0 --search_epochs 500 > LOG/Nohup/Ablation+MovieLens+5+AutoCASH-Meta+nohup_5.txt & 


sleep 20
nohup python main.py --device 6 --search_method Random --full_data_name AmazonElectronics --epochs 5 --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/AmazonElectronics+6+Random+nohup_5_.txt & 
sleep 20
nohup python main.py --device 6 --search_method TPE_SMBO --full_data_name AmazonElectronics --epochs 5 --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/AmazonElectronics+6+TPE_SMBO+nohup_5_.txt & 
sleep 20
nohup python main.py --device 6 --search_method SMAC_SMBO --full_data_name AmazonElectronics --epochs 5 --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/AmazonElectronics+6+SMAC_SMBO+nohup_5_.txt & 
sleep 20
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonElectronics --epochs 5 --gcn_flag --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/AmazonElectronics+7+AutoCASH+nohup_5_.txt & 
sleep 20
nohup python main.py --device 7 --search_method MONO_MAB_SMBO --full_data_name AmazonElectronics --epochs 5 --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/AmazonElectronics+7+MONO_MAB_SMBO+nohup_5_.txt & 


sleep 20
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonElectronics --epochs 5 --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/Ablation+AmazonElectronics+7+AutoCASH-Gcn+nohup_5_.txt & 
------
sleep 20
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics --epochs 5 --gcn_flag --retain_metapredictorepochs 0 --sample_ratio 0.5 --search_epochs 500 > LOG/Nohup/Ablation+AmazonElectronics+5+AutoCASH-Meta+nohup_5_.txt & 





nohup python main.py --device 1 --search_method AutoCASH --full_data_name MovieLens1m --epochs 5 --gcn_flag --search_epochs 500 > LOG/Nohup/MovieLens1m+1+AutoCASH+nohup_5.txt & 

nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics --epochs 5 --gcn_flag --search_epochs 500 > LOG/Nohup/AmazonElectronics+5+AutoCASH+nohup_5.txt & 
nohup python main.py --device 5 --search_method Random --full_data_name AmazonElectronics --epochs 5 --search_epochs 500 > LOG/Nohup/AmazonElectronics+5+Random+nohup_5.txt & 
nohup python main.py --device 6 --search_method AutoCASH --full_data_name MovieLens1m --epochs 5 --gcn_flag --search_epochs 500 > LOG/Nohup/MovieLens1m+6+AutoCASH+nohup_5.txt & 
nohup python main.py --device 6 --search_method Random --full_data_name MovieLens1m --epochs 5 --search_epochs 500 > LOG/Nohup/MovieLens1m+6+Random+nohup_5.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonBeauty --epochs 5 --gcn_flag --search_epochs 500 > LOG/Nohup/AmazonBeauty+7+AutoCASH+nohup_5.txt & 
nohup python main.py --device 7 --search_method Random --full_data_name AmazonBeauty --epochs 5 --search_epochs 500 > LOG/Nohup/AmazonBeauty+7+Random+nohup_5.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens --epochs 5 --gcn_flag --search_epochs 500 > LOG/Nohup/MovieLens+7+AutoCASH+nohup_5.txt & 
nohup python main.py --device 7 --search_method Random --full_data_name MovieLens --epochs 5 --search_epochs 500 > LOG/Nohup/MovieLens+7+Random+nohup_5.txt & 

nohup python main.py --device 7 --search_method AutoCASH --full_data_name AmazonBeauty --epochs 5 --gcn_flag --search_epochs 500 --finetune_infonum 16 > LOG/Nohup/AmazonBeauty+7+AutoCASH+nohup_5.txt & 
nohup python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens --epochs 5 --gcn_flag --search_epochs 500 --finetune_infonum 16 > LOG/Nohup/MovieLens+7+AutoCASH+nohup_5.txt & 
nohup python main.py --device 5 --search_method AutoCASH --full_data_name AmazonElectronics --epochs 5 --gcn_flag --search_epochs 500 --finetune_infonum 16 > LOG/Nohup/AmazonElectronics+5+AutoCASH+nohup_5.txt & 



python main.py --device 5 --search_method AutoCASH --full_data_name AmazonBeauty --epochs 5 --gcn_flag --search_epochs 500 --retain_metapredictorepochs 200 --sample_num 100 --finetune_epoch 1
python main.py --device 6 --search_method AutoCASH --full_data_name MovieLens --epochs 5 --gcn_flag --search_epochs 500 --retain_metapredictorepochs 20 --sample_num 300 
python main.py --device 7 --search_method AutoCASH --full_data_name MovieLens --epochs 5 --gcn_flag --search_epochs 500 --retain_metapredictorepochs 20 --sample_num 300 


#python main.py --device 6 --search_method AutoCASH --full_data_name MovieLens --epochs 5 --gcn_flag --search_epochs 500 --retain_metapredictorepochs 20 --sample_num 100

python main.py --device 5 --search_method AutoCASH --full_data_name MovieLens --gcn_flag 
python main.py --device 6 --search_method AutoCASH --full_data_name AmazonBeauty --gcn_flag 
python main.py --device 7 --search_method AutoCASH --full_data_name AmazonElectronics --gcn_flag 
