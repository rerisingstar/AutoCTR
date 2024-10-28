# performance comparison
nohup python main.py --device 0 --search_method Random --full_data_name MovieLens10m --epochs 5 --sample_ratio 0.1 --search_epochs 100 > LOG/Nohup/MovieLens10m+5+Random+nohup.txt & 
sleep 20
nohup python main.py --device 0 --search_method TPE_SMBO --full_data_name MovieLens10m --epochs 8 --sample_ratio 0.3 --search_epochs 240 > LOG/Nohup/MovieLens10m+5+TPE_SMBO+nohup.txt & 
sleep 20
nohup python main.py --device 0 --search_method SMAC_SMBO --full_data_name MovieLens10m --epochs 5 --sample_ratio 0.1 --search_epochs 100 > LOG/Nohup/MovieLens10m+5+SMAC_SMBO+nohup.txt & 
sleep 20
nohup python main.py --device 0 --search_method AutoCASH --full_data_name MovieLens10m --epochs 8 --gcn_flag --sample_ratio 0.3 --search_epochs 240 > LOG/Nohup/MovieLens10m+5+AutoCASH2+nohup.txt & 
sleep 20
nohup python main.py --device 0 --search_method MONO_MAB_SMBO --full_data_name MovieLens10m --epochs 5 --sample_ratio 0.1 --search_epochs 60 > LOG/Nohup/MovieLens10m+6+MONO_MAB_SMBO+nohup.txt & 


nohup python eval.py > LOG/Nohup/MovieLens10m+TPE+nohup.txt & 
nohup python eval.py > LOG/Nohup/MovieLens10m+Random+nohup.txt & 
nohup python eval.py > LOG/Nohup/MovieLens10m+SMAC+nohup.txt & 

Namespace(device=0, epochs=8, finetune_epoch=1, finetune_infonum=-1, full_data_name='MovieLens', gcn_flag=False, meta_ratio=1.0, reindex=False, retain_metapredictorepochs=-1, rising_c=10, rising_mode=1, sample_num=100, sample_ratio=1.0, search_epochs=640, search_method='Random', seed=1024, solution_randomelpsion=0.3)