# Collaborative-based Pre-training Stage
python run_seq.py --dataset='beauty' --gpu_id=0  --metrics="['Recall', 'NDCG', 'MRR']" --valid_metric="MRR@10" --train_batch_size=1024 --lmd=0.1 --lmd_sem=0.1 --model='DuoRec' --contrast='us_x' --sim='dot' --tau=1 --nproc=2 --epochs=100 --data_path="./recbole/dataset"
# Retrieval-Augmented Fine-tuning Stage
python run_seq.py --dataset='Amazon_Beauty' --data_path="./recbole/dataset" --pre_training_ckt="./log/DuoRec/Amazon_Beauty/bs1024-lmd0.1-sem0.1-us_x-Mar-19-2025_21-16-57-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth" --model='RaSeRec'
#gru4_cmd
python run_seq.py --dataset='Amazon_Beauty' --gpu_id=0 --metrics="['Recall', 'NDCG', 'MRR']" --valid_metric="MRR@10" --train_batch_size=1024 --model='GRU4Rec' --epochs=100 --data_path="./recbole/dataset"
#ftragrec
python run_ftragrec.py --dataset="Amazon_Beauty" --data_path="./recbole/dataset" --duorec_path="./log/DuoRec/Amazon_Beauty/bs1024-lmd0.1-sem0.1-us_x-Mar-19-2025_21-16-57-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/model.pth"