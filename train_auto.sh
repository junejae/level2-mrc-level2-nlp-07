python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 1 \
--weight_decay 0.0 \
--warmup_ratio 0.1 \
--learning_rate 3e-5 \
--report_to wandb \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] bert-base epoch 1" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \
--model_name_or_path "klue/bert-base" \
--overwrite_cache \
--max_seq_length 384 \
--doc_stride 32

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--overwrite_output_dir True \
--predict_file_name bert_e1_final \
--top_k_retrieval 30 \
--elastic True \
--do_predict



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 3 \
--weight_decay 0.0 \
--warmup_ratio 0.1 \
--learning_rate 3e-5 \
--report_to wandb \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] bert-base epoch 3" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \
--model_name_or_path "klue/bert-base" \
--overwrite_cache \
--max_seq_length 384 \
--doc_stride 32

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--overwrite_output_dir True \
--predict_file_name bert_e3_final \
--top_k_retrieval 30 \
--elastic True \
--do_predict



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 4 \
--weight_decay 0.0 \
--warmup_ratio 0.1 \
--learning_rate 3e-5 \
--report_to wandb \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] bert-base epoch 4" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \
--model_name_or_path "klue/bert-base" \
--overwrite_cache \
--max_seq_length 384 \
--doc_stride 32

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--overwrite_output_dir True \
--predict_file_name bert_e4_final \
--top_k_retrieval 30 \
--elastic True \
--do_predict



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 5 \
--weight_decay 0.0 \
--warmup_ratio 0.1 \
--learning_rate 3e-5 \
--report_to wandb \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] bert-base epoch 5" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \
--model_name_or_path "klue/bert-base" \
--overwrite_cache \
--max_seq_length 384 \
--doc_stride 32

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--overwrite_output_dir True \
--predict_file_name bert_e5_final \
--top_k_retrieval 30 \
--elastic True \
--do_predict



