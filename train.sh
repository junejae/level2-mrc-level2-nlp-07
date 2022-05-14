python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--fp16 True \
--overwrite_output_dir True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 2 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 3e-5 \
--report_to wandb \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] roberta-large sota no fp16" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \
--model_name_or_path "klue/roberta-large" \
--overwrite_cache \
--max_seq_length 384 \
--doc_stride 32 \
--custom_model False 

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--overwrite_output_dir True \
--predict_file_name sota_no_fp16 \
--top_k_retrieval 30 \
--elastic True \
--do_predict


# --model_name_or_path "klue/bert-base" \
# --embedding_type "Dense" 
# --train_dense_retrieval \
# --train_retrieval \
