python retriever_test.py \
--output_dir ./models/train_dataset \
# --do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 2 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 3e-5 \
--report_to wandb \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] roberta-large top_k 100" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \
--train_retrieval \
--model_name_or_path "klue/roberta-large" \
# --train_dense_retrieval \
--overwrite_cache \
--embedding_type "Dense" 
# --max_seq_length 512 \
# --doc_stride 32 
