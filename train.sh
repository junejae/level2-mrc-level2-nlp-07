python train.py \
--output_dir ./models/train_dataset \
--do_train \
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
<<<<<<< HEAD
--wandb_run_name "[lkm] roberta-large max384 doc128" \
=======
--wandb_run_name "[lkm] robert-large test" \
>>>>>>> 7d7bbfedf83445a6917eef9ab702e2032fab4af5
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \
--model_name_or_path "klue/roberta-large" \
--overwrite_cache \
--max_seq_length 384 \
--doc_stride 128 


# --model_name_or_path "klue/bert-base" \
# --embedding_type "Dense" 
# --train_dense_retrieval \
# --train_retrieval \
