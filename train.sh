python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
<<<<<<< HEAD
--num_train_epochs 2 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 3e-5 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large 2 epoch 3e5" \
=======
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 2e-5 \
--report_to wandb \
--project_name "[MRC] wandb_test" \
--entity_name growing_sesame \
--wandb_run_name "roberta-large standard epochs 5 batch 8" \
>>>>>>> 319a8dd9639d4b6e1e3d8c5a3d73dca49d5be92a
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

