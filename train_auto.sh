python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 9e-6 \
--project_name "[MRC] baseline" \
--entity_name growing_sesame \
--wandb_run_name "robert-large lr 9e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 8e-6 \
--project_name "[MRC] baseline" \
--entity_name growing_sesame \
--wandb_run_name "robert-large lr 8e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \


python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 7e-6 \
--project_name "[MRC] baseline" \
--entity_name growing_sesame \
--wandb_run_name "robert-large lr 7e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \


python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 6e-6 \
--project_name "[MRC] baseline" \
--entity_name growing_sesame \
--wandb_run_name "robert-large lr 6e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \


python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 5e-6 \
--project_name "[MRC] baseline" \
--entity_name growing_sesame \
--wandb_run_name "robert-large lr 5e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

