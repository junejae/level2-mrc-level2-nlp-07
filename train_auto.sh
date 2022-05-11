python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 4 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 8e-6 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e4 lr 8e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e4 lr 8e6" \



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 4 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 9e-6 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e4 lr 9e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e4 lr 9e6" \



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 4 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 1e-5 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e4 lr 1e5" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e4 lr 1e5" \



python train.py \
--output_dir ./models/train_dataset \
--do_train True \
--do_eval True \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 4 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 2e-5 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e4 lr 2e5" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e4 lr 2e5" \



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 4 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 4e-5 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e4 lr 4e5" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e4 lr 4e5" \



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 4 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 5e-5 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e4 lr 5e5" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e4 lr 5e5" \



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 6e-6 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e5 lr 6e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e5 lr 6e6" \



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 7e-6 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e5 lr 7e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e5 lr 7e6" \




python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 8e-6 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e5 lr 8e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e5 lr 8e6" \



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 9e-6 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e5 lr 9e6" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e5 lr 9e6" \



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 1e-5 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e5 lr 1e5" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e5 lr 1e5" \



python train.py \
--output_dir ./models/train_dataset \
--do_train True \
--do_eval True \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 2e-5 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e5 lr 2e5" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e5 lr 2e5" \



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 4e-5 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e5 lr 4e5" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e5 lr 4e5" \



python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--fp16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 5e-5 \
--project_name "[MRC] hp-tuning" \
--entity_name growing_sesame \
--wandb_run_name "[lkm] robert-large b16 e5 lr 5e5" \
--evaluation_strategy steps \
--eval_steps 500 \
--save_total_limit 5 \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name "b16 e5 lr 5e5" \

