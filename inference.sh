python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--overwrite_output_dir True \
--predict_file_name 99800 \
--top_k_retrieval 200 \
--dataset_name ../data/test_dataset/ \
--do_predict

# --do_eval \
# --dataset_name ../data/train_dataset/ \
# --model_name_or_path ./models/train_dataset/checkpoint-1000/ \
# --model_name_or_path ./models/train_dataset/ \
