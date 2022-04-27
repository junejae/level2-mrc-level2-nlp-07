python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--do_predict \
--overwrite_output_dir True \
--predict_file_name 9980 \

# --model_name_or_path ./models/train_dataset/checkpoint-1000/ \
# --model_name_or_path ./models/train_dataset/ \
