python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--overwrite_output_dir True \
--predict_file_name top_40 \
--top_k_retrieval 40 \
--dataset_name ../data/test_dataset/ \
--do_predict \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--overwrite_output_dir True \
--predict_file_name top_30 \
--top_k_retrieval 30 \
--dataset_name ../data/test_dataset/ \
--do_predict \


# --use_faiss True \

# python retriever_test.py \
# --output_dir ./outputs/train_dataset/ \
# --overwrite_output_dir True \
# --dataset_name ../data/train_dataset/ \
# --top_k_retrieval 20



# --do_eval \
# --dataset_name ../data/train_dataset/ \
# --model_name_or_path ./models/train_dataset/checkpoint-1000/ \
# --model_name_or_path ./models/train_dataset/ \
