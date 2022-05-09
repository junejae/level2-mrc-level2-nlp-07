python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--overwrite_output_dir True \
--predict_file_name bm25_top_500 \
--top_k_retrieval 200 \
--dataset_name ../data/test_dataset/ \
--bm25 True \
--do_predict


# --use_faiss True \

# python retriever_test.py \
# --output_dir ./outputs/train_dataset/ \
# --overwrite_output_dir True \
# --dataset_name ../data/train_dataset/ \
# --bm25 True \
# --top_k_retrieval 500



# --do_eval \
# --dataset_name ../data/train_dataset/ \
# --model_name_or_path ./models/train_dataset/checkpoint-1000/ \
# --model_name_or_path ./models/train_dataset/ \
