# python inference.py \
# --output_dir ./outputs/test_dataset/ \
# --model_name_or_path ./models/train_dataset/ \
# --dataset_name ../data/test_dataset/ \
# --do_predict \
# --overwrite_output_dir True \
# --predict_file_name 1996 \


# python inference.py \
# --output_dir ./outputs/test_dataset/ \
# --model_name_or_path ./models/train_dataset/ \
# --dataset_name ../data/test_dataset/ \
# --overwrite_output_dir True \
# --predict_file_name 384_128_elastic_30_no_dup \
# --top_k_retrieval 30 \
# --elastic True \
# --do_predict


# --bm25 True \
# --use_faiss True \

# python retriever_test.py \
# --output_dir ./outputs/train_dataset/ \
# --overwrite_output_dir True \
# --dataset_name ../data/train_dataset/ \
# --elastic True \
# --top_k_retrieval 20



# --do_eval \
# --dataset_name ../data/train_dataset/ \
# --model_name_or_path ./models/train_dataset/checkpoint-1000/ \
# --model_name_or_path ./models/train_dataset/ \

python inference.py \
--output_dir ./outputs/test_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--dataset_name ../data/test_dataset/ \
--overwrite_output_dir True \
--predict_file_name sota_fp16 \
--top_k_retrieval 30 \
--elastic True \
--fp16 True \
--do_predict

# python inference.py \
# --output_dir ./outputs/test_dataset/ \
# --model_name_or_path ./models/train_dataset/checkpoint-1500/ \
# --dataset_name ../data/test_dataset/ \
# --overwrite_output_dir True \
# --predict_file_name 384_128_1500step_elastic_30_no_dup \
# --top_k_retrieval 30 \
# --elastic True \
# --do_predict


