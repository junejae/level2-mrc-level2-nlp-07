"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.
대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple
import torch
import numpy as np
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from retrieval import SparseRetrieval
#from dense_retrieval import *
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )

    if data_args.embedding_type == "Sparse":
        datasets = run_sparse_retrieval(
            tokenizer.tokenize, datasets, training_args, data_args,
    )
    #else: # "Dense"
    #    datasets = run_dense_retrieval(
    #        tokenizer.tokenize, datasets, model_args, data_args,
    #    )


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    count = 0
    for i in range(len(df)):
        ground = df['original_context'][i]
        context = df['context'][i]

        if ground in context:
            count += 1


    print("Accuracy: ", count / len(df))

"""
###수정중###
def run_dense_retrieval(
    tokenizer: AutoTokenizer,
    datasets: DatasetDict,
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    p_encoder = BertEncoder.from_pretrained(model_args.model_name_or_path).cuda()
    q_encoder = BertEncoder.from_pretrained(model_args.model_name_or_path).cuda()
    
    model_dict = torch.load("./dense_encoder/encoder.pth")
    p_encoder.load_state_dict(model_dict['p_encoder'])
    q_encoder.load_state_dict(model_dict['q_encoder'])
    
    args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01
    )
                
    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = DenseRetrieval(args=args, dataset=datasets, num_neg=2, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)
    retriever.get_dense_embedding()
    print(datasets)
    df = retriever.retrieve(datasets, topk=data_args.top_k_retrieval)
    count = 0
    for i in range(len(df)):
        ground = df['original_context'][i]
        context = df['context'][i]
        if ground in context:
            count += 1
    print("Accuracy: ", count / len(df))
"""


if __name__ == "__main__":
    main()