import logging
import sys
import pandas as pd
from elasticsearch import Elasticsearch
from subprocess import Popen, PIPE, STDOUT
from arguments import DataTrainingArguments, ModelArguments, WandbArguments
from datasets import (
    load_from_disk,

    load_dataset, 
    concatenate_datasets
)
from retrieval import SparseRetrieval
from retrieval_dense import *
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
from bm25 import bm25_func
from utils_qa import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)

def set_elastic_server():
    path_to_elastic = "/opt/ml/input/code/elasticsearch-6.7.2/bin/elasticsearch"
    es_server = Popen(
        [path_to_elastic],
        stdout=PIPE,
        stderr=STDOUT,
        preexec_fn=lambda: os.setuid(1),
    )
    config = {"host": "localhost", "port": 30001}

    es = Elasticsearch([config])

    if es.ping():
        print("Connected")
        return es
    else:
        print("Connection Failed")
        return None
        
def get_top_k_passages(self, question: str, k: int) -> list:
    query = {"query": {"match": {"document_text": question}}}
    result = self.es.search(index=self.index_name, body=query, size=k)
    return result["hits"]["hits"]
    
    
if __name__ == "__main__":
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, WandbArguments)
    )
    model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()

    # wandb.init(project=wandb_args.project_name, entity=wandb_args.entity_name)
    # wandb.run.name = wandb_args.wandb_run_name

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


    es = set_elastic_server()

    # Elasticsearch 는 데이터를 저장하기 전에 Index 라는 것을 생성해야 합니다.
    # 해당 Index 에 데이터를 저장할 수 있습니다
    INDEX = 'wiki'
    if es.indices.exists(index_name):
        print("Index Mapping already exists.")
    
    with open(os.path.join("../data", "wikipedia_documents.json"), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    wiki_data = list(
        dict.fromkeys([v["text"] for v in wiki.values()])
    )  # set 은 매번 순서가 바뀌므로
          
    
    for idx, text in enumerate(wiki_data):
        body = {'text': text}
        es.index(index=INDEX, doc_type='news', id=idx+1, body=body) 
        
        
        

    cnt = 0
    for data in tqdm(datasets['validation']):
        question_text = data["question"]
        context = data["context"]
        result = retriever.get_top_k_passages(question_text, data_args.top_k_retrieval)
        print(result)
        print(dd)
        print(f"question_text : {question_text}, context : {context} ")
        for res in result:
            if res["_source"]["document_text"] == context:
                match_cnt += 1
                break
    print(f"matching score is {match_cnt/3952:.3f}")
    
    


  
        


