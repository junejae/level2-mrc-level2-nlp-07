from elasticsearch import Elasticsearch
from datasets import load_from_disk, Dataset
from typing import Optional
import json
import re
import pandas as pd
from tqdm import tqdm


def elastic_func(datasets: Dataset, topk: Optional[int] = 1):
    try:
        es.transport.close()
    except:
        pass
    es = Elasticsearch(timeout=30)

    INDEX_NAME = "wiki"
    index_config = {
    "settings": {
        "analysis": {
            "analyzer": {
                "my_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["shingle"]
                }
            },    
        }
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "document_text": {"type": "text", "analyzer": "my_analyzer"}
        }    
    }
    }

    path = "/opt/ml/input/data/wikipedia_documents_no_dup.json"
    with open(path, "r") as f:
        wiki = json.load(f)


    if es.indices.exists(INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
    es.indices.create(index=INDEX_NAME, body=index_config)


    for doc_id, doc in tqdm(wiki.items(), total=len(wiki)):
        temp_d = {}
        temp_d['document_text'] = doc['text']
        es.index(index=INDEX_NAME, id=doc_id, body=temp_d)

    query = datasets["question"]
    ids = datasets["id"]
    context = datasets["context"] if "context" in datasets.column_names else []
    answer = datasets["answers"] if "answers" in datasets.column_names else []

    total = []
    for i, (q, idx) in enumerate(tqdm(zip(query, ids), desc="Elastic search: ", total=len(query))):
        q = q.replace("~", "-")
        q = q.replace("/", "")
        res = es.search(index=INDEX_NAME, q=q, size=topk)  # topk개의 문서를 반환합니다
        
        total_context = [res["hits"]["hits"][j]["_source"]["document_text"] for j in range(topk)]
        tmp = {
            # Query와 해당 id를 반환합니다.
            "question": q,
            "id": idx,
            # Retrieve한 Passage의 context를 반환합니다.
            "context": " ".join(total_context),
        }
        if context and answer:
            # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
            tmp["original_context"] = context[i]
            tmp["answers"] = answer[i]
        total.append(tmp)    

    cqas = pd.DataFrame(total)
    return cqas
