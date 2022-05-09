import pandas as pd
from tqdm import tqdm
from typing import Optional
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import json
from datasets import Dataset


def bm25_func(datasets: Dataset, topk: Optional[int] = 1):
    path = "../data/wikipedia_documents.json"
    with open(path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
    contexts = list(
        dict.fromkeys([v["text"] for v in wiki.values()])
    )
    print(f"Lengths of unique contexts : {len(contexts)}")
    tokenized_context = [doc.split(" ") for doc in contexts]
    bm25 = BM25Plus(tokenized_context)

    pred = []
    for i in tqdm(range(len(datasets['question']))):
        query = datasets['question'][i].split(" ")
        top_k = bm25.get_top_n(query, contexts, n=topk)
        pred.append(top_k)

    total = []
    for idx, example in enumerate(
        tqdm(datasets, desc="BM25 retrieval: ")
    ):
        tmp = {
            # Query와 해당 id를 반환합니다.
            "question": example["question"],
            "id": example["id"],
            # Retrieve한 Passage의 context를 반환합니다.
            "context": " ".join(pred[idx]),
        }
        if "context" in example.keys() and "answers" in example.keys():
            # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
            tmp["original_context"] = example["context"]
            tmp["answers"] = example["answers"]
        total.append(tmp)

    cqas = pd.DataFrame(total)
    return cqas
