import os
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
from typing import List, Optional, Tuple, Union
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import wandb
import time
from contextlib import contextmanager
from datasets import Dataset, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,AutoModel, AutoConfig,
    AdamW, get_linear_schedule_with_warmup,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from arguments import *

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class DenseRetrieval:

    def __init__(self, args, data_args, dataset, num_neg, 
                 tokenizer, p_encoder=None, q_encoder=None,
                 data_path: Optional[str] = "../data/",
                 context_path: Optional[str] = "wikipedia_documents.json",):

        '''
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        '''

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        
        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg
        self.is_negative = data_args.is_negative_sampling

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        
        self.p_embedding = None # get_sparse_embedding()으로 생성

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.        
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in dataset['context']:
            
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        train_dataset, valid_dataset =  train_test_split(train_dataset, test_size=0.2, random_state=42)

        self.train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=self.args.per_device_train_batch_size, drop_last=True)
        self.valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=self.args.per_device_eval_batch_size, drop_last=True)

    def train(self, args=None):

        if self.is_negative:
            self.prepare_in_batch_negative(num_neg=self.num_neg)

        
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    wandb.log({
                        "Train/Loss": loss
                        })
            
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

            """
            validation
            """
            with tqdm(self.valid_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.eval()
                    self.q_encoder.eval()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)


                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    wandb.log({
                        "Valid/Loss": loss
                        })
            

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs


    def get_dense_embedding(self, model_args=None):

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """
        
        # Pickle을 저장합니다.
        data_path = "../data/"
        p_encoder_name = f"dense_embedding_p_encoder.bin"
        p_embedding_name = f"dense_embedding.bin"
        q_encoder_name = f"dense_embedding_q_encoder.bin"
        p_encoder_path = os.path.join(data_path, p_encoder_name)
        p_embedding_path = os.path.join(data_path, p_embedding_name)
        q_encoder_path = os.path.join(data_path, q_encoder_name)

        if os.path.isfile(p_encoder_path) and os.path.isfile(q_encoder_path) and os.path.isfile(p_embedding_path):
            print("Found P_Encoder & Q_Encoder & Dense Embedding")
            with open(p_encoder_path, "rb") as file:
                self.p_encoder = pickle.load(file)
            with open(p_embedding_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(q_encoder_path, "rb") as file:
                self.q_encoder = pickle.load(file)
            print("Encoder pickle load.")
        else:
            print("Build P_Encoder & Q_Encoder")
            model_checkpoint = model_args.model_name_or_path

            self.p_encoder = Encoder(model_checkpoint)
            self.q_encoder = Encoder(model_checkpoint)
            if torch.cuda.is_available():
                self.p_encoder.cuda()
                self.q_encoder.cuda()
            self.train()
            
            # p_embedding
            with torch.no_grad():
                self.p_encoder.eval()

                p_embs = []
                for p in self.contexts:
                    p = self.tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                    
                    p_inputs = {
                        'input_ids': p['input_ids'].to(self.args.device),
                        'attention_mask': p['attention_mask'].to(self.args.device)
                    }
                                        
                    p_emb = self.p_encoder(**p_inputs).to('cpu').numpy()
                    p_embs.append(p_emb)
                    torch.cuda.empty_cache()
                    
                    del p

            p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)
            self.p_embedding = p_embs
            
            with open(p_encoder_path, "wb") as file:
                pickle.dump(self.p_encoder, file)
            with open(p_embedding_path, "wb") as file:
                pickle.dump(p_embs, file)
            with open(q_encoder_path, "wb") as file:
                pickle.dump(self.q_encoder, file)
            print("Encoder pickle saved.")

    def retrieve(
            self, dataset, topk: Optional[int] = 1
        ) -> Union[Tuple[List, List], pd.DataFrame]:

            assert self.p_embedding is not None, "get_dense_embedding() 메소드를 먼저 수행해줘야합니다."

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_indices = self.get_relevant_doc_bulk(
                    dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
            
    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

        pprint("in relevant")
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return rank[:k]

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1, args=None, p_encoder=None, q_encoder=None
    ) -> Tuple[List, List]:

        if args is None:
            args = self.args

        if q_encoder is None:
            q_encoder = self.q_encoder
        
        # p_embedding
        with torch.no_grad():
            self.q_encoder.eval()
            query_vec = []
            for q in queries:
                q = self.tokenizer(q, padding="max_length", truncation=True, return_tensors='pt').to('cuda')\
                
                q_inputs = {
                    'input_ids': q['input_ids'].to(args.device),
                    'attention_mask': q['attention_mask'].to(args.device)
                }
                                    
                q_emb = self.q_encoder(**q_inputs).to('cpu').numpy()
                query_vec.append(q_emb)

        query_vec = torch.Tensor(query_vec).squeeze()  # (num_passage, emb_dim)
        result = torch.matmul(query_vec, torch.transpose(self.p_embedding, 0, 1))
        

        doc_indices = []
        for i in range(result.shape[0]):
            rank = torch.argsort(result[i, :], descending=True).squeeze()
            #doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(rank.tolist()[:k])
        return doc_indices
    

class Encoder(nn.Module):
    
    def __init__(self, model_checkpoint=None):
        super(Encoder, self).__init__()

        self.model_checkpoint = model_checkpoint
        config = AutoConfig.from_pretrained(self.model_checkpoint)
        self.model = AutoModel.from_pretrained(model_checkpoint, config=config)
            
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )


        pooled_output = outputs[1]

        return pooled_output



if __name__ == "__main__":

    
    model_checkpoint = "klue/roberta-large"

    datasets = load_from_disk("../data/train_dataset")
    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )

    p_encoder = Encoder(model_checkpoint)
    q_encoder = Encoder(model_checkpoint)

    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    train_dataset = datasets['train']
    retriever = DenseRetrieval(args=args, dataset=train_dataset, num_neg=2, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)
    retriever.train()

    test_n = 3
    quesetions = datasets['validation']['question']
    contexts = datasets['validation']['context']
    for query, ground_truth in zip(quesetions[:test_n], contexts[:test_n]):
        results = retriever.get_relevant_doc(query=query, k=5)
        print(f"[Search Query] {query}")
        print(f"[Ground Truth] {ground_truth}\n")

        indices = results.tolist()
        for i, idx in enumerate(indices):
            print(f"Top-{i + 1}th Passage (Index {idx})")
            pprint(retriever.dataset['context'][idx])