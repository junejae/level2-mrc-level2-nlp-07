from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    predict_file_name: Optional[str] = field(
        default=None,
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    embedding_type: str = field(
        default="Sparse",
        metadata={"help": "Whether to run passage retrieval using sparse embedding or dense embedding."},
    )
    train_retrieval: bool = field(
        default=False,
        metadata={"help": "whether to train sparse embedding (prepare for retrieval)."}
    )
    train_dense_retrieval: bool = field(
        default=False,
        metadata={"help": "whether to train dense embedding (prepare for retrieval)."}
    )
    other_dataset_name: str = field(
        default="KETI-AIR/korquad" ,
        metadata={"help": "state the name of the dataset to be trained"}
    )
    other_dataset_ver: str = field(
        default="v1.0" ,
        metadata={"help": "state the version of the dataset to be trained"}
    )
    is_using_ex_dataset: bool = field(
        default=False,
    ),
    is_negative_sampling: bool = field(
        default=True,
    )
    is_elastic: bool = field(
        default=True,
        metadata={"help": "whether to train different datasets."}
    )
    is_multiple_training: bool = field(
        default=False,
        metadata={"help": "whether to train after train"}
    )
    bm25: bool = field(
        default=False,
        metadata={"help": "whether to use bm25 (instead of tf-idf)."}
    )
    is_using_augmented_dataset: bool = field(
        default=False,
        metadata={"help": "whether to train with augmented dataset"}
    )
    augmented_dataset_dir: str = field(
        default="DataAug/train_with_no_answer",
        metadata={"help": "declare directory of the augmented dataset"}
    )
    elastic: bool = field(
        default=False,
        metadata={"help": "whether to use elastic (instead of tf-idf)."}
    )
    is_using_title_attatchment: bool = field(
        default=False,
        metadata={"help": "whether to train with context mixed with title"}
    )
    title_position: str = field(
        default="front",
        metadata={"help": "declare title position"}
    )
    


@dataclass
class WandbArguments:
    """
    Arguments related to wandb.
    """
    project_name: str = field(
        default="[MRC] dense embedding" ,
    )
    entity_name: Optional[str] = field(
        default="growing_sesame",
    )
    wandb_run_name: Optional[str] = field(
        default="eval test",
    )
