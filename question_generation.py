import json
import pandas as pd
from pororo import Pororo
import os
from transformers import set_seed
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def find_idx(text, title):
    idx =text.find(title)
    
    if idx < 0:
        return []
    else:
        return [idx]
    
def main():

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(42)
    
    # wiki dataset을 가져옵니다.
    with open(os.path.join("../data", "wikipedia_documents.json"), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    wiki_df = pd.DataFrame.from_dict(wiki, orient='index')
    
    # Question Generation 
    qg = Pororo(task="qg", lang="ko")

    title = []
    context = []
    question = []
    id = []
    answers = []
    document_id = []

    # Data Augmentation
    for i in tqdm(range(len(wiki_df))):
        # question generation
        q = qg(wiki_df['text'][i], wiki_df['title'][i])
        title.append(wiki_df['title'][i])
        context.append(wiki_df['text'][i])
        question.append(q)
        id.append('wiki-0-{0:06d}'.format(i))
        
        tmp = {}
        tmp['answer_start'] = find_idx(wiki_df['text'][i], wiki_df['title'][i])
        tmp['text'] = [wiki_df['title'][i]]
        answers.append(tmp)
        document_id.append(i)
    
    aug_df = pd.DataFrame(zip(title, context, question, id, answers, document_id), columns=['title', 'context', 'question', 'id', 'answers', 'document_id'])
    aug_df.to_csv("../data/augmented_data.csv", encoding='utf-8')
    
if __name__ == "__main__":
    main()