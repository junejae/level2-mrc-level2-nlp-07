import os
import pandas as pd
from collections import defaultdict, Counter

def hard_voting_by_n_best(data_dir, input_filenames, output_filename):
    '''
    parameter :
        data_dir (str) = "../outputs/test_dataset"
        input_filenames (list) = ["nbest_pred.json", "nbest_pred_9980.json"]
        output_filename (str) = "../outputs/test_dataset/ens.json"
    description :
        앙상블할 파일 목록을 입력하면, mrc_id에 따라 output_filename으로 파일 하드보팅한 파일 생성 및 반환
    output : 
        output_filename (json 파일)
        output (pd.Series)
    '''
    mrc_dict = defaultdict(list)
    for filename in input_filenames:
        df = pd.read_json(os.path.join(data_dir, filename))
        for k, topk in df.items():
            candidate = [unit['text'] for unit in [*topk.values]]
            mrc_dict[k].extend(candidate)
    
    pred_dict = {}
    for k, candidate in mrc_dict.items():
        pred = Counter(candidate).most_common(1)[0][0]
        pred_dict[k] = pred
    output = pd.Series(pred_dict)
    output.to_json(output_filename)
    return output


def hard_voting_by_pred(data_dir, input_filenames, output_filename):
    '''
    parameter :
        data_dir (str) = "../outputs/test_dataset"
        input_filenames (list) = ["pred.json", "pred_9980.json"]
        output_filename (str) = "../outputs/test_dataset/ens_pred.json"
    description :
        앙상블할 파일 목록을 입력하면, mrc_id에 따라 output_filename으로 파일 하드보팅한 파일 생성 및 반환
    output : 
        output_filename (json 파일)
        output (pd.Series)
    '''
    mrc_dict = defaultdict(list)
    for filename in input_filenames:
        vec = pd.read_json(os.path.join(data_dir, filename), orient='records', typ='series')
        for k, pred in vec.items():
            mrc_dict[k].append(pred)
    
    pred_dict = {}
    for k, candidate in mrc_dict.items():
        pred = Counter(candidate).most_common(1)[0][0]
        pred_dict[k] = pred
    output = pd.Series(pred_dict)
    output.to_json(output_filename)
    return mrc_dict