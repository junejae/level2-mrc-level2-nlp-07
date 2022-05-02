from torch.nn import CrossEntropyLoss
import torch

# 2강 슬라이드 21페이지 참고
def eval_loss(predictions, labels):
    pred_start, pred_end = predictions

    label_start, label_end = labels

    loss_fnc = CrossEntropyLoss()

    start_loss = loss_fnc(torch.tensor(pred_start), torch.tensor(label_start))
    end_loss = loss_fnc(torch.tensor(pred_end), torch.tensor(label_end))

    total_loss = (start_loss + end_loss) / 2

    return total_loss.item()