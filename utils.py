import json


def calc_acc(preds, target):
    preds = preds.view(-1)
    target = target.view(-1)
    return (preds == target).float().mean()


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
