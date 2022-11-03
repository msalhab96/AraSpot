def calc_acc(preds, target):
    preds = preds.view(-1)
    target = target.view(-1)
    return (preds == target).mean()
