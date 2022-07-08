from sklearn.metrics import precision_recall_fscore_support


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def ee_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='micro')