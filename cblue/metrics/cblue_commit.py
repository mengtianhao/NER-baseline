import os
import json


def ee_commit_prediction(dataset, preds, output_dir):
    orig_text = dataset.orig_text

    pred_result = []
    for item in zip(orig_text, preds):
        tmp_dict = {'text': item[0], 'entities': item[1]}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'CMeEE_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))
