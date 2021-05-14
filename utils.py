import torch
import random
import numpy as np
import json


TRAIN = 'train'
DEV = 'dev'
TEST = 'test'


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["label"] for f in batch]
    h_pos = [f["h_pos"] for f in batch]
    t_pos = [f["t_pos"] for f in batch]
    sent_ids = [f["id"] for f in batch]
    hts = [f.get('ht', None) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)  # (B, max_len)
    input_mask = torch.tensor(input_mask, dtype=torch.float)  # (B, max_len)
    # h_pos = torch.tensor(h_pos, dtype=torch.long)  # (B, )
    # t_pos = torch.tensor(t_pos, dtype=torch.long)  # (B, )
    labels = torch.tensor(labels)  # (B, )
    # output = (input_ids, input_mask, h_pos, t_pos, labels)
    return {
        'input_ids': input_ids,
        'attention_mask': input_mask,
        'h_pos': h_pos,
        't_pos': t_pos,
        'labels': labels,
        'id': sent_ids,
        'ht': hts
    }


def export_predict_results(args, tag, outs):
    file = args.dev_file if tag == DEV else args.test_file
    file = f"{args.data_dir}/{file}"
    data = json.load(open(file, 'r', encoding='utf8'))
    for d in data:
        d['relations'] = outs.get(d['id'], [])
    out_file = f"{file[:-5]}_pred.json"
    json.dump(data, open(out_file, 'w', encoding='utf8'), ensure_ascii=False, indent=2)
    print("输出预测结果到 {}".format(out_file))


def report_string(report_dict):
    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=12)
    report += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
    for metric in report_dict:
        l = report_dict[metric]
        report += row_fmt.format(
            metric, l['precision']*100, l['recall']*100, l['f1-score']*100, l['support'], width=12, digits=2)
    report += '\n\n'
    return report
