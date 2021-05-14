import json
from tqdm import tqdm
import random
from transformers import AutoTokenizer

from utils import TRAIN, DEV, TEST

HEAD_START = 'H'
HEAD_END = '/H'
TAIL_START = 'T'
TAIL_END = '/T'
NONE_TYPE = '[NONE]'


def get_marker_to_bert_token(entity_types):
    """

    :param entity_types: list, 实体类型列表，如 ['Task', 'Method', 'Generic', ...]
        `read_data()` 调用该函数时默认读取 `./meta/entity_types.json`
    :return: marker_to_bert_token: dict, 头尾实体标记 与 scibert_scivocab_cased vocab token 的映射；
        key 为实体标记，value 为 Bert token
        Ref: https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/allenai/scibert_scivocab_cased/vocab.txt
    """
    marker_to_bert_token = dict()
    unused_tokens = [f'[unused{i}]' for i in range(1, 50)]
    cnt = 0
    for H_or_T in [HEAD_START, HEAD_END, TAIL_START, TAIL_END]:
        for et in entity_types:
            marker = f"{H_or_T}-{et}"
            marker_to_bert_token[marker] = unused_tokens[cnt]
            cnt += 1
    return marker_to_bert_token


def tokenize_entity_pair(tokens, h, t, marker_to_bert_token, tokenizer):
    use_entity_type = True
    if f'{HEAD_START}-{NONE_TYPE}' in marker_to_bert_token:
        use_entity_type = False
    out_tokens = []
    h_pos, t_pos = 0, 0
    for tok_idx, tok in enumerate(tokens):
        if tok_idx == h['pos'][0]:  # H-entity_type
            entity_type = h['entity_type'] if use_entity_type else NONE_TYPE
            marker = f"{HEAD_START}-{entity_type}"
            out_tokens.append(marker_to_bert_token[marker])
            h_pos = len(out_tokens) - 1
        if tok_idx == t['pos'][0]:  # T-entity_type
            entity_type = t['entity_type'] if use_entity_type else NONE_TYPE
            marker = f"{TAIL_START}-{entity_type}"
            out_tokens.append(marker_to_bert_token[marker])
            t_pos = len(out_tokens) - 1
        wordpieces = tokenizer.tokenize("-".join(tok.split()))
        out_tokens.extend(wordpieces)
        if tok_idx == h['pos'][1] - 1:  # /H-entity_type
            entity_type = h['entity_type'] if use_entity_type else NONE_TYPE
            marker = f"{HEAD_END}-{entity_type}"
            out_tokens.append(marker_to_bert_token[marker])
        if tok_idx == t['pos'][1] - 1:  # /T-entity_type
            entity_type = t['entity_type'] if use_entity_type else NONE_TYPE
            marker = f"{TAIL_END}-{entity_type}"
            out_tokens.append(marker_to_bert_token[marker])
    return out_tokens, h_pos, t_pos


def get_feature(tokens, tokenizer, h, t, rel_id, marker_to_bert_token, max_seq_length):
    out_tokens, h_pos, t_pos = tokenize_entity_pair(
        tokens, h, t, marker_to_bert_token, tokenizer)
    out_tokens = out_tokens[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(out_tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    return {
        'input_ids': input_ids,
        'h_pos': h_pos,
        't_pos': t_pos,
        'label': rel_id
    }


def check_args(split, na_rate, drop_exceed_max_len):
    assert split in [TRAIN, DEV, TEST]
    if split == DEV and na_rate is not None:
        print("验证集, na_rate 设置为 None")
        na_rate = None
    if split == TRAIN:
        if na_rate is not None:
            assert na_rate > 1.0, f"训练集, na_rate = {na_rate}, na_rate 必须大于1."
    if na_rate is not None:
        assert na_rate > 0., f'na_rate={na_rate}, na_rate 必须大于0.'
    if split == TEST:
        assert not drop_exceed_max_len, f'测试集, `drop_exceed_max_len must` 必须为 False'
    return na_rate


def read_data(data_files, tokenizer, split, max_seq_length=128, na_rate=None,
              meta_dir='./meta', drop_exceed_max_len=False, use_entity_type=False):
    na_rate = check_args(split, na_rate, drop_exceed_max_len)
    if not isinstance(data_files, list):
        data_files = [data_files]
    data = []
    print(f"\n数据集：{split}:")
    for data_file in data_files:
        print(f"读取数据: {data_file}")
        data += json.load(open(data_file, 'r', encoding='utf8'))
    print("\n")
    if use_entity_type:
        entity_types = json.load(open(f'{meta_dir}/entity_types.json', 'r', encoding='utf8'))
    else:
        entity_types = [NONE_TYPE]  # Single-dummy entity type <==> do not use entity_type
    rel2id = json.load(open(f'{meta_dir}/rel2id.json', 'r', encoding='utf8'))
    marker_to_bert_token = get_marker_to_bert_token(entity_types)
    print(marker_to_bert_token)
    pos_features, na_features = [], []
    exceed_cnt = 0
    for sample in tqdm(data, desc=f'{split}-Example'):
        tokens = list(sample['tokens'])
        # 4 for markers <H></H> <T></T>; 2 for [CLS], [SEP]
        if drop_exceed_max_len and len(tokens) + 6 > max_seq_length:
            exceed_cnt += 1
            continue
        entities = sample['entities']
        h_t_pairs = set()
        # positive entity pairs
        if split in [TRAIN, DEV]:
            for rel in sample['relations']:
                feature = get_feature(
                    tokens, tokenizer,
                    entities[rel['h_idx']], entities[rel['t_idx']], rel2id[rel['r']],
                    marker_to_bert_token, max_seq_length)
                feature['id'] = sample['id']  # For computing scores
                if split == DEV:
                    feature['ht'] = (rel['h_idx'], rel['t_idx'])
                pos_features.append(feature)
                h_t_pairs.add((rel['h_idx'], rel['t_idx']))

        # NA entity pairs
        for h_idx in range(len(entities)):
            for t_idx in range(len(entities)):
                if h_idx != t_idx and (h_idx, t_idx) not in h_t_pairs:
                    feature = get_feature(
                        tokens, tokenizer,
                        entities[h_idx], entities[t_idx], rel2id['NA'],
                        marker_to_bert_token, max_seq_length)
                    feature['id'] = sample['id']  # For computing scores
                    if split in [TEST, DEV]:
                        feature['ht'] = (h_idx, t_idx)
                    na_features.append(feature)

    if split in [TRAIN, DEV]:  # train/dev
        if split == TRAIN and na_rate:
            random.shuffle(na_features)
            num_na_features = int(na_rate * len(pos_features))
            na_features = na_features[:num_na_features]
        p, n = len(pos_features), len(na_features)
        if drop_exceed_max_len:
            print("训练时丢弃句子数目: {} (这些句子大于 max_seq_len {})".format(exceed_cnt, max_seq_length))
        print("数据集：{} || 有关系的实体对数目: {}; 没关系的实体对(NA)数目: {}, 即 1:{:.1f}".format(split, p, n, n / float(p)))
        return pos_features + na_features
    else:
        print("数据集：{} || 待预测实体对数目: {}".format(split, len(na_features)))
        return na_features  # test, labels are dummy label 'NA'


if __name__ == '__main__':
    data_file = 'E:/workspace/aviation_ie/prepro/easy_2_000000_000300.json'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    features = read_data(data_file, tokenizer, split=TRAIN, na_rate=5)
    print(len(features))
    for i in range(10):
        print(features[i])
