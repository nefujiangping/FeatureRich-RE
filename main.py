import argparse
import numpy as np
from models.re_model import RE
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import os
import json
import wandb
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils import collate_fn, set_seed
from prepro import read_data
from utils import export_predict_results, report_string

from utils import TRAIN, DEV, TEST

os.environ["WANDB_API_KEY"] = "这里填写API Key"
os.environ["WANDB_MODE"] = "dryrun"


def evaluate(args, model: RE, features, tag=DEV, id2rel=None, output=False):
    assert tag in [TEST, DEV], f"tag = {tag}, not in {TEST, DEV}"
    data_loader = DataLoader(
        features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    y_pred, y_true = [], []
    outs = dict()
    for batch in tqdm(data_loader, desc=f'{tag}-batch'):
        inputs = {
            'input_ids': batch['input_ids'].to(args.device),
            'attention_mask': batch['attention_mask'].to(args.device),
            'h_pos': batch['h_pos'],  # list of length B
            't_pos': batch['t_pos']  # list of length B
        }
        with torch.no_grad():
            logits = model(inputs)  # (B, num_labels)
            _, pred = torch.max(logits, -1)  # (B, )
            pred = pred.cpu().numpy()  # (B, )

        if tag == DEV:
            y_pred.append(pred)
            label = batch['labels'].cpu().numpy()  # (B, )
            y_true.append(label)

        if output:
            sent_ids = batch['id']
            hts = batch['ht']
            for i, rel_id in enumerate(pred):
                if rel_id < 1:  # 0 is NA
                    continue
                if sent_ids[i] not in outs:
                    outs[sent_ids[i]] = []
                outs[sent_ids[i]].append({"h_idx": hts[i][0], "t_idx": hts[i][1], "r": id2rel[rel_id]})

    if output:
        export_predict_results(args, tag, outs)

    if tag == DEV:
        # number of entity pairs
        y_pred = np.concatenate(y_pred, axis=0).astype(np.int32)
        y_true = np.concatenate(y_true, axis=0).astype(np.int32)
        report_dict = classification_report(
            y_true, y_pred, labels=list(range(1, args.num_labels)), output_dict=True, zero_division=0)
        print(report_string(report_dict))
        print([id2rel[rid] for rid in range(1, args.num_labels)])
        return report_dict['micro avg']['f1-score']


def train(args, model: RE, train_features: list, dev_features: list, id2rel=None):

    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        print("训练总步数: {}".format(total_steps))
        print("Warmup步数: {}".format(warmup_steps))
        for epoch in tqdm(range(int(num_epoch)), desc='Epoch'):
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {
                    'input_ids': batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'h_pos': batch['h_pos'],  # list of length B
                    't_pos': batch['t_pos']  # list of length B
                }
                labels = batch['labels'].to(args.device)  # (B, )
                logits = model(inputs)
                loss = model.loss(logits, labels)
                loss = loss / args.gradient_accumulation_steps
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                wandb.log({"loss": loss.item()}, step=num_steps)

            print("loss:", loss.item())
            # do eval
            dev_micro_f1 = evaluate(args, model, dev_features, tag=DEV, id2rel=id2rel)
            wandb.log({"dev_f1": dev_micro_f1*100}, step=num_steps)
            if dev_micro_f1 > best_score:
                # 保存效果最好的模型到 save_path 路径
                best_score = dev_micro_f1
                save_path = args.save_path
                print(f'Epoch {epoch} 验证集f1={dev_micro_f1}: 保存模型到 {save_path} ...')
                torch.save(model.state_dict(), save_path)
                print(f'Epoch {epoch}: 成功保存模型到 {save_path} ！')

    new_layer = ["classifier"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def check_args(args):
    def __assert_file__(basename, file, meta=('训练', '--train_file train.json')):
        assert basename, "当前为{p1}模式，请指定{p1}数据文件，如: {p2}".format(p1=meta[0], p2=meta[1])
        assert os.path.exists(file), "当前为{p1}模式，请确保{p1}文件存在，给定{p1}文件路径为：{p2}".format(p1=meta[0], p2=file)
    mode = args.mode
    if mode == TRAIN:
        __assert_file__(args.train_file, f'{args.data_dir}/{args.train_file}', ('训练', '--train_file train.json'))
        assert args.save_path, "当前为训练模式，请给定模型保存位置，如 --save_path ./checkpoint/FeatureRich-RE.pt"
        save_model_dir = os.path.dirname(args.save_path)
        os.makedirs(save_model_dir, exist_ok=True)
    elif mode == DEV:
        __assert_file__(args.dev_file, f'{args.data_dir}/{args.dev_file}', ('验证', '--dev_file dev.json'))
        if (not args.load_path) or (not os.path.exists(args.load_path)):
            raise FileNotFoundError(
                "当前为验证模式，请给定训练好的模型的位置，如 --load_path ./output/re/best_checkpoint/avionics_re.pt ;"
                + "当前给定 load_path 为：" + args.load_path)
    elif mode == TEST:
        __assert_file__(args.test_file, f'{args.data_dir}/{args.test_file}', ('预测', '--test_file test.json'))
        if (not args.load_path) or (not os.path.exists(args.load_path)):
            raise FileNotFoundError(
                "当前为测试模式，请给定训练好的模型的位置，如 --load_path ./checkpoint/FeatureRich-RE.pt ;"
                + "当前给定 load_path 为：" + args.load_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data", type=str, help="训练集/验证集/测试集数据所在目录")
    parser.add_argument("--meta_dir", default="./meta", type=str, help="包含实体/关系类型的目录")
    parser.add_argument("--transformer_type", default="bert", type=str, help="transformer 类型")
    parser.add_argument("--model_name_or_path", default="./sciBERT", type=str,
                        help="Hugging-face transformers 模型名/模型目录")

    parser.add_argument("--mode", type=str, required=True, choices=[TRAIN, DEV, TEST], help="运行模式：训练/验证/测试")
    parser.add_argument("--train_file", default="train.json", type=str, help="训练所用数据，多个文件可用半角逗号(,)隔开")
    parser.add_argument("--dev_file", default="dev.json", type=str, help="验证所用数据，格式与训练集相同，即每个样本必须包含 relations 字段")
    parser.add_argument("--test_file", default="test.json", type=str, help="测试所用数据，格式与训练集相同，可不包含 relations 字段")
    parser.add_argument("--save_path", default="", type=str, help="训练时保存模型时，模型保存路径")
    parser.add_argument("--load_path", default="", type=str, help="预测时，模型加载路径")
    parser.add_argument("--export_prediction", action='store_true', default=False,
                        help="是否导出预测结果到文件，测试时会被自动设置为 True；验证时请根据需求设置。输出文件名为待预测文件名后加`_pred`后缀。")

    parser.add_argument("--max_seq_length", default=150, type=int,
                        help="输入序列最大长度，注：预处理会在原始序列基础上添加 6 个特殊token，故请保证 max_seq_length >= (最长句子长度+6)。"
                             "训练时若句子长度超出设置，会被丢弃；验证/预测时会报错")

    parser.add_argument("--na_rate", default=None, type=float,
                        help="用于设置训练时使用没有关系的实体对(NA)的比例，默认为None，表示使用训练集里所有的没有关系的实体对；"
                             ">0时，使用没有关系的实体对训练数目 = (na_rate*有关系实体对数量)")

    parser.add_argument("--use_entity_type", action='store_true', default=False, help="是否使用实体类型作为输入特征")

    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="训练 batch_size，即实体对数目")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="验证/测试 batch_size，即实体对数目")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="Adam 优化器初始学习率，一般为 e-5 级别")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Adam 优化器 epsilon.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="最大 gradient norm，超过该值会被截断")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Adam优化器 Warm up 步骤比例")
    parser.add_argument("--num_train_epochs", default=int(15), type=int,
                        help="训练轮数，数据集较小时(只有几百/几千条数据)一般设置为 < 10")
    parser.add_argument("--seed", type=int, default=66,
                        help="随机种子，便于复现")
    parser.add_argument("--num_labels", type=int, default=5,
                        help="关系类别数目（包括NA），例如有4个关系，则设置为5")
    parser.add_argument("--exp_note", type=str, default=None,
                        help="关于这次实验的一些说明。")

    args = parser.parse_args()

    # 设置随机种子，保证结果可复现
    set_seed(args)
    check_args(args)

    if args.mode == TEST:
        args.export_prediction = True
    print("Mode: ", args.mode)
    if args.exp_note:
        print("备注: ", args.exp_note)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    id2rel = {v: k for k, v in json.load(open(f'{args.meta_dir}/rel2id.json', 'r', encoding='utf8')).items()}

    model = RE(args.model_name_or_path, num_labels=args.num_labels)
    model.to(0)

    if args.mode == TRAIN:  # Train
        train_files = [os.path.join(args.data_dir, file) for file in args.train_file.split(',')]
        train_features = read_data(train_files, tokenizer=tokenizer, split=TRAIN,
                                   max_seq_length=args.max_seq_length,
                                   meta_dir=args.meta_dir, drop_exceed_max_len=True,
                                   na_rate=args.na_rate, use_entity_type=args.use_entity_type)
        dev_file = os.path.join(args.data_dir, args.dev_file)
        dev_features = read_data(dev_file, tokenizer=tokenizer, split=DEV, max_seq_length=args.max_seq_length,
                                 meta_dir=args.meta_dir, drop_exceed_max_len=True, use_entity_type=args.use_entity_type)
        train(args, model, train_features, dev_features, id2rel)
    elif args.mode in [DEV, TEST]:
        test_file = os.path.join(args.data_dir, args.test_file if args.mode == TEST else args.dev_file)
        features = read_data(test_file, tokenizer=tokenizer, split=args.mode, max_seq_length=args.max_seq_length,
                             meta_dir=args.meta_dir, drop_exceed_max_len=False, use_entity_type=args.use_entity_type)
        print(f"加载模型 {args.load_path} ...")
        model.load_state_dict(torch.load(args.load_path))
        print(f"\n成功加载模型 {args.load_path} ！\n")
        output = True if args.mode == TEST else args.export_prediction
        evaluate(args, model, features, tag=args.mode, id2rel=id2rel, output=output)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    wandb.init(project="feature_rich_re")
    main()
