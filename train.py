import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import time
import argparse
import numpy as np
import random
from tqdm import tqdm

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader, TensorDataset, dataloader, dataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch._C import device
import torch

from preprocess import gen_vocab_and_data_cache
from MyTransformer import Transformer
from utils import ScheduledOptimizer


parser = argparse.ArgumentParser()

#! task parameters
parser.add_argument('-src_lang', type=str, default='en')
parser.add_argument('-tgt_lang', type=str, default='zh')
parser.add_argument('-task_name', type=str, default='en_zh_translation')

#! path
parser.add_argument('-data_dir', type=str, default='./data/news-commentary-v16.en-zh.tsv')
parser.add_argument('-train_data_cache', type=str, default='./data/train_data_cache.pkl')
parser.add_argument('-val_data_cache', type=str, default='./data/val_data_cache.pkl')
parser.add_argument('-overwrite_data_cache', type=bool, default=False)
parser.add_argument('-src_tokenizer_dir', type=str, default='./models/bert-base-uncased')
parser.add_argument('-tgt_tokenizer_dir', type=str, default='./models/bert-base-chinese')
parser.add_argument('-transform_dir', type=str, default='./checkpoints/transforms.pkl')
parser.add_argument('-overwrite_transform', type=bool, default=False)
parser.add_argument('-log_dir', type=str, default='./results/logs')
parser.add_argument('-result_dir', type=str, default='./results')
parser.add_argument('-checkpoints', type=str, default='./checkpoints')

#! model parameters
parser.add_argument('-src_max_seq_len', type=int, default=100)
parser.add_argument('-tgt_max_seq_len', type=int, default=100)
parser.add_argument('-min_freq', type=int, default=1)
parser.add_argument('-pad_idx', type=int, default=1, help='keep src_pad_idx=tgt_pad_idx')
parser.add_argument('-num_layer', type=int, default=6)
parser.add_argument('-num_head', type=int, default=8)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)
parser.add_argument('-d_model', type=int, default=512)
parser.add_argument('-d_ff', type=int, default=2048)
parser.add_argument('-drop', type=float, default=0.1)
parser.add_argument('-scale_emb', type=bool, default=True)
parser.add_argument('-share_proj_weight', type=bool, default=True)
parser.add_argument('-share_emb_weight', type=bool, default=False)


#! train parameters
parser.add_argument('-seed', type=int, default=123)
parser.add_argument('-epochs', type=int, default=30)
parser.add_argument('-batch_size', type=int, default=200)
parser.add_argument('-use_tb', type=bool, default=True)
parser.add_argument('-warmup_proportion', type=float, default=0.1)
parser.add_argument('-cuda', type=bool, default=True)
parser.add_argument('-train', type=bool, default=False)
parser.add_argument('-eval', type=bool, default=False)
parser.add_argument('-log', type=bool, default=True)
parser.add_argument('-log_step', type=int, default=200)  # 2000
parser.add_argument('-save_step', type=int, default=1000)  # 5000
parser.add_argument('-labelsmooth', type=bool, default=True)
parser.add_argument('-smooth_eps', type=float, default=0.1)
parser.add_argument('-beta1', type=float, default=0.9)
parser.add_argument('-beta2', type=float, default=0.98)
parser.add_argument('-adam_eps', type=float, default=1e-9)


opt = parser.parse_args()


def get_logger(log_dir, verbosity=1, exp_tag=None, name=None):

    now_time = time.strftime("%F-%H-%M-%S")
    log_file = os.path.join(log_dir, now_time+('_{}.log'.format(exp_tag) if exp_tag else '.log'))
    #! level:DEBUG<INFO<WARING  只能输出等级大于等于当前设置等级的信息
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(log_file, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


logger = get_logger(opt.log_dir, exp_tag=None)


def get_dataloader(tag='train'):
    if tag == 'train':
        data_cache_dir = opt.train_data_cache
    if tag == 'val':
        data_cache_dir = opt.val_data_cache
    data = torch.load(data_cache_dir)
    dataset = TensorDataset(data['src'], data['tgt'])
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False)
    return dataloader


def criterion(outputs, targets):
    if opt.labelsmooth:
        n_class = outputs.shape[-1]
        one_hot = F.one_hot(targets, n_class)
        soft_logits = one_hot*(1-opt.smooth_eps)+(1-one_hot)*(opt.smooth_eps/(n_class-1))

        log_prob = F.log_softmax(outputs, dim=1)
        loss = -(soft_logits*log_prob).sum(dim=1)
        no_pad = targets.ne(opt.pad_idx)
        loss = loss.masked_select(no_pad).sum()
    else:
        loss = F.cross_entropy(outputs, targets, ignore_index=opt.pad_idx, reduction='sum')
    return loss


def train(model):
    #* tensorboard,data,optimizer,criterion,global_parameters
    if opt.use_tb:
        writer = SummaryWriter(comment='test')

    dataloader = get_dataloader(tag='train')

    total_step = len(dataloader)*opt.epochs
    warmup_step = int(total_step*opt.warmup_proportion)#! 必须使warm_step>=4000
    optimizer = optim.Adam(model.parameters(), betas=(opt.beta1, opt.beta2), eps=opt.adam_eps)
    scheduledOptimizer = ScheduledOptimizer(optimizer, opt.d_model, warmup_step)

    logger.info(f'Total train steps: {total_step}')
    logger.info('################################')

    global_step = 0
    total_loss = 0.0
    total_num_words = 0

    for epoch in range(opt.epochs):

        model.train()
        for src, tgt in tqdm(dataloader, desc='Epoch: {}'.format(epoch+1)):
            tgt_inputs, tgt_outputs = tgt[:, :-1], tgt[:, 1:].reshape(-1)
            src, tgt_inputs, tgt_outputs = src.to(opt.device), tgt_inputs.to(opt.device), tgt_outputs.to(opt.device)
            logits = model(src, tgt_inputs)
            loss = criterion(logits, tgt_outputs)

            optimizer.zero_grad()
            loss.backward()
            # todo gradient accumulation
            scheduledOptimizer.step_and_update_lr()

            global_step += 1
            total_loss += loss.item()
            num_words = tgt_outputs.ne(opt.pad_idx).sum().item()
            total_num_words += num_words
            print("loss_per_word: {:.3f}".format(loss.item()/num_words))
            if global_step % opt.log_step == 0:
                logs = {}
                logs['train_loss'] = total_loss/total_num_words
                logs['lr'] = scheduledOptimizer.get_lr()[0]
                result = evaluate(model, global_step)
                logs.update(result)
                for key, value in logs.items():
                    writer.add_scalar(key, value, global_step)
                logger.info(f"#####Evaluate: global step {global_step}#####")
                for key, value in result.items():
                    logger.info("{} = {}".format(key, value))

                total_loss = 0.0
                total_num_words = 0

            if (global_step % opt.save_step == 0) or global_step == total_step:
                save_model_dir = os.path.join(opt.checkpoints, f'{global_step}.pth')
                torch.save(model.state_dict(), save_model_dir)  # todo save best model
                logger.info(f"global step {global_step}: save model to {save_model_dir}")

    writer.close()
    return global_step


def evaluate(model, global_step):
    dataloader = get_dataloader(tag='val')
    model.eval()
    total_loss = 0.0
    total_num_words = 0
    total_num_corrects = 0
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc='Evaluating'):
            tgt_inputs, tgt_outputs = tgt[:, :-1], tgt[:, 1:].reshape(-1)
            src, tgt_inputs, tgt_outputs = src.to(opt.device), tgt_inputs.to(opt.device), tgt_outputs.to(opt.device)
            logits = model(src, tgt_inputs)
            loss = criterion(logits, tgt_outputs)
            # todo 验证num_words num_corrects计算是否正确
            no_pad_mask = tgt_outputs.ne(opt.pad_idx)
            num_words = no_pad_mask.sum().item()
            preds = logits.argmax(dim=1)
            num_corrects = preds.eq(tgt_outputs).masked_select(no_pad_mask).sum().item()

            total_loss += loss.item()
            total_num_words += num_words
            total_num_corrects += num_corrects

    loss_per_word = total_loss/total_num_words
    ppl = np.exp(min(loss_per_word, 100))
    acc = total_num_corrects/total_num_words
    result = {'eval_loss_per_word': loss_per_word, 'eval_ppl': ppl, 'eval_acc': acc}
    eval_result_file = os.path.join(opt.result_dir, f'{global_step}_eval_result.txt')
    with open(eval_result_file, 'w') as f:
        for key in result.keys():
            f.write('%s = %s \n' % (key, str(result[key])))
    return result


def main():
    start_time = time.time()

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    opt.cuda = opt.cuda & torch.cuda.is_available()
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    # generate vocab and transform seq data to tensor data
    gen_vocab_and_data_cache(opt, logger)

    # * check model again(emb share,vocab share)
    transformer = Transformer(
        src_max_seq_len=opt.src_max_seq_len,
        tgt_max_seq_len=opt.tgt_max_seq_len,
        src_vocab_size=opt.src_vocab_size,
        tgt_vocab_size=opt.tgt_vocab_size,
        src_pad_idx=opt.pad_idx,
        tgt_pad_idx=opt.pad_idx,
        num_layer=opt.num_layer,
        num_head=opt.num_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_ff=opt.d_ff,
        drop=opt.drop,
        scale_emb=opt.scale_emb,
        share_proj_weight=opt.share_proj_weight,
        share_emb_weight=opt.share_emb_weight
    )
    transformer = transformer.to(opt.device)
    # todo 断点继训
    logger.debug("==== Model Structure: =====")
    logger.debug(transformer)

    logger.info("==== Training/Evaluation Parameters: =====")
    for attr, value in sorted(opt.__dict__.items()):
        logger.info('\t{}={}'.format(attr, value))
    logger.info("==== Parameters End =====\n")

    if opt.train:
        logger.info('==== Train Start=====')
        global_step = train(transformer)  # * check
        logger.info('==== Train End =====')

    if opt.eval:
        logger.info('==== Eval Start=====')
        result = evaluate(transformer, global_step)  # * check
        logger.info(f'==== Eval result: globalstep={global_step} =====')
        for key, value in result.items():
            logger.info("eval_{} = {}".format(key, value))
        logger.info('==== Eval End =====')

    logger.info(f'==== Experiment End-total time:{time.time()-start_time}=====')


if __name__ == "__main__":
    main()
