import pandas as pd
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertTokenizer
import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

UNK, PAD, BOS, EOS = '[UNK]', '[PAD]', '[BOS]', '[EOS]'


def read_tsv_corpus(data_dir):
    data = pd.read_csv(data_dir, sep='\t', encoding='utf-8', names=['src', 'tgt'])
    data = data.dropna(axis=0).reset_index(drop=True)
    return {'src': data['src'].tolist(), 'tgt': data['tgt'].tolist()}


class TextTransform(object):
    def __init__(self, tokenizer, max_seq_len, unk_token, pad_token, bos_token, eos_token, min_freq=1, truncate_first=False):
        self.min_freq = min_freq
        self.special_tokens = [unk_token, pad_token, bos_token, eos_token]
        self.unk_idx, self.pad_idx, self.bos_idx, self.eos_idx = range(4)
        self.tokenizer = tokenizer
        assert max_seq_len >= 2
        self.max_seq_len = max_seq_len-2
        self.truncate_first = truncate_first
        self.vocab = None

    def build_vocab(self, corpus):
        def yield_tokens(corpus, tokenizer):
            for sentence in corpus:
                yield tokenizer.tokenize(sentence)
        corpus_tokens = yield_tokens(corpus, self.tokenizer)
        self.vocab = build_vocab_from_iterator(corpus_tokens, min_freq=self.min_freq, specials=self.special_tokens, special_first=True)
        self.vocab.set_default_index(self.unk_idx)

    def load_vocab(self, vocab_dir):
        self.vocab = torch.load(vocab_dir)

    def pad(self, seq):
        padded_seq = [self.bos_idx]+(seq[-self.max_seq_len:] if self.truncate_first else seq[:self.max_seq_len])+[self.eos_idx]+[self.pad_idx]*max(0, self.max_seq_len-len(seq))
        return padded_seq

    def vocab_size(self):
        return len(self.vocab)

    def __call__(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        token_ids = self.vocab(tokens)
        input_ids = self.pad(token_ids)
        return input_ids


def sequential_transform(src_data, tgt_data, src_trans, tgt_trans):
    src_examples, tgt_examples = [], []
    for src_seq, tgt_seq in tqdm(zip(src_data, tgt_data)):
        src_examples.append(src_trans(src_seq))
        tgt_examples.append(tgt_trans(tgt_seq))
    return torch.tensor(src_examples), torch.tensor(tgt_examples)


def gen_vocab_and_data_cache(opt, logger):

    src_tokenizer = BertTokenizer.from_pretrained(opt.src_tokenizer_dir)
    tgt_tokenizer = BertTokenizer.from_pretrained(opt.tgt_tokenizer_dir)

    data = read_tsv_corpus(opt.data_dir)  # data:{'src':**,'tgt':**}
    src_train, src_val, tgt_train, tgt_val = train_test_split(
        data['src'], data['tgt'], test_size=0.1, shuffle=True, random_state=opt.seed
    )

    src_textTransform = TextTransform(src_tokenizer, opt.src_max_seq_len, UNK, PAD, BOS, EOS, min_freq=opt.min_freq)
    tgt_textTransform = TextTransform(tgt_tokenizer, opt.tgt_max_seq_len, UNK, PAD, BOS, EOS, min_freq=opt.min_freq)

    # ! 'To sharing word embedding the src/trg word2idx table shall be the same.'

    if (not os.path.exists(opt.transform_dir)) or opt.overwrite_transform:
        logger.info('building src vocabulary...')
        src_textTransform.build_vocab(data['src'])
        logger.info('building tgt vocabulary...')
        tgt_textTransform.build_vocab(data['tgt'])
        textTransforms = {'src': src_textTransform, 'tgt': tgt_textTransform}
        torch.save(textTransforms, opt.transform_dir)

    else:
        logger.info("sequential_transform cache alreadly existed")
        logger.info(f"load sequential_transform cache from {opt.transform_dir}")
        textTransforms = torch.load(opt.transform_dir)
        src_textTransform, tgt_textTransform = textTransforms['src'], textTransforms['tgt']

    if (not (os.path.exists(opt.train_data_cache) and os.path.exists(opt.val_data_cache))) or opt.overwrite_data_cache:
        src_train_examples, tgt_train_examples = sequential_transform(src_train, tgt_train, src_textTransform, tgt_textTransform)
        src_val_examples, tgt_val_examples = sequential_transform(src_val, tgt_val, src_textTransform, tgt_textTransform)
        train_data_cache = {'src': src_train_examples, 'tgt': tgt_train_examples}
        val_data_cache = {'src': src_val_examples, 'tgt': tgt_val_examples}
        torch.save(train_data_cache, opt.train_data_cache)
        torch.save(val_data_cache, opt.val_data_cache)
        logger.info(f'save train data cache to {opt.train_data_cache}')
        logger.info(f'save val data cache to {opt.val_data_cache}')
    else:
        logger.info("train and val data cache alreadly existed")

    logger.info('src vocab size:{} tgt vocab size:{} train examples:{} val examples:{}'.format(
        src_textTransform.vocab_size(),
        tgt_textTransform.vocab_size(),
        len(src_train),
        len(src_val)
    ))

    opt.src_vocab_size, opt.tgt_vocab_size = src_textTransform.vocab_size(), tgt_textTransform.vocab_size()

    # print('我爱你 铪:{}'.format(tgt_textTransform('我爱你 铪')))
    # print('I 铪 love you{}'.format(src_textTransform('I 铪 love you')))
    # print(src_data[:2])
    # print(len(src_data[0]),len(tgt_data[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_lang', type=str, default='en')
    parser.add_argument('-tgt_lang', type=str, default='zh')

    parser.add_argument('-data_dir', type=str, default='./data/news-commentary-v16.en-zh.tsv')
    parser.add_argument('-train_data_cache', type=str, default='./data/train_data_cache.pkl')
    parser.add_argument('-val_data_cache', type=str, default='./data/val_data_cache.pkl')
    parser.add_argument('-overwrite_data_cache', type=bool, default=False)
    parser.add_argument('-src_tokenizer_dir', type=str, default='./models/bert-base-uncased')
    parser.add_argument('-tgt_tokenizer_dir', type=str, default='./models/bert-base-chinese')
    parser.add_argument('-vocab_dir', type=str, default='./checkpoints/vocabs')
    parser.add_argument('-overwrite_vocab', type=bool, default=False)

    parser.add_argument('-max_src_len', type=int, default=128)
    parser.add_argument('-max_tgt_len', type=int, default=128)
    parser.add_argument('-min_freq', type=int, default=1)
    opt = parser.parse_args()

    src_tokenizer = BertTokenizer.from_pretrained(opt.src_tokenizer_dir)
    tgt_tokenizer = BertTokenizer.from_pretrained(opt.tgt_tokenizer_dir)

    data = read_tsv_corpus(opt.data_dir)  # data:{'src':**,'tgt':**}
    src_train, src_val, tgt_train, tgt_val = train_test_split(data['src'], data['tgt'], test_size=0.1, shuffle=True)

    opt.src_vocab_dir = os.path.join(opt.vocab_dir, '{}-src_vocab.pkl'.format(opt.src_lang))
    opt.tgt_vocab_dir = os.path.join(opt.vocab_dir, '{}-tgt_vocab.pkl'.format(opt.tgt_lang))

    src_textTransform = TextTransform(src_tokenizer, opt.max_src_len, UNK, PAD, BOS, EOS, min_freq=opt.min_freq)
    tgt_textTransform = TextTransform(tgt_tokenizer, opt.max_tgt_len, UNK, PAD, BOS, EOS, min_freq=opt.min_freq)

    # ! 'To sharing word embedding the src/trg word2idx table shall be the same.'
    if (not os.path.exists(opt.src_vocab_dir)) or opt.overwrite_vocab:
        src_textTransform.build_vocab(data['src'], opt.src_vocab_dir)
    else:
        src_textTransform.load_vocab(opt.src_vocab_dir)
        print(f'load src_vocab from {opt.src_vocab_dir}')

    if (not os.path.exists(opt.tgt_vocab_dir)) or opt.overwrite_vocab:
        tgt_textTransform.build_vocab(data['tgt'], opt.tgt_vocab_dir)
    else:
        tgt_textTransform.load_vocab(opt.tgt_vocab_dir)
        print(f'load tgt_vocab from {opt.tgt_vocab_dir}')

    if (not (os.path.exists(opt.train_data_cache) and os.path.exists(opt.val_data_cache))) or opt.overwrite_data_cache:
        src_train_examples, tgt_train_examples = sequential_transform(src_train, tgt_train, src_textTransform, tgt_textTransform)
        src_val_examples, tgt_val_examples = sequential_transform(src_val, tgt_val, src_textTransform, tgt_textTransform)
        train_data_cache = {'src': src_train_examples, 'tgt': tgt_train_examples}
        val_data_cache = {'src': src_val_examples, 'tgt': tgt_val_examples}
        torch.save(train_data_cache, opt.train_data_cache)
        torch.save(val_data_cache, opt.val_data_cache)
        print(f'save train data cache to {opt.train_data_cache}')
        print(f'save val data cache to {opt.val_data_cache}')
    else:
        print("train and val data cache has existed")
    # !test vocab
    print('我爱你 铪:{}'.format(tgt_textTransform('我爱你 铪')))
    print('I 铪 love you{}'.format(src_textTransform('I 铪 love you')))

    # !test data
    test_data = torch.load(opt.train_data_cache)
    test = test_data['tgt']

    sen = test[np.random.randint(0, 10)]
    for w in sen:
        if w == 1:
            break
        print(tgt_textTransform.vocab.lookup_token(w))


if __name__ == '__main__':
    main()
    import torch
    torch.nn.Transformer