import torch
from translator import Translator
import argparse
from MyTransformer import Transformer,PadMask
import numpy as np
from torch.utils.data import TensorDataset


parser = argparse.ArgumentParser()

parser.add_argument('-model_checkpoint', type=str, default='./checkpoints/8000.pth')
parser.add_argument('-cuda', type=bool, default=True)
parser.add_argument('-transform_dir', type=str, default='./checkpoints/transforms.pkl')
#! model parameters
parser.add_argument('-src_max_seq_len', type=int, default=100)
parser.add_argument('-tgt_max_seq_len', type=int, default=100)
parser.add_argument('-min_freq', type=int, default=1)
parser.add_argument('-pad_idx', type=int, default=1)
parser.add_argument('-bos_idx', type=int, default=2)
parser.add_argument('-eos_idx', type=int, default=3)
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


opt = parser.parse_args()


def main():

    opt.cuda = opt.cuda & torch.cuda.is_available()
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    textTransforms = torch.load(opt.transform_dir)
    src_textTransform, tgt_textTransform = textTransforms['src'], textTransforms['tgt']

    opt.src_vocab_size, opt.tgt_vocab_size = src_textTransform.vocab_size(), tgt_textTransform.vocab_size()
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
    transformer.load_state_dict(torch.load(opt.model_checkpoint))

    translator = Translator(transformer, 3, opt.tgt_max_seq_len, opt.pad_idx, opt.pad_idx, opt.bos_idx, opt.eos_idx)
    translator=translator.to(opt.device)
    
    # src_seq='I love you'
    # tgt_seq='我爱'
    # src_idx=src_textTransform(src_seq)
    # src_tensor=torch.tensor(src_idx,dtype=torch.long)
    # src_tensor=src_tensor.to(opt.device)
    # tgt_idx=tgt_textTransform(tgt_seq)
    # tgt_tensor=torch.tensor(tgt_idx,dtype=torch.long)
    # tgt_tensor=tgt_tensor.to(opt.device)
    # # enc_output, gen_seq, scores=translator._get_init_state(src_tensor,src_mask)
    # # print(enc_output)
    # # print(gen_seq)
    # # print(scores)
    # src_tensor=src_tensor.reshape(1,-1)
    # tgt_tensor=src_tensor.reshape(1,-1)

    data = torch.load('./data/val_data_cache.pkl')
    dataset = TensorDataset(data['src'], data['tgt'])

    transformer.eval()
    with torch.no_grad():
        outputs=transformer(src_tensor,tgt_tensor)
    print(outputs.shape)
    res=np.argmax(outputs.detach().cpu().numpy(),axis=1)
    # res=translator.translate_sentence(src_tensor)
    res=tgt_textTransform.vocab.lookup_tokens(res)
    print(res)


if __name__ == '__main__':
    print(opt.__dict__)
    import json
    with open('translate_args.json','w',encoding='utf-8') as f:
        json.dump(opt.__dict__,f,ensure_ascii=False,indent=2)

    import torchtext
    torchtext.datasets.Multi30k()
    #main()
    

