"""
This file is to generate samples via diverse beam search.
"""

# coding=utf-8

import argparse
import json
import logging
import os

import numpy as np
import torch
from torch import Tensor
from transformers.trainer_utils import set_seed

from inputters import inputters
from inputters.inputter_utils import _norm
from utils.building_utils import boolean_string, build_model, deploy_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def cut_seq_to_eos(sentence, eos, remove_id=None):
    if remove_id is None:
        remove_id = [-1]
    sent = []
    for s in sentence:
        if s in remove_id:
            continue
        if s != eos:
            sent.append(s)
        else:
            break
    return sent


parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True)
parser.add_argument('--inputter_name', type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_checkpoint", '-c', type=str, default=None)

parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument("--max_input_length", type=int, default=160)
parser.add_argument("--max_src_turn", type=int, default=None)
parser.add_argument("--max_decoder_input_length", type=int, default=40)
parser.add_argument("--max_knowledge_length", type=int, default=None)
parser.add_argument('--label_num', type=int, default=None)
parser.add_argument('--multi_knl', action='store_true', help='allow candidate knowledge items')

parser.add_argument('--chinese', action='store_true', help='chinese language')
parser.add_argument('--add_nlg_eval', action='store_true', help='add nlg-eval')

parser.add_argument("--min_length", type=int, default=None)
parser.add_argument("--max_length", type=int, default=None)
parser.add_argument("--num_return_sequences", type=int, default=10)

parser.add_argument("--infer_batch_size", type=int, default=16)
parser.add_argument('--infer_input_file', type=str, nargs='+', required=True)

parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1)
parser.add_argument('--num_beams', type=int, default=10)
parser.add_argument("--num_beam_groups", type=int, default=10)
parser.add_argument("--length_penalty", type=float, default=1.0)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
parser.add_argument("--diversity_penalty", type=int, default=1.0)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
args.device, args.n_gpu = device, n_gpu

logger.info('initializing cuda...')
_ = torch.tensor([1.], device=args.device)

# occupy_mem(os.environ["CUDA_VISIBLE_DEVICES"])

set_seed(args.seed)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

names = {
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
}

toker, model = build_model(checkpoint=args.load_checkpoint, **names)
model = deploy_model(model, args)

model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

if args.fp16:
    from apex import amp

    model, optimizer = amp.initialize(model, opt_level="O1")

model.eval()

inputter = inputters[args.inputter_name]()
dataloader_kwargs = {
    'max_src_turn': args.max_src_turn,
    'max_input_length': args.max_input_length,
    'max_decoder_input_length': args.max_decoder_input_length,
    'max_knowledge_length': args.max_knowledge_length,
    'label_num': args.label_num,
    'multi_knl': args.multi_knl,
    'infer_batch_size': args.infer_batch_size,
}

pad = toker.pad_token_id
if pad is None:
    pad = toker.eos_token_id
    assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
bos = toker.bos_token_id
if bos is None:
    bos = toker.cls_token_id
    assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
eos = toker.eos_token_id
if eos is None:
    eos = toker.sep_token_id
    assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

generation_kwargs = {
    'max_length': args.max_length,
    'min_length': args.min_length,
    'do_sample': False,
    'temperature': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'num_beams': args.num_beams,
    'num_beam_groups': args.num_beam_groups,
    'num_return_sequences': args.num_return_sequences,
    'diversity_penalty': args.diversity_penalty,
    'length_penalty': args.length_penalty,
    'repetition_penalty': args.repetition_penalty,
    'no_repeat_ngram_size': args.no_repeat_ngram_size,
    'encoder_no_repeat_ngram_size': args.no_repeat_ngram_size if model.config.is_encoder_decoder else None,
    'pad_token_id': pad,
    'bos_token_id': bos,
    'eos_token_id': eos,
}
print(json.dumps(generation_kwargs, indent=2, ensure_ascii=False))

for infer_idx, infer_input_file in enumerate(args.infer_input_file):
    set_seed(args.seed)
    infer_dataloader = inputter.infer_dataloader(
        infer_input_file,
        toker,
        **dataloader_kwargs
    )

    res = []
    other_res = {}
    decode = lambda x: _norm(toker.decode(x))
    for batch, posts, references, sample_ids in infer_dataloader:
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        batch.update(generation_kwargs)
        encoded_info, generations = model.generate(**batch)

        batch_other_res = None
        if 'other_res' in batch:
            batch_other_res = batch.pop('other_res')
            add_acc = 'acc_map' in batch_other_res and any(
                k in batch_other_res and v in encoded_info for k, v in batch_other_res['acc_map'].items())
            if add_acc:
                if 'acc' not in other_res:
                    other_res['acc'] = {}
                if 'acc_map' not in other_res:
                    other_res['acc_map'] = batch_other_res['acc_map']

                for k, v in batch_other_res['acc_map'].items():
                    if k not in batch_other_res or v not in encoded_info:  # TODO
                        continue  # TODO
                    batch_other_res[k] = batch_other_res[k].tolist()
                    encoded_info[v] = encoded_info[v].tolist()
                    if f'{v}_top1' in encoded_info:
                        encoded_info[f'{v}_top1'] = encoded_info[f'{v}_top1'].tolist()
                    if f'{v}_top3' in encoded_info:
                        encoded_info[f'{v}_top3'] = encoded_info[f'{v}_top3'].tolist()
                    if f'{v}_dist' in encoded_info:
                        encoded_info[f'{v}_dist'] = encoded_info[f'{v}_dist'].tolist()

                    if k not in other_res['acc']:
                        other_res['acc'][k] = []
                    other_res['acc'][k].extend(batch_other_res[k])

                    if v not in other_res['acc']:
                        other_res['acc'][v] = []
                    other_res['acc'][v].extend(encoded_info[v])

                    if f'{v}_top1' in encoded_info:
                        if f'{v}_top1' not in other_res['acc']:
                            other_res['acc'][f'{v}_top1'] = []
                        other_res['acc'][f'{v}_top1'].extend(encoded_info[f'{v}_top1'])
                    if f'{v}_top3' in encoded_info:
                        if f'{v}_top3' not in other_res['acc']:
                            other_res['acc'][f'{v}_top3'] = []
                        other_res['acc'][f'{v}_top3'].extend(encoded_info[f'{v}_top3'])

                    if f'{v}_dist' in encoded_info:
                        if f'{v}_dist' not in other_res['acc']:
                            other_res['acc'][f'{v}_dist'] = []
                        other_res['acc'][f'{v}_dist'].extend(encoded_info[f'{v}_dist'])

        generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]

        for idx in range(len(sample_ids)):
            p = posts[idx]
            r = references[idx]
            if args.num_return_sequences > 1:
                g = []
                for gg in generations[idx * args.num_return_sequences: (idx + 1) * args.num_return_sequences]:
                    g.append(gg)
            else:
                g = generations[idx]

            if isinstance(g[0], list):
                g = [decode(gg) for gg in g]
            else:
                g = decode(g)

            tmp_res_to_append = {'sample_id': sample_ids[idx], 'post': p, 'response': r, 'generation': g}
            # print('> context:   ', p)
            # print('> generation:', g)

            other_res_to_append = {}
            if batch_other_res is not None:
                if add_acc:
                    for k, v in batch_other_res['acc_map'].items():
                        if k not in batch_other_res or v not in encoded_info:  # TODO
                            continue  # TODO
                        other_res_to_append[v] = encoded_info[v][idx]
                        if f'{v}_top1' in encoded_info:
                            other_res_to_append[f'{v}_top1'] = encoded_info[f'{v}_top1'][idx]
                        if f'{v}_top3' in encoded_info:
                            other_res_to_append[f'{v}_top3'] = encoded_info[f'{v}_top3'][idx]
                        if f'{v}_dist' in encoded_info:
                            other_res_to_append[f'{v}_dist'] = ' '.join(map(str, encoded_info[f'{v}_dist'][idx]))

            tmp_res_to_append.update(other_res_to_append)
            res.append(tmp_res_to_append)

        # raise EOFError

    checkpoint_dir_path = '/'.join(args.load_checkpoint.split('/')[:-1])
    checkpoint_name = args.load_checkpoint.split('/')[-1]
    infer_input_file_name = infer_input_file.split('/')[-1]
    infer_input_file_name = '.'.join(infer_input_file_name.split('.')[:-1])
    save_dir = f'{checkpoint_dir_path}/candidates_{args.num_return_sequences}_{checkpoint_name}_{infer_input_file_name}'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("Make dir:", save_dir)

    with open(os.path.join(save_dir, f'candidates.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, sort_keys=False)

    with open(os.path.join(save_dir, f'candidates.txt'), 'w') as f:
        for line in res:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
