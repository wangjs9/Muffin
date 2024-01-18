# hide
# Imports

"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
from cProfile import label
from multiprocessing import context
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from metric.myMetrics import Metric
import glob
import logging
import pickle
import random
import sys
import re
import shutil
from typing import Dict, List, Tuple
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import torch
import nltk

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import time
from pathlib import Path
import json
from src.transformers import (
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from src.transformers import (BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig)

# from utils.data_parallel import BalancedDataParallel
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

# Configs
logger = logging.getLogger(__name__)


class InputFeatures_train(object):
    def __init__(self, input_ids, kws_enc, trans_cls_index,
                 lm_labels, trans_emotion_labels, strategy_ids, strat_labels,
                 emo_trans, trans_graph, inter_graph, trans_edge_type, inter_edge_type):
        self.input_ids = input_ids
        self.trans_cls_index = trans_cls_index
        self.lm_labels = lm_labels
        self.trans_emotion_labels = trans_emotion_labels
        self.strategy_ids = strategy_ids
        self.strat_labels = strat_labels
        self.emo_trans = emo_trans
        self.trans_graph = trans_graph
        self.inter_graph = inter_graph
        self.trans_edge_type = trans_edge_type
        self.inter_edge_type = inter_edge_type
        self.kws_enc = kws_enc


class InputFeatures_blender(object):
    def __init__(self, encoder_feature, decoder_feature):
        self.input_ids = encoder_feature.input_ids
        self.trans_cls_index = encoder_feature.trans_cls_index
        self.trans_emotion_labels = encoder_feature.trans_emotion_labels
        self.decoder_input_ids = decoder_feature.input_ids
        self.decoder_lm_labels = decoder_feature.lm_labels
        self.decoder_strategy_ids = decoder_feature.strategy_ids
        self.strategy_labels = encoder_feature.strat_labels + [decoder_feature.strategy_ids[0]]
        self.emo_trans = encoder_feature.emo_trans
        self.trans_graph = encoder_feature.trans_graph
        self.inter_graph = encoder_feature.inter_graph
        self.trans_edge_type = encoder_feature.trans_edge_type
        self.inter_edge_type = encoder_feature.inter_edge_type
        self.keywords = encoder_feature.kws_enc


def process_row_to_comet_query(row):
    sents = row.strip().split('EOS')
    n_sent = len(sents)
    all_seeker_uttrs = []
    for i in range(n_sent - 1, -1, -1):
        tokens = sents[i].strip().split(' ')
        if int(tokens[1]) == 0:
            if int(tokens[1]) == 0:
                return ' '.join(tokens[3:])


def summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, chat_texts,
            test_situation_file_path):
    with open(test_file_path, "r", encoding="utf-8") as f:
        ctx = f.read().split("\n")
    with open(test_situation_file_path, "r", encoding="utf-8") as f:
        st = f.read().split("\n")
    ctx = ctx[:-1]
    st = st[:-1]
    with open(generate_file_path, "r", encoding="utf-8") as f:
        gen_rep = json.load(f)
    with open(reference_file_path, "r", encoding="utf-8") as f:
        ref_rep = json.load(f)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        for (ctx_row, ref_rep_row, gen_rep_row, chat_text, st_row) in zip(ctx, ref_rep, gen_rep, chat_texts, st):
            query = process_row_to_comet_query(chat_text)
            if query is None:
                query = ""

            context = ctx_row.split(' EOS')
            utterances = []
            for item in context[:-1]:
                _, src_role, _, src = _norm_text(item)
                if src_role == 0:
                    utt = src
                else:
                    utt = src.split("] ")[1]
                utterances.append(utt)

            line = '\t'.join(utterances) + '\t' + gen_rep_row + '\n'
            # line = '[contxt]\t' + ctx_row + '\n[reference_response]\t' + ref_rep_row + '\n[hypothesis_response]\t' + gen_rep_row + '\n[comet query]\t' + query +'\n[situation]\t' + st_row + '\n[situation comet blocks (attention top5)]\t' + '\n' * 2

            f.writelines(line)


def _Emotion_tag_to_tag_idx():
    return {'angry': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'neutral': 5}


def _construct_graph(trans_cls_index, trans_role):
    trans_graph = np.zeros((3 * len(trans_cls_index), 3 * len(trans_cls_index)))
    inter_graph = np.zeros((3 * len(trans_cls_index), 3 * len(trans_cls_index)))
    trans_edge_type = np.zeros((3 * len(trans_cls_index), 3 * len(
        trans_cls_index)))  # 1: Sem-Sem, 2: Sem-Emo, 3: Sem-Str, 4: Emo-Emo, 5: Emo-Str, 6: Str-Str, 7: Str-Emo
    inter_edge_type = np.zeros((3 * len(trans_cls_index), 3 * len(trans_cls_index)))

    trans_graph_wo_sem = np.zeros((2 * len(trans_cls_index), 2 * len(trans_cls_index)))
    inter_graph_wo_sem = np.zeros((2 * len(trans_cls_index), 2 * len(trans_cls_index)))
    trans_edge_type_wo_sem = np.zeros((2 * len(trans_cls_index), 2 * len(
        trans_cls_index)))  # 1: Sem-Sem, 2: Sem-Emo, 3: Sem-Str, 4: Emo-Emo, 5: Emo-Str, 6: Str-Str, 7: Str-Emo
    inter_edge_type_wo_sem = np.zeros((2 * len(trans_cls_index), 2 * len(trans_cls_index)))

    trans_graph_wo_str = np.zeros((2 * len(trans_cls_index), 2 * len(trans_cls_index)))
    inter_graph_wo_str = np.zeros((2 * len(trans_cls_index), 2 * len(trans_cls_index)))
    trans_edge_type_wo_str = np.zeros((2 * len(trans_cls_index), 2 * len(
        trans_cls_index)))  # 1: Sem-Sem, 2: Sem-Emo, 3: Sem-Str, 4: Emo-Emo, 5: Emo-Str, 6: Str-Str, 7: Str-Emo
    inter_edge_type_wo_str = np.zeros((2 * len(trans_cls_index), 2 * len(trans_cls_index)))

    trans_graph_wo_emo = np.zeros((2 * len(trans_cls_index), 2 * len(trans_cls_index)))
    inter_graph_wo_emo = np.zeros((2 * len(trans_cls_index), 2 * len(trans_cls_index)))
    trans_edge_type_wo_emo = np.zeros((2 * len(trans_cls_index), 2 * len(
        trans_cls_index)))  # 1: Sem-Sem, 2: Sem-Emo, 3: Sem-Str, 4: Emo-Emo, 5: Emo-Str, 6: Str-Str, 7: Str-Emo
    inter_edge_type_wo_emo = np.zeros((2 * len(trans_cls_index), 2 * len(trans_cls_index)))

    for i in range(len(trans_cls_index)):
        if i > 0:
            j = i - 1
            cur_role = trans_role[i]
            usr_cnt, sys_cnt = 0, 0
            while j >= 0:
                pre_role = trans_role[j]
                if pre_role == cur_role:
                    if cur_role == 0:  # User-User Connection

                        trans_graph[3 * i][3 * j] = 1
                        trans_graph[3 * i + 2][3 * j + 2] = 1
                        inter_graph[3 * i + 2][3 * j] = 1

                        trans_graph_wo_sem[2 * i + 1][2 * j + 1] = 1

                        trans_graph_wo_str[2 * i][2 * j] = 1
                        trans_graph_wo_str[2 * i + 1][2 * j + 1] = 1
                        inter_graph_wo_str[2 * i + 1][2 * j] = 1

                        trans_graph_wo_emo[2 * i][2 * j] = 1

                        trans_edge_type[3 * i][3 * j] = 1
                        inter_edge_type[3 * i + 2][3 * j] = 2
                        trans_edge_type[3 * i + 2][3 * j + 2] = 4

                        trans_edge_type_wo_sem[2 * i + 1][2 * j + 1] = 4

                        trans_edge_type_wo_str[2 * i][2 * j] = 1
                        inter_edge_type_wo_str[2 * i + 1][2 * j] = 2
                        trans_edge_type_wo_str[2 * i + 1][2 * j + 1] = 4

                        trans_edge_type_wo_emo[2 * i][2 * j] = 1

                        usr_cnt += 1
                    else:  # Sys-Sys Connection

                        trans_graph[3 * i][3 * j] = 1
                        inter_graph[3 * i + 1][3 * j] = 1
                        trans_graph[3 * i + 1][3 * j + 1] = 1

                        trans_graph_wo_sem[2 * i][2 * j] = 1

                        trans_graph_wo_str[2 * i][2 * j] = 1

                        trans_graph_wo_emo[2 * i][2 * j] = 1
                        trans_graph_wo_emo[2 * i + 1][2 * j + 1] = 1
                        inter_graph_wo_emo[2 * i + 1][2 * j] = 1

                        trans_edge_type[3 * i][3 * j] = 1
                        inter_edge_type[3 * i + 1][3 * j] = 3
                        trans_edge_type[3 * i + 1][3 * j + 1] = 6

                        trans_edge_type_wo_sem[2 * i][2 * j] = 6

                        trans_edge_type_wo_str[2 * i][2 * j] = 1

                        trans_edge_type_wo_emo[2 * i][2 * j] = 1
                        inter_edge_type_wo_emo[2 * i + 1][2 * j] = 3
                        trans_edge_type_wo_emo[2 * i + 1][2 * j + 1] = 6
                        sys_cnt += 1
                else:
                    if cur_role == 0:  # Sys-User Connection

                        trans_graph[3 * i][3 * j] = 1
                        inter_graph[3 * i + 2][3 * j] = 1
                        inter_graph[3 * i + 2][3 * j + 1] = 1

                        inter_graph_wo_sem[2 * i + 1][2 * j] = 1

                        trans_graph_wo_str[2 * i][2 * j] = 1
                        inter_graph_wo_str[2 * i + 1][2 * j] = 1

                        trans_graph_wo_emo[2 * i][2 * j] = 1

                        trans_edge_type[3 * i][3 * j] = 1
                        inter_edge_type[3 * i + 2][3 * j] = 2
                        inter_edge_type[3 * i + 2][3 * j + 1] = 7

                        inter_edge_type_wo_sem[2 * i + 1][2 * j] = 7

                        trans_edge_type_wo_str[2 * i][2 * j] = 1
                        inter_edge_type_wo_str[2 * i + 1][2 * j] = 2

                        trans_edge_type_wo_emo[2 * i][2 * j] = 1
                        sys_cnt += 1
                    else:  # User-Sys Connection

                        trans_graph[3 * i][3 * j] = 1
                        inter_graph[3 * i + 1][3 * j] = 1
                        inter_graph[3 * i + 1][3 * j + 2] = 1

                        inter_graph_wo_sem[2 * i][2 * j + 1] = 1

                        trans_graph_wo_str[2 * i][2 * j] = 1

                        trans_graph_wo_emo[2 * i][2 * j] = 1
                        inter_graph_wo_emo[2 * i + 1][2 * j] = 1

                        trans_edge_type[3 * i][3 * j] = 1
                        inter_edge_type[3 * i + 1][3 * j] = 3
                        inter_edge_type[3 * i + 1][3 * j + 2] = 5

                        inter_edge_type_wo_sem[2 * i][2 * j + 1] = 5

                        trans_edge_type_wo_str[2 * i][2 * j] = 1

                        trans_edge_type_wo_emo[2 * i][2 * j] = 1
                        inter_edge_type_wo_emo[2 * i + 1][2 * j] = 3
                        usr_cnt += 1
                j -= 1
    return trans_graph, inter_graph, trans_edge_type, inter_edge_type, trans_graph_wo_sem, inter_graph_wo_sem, trans_edge_type_wo_sem, inter_edge_type_wo_sem, \
        trans_graph_wo_str, inter_graph_wo_str, trans_edge_type_wo_str, inter_edge_type_wo_str, trans_graph_wo_emo, inter_graph_wo_emo, trans_edge_type_wo_emo, inter_edge_type_wo_emo


def _make_feature(args, id_, sents, kws_enc, utt_emotion, rls, ts, cls, bos, eos, block_size=512, strategy_labels=None,
                  evaluate=False, decoder=False, comet_repre=None):
    # we did't use role label and turn number in modeling as they did't carry significant improvement. However, codes still remain here.
    emo_label_map = _Emotion_tag_to_tag_idx()
    emo_labels = [emo_label_map[item] for item in utt_emotion]
    for i in range(len(rls)):
        if rls[i] == 1:
            emo_labels[i] = 8

    if len(sents) == 0:
        return InputFeatures_train([], [], [], [], [],
                                   [], [], [], [])

    if decoder:
        input_ids = [bos] + sents[0]
    else:
        input_ids = [i for s in sents for i in s + [cls]]  # eos [sem]+[str]+[emo] / [cls]

    lm_labels = []
    if decoder:
        lm_labels = sents[0] + [eos]

    strategy_ids = [strategy_labels[0]]

    i = len(lm_labels) - 1

    if len(input_ids) == 0:
        import pdb
        pdb.set_trace()

    if not decoder:
        if len(input_ids) >= block_size:
            input_ids = input_ids[-block_size:]
            input_ids[0] = cls
        else:
            input_ids = [cls] + input_ids

    j = 0
    cls_index = []
    while j < len(input_ids):
        if input_ids[j] == cls or input_ids[j] == eos:  # cls
            cls_index.append(j)
        j += 1

    trans_cls_index = []
    sys_index = []
    trans_turn = []
    if len(rls) < 3:
        trans_cls_index = cls_index
        strat_labels = strategy_labels
        trans_role = rls
        trans_turn = ts
        trans_emotion_labels = emo_labels
    else:
        j = len(rls) - 1
        cnt = 0
        while j >= 0:
            if rls[j] == 1:
                cnt += 1
                sys_index.append(j)
                if cnt == 2:
                    break
            j -= 1

        if j < 0:
            j = 0

        trans_num = len(rls) - j
        if trans_num >= len(cls_index):
            trans_num = len(cls_index) - 1
            trans_cls_index = cls_index
        else:
            trans_cls_index = cls_index[len(cls_index) - trans_num - 1:]

        trans_role = rls[len(rls) - trans_num:]
        trans_emotion_labels = emo_labels[len(emo_labels) - trans_num - 1:]
        strat_labels = strategy_labels[len(strategy_labels) - trans_num:]
        trans_turn = ts[len(ts) - trans_num:]
        assert len(trans_cls_index) - 1 == len(strat_labels) == len(trans_turn)  # == len(trans_emotion_labels) - 1

    trans_role += [1]
    trans_graph, inter_graph, trans_edge_type, inter_edge_type, \
        trans_graph_wo_sem, inter_graph_wo_sem, trans_edge_type_wo_sem, inter_edge_type_wo_sem, \
        trans_graph_wo_str, inter_graph_wo_str, trans_edge_type_wo_str, inter_edge_type_wo_str, \
        trans_graph_wo_emo, inter_graph_wo_emo, trans_edge_type_wo_emo, inter_edge_type_wo_emo, = _construct_graph(
        trans_cls_index, trans_role)

    if args.wo_sem:
        trans_graph, inter_graph, trans_edge_type, inter_edge_type = trans_graph_wo_sem, inter_graph_wo_sem, trans_edge_type_wo_sem, inter_edge_type_wo_sem
    if args.wo_str:
        trans_graph, inter_graph, trans_edge_type, inter_edge_type = trans_graph_wo_str, inter_graph_wo_str, trans_edge_type_wo_str, inter_edge_type_wo_str
    if args.wo_emo:
        trans_graph, inter_graph, trans_edge_type, inter_edge_type = trans_graph_wo_emo, inter_graph_wo_emo, trans_edge_type_wo_emo, inter_edge_type_wo_emo

    if evaluate and strategy_labels[-1] != 8:
        try:
            lm_labels[lm_labels.index(strategy_labels[-1] + 54944)] = -100
        except Exception:
            pass

    # emotion transition with COMET (xReact)
    if not decoder:
        emo_trans = []
        want_trans = []
        effect_trans = []
        intent_trans = []
        length = len(trans_cls_index) - 1
        for i in range(length):
            if trans_role[i] == 1:
                oReact_csk = comet_repre['oReact'][:-1]  # exclude the last sentence which is response
                oWant_csk = comet_repre['oWant'][:-1]
                oEffect_csk = comet_repre['oEffect'][:-1]
                emo_trans.append(oReact_csk[i - length])
                want_trans.append(oWant_csk[i - length])
                effect_trans.append(oEffect_csk[i - length])
                intent_trans.append(np.zeros(1024))
            else:
                xReact_csk = comet_repre['xReact'][:-1]  # exclude the last sentence which is response
                xWant_csk = comet_repre['xWant'][:-1]
                xEffect_csk = comet_repre['xEffect'][:-1]
                xIntent_csk = comet_repre['xIntent'][:-1]
                emo_trans.append(xReact_csk[i - length])
                want_trans.append(xWant_csk[i - length])
                effect_trans.append(xEffect_csk[i - length])
                intent_trans.append(xIntent_csk[i - length])
        assert len(trans_cls_index) - 1 == len(strat_labels) == len(emo_trans)
        emo_trans.append(np.zeros(1024))
        want_trans.append(np.zeros(1024))
        effect_trans.append(np.zeros(1024))
        intent_trans.append(np.zeros(1024))
    else:
        emo_trans = None
        want_trans = None
        effect_trans = None
        intent_trans = None

    kws_enc = kws_enc[-len(trans_cls_index):]

    feature = InputFeatures_train(input_ids, kws_enc, trans_cls_index, \
                                  lm_labels, trans_emotion_labels, \
                                  strategy_ids, strat_labels, emo_trans, \
                                  trans_graph, inter_graph, trans_edge_type, inter_edge_type)
    return feature


def _norm_text(text):
    emo, r, t, *toks = text.strip().split()
    try:
        emo = 0  # int(emo)
        r = int(r)
        t = int(t)
        toks = ' '.join(toks[:len(toks)])
    except Exception as e:
        raise e
    return emo, r, t, toks


def _get_inputs_from_text(text, utt_kws, tokenizer, strategy=True, cls=False):
    srcs = text.strip()
    inputs = []
    kws = []
    roles = []
    turns = []
    strategy_labels = []
    srcs = srcs.split(" EOS")
    emotion = None
    strategy_map = {"[Question]": 0, "[Reflection of feelings]": 1, "[Information]": 2,
                    "[Restatement or Paraphrasing]": 3, "[Others]": 4, "[Self-disclosure]": 5,
                    "[Affirmation and Reassurance]": 6, "[Providing Suggestions]": 7}
    for idx, src in enumerate(srcs):

        if src == "":
            continue

        src_emo, src_role, src_turn, src = _norm_text(src)

        if src_role == 0:
            context_id = tokenizer.encode(src)
        else:
            utt = src.split("] ")[1]
            context_id = tokenizer.encode(utt)

        if len(utt_kws[idx]) == 0:
            kws_id = [0]
        else:
            kws_seq = ' '.join(utt_kws[idx])
            kws_id = tokenizer.encode(kws_seq)

        if emotion is None:
            emotion = src_emo

        if not strategy:
            context_id = [i for i in context_id if i < 54944]
        elif cls:
            context_id = tokenizer.cls + [i for i in context_id if i < 54944]
        else:
            pass

        if src_role == 1:
            try:
                label = "[" + src.split("[")[1].split("]")[0] + "]"
            except Exception as e:
                strategy_labels.append(8)
            else:
                strategy_labels.append(strategy_map[label])  # Strategy_map tokenizer.encode([label])[0] - 54944
        else:
            strategy_labels.append(8)

        inputs.append(context_id)
        kws.append(kws_id)
        roles.append(src_role)
        turns.append(src_turn)

    if len(utt_kws[-1]) == 0:
        last_kws_id = [0]
    else:
        kws_seq = ' '.join(utt_kws[-1])
        last_kws_id = tokenizer.encode(kws_seq)
    kws.append(last_kws_id)
    max_kws_len = 5
    pad_kws = []
    for item in kws:
        if len(item) < max_kws_len:
            pad_kws.append(item + [-100] * (max_kws_len - len(item)))
        else:
            pad_kws.append(item[:max_kws_len])
    assert len(inputs) == len(pad_kws) - 1

    return inputs, pad_kws, roles, turns, strategy_labels, emotion


def construct_conv_ESD(args, idx, row, utt_kws, utt_emotion, comet_repre, tokenizer, evaluate=False, strategy=True):
    #  process input text
    inputs, kws_enc, roles, turns, strategy_labels, _ = _get_inputs_from_text("EOS".join(row.split("EOS")[:-1]),
                                                                              utt_kws, tokenizer, strategy=strategy)
    # process output (decoder input) text
    d_inputs, _, d_roles, d_turns, d_strategy_labels, _ = _get_inputs_from_text(row.split("EOS")[-1], utt_kws,
                                                                                tokenizer, strategy=strategy)

    # make feature for input text
    feature = _make_feature(args, idx, inputs, kws_enc, utt_emotion, roles, turns,
                            tokenizer.encode(tokenizer.cls_token)[0], tokenizer.bos_token_id, tokenizer.eos_token_id,
                            strategy_labels=strategy_labels, evaluate=evaluate, decoder=False, comet_repre=comet_repre)
    # make feature for output (decoder input) text
    d_feature = _make_feature(args, idx, d_inputs, kws_enc, utt_emotion, d_roles, d_turns,
                              tokenizer.encode(tokenizer.cls_token)[0], tokenizer.bos_token_id, tokenizer.eos_token_id,
                              strategy_labels=d_strategy_labels, evaluate=evaluate, decoder=True,
                              comet_repre=comet_repre)

    # comet_st_ids, comet_st_mask = _get_comet_input(comet_st_row, tokenizer, max_num_attr=20)
    feature = InputFeatures_blender(feature, d_feature)
    return feature


def pad_matrix(matrix, padding_index=0):
    max_len = max(len(i) for i in matrix)
    batch_matrix = []
    for item in matrix:
        batch_matrix.append(np.pad(item, ((0, max_len - len(item)), (0, max_len - len(item))), 'constant',
                                   constant_values=(padding_index, padding_index)))
    return torch.tensor(batch_matrix).long()


class ESDDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, keywords, emotion, comet_repre, block_size=512,
                 evaluate=False, strategy=True, test=False):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        self.tokenizer = tokenizer
        directory = args.data_cache_dir
        if not os.path.exists(directory):
            os.makedirs(directory)

        if evaluate:
            if not test:
                cached_features_file = os.path.join(
                    directory, "val_cached_lm_" + str(block_size)
                )
            else:
                cached_features_file = os.path.join(
                    directory, "test_cached_lm_" + str(block_size)
                )
        else:
            cached_features_file = os.path.join(
                directory, "trn_cached_lm_" + str(block_size)
            )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.features = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            self.features = []
            for idx, (row, emo, kws, cmt_repre) in enumerate(zip(df[:-1], emotion, keywords, comet_repre)):
                conv = construct_conv_ESD(args, idx, row, kws['keywords'], emo['emotions'], cmt_repre, tokenizer,
                                          strategy=strategy, evaluate=evaluate)
                self.features.append(conv)

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Finished~")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    @staticmethod
    def collate(features):
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)

        trans_cls_index = pad_sequence([torch.tensor(f.trans_cls_index, dtype=torch.long) for f in features],
                                       batch_first=True, padding_value=1)

        trans_graph = pad_matrix([f.trans_graph for f in features])
        inter_graph = pad_matrix([f.inter_graph for f in features])
        trans_edge_type = pad_matrix([f.trans_edge_type for f in features])
        inter_edge_type = pad_matrix([f.inter_edge_type for f in features])

        keyword_ids = []
        for f in features:
            per_conv_idx = []
            for item in f.keywords:
                per_conv_idx.append(torch.tensor(item, dtype=torch.long))
            per_conv_idx = pad_sequence(per_conv_idx, batch_first=True, padding_value=-100)
            keyword_ids.append(per_conv_idx)
        keyword_ids = pad_sequence(keyword_ids, batch_first=True, padding_value=-100)

        trans_emotion_labels = pad_sequence([torch.tensor(f.trans_emotion_labels, dtype=torch.long)
                                             for f in features],
                                            batch_first=True, padding_value=8)

        decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long)
                                          for f in features],
                                         batch_first=True, padding_value=0)

        decoder_labels = pad_sequence([torch.tensor(f.decoder_lm_labels, dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=-100)

        decoder_strategy_ids = pad_sequence([torch.tensor(f.decoder_strategy_ids, dtype=torch.long)
                                             for f in features], batch_first=True, padding_value=8)

        strategy_labels = pad_sequence([torch.tensor(f.strategy_labels, dtype=torch.long) for f in features],
                                       batch_first=True, padding_value=8)

        emo_trans = pad_sequence([torch.tensor(f.emo_trans, dtype=torch.float) for f in features], batch_first=True,
                                 padding_value=0)

        return (input_ids, keyword_ids, trans_cls_index, trans_emotion_labels, decoder_input_ids, decoder_labels,
                decoder_strategy_ids, strategy_labels, emo_trans, trans_graph, inter_graph, trans_edge_type,
                inter_edge_type)


def load_and_cache_examples(args, tokenizer, df, kws, emo, comet_repre, evaluate=False, strategy=True, test=False):
    return ESDDataset(tokenizer, args, df, kws, emo, comet_repre, evaluate=evaluate, strategy=strategy, test=test)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


# Training of model
def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, collate_fn=ESDDataset.collate, shuffle=True, drop_last=False
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    other = ["trans_interact", "encoder_attn_emotion", "fusion"]
    no_main = no_decay + other

    params = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_main)], 'weight_decay': args.weight_decay,
         'lr': 2e-5},
        {'params': [p for n, p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 2e-5},
        {'params': [p for n, p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 5e-5},
        {'params': [p for n, p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': 5e-5},
    ]

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if False and (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        # model = BalancedDataParallel(2,model, dim=0).to(args.device)
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        ).to(args.device)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss, tr_lm_loss, logging_lm_loss, tr_emo_loss, \
        logging_emo_loss, tr_strategy_loss, logging_strategy_loss, = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    tr_bow_loss, logging_bow_loss = 0.0, 0.0
    best_acc = 0
    nb_tr_steps = 0.0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", leave=True, position=0)
        model.train()
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            input_ids, keyword_ids, trans_cls_index, trans_emotion_labels, decoder_input_ids, \
                decoder_labels, decoder_strategy_ids, strategy_labels, emo_trans, trans_graph, inter_graph, trans_edge_type, inter_edge_type = batch

            decoder_strategy_ids = decoder_strategy_ids[:, 0]
            decoder_strategy_ids = decoder_strategy_ids.to(args.device)

            if input_ids.shape[1] > 512: continue

            input_ids = input_ids.to(args.device)
            keyword_ids = keyword_ids.to(args.device)
            trans_cls_index = trans_cls_index.to(args.device)
            decoder_input_ids = decoder_input_ids.to(args.device)
            decoder_label_ids = decoder_labels.to(args.device)
            strategy_labels = strategy_labels.to(args.device)
            trans_emotion_labels = trans_emotion_labels.to(args.device)
            emo_trans = emo_trans.to(args.device)
            trans_graph = trans_graph.to(args.device)
            inter_graph = inter_graph.to(args.device)
            trans_edge_type = trans_edge_type.to(args.device)
            inter_edge_type = inter_edge_type.to(args.device)

            # we did't use role label and turn number in modeling as they did't carry significant improvement. Codes still remain.

            outputs = model(
                input_ids,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
                decoder_input_ids=decoder_input_ids,
                trans_cls_index=trans_cls_index,
                emo_trans=emo_trans,
                labels=decoder_label_ids,
                trans_emotion_labels=trans_emotion_labels,
                strategy_labels=strategy_labels,
                decoder_emotion_mask=trans_cls_index.ne(1),
                trans_graph=trans_graph,
                inter_graph=inter_graph,
                trans_edge_type=trans_edge_type,
                inter_edge_type=inter_edge_type,
                keyword_ids=keyword_ids
            )

            loss = outputs.loss
            lm_loss = ppl = outputs.lm_loss
            strategy_loss = outputs.strategy_loss
            emotion_loss = outputs.emotion_loss
            bow_loss = outputs.bow_loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                backward_loss = loss
                backward_loss.backward()

            tr_loss += loss.item()
            tr_lm_loss += lm_loss.item()
            if strategy_loss is not None:
                tr_strategy_loss += strategy_loss.item()

            if emotion_loss is not None:
                tr_emo_loss += emotion_loss.item()

            if bow_loss is not None:
                tr_bow_loss += bow_loss.item()

            nb_tr_steps += 1
            mean_loss = tr_loss / nb_tr_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 \
                        and global_step % args.logging_steps == 0 and global_step > t_total * 0.0:
                    model.eval()
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, args.eval_dataset,
                                           "{}-{}".format("checkpoint", global_step))
                        test_results = evaluate(args, model, tokenizer, args.test_dataset, "of test set")
                        # print('Test Results:', test_results)
                    model.train()
                    logger.info(
                        "lr: %f, step: %d, loss: %f, lm_loss: %f, strategy_loss: %f, bow_loss: %f, emotion_loss: %f",
                        scheduler.get_last_lr()[0],
                        global_step, (tr_loss - logging_loss) / args.logging_steps,
                                     (tr_lm_loss - logging_lm_loss) / args.logging_steps,
                                     (tr_strategy_loss - logging_strategy_loss) / args.logging_steps,
                                     (tr_bow_loss - logging_bow_loss) / args.logging_steps,
                                     (tr_emo_loss - logging_emo_loss) / args.logging_steps)

                    logging_loss = tr_loss
                    logging_lm_loss = tr_lm_loss
                    logging_emo_loss = tr_emo_loss
                    logging_strategy_loss = tr_strategy_loss

                    logging_bow_loss = tr_bow_loss

                    if results['eval_strategy_predict_accuracy'] > best_acc:
                        best_acc = results['eval_strategy_predict_accuracy']

                        if args.save:
                            checkpoint_prefix = "checkpoint"

                            output_dir = args.output_dir
                            os.makedirs(output_dir, exist_ok=True)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            _rotate_checkpoints(args, checkpoint_prefix)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    print("Train finished~")
    return global_step, tr_loss / global_step


# Evaluation of some model
def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix="") -> Dict:
    import numpy as np
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=ESDDataset.collate,
        drop_last=False
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss, eval_emotion_loss, eval_strategy_loss = 0.0, 0.0, 0.0
    nb_eval_steps = 0

    num_samples = []
    emo_hits = []
    # strategy_hits_topk = [[] for _ in range(7)]
    strategy_hits, emotion_hits, encoder_strategy_hits = [], [], []

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):

        input_ids, keyword_ids, trans_cls_index, trans_emotion_labels, decoder_input_ids, \
            decoder_labels, decoder_strategy_ids, strategy_labels, emo_trans, trans_graph, inter_graph, trans_edge_type, inter_edge_type = batch

        decoder_strategy_ids = decoder_strategy_ids[:, 0]
        decoder_strategy_ids = decoder_strategy_ids.to(args.device)
        if input_ids.shape[1] > 512: continue

        input_ids = input_ids.to(args.device)
        keyword_ids = keyword_ids.to(args.device)
        trans_cls_index = trans_cls_index.to(args.device)
        decoder_input_ids = decoder_input_ids.to(args.device)
        decoder_label_ids = decoder_labels.to(args.device)
        strategy_labels = strategy_labels.to(args.device)
        trans_emotion_labels = trans_emotion_labels.to(args.device)
        emo_trans = emo_trans.to(args.device)
        trans_graph = trans_graph.to(args.device)
        inter_graph = inter_graph.to(args.device)
        trans_edge_type = trans_edge_type.to(args.device)
        inter_edge_type = inter_edge_type.to(args.device)

        with torch.no_grad():
            if not args.role:
                role_ids = None
            if not args.turn:
                turn_ids = None

            outputs = model(input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id),
                            decoder_input_ids=decoder_input_ids, trans_cls_index=trans_cls_index, emo_trans=emo_trans,
                            labels=decoder_label_ids, trans_emotion_labels=trans_emotion_labels,
                            strategy_labels=strategy_labels, decoder_emotion_mask=trans_cls_index.ne(1),
                            trans_graph=trans_graph, inter_graph=inter_graph, trans_edge_type=trans_edge_type,
                            inter_edge_type=inter_edge_type, keyword_ids=keyword_ids)

            strategy_logits = outputs.strategy_logits
            emotion_logits = outputs.emotion_logits

            encoder_label_num = torch.sum(trans_cls_index != 1, dim=-1) - 1
            if strategy_logits is not None:
                for batch, (label, num) in enumerate(zip(strategy_labels, encoder_label_num)):
                    j = len(label) - 1
                    while j >= 0:
                        if label[j] != 8:
                            break
                        j -= 1
                    assert strategy_labels[batch][j] != 8
                    if strategy_logits[batch][j].argmax() == strategy_labels[batch][j]:
                        strategy_hits.append(1)
                    else:
                        strategy_hits.append(0)

                    for k in range(num):
                        if strategy_labels[batch][k] != 8:
                            if strategy_logits[batch][k].argmax() == strategy_labels[batch][k]:
                                encoder_strategy_hits.append(1)
                            else:
                                encoder_strategy_hits.append(0)

            if emotion_logits is not None:
                for batch, emotion_logit in enumerate(emotion_logits):
                    for idx, logit in enumerate(emotion_logit):
                        if trans_emotion_labels[batch][idx] != 8:
                            if logit.argmax() == trans_emotion_labels[batch][idx]:
                                emotion_hits.append(1)
                            else:
                                emotion_hits.append(0)

            lm_loss = outputs.lm_loss
            strategy_loss = outputs.strategy_loss
            emotion_loss = outputs.emotion_loss
            num_samples.append((decoder_label_ids.cpu().numpy() != -100).astype(int).sum())
            eval_loss += lm_loss.sum().item() * (decoder_label_ids.cpu().numpy() != -100).astype(int).sum()
            if strategy_logits is not None:
                eval_strategy_loss += strategy_loss.sum().item() \
                                      * (strategy_labels.cpu().numpy() != 8).astype(int).sum()
            if emotion_logits is not None:
                eval_emotion_loss += emotion_loss.sum().item() \
                                     * (trans_emotion_labels.cpu().numpy() != 8).astype(int).sum()

        nb_eval_steps += 1

    eval_loss = eval_loss / sum(num_samples)
    perplexity = torch.exp(torch.tensor(eval_loss)).item()
    eval_strategy_loss = eval_strategy_loss / sum(num_samples)
    if emotion_logits is not None:
        eval_emotion_loss = eval_emotion_loss / sum(num_samples)
    if emotion_logits is not None:
        result = {"eval_perplexity": perplexity, "eval_strategy_loss": eval_strategy_loss,
                  "eval_emotion_loss": eval_emotion_loss,
                  "eval_strategy_predict_accuracy": sum(strategy_hits) / len(strategy_hits),
                  "eval_emotion_predict_accuracy": sum(emotion_hits) / len(emotion_hits),
                  "eval_number_of_evaluated_examples": len(
                      strategy_hits)}  # "eval_encoder_strategy_predict_accuracy": sum(encoder_strategy_hits) / len(encoder_strategy_hits)
    elif strategy_logits is not None:
        result = {"eval_perplexity": perplexity, "eval_strategy_loss": eval_strategy_loss,
                  "eval_strategy_predict_accuracy": sum(strategy_hits) / len(strategy_hits),
                  "eval_number_of_evaluated_examples": len(strategy_hits)}
    else:
        result = {"eval_perplexity": perplexity}
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")

    with open(output_eval_file, "a+") as writer:
        # print("***** Eval results {} *****".format(prefix))
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("***** Eval results {} *****".format(prefix) + "\n")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            # print("  %s = %s" % (key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main(args):
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
            and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if not args.no_cuda:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        device = torch.device("cpu")
        args.device = device
        args.n_gpu = 0

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_or_path,
                                                                    cache_dir=args.model_cache_dir)

    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    comet_repre_trn = pickle.load(open(args.data_path + "/" + args.train_comet_repre, 'rb'), encoding='latin1')
    with open(args.data_path + "/" + args.train_file_name, "r", encoding="utf-8") as f:
        df_trn = f.read().split("\n")
    with open(args.data_path + "/" + args.train_emotion_name, "r", encoding="utf-8") as f:
        emo_trn = json.load(f)
    with open(args.data_path + "/" + args.train_keywords, "r", encoding="utf-8") as f:
        kws_trn = json.load(f)

    comet_repre_val = pickle.load(open(args.data_path + "/" + args.eval_comet_repre, 'rb'), encoding='latin1')
    with open(args.data_path + "/" + args.eval_file_name, "r", encoding="utf-8") as f:
        df_val = f.read().split("\n")
    with open(args.data_path + "/" + args.eval_emotion_name, "r", encoding="utf-8") as f:
        emo_val = json.load(f)
    with open(args.data_path + "/" + args.eval_keywords, "r", encoding="utf-8") as f:
        kws_val = json.load(f)

    comet_repre_test = pickle.load(open(args.data_path + "/" + args.test_comet_repre, 'rb'), encoding='latin1')
    with open(args.data_path + "/" + args.test_file_name, "r", encoding="utf-8") as f:
        df_test = f.read().split("\n")
    with open(args.data_path + "/" + args.test_emotion_name, "r", encoding="utf-8") as f:
        emo_test = json.load(f)
    with open(args.data_path + "/" + args.test_keywords, "r", encoding="utf-8") as f:
        kws_test = json.load(f)

    args.eval_dataset = load_and_cache_examples(args, tokenizer, df_val, kws_val, emo_val, comet_repre_val,
                                                evaluate=True, strategy=args.strategy, test=False)
    args.test_dataset = load_and_cache_examples(args, tokenizer, df_test, kws_test, emo_test, comet_repre_test,
                                                evaluate=True, strategy=args.strategy, test=True)

    # Training
    if args.do_train:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)
        args.train_dataset = load_and_cache_examples(args, tokenizer, df_trn, kws_trn, emo_trn, comet_repre_trn,
                                                     evaluate=False, strategy=args.strategy)
        global_step, tr_loss = train(args, args.train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.output_dir, from_tf=False)
        model.to(args.device)
        model.eval()


def generate(args):
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.output_dir, from_tf=False)
    model.resize_token_embeddings(len(tokenizer))

    # Setup CUDA, GPU & distributed training
    if not args.no_cuda:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        device = torch.device("cpu")
        args.device = device
        args.n_gpu = 0
    set_seed(args)

    with open(args.data_path + "/" + args.test_file_name, "r") as f:
        chat_texts = f.read().split("\n")

    comet_repre = pickle.load(open(args.data_path + "/" + args.test_comet_repre, 'rb'), encoding='latin1')

    with open(args.data_path + "/" + args.test_emotion_name, "r", encoding="utf-8") as f:
        emo_test = json.load(f)

    with open(args.data_path + "/" + args.test_keywords, "r", encoding="utf-8") as f:
        kws_test = json.load(f)

    ref_list = []
    hyp_list = []
    strategy_logit_str = []
    model.to(args.device)
    # Let's chat for 5 lines
    strategy_hits = []
    strategy_record = []
    strategy_hits_topk = [[] for _ in range(8)]

    model.eval()
    args.test_dataset = load_and_cache_examples(args, tokenizer, chat_texts, kws_test, emo_test, comet_repre,
                                                evaluate=True, strategy=args.strategy, test=True)

    test_results = evaluate(args, model, tokenizer, args.test_dataset, "of test set")
    print('Test Results:', test_results)

    for idx, (c_text, kws, emo, cmt_repre) in tqdm(
            enumerate(zip(chat_texts[:-1], kws_test, emo_test, comet_repre)), desc="Testing", total=len(emo_test)):

        chat_history = c_text
        f = construct_conv_ESD(args, idx, chat_history, kws['keywords'], emo['emotions'], cmt_repre, tokenizer,
                               strategy=False)

        next_strategy_id = f.decoder_strategy_ids[0]
        # decoder_strategy_ids = torch.tensor([f.decoder_strategy_ids], dtype=torch.long)
        # decoder_strategy_ids = decoder_strategy_ids.to(device)
        # decoder_strategy_ids = decoder_strategy_ids[:, 0]
        strategy_labels = torch.tensor([f.strategy_labels], dtype=torch.long).to(device)

        ref_list.append(tokenizer.decode(f.decoder_input_ids, skip_special_tokens=True))

        paras = {}
        input_ids = torch.tensor([f.input_ids], dtype=torch.long).to(args.device)

        paras["trans_cls_index"] = torch.tensor([f.trans_cls_index], dtype=torch.long).to(args.device)
        paras["emo_trans"] = torch.tensor([f.emo_trans], dtype=torch.float).to(args.device)
        paras["trans_graph"] = torch.tensor([f.trans_graph]).long().to(args.device)
        paras["inter_graph"] = torch.tensor([f.inter_graph]).long().to(args.device)
        paras["trans_edge_type"] = torch.tensor([f.trans_edge_type]).long().to(args.device)
        paras["inter_edge_type"] = torch.tensor([f.inter_edge_type]).long().to(args.device)
        paras["attention_mask"] = input_ids.ne(tokenizer.pad_token_id)

        chat_history_ids, strategy_logits = model.generate(
            input_ids,
            **paras,
            max_length=512,
            min_length=5,
            num_beams=1,
            pad_token_id=0,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.3,
            top_k=30,
            do_sample=True,
            repetition_penalty=1.03
        )  # top_p 0.9, topk 30
        chat_history_ids = chat_history_ids.cpu()

        hyp_list.append(tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True))

        id2strategy = {
            0: "[Question]", 1: "[Reflection of feelings]", 2: "[Information]",
            3: "[Restatement or Paraphrasing]", 4: "[Others]", 5: "[Self-disclosure]",
            6: "[Affirmation and Reassurance]", 7: "[Providing Suggestions]", 8: "[No Strategy]"
        }
        strategy_record.append({
            "ref strategy": id2strategy[next_strategy_id],
            "hyp strategy": id2strategy[strategy_logits[0][-1].argmax().item()]
        })

        for batch, label in enumerate(strategy_labels):  # strategy_labels
            j = len(label) - 1
            while j >= 0:
                if label[j] != 8:
                    break
                j -= 1
            assert next_strategy_id == strategy_labels[batch][j] != 8
            if strategy_logits[batch][j].argmax() == strategy_labels[batch][j]:
                strategy_hits.append(1)
            else:
                strategy_hits.append(0)

        decoder_strategy_logits = strategy_logits[:, -1, :]
        for k in range(8):
            _, topk = decoder_strategy_logits[0].topk(k + 1, -1)

            strategy_hits_topk[k].append(sum((topk == next_strategy_id).cpu().numpy().tolist()))
        decoder_strategy_logits = decoder_strategy_logits[0].cpu().numpy().tolist()
        decoder_strategy_logits = ["%.4f" % logit for logit in decoder_strategy_logits]
        strategy_logit_str.append('\t'.join(decoder_strategy_logits))

    for i in range(8):
        print(sum(strategy_hits_topk[i]) / len(strategy_hits_topk[i]))

    if not os.path.exists(args.generation_dir):
        os.makedirs(args.generation_dir)
    test_file_path = "dataset/testWithStrategy_short.tsv"
    test_situation_file_path = "dataset/testSituation.txt"
    strategy_record_file_path = os.path.join(args.generation_dir, "strategy_record.json")
    generate_file_path = os.path.join(args.generation_dir, "hyp_strategy.json")
    reference_file_path = os.path.join(args.generation_dir, "ref_strategy.json")
    summary_file_path = os.path.join(args.generation_dir, "summary.txt")
    strategy_logits_file = os.path.join(args.generation_dir, "strategy_logits.txt")
    with open(strategy_logits_file, "w", encoding="utf-8") as f:
        for item in strategy_logit_str:
            f.write(item + '\n')

    with open(strategy_record_file_path, "w", encoding="utf-8") as f:
        json.dump(strategy_record, f, indent=2, ensure_ascii=False)
    with open(generate_file_path, "w", encoding="utf-8") as f:
        json.dump(hyp_list, f, indent=2, ensure_ascii=False)
    with open(reference_file_path, "w", encoding="utf-8") as f:
        json.dump(ref_list, f, indent=2, ensure_ascii=False)
    summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, chat_texts,
            test_situation_file_path)

    print("write result to:", summary_file_path)
    print("Generate finished~")
    metric = Metric(toker=tokenizer, hyp_path=generate_file_path, ref_path=reference_file_path)
    result, result_list = metric.close()
    print(result)
    with open(os.path.join(args.generation_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.generation_dir, "result_list.json"), "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)

    from metric import NLGEval

    metric = NLGEval()
    ref_list = list(map(lambda ref: ' '.join(nltk.word_tokenize(ref.lower())), ref_list))
    hyp_list = list(map(lambda hyp: ' '.join(nltk.word_tokenize(hyp.lower())), hyp_list))
    metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list)
    with open(os.path.join(args.generation_dir, "metric_nlgeval_res.json"), "w", encoding="utf-8") as f:
        json.dump(metric_res, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.generation_dir, "metric_nlgeval_list.json"), "w", encoding="utf-8") as f:
        json.dump(metric_res_list, f, indent=2, ensure_ascii=False)

    print("=" * 100)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./blender_strategy', help="Path of output dir")
    parser.add_argument("--generation_dir", type=str, default='./generated_data', help="Path of output dir")
    parser.add_argument("--model_type", type=str, default='mymodel')
    parser.add_argument("--use_emotion", action='store_true', default=False)
    parser.add_argument("--use_bow", action='store_true', default=False)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/blenderbot_small-90M")
    parser.add_argument("--config_name", type=str, default="facebook/blenderbot_small-90M")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/blenderbot_small-90M")
    parser.add_argument("--data_path", type=str, default="./dataset")
    parser.add_argument("--train_file_name", type=str, default="trainWithStrategy_short.tsv")
    parser.add_argument("--eval_file_name", type=str, default="devWithStrategy_short.tsv")
    parser.add_argument("--test_file_name", type=str, default="testWithStrategy_short.tsv")
    parser.add_argument("--train_emotion_name", type=str, default="train_emotion.json")
    parser.add_argument("--eval_emotion_name", type=str, default="dev_emotion.json")
    parser.add_argument("--test_emotion_name", type=str, default="test_emotion.json")
    parser.add_argument("--train_keywords", type=str, default="train_keywords.json")
    parser.add_argument("--eval_keywords", type=str, default="dev_keywords.json")
    parser.add_argument("--test_keywords", type=str, default="test_keywords.json")
    parser.add_argument("--train_comet_repre", type=str, default="train_csk.pkl")
    parser.add_argument("--eval_comet_repre", type=str, default="dev_csk.pkl")
    parser.add_argument("--test_comet_repre", type=str, default="test_csk.pkl")

    parser.add_argument("--model_cache_dir", type=str, default="./blender-small")
    parser.add_argument("--data_cache_dir", type=str, default="./cached")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--generation", type=bool, default=False)
    parser.add_argument("--generate_and_eval", type=bool, default=False)
    parser.add_argument("--save", action='store_true', default=True)
    parser.add_argument("--evaluate_during_training", type=bool, default=True)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=20)
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=120)  # 100
    parser.add_argument("--logging_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=30)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--eval_all_checkpoints", type=bool, default=False)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--overwrite_output_dir", type=bool, default=True)
    parser.add_argument("--overwrite_cache", type=bool, default=False)
    parser.add_argument("--should_continue", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--fp16_opt_level", type=str, default='O1')
    parser.add_argument("--strategy", type=bool, default=False)
    parser.add_argument("--turn", type=bool, default=False)
    parser.add_argument("--role", type=bool, default=False)
    parser.add_argument("--wo_sem", type=bool, default=False)
    parser.add_argument("--wo_str", type=bool, default=False)
    parser.add_argument("--wo_emo", type=bool, default=False)
    parser.add_argument("--test", action='store_true', default=False)

    args = parser.parse_args()

    if args.test:
        generate(args)
    else:
        main(args)
