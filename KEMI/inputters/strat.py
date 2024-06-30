# coding=utf-8

import json
import tqdm
import torch
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
import random
from functools import partial
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputters.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader
from .PARAMS import GOLDEN_TRUTH


class Inputter(object):
    def __init__(self):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features

        # train
        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader

        # valid
        self.valid_dataloader = DynamicBatchingLoader

        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


# basic utils
class InputFeatures(object):
    def __init__(
            self,
            input_ids,
            decoder_input_ids, labels,
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels

        self.input_len = self.input_length + self.decoder_input_length


def featurize(
        bos, eos,
        context, knowledge, max_input_length,
        response, max_decoder_input_length, strat_id,
):
    context = [c + [eos] for c in context]
    context += [knowledge + [eos]]
    input_ids = sum(context, [])[:-1]
    input_ids = input_ids[-max_input_length:]

    labels = ([strat_id] + response + [eos])[:max_decoder_input_length + 1]
    decoder_input_ids = [bos] + labels[:-1]

    assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]

    return InputFeatures(
        input_ids,
        decoder_input_ids, labels,
    )


def convert_data_to_inputs(data, data_name, knowledge_name, toker: PreTrainedTokenizer, **kwargs):
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))

    if data_name == 'esconv':
        dialog = data['dialog']
        inputs = []
        context = []
        knowledge = []

        for i in range(len(dialog)):
            text = _norm(dialog[i]['text'])
            text = process(text)

            if dialog[i]['speaker'] == 'sys':
                strat_id = process('[' + dialog[i]['strategy'] + ']')
                assert len(strat_id) == 1
                strat_id = strat_id[0]

                if knowledge_name == 'oracle':
                    heal = process('[knowledge]') + process(dialog[i]['heal'])
                elif knowledge_name in ['sbert', 'graph']:
                    heal = process(dialog[i]['heal'])
                else:
                    heal = []

            else:
                if knowledge_name in ['basic', 'oracle', 'sbert', 'graph']:
                    knowledge = process(dialog[i]['knowledge'])
                elif knowledge_name == 'bm25':
                    knowledge = process('[knowledge]') + process(dialog[i]['knowledge'])
                else:
                    knowledge = []

            if i > 0 and dialog[i]['speaker'] == 'sys':
                res = {
                    'context': context.copy(),
                    'knowledge': knowledge + heal,
                    'response': text,
                    'strat_id': strat_id,
                }

                inputs.append(res)

            # if dialog[i]['speaker'] == 'sys':
            #    text = [strat_id] + text

            context = context + [text]
    elif data_name == 'mi':
        strat_id = process('[' + data['strategy'] + ']')
        assert len(strat_id) == 1
        strat_id = strat_id[0]
        if knowledge_name == 'basic':
            knowledge = process(data['knowledge'])
        elif knowledge_name == 'bm25':
            knowledge = process('[knowledge]') + process(data['knowledge'])
        elif knowledge_name in ['sbert', 'graph']:
            knowledge = process(data['knowledge']) + process(data['heal'])
        else:
            knowledge = []
        inputs = [{
            'context': [process(text) for text in data['dialog']],
            'knowledge': knowledge,
            'response': process(data['target']),
            'strat_id': strat_id
        }]
    else:
        raise ValueError('Invalid data name.')

    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    if len(inputs) == 0:
        return []

    assert kwargs.get('max_input_length', None) is not None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')
    assert kwargs.get('max_decoder_input_length', None) is not None, 'you should give max_decoder_input_length'
    max_decoder_input_length = kwargs.get('max_decoder_input_length')

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

    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            bos, eos,
            ipt['context'], ipt['knowledge'], max_input_length,
            ipt['response'], max_decoder_input_length, ipt['strat_id'],
        )
        features.append(feat)
    return features


# for training
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer, data_name: str, knowledge_name: str,
                infer=False):
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

        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features],
                                 batch_first=True, padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
                                      batch_first=True, padding_value=0.)
        input_length = torch.tensor([f.input_length for f in features], dtype=torch.long)

        if not infer:
            decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                                             batch_first=True, padding_value=pad)
            labels = pad_sequence([torch.tensor(f.labels, dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=-100)
        else:
            decoder_input_ids = torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long)
            labels = None

        if data_name == 'esconv':
            strat_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(toker) + 8
        elif data_name == 'mi':
            strat_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(toker) + 10

        if knowledge_name == 'basic':
            strat_id += 5
        elif knowledge_name == 'bm25':
            strat_id += 1
        elif knowledge_name == 'oracle':
            strat_id += 6
        elif knowledge_name in ['sbert', 'graph']:
            strat_id += 8

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'input_length': input_length,

            'decoder_input_ids': decoder_input_ids,
            'labels': labels,

            'strat_id': strat_id,
        }

        return res


# for validation
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """

    def __init__(self, corpus_file, toker, batch_size, data_name, knowledge_name, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.data_name = data_name
        self.knowledge_name = knowledge_name
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()

            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating", position=0, leave=True):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.data_name, self.knowledge_name, self.toker, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []

            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch

        except StopIteration:
            pass

    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker, self.data_name, self.knowledge_name)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        if self.data_name == 'esconv':
            return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))
        elif self.data_name == 'mi':
            return len(reader)


# for inference
def prepare_infer_batch(features, toker, data_name, knowledge_name, interact=None):
    res = FeatureDataset.collate(features, toker, data_name, knowledge_name, True)

    res['batch_size'] = res['input_ids'].size(0)

    other_res = res['other_res'] = {}
    other_res['acc_map'] = {
        'cls_strat_id': 'pred_strat_id',
    }

    if interact is None and GOLDEN_TRUTH:
        other_res['cls_strat_id'] = res.get('strat_id')
    else:
        other_res['cls_strat_id'] = res.pop('strat_id')

    return res


def get_infer_batch(infer_input_file, toker, data_name, knowledge_name, **kwargs):
    assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
    infer_batch_size = kwargs.get('infer_batch_size')

    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()

    features = []
    sample_ids = []
    posts = []
    references = []
    for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc=f"inferring", position=0, leave=True):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, data_name, knowledge_name, toker, **kwargs)
        tmp_features = convert_inputs_to_features(inputs, toker, **kwargs)
        for i in range(len(inputs)):
            features.append(tmp_features[i])
            ipt = inputs[i]
            posts.append(toker.decode(ipt['context'][-1]))
            references.append(toker.decode(ipt['response']))
            sample_ids.append(sample_id)

            if len(sample_ids) == infer_batch_size:
                yield prepare_infer_batch(features, toker, data_name, knowledge_name), posts, references, sample_ids
                features = []
                sample_ids = []
                posts = []
                references = []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker, data_name, knowledge_name), posts, references, sample_ids
