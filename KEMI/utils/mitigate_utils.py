# coding=utf-8

import os
import sys
import json
import pickle
import numpy as np
from typing import List
from functools import partial
from tqdm import tqdm

import nltk
import torch
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils import PreTrainedTokenizer

from inputters.inputter_utils import BucketSampler, _norm
from utils.eval_utils import eval_model_loss

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from metric.myMetrics import Metric
from metric import NLGEval


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


def test_model(model, toker, args, loss_loader, infer_dataloader):
    model.eval()
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
        'max_length': 40,
        'min_length': 15,
        'do_sample': True,
        'temperature': 0.7,
        'top_k': 30,
        'top_p': 0.3,
        'num_beams': 1,
        'num_return_sequences': 1,
        'length_penalty': 1.0,
        'repetition_penalty': 1.0,
        'no_repeat_ngram_size': 3,
        'encoder_no_repeat_ngram_size': 3 if model.config.is_encoder_decoder else None,
        'pad_token_id': pad,
        'bos_token_id': bos,
        'eos_token_id': eos,
    }
    print(json.dumps(generation_kwargs, indent=2, ensure_ascii=False))

    metric_results = {}
    infer_loss, _, infer_samples, pointwise_loss, pointwise_sample = eval_model_loss(
        model=model,
        eval_dataloader=loss_loader,
        infer=True,
        args=args,
    )
    assert len(pointwise_loss) == len(pointwise_sample)
    metric_results["perplexity"] = float(np.exp(infer_loss))
    ptr = 0
    metric = Metric(toker)

    results = []
    other_res = {}
    decode = lambda x: _norm(toker.decode(x))

    with torch.no_grad():
        for batch, contexts, references, sample_ids in infer_dataloader:
            batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            batch.update(generation_kwargs)
            encoded_info, generations = model.generate(
                data_name=args.data_name, knowledge_name=args.knowledge_name, **batch
            )

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
                c = contexts[idx]
                r = references[idx]
                g = generations[idx]
                ref, gen = [r], toker.decode(g) if not isinstance(g[0], list) else toker.decode(g[0])
                metric.forword(ref, gen, chinese=args.chinese)
                g = decode(g)
                tmp_res_to_append = {"sample_id": sample_ids[idx], "context": c, "response": r, "generation": g}

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

                ptr_loss = pointwise_loss[ptr]
                ptr_sample = pointwise_sample[ptr]
                turn_loss = ptr_loss / ptr_sample
                turn_ppl = np.exp(turn_loss)
                tmp_res_to_append["token_num"] = ptr_sample
                tmp_res_to_append["loss"] = turn_loss
                tmp_res_to_append["ppl"] = turn_ppl
                ptr += 1
                results.append(tmp_res_to_append)

    assert ptr == len(pointwise_loss)

    save_dir = os.path.join(args.checkpoint_dir, f"inference_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "test_generations.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, sort_keys=False)

    with open(os.path.join(save_dir, "test_generations.txt"), "w") as f:
        for line in results:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    metric_result_list = {}
    closed_result = metric.close()
    metric_results.update(closed_result[0])
    metric_result_list.update(closed_result[1])

    if 'acc' in other_res:
        for k, v in other_res['acc_map'].items():
            if k not in other_res['acc'] or v not in other_res['acc']:  # TODO
                continue  # TODO
            kk = np.array(other_res['acc'][k], dtype=int)
            vv = np.array(other_res['acc'][v], dtype=int)
            print(f'{k}: classification_report\n', classification_report(kk, vv))
            with open(os.path.join(save_dir, f'confusion_matrix_{k}.json'), 'w') as f:
                json.dump(confusion_matrix(kk, vv).tolist(), f)
                print(f'{k}: confusion_matrix\n', confusion_matrix(kk, vv))

            metric_results[f'acc_{k}'] = np.mean(kk == vv)
            metric_results[f'f1_micro_{k}'] = f1_score(kk, vv, average='micro')
            metric_results[f'f1_macro_{k}'] = f1_score(kk, vv, average='macro')
            metric_results[f'p_micro_{k}'] = precision_score(kk, vv, average='micro')
            metric_results[f'p_macro_{k}'] = precision_score(kk, vv, average='macro')
            metric_results[f'r_micro_{k}'] = recall_score(kk, vv, average='micro')
            metric_results[f'r_macro_{k}'] = recall_score(kk, vv, average='macro')
            if metric_result_list is None:
                metric_result_list = {}
            metric_result_list[f'acc_{k}'] = (kk == vv).astype(int).tolist()

            if f'{v}_top1' in other_res['acc']:
                vv_top1 = np.array(other_res['acc'][f'{v}_top1'], dtype=int)
                metric_results[f'acc_{k}_top1'] = np.mean(np.sum((kk.reshape(-1, 1) - vv_top1) == 0, axis=-1) != 0)
                metric_result_list[f'acc_{k}_top1'] = (
                        np.sum((kk.reshape(-1, 1) - vv_top1) == 0, axis=-1) != 0).astype(int).tolist()
            if f'{v}_top3' in other_res['acc']:
                vv_top3 = np.array(other_res['acc'][f'{v}_top3'], dtype=int)
                metric_results[f'acc_{k}_top3'] = np.mean(np.sum((kk.reshape(-1, 1) - vv_top3) == 0, axis=-1) != 0)
                metric_result_list[f'acc_{k}_top3'] = (
                        np.sum((kk.reshape(-1, 1) - vv_top3) == 0, axis=-1) != 0).astype(int).tolist()

    with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metric_results, f, indent=2, ensure_ascii=False, sort_keys=False)

    if metric_result_list != None:
        with open(os.path.join(save_dir, "test_metrics_list.json"), "w", encoding="utf-8") as f:
            json.dump(metric_result_list, f, indent=2, ensure_ascii=False, sort_keys=False)

    ref_list = []
    hyp_list = []
    for line in results:
        if isinstance(line['response'], list):
            ref = line['response'][0]
        else:
            ref = line['response']
        ref = ' '.join(nltk.word_tokenize(ref.lower()))

        if isinstance(line['generation'], list):
            hyp = line['generation'][0]
        else:
            hyp = line['generation']
        hyp = ' '.join(nltk.word_tokenize(hyp.lower()))

        ref_list.append(ref)
        hyp_list.append(hyp)

    metric = NLGEval()
    metric_results, metric_result_list = metric.compute_metrics([ref_list], hyp_list)
    with open(os.path.join(save_dir, f'metric_nlgeval.json'), 'w') as f:
        json.dump(metric_results, f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(save_dir, f'metric_nlgeval_list.json'), 'w') as f:
        json.dump(metric_result_list, f, ensure_ascii=False)


def eval_model(model, toker, args, loss_loader, infer_dataloader):
    model.eval()

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

    # generation_kwargs = {
    #     'max_length': 40,
    #     'min_length': 15,
    #     'do_sample': True,
    #     'temperature': 0.7,
    #     'top_k': 30,
    #     'top_p': 0.3,
    #     'num_beams': 1,
    #     'num_return_sequences': 1,
    #     'length_penalty': 1.0,
    #     'repetition_penalty': 1.0,
    #     'no_repeat_ngram_size': 3,
    #     'encoder_no_repeat_ngram_size': 3,
    #     'pad_token_id': pad,
    #     'bos_token_id': bos,
    #     'eos_token_id': eos,
    # }
    infer_loss, infer_ppl, infer_samples, pointwise_loss, pointwise_sample = eval_model_loss(
        model=model,
        eval_dataloader=loss_loader,
        infer=True,
        args=args,
    )
    assert len(pointwise_loss) == len(pointwise_sample)
    metric_results = {"perplexity": float(np.exp(infer_loss))}
    # ptr = 0
    # metric = Metric(toker)

    results = []
    # decode = lambda x: _norm(toker.decode(x))
    #
    # with torch.no_grad():
    #     for batch, contexts, references, sample_ids in infer_dataloader:
    #         batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
    #         batch.update(generation_kwargs)
    #         encoded_info, generations = model.generate(
    #             data_name=args.data_name, knowledge_name=args.knowledge_name, **batch
    #         )
    #
    #         generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]
    #
    #         for idx in range(len(sample_ids)):
    #             c = contexts[idx]
    #             r = references[idx]
    #             g = generations[idx]
    #             ref, gen = [r], toker.decode(g) if not isinstance(g[0], list) else toker.decode(g[0])
    #             metric.forword(ref, gen, chinese=args.chinese)
    #             g = decode(g)
    #             tmp_res_to_append = {"sample_id": sample_ids[idx], "context": c, "response": r, "generation": g}
    #
    #             ptr_loss = pointwise_loss[ptr]
    #             ptr_sample = pointwise_sample[ptr]
    #             turn_loss = ptr_loss / ptr_sample
    #             turn_ppl = np.exp(turn_loss)
    #             tmp_res_to_append["token_num"] = ptr_sample
    #             tmp_res_to_append["loss"] = turn_loss
    #             tmp_res_to_append["ppl"] = turn_ppl
    #             ptr += 1
    #             results.append(tmp_res_to_append)
    #
    # assert ptr == len(pointwise_loss)

    metric_result_list = {}
    # closed_result = metric.close()
    # metric_results.update(closed_result[0])
    # metric_result_list.update(closed_result[1])

    return infer_loss, infer_ppl, metric_results, metric_result_list, results


class MuffinFeature(object):
    def __init__(self, feature, candidates, candidate_labels, max_length, inputter_is_strat=False):
        """

        :param feature: features of the input data, including input_ids, input_length
        :param candidates: additional samples
        :param labels: labels to indicate whether the candidates are helpful
        :param max_length:
        :param inputter_is_strat:
        """
        self.input_ids = feature.input_ids
        self.input_length = feature.input_length

        if inputter_is_strat:
            strat = torch.tensor([feature.decoder_input_ids[:2]] * candidates.size(0), dtype=candidates.dtype)
            self.candidates = torch.cat((strat, candidates[:, 1:]), dim=-1)[:, :max_length]
        else:
            self.candidates = candidates[:, :max_length]

        self.candidate_labels = candidate_labels
        self.decoder_input_ids = feature.decoder_input_ids
        self.decoder_input_length = feature.decoder_input_length
        self.labels = feature.labels
        self.input_len = feature.input_len


class MitigationDataLoader(object):
    def __init__(
            self, toker, data_dir, max_length=40, bucket=100, is_sorted=True, batch_size=1, shuffle=True, **kwargs
    ):
        """ :returns format: context, golden_response, [[candidiate_i, score_i], ...] """
        self.max_length = max_length
        self.batch_size = batch_size
        self.toker = toker
        self.sorted = is_sorted
        self.bucket_size = bucket * self.batch_size
        self.shuffle = shuffle

        # load generated responses and corresponding scores
        input_generation_file = os.path.join(data_dir, "candidates.txt")
        with open(input_generation_file, "r", encoding='utf-8') as file:
            candidates = []
            for line in file.readlines():
                line = json.loads(line)
                candidates.append([line["response"]] + line["generation"])
        input_label_file = os.path.join(data_dir, "candidate_feedback.npy")
        with open(input_label_file, "rb") as file:
            label_list = np.load(file, allow_pickle=True)

        # load processed data
        # can be found in corresponding inputter_utils.py
        assert 'inputter_name' in kwargs
        assert 'config_name' in kwargs
        assert 'data_name' in kwargs
        assert 'knowledge_name' in kwargs
        inputter_name = kwargs.pop('inputter_name')
        config_name = kwargs.pop('config_name')
        data_name = kwargs.pop('data_name')
        knowledge_name = kwargs.pop('knowledge_name')
        self.data_name = data_name
        self.knowledge_name = knowledge_name
        with open(f"DATA/{inputter_name}.{config_name}.{data_name}.{knowledge_name}/data.pkl", "rb") as f:
            data = pickle.load(f)
        self.trunc_chunk, self.lens = self.process_data(data, candidates, label_list, inputter_name == "strat")

    def process_data(self, data, candidates, label_list, inputter_is_strat=False):
        trunc_chunk = []
        lens = []
        for feat, cand, labels in \
                tqdm(zip(data, candidates, label_list), total=len(data), desc="Preparing DataLoader"):
            cand = self.toker.batch_encode_plus(cand, return_tensors="pt", pad_to_max_length=False, padding=True)
            cand = cand["input_ids"]
            bos_tensor = torch.ones(cand.size(0), 1, dtype=cand.dtype) * self.toker.bos_token_id
            new_cand = torch.cat((bos_tensor, cand), -1)
            feat = MuffinFeature(feat, new_cand, labels, self.max_length, inputter_is_strat)
            trunc_chunk.append(feat)
            lens.append(feat.input_len)

        return trunc_chunk, lens

    def __len__(self):
        return len(self.trunc_chunk)

    def __iter__(self):
        dataset = MitigationDataset(self.trunc_chunk)
        sampler = BucketSampler(self.lens, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=4,  # can test multi-worker
            collate_fn=partial(
                dataset.collate,
                toker=self.toker,
                data_name=self.data_name,
                knowledge_name=self.knowledge_name
            )
        )
        yield from loader


class MitigationDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[MuffinFeature], toker: PreTrainedTokenizer, data_name: str, knowledge_name: str):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'

        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features], batch_first=True,
                                 padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
                                      batch_first=True, padding_value=0.)
        candidate_labels = pad_sequence([torch.tensor(f.candidate_labels, dtype=torch.long) for f in features],
                                        batch_first=True, padding_value=-1)
        cand_max_len = max([f.candidates.size(1) for f in features])
        # candidate_ids = [pad_to_max(f.candidates, pad, cand_max_len) for f in features]
        if len(features) > 1:
            cand_max_num = max([f.candidates.size(0) for f in features])
            cand_total_max = cand_max_len * cand_max_num
            candidate_ids = [pad_to_max(f.candidates, pad, cand_max_len).reshape(1, -1) for f in features]
            candidate_ids = [pad_to_max(cand, pad, cand_total_max).reshape(cand_max_num, -1) for cand in candidate_ids]
        else:
            candidate_ids = [pad_to_max(f.candidates, pad, cand_max_len) for f in features]
        candidate_ids = torch.stack(candidate_ids)

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

        res = {"input_ids": input_ids, "attention_mask": attention_mask, "candidate_ids": candidate_ids,
               "candidate_labels": candidate_labels, "strat_id": strat_id}

        return res


def pad_to_max(X, pad_token_id, max_len=-1):
    if max_len < 0:
        max_len = max(x.size(0) for x in X)
    seq_num, seq_len = X.size()
    if seq_len == max_len:
        return X
    else:
        pad_tensor = torch.ones(seq_num, (max_len - seq_len), dtype=X[0].dtype) * pad_token_id
        result = torch.cat((X, pad_tensor), -1)
        return result


def RankingLoss(score, label, margin=0.001):
    if score.sum() == 0:
        return 0
    LossFunc = torch.nn.MarginRankingLoss(0.0, reduction='sum')
    loss_mask = (score != 0).long()
    total_loss = LossFunc(score, score, loss_mask) / loss_mask.sum()

    n = score.size(1)
    LossFunc = torch.nn.MarginRankingLoss(margin, reduction='sum')
    for i in range(1, n):
        neg_score, neg_label = score[:, :-i], label[:, :-i]
        pos_score, pos_label = score[:, i:], label[:, i:]
        neg_score = neg_score.contiguous().view(-1)
        pos_score = pos_score.contiguous().view(-1)
        label_mask = ((pos_label != -1) & (neg_label != -1)).long()
        loss_mask = ((pos_label - neg_label) * label_mask).view(-1)
        if loss_mask.sum() == 0:
            continue
        loss = LossFunc(pos_score, neg_score, loss_mask) / loss_mask.abs().sum()
        total_loss += loss

    return total_loss
