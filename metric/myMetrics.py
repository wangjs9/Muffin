# coding=utf-8

import json
import warnings
import numpy as np
import nltk
from typing import List
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import bert_score


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for _ in range(0, len(sub) + 1)] for _ in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Metric(object):
    def __init__(self, toker, hyp_path=None, ref_path=None):
        self.refs = []
        self.hyps = []
        self.toker = toker
        if hyp_path != None and ref_path != None:
            with open(hyp_path, 'r', encoding='utf-8') as f:
                hyps = json.load(f)
            with open(ref_path, 'r', encoding='utf-8') as f:
                refs = json.load(f)
            assert len(hyps) == len(refs)

            for i in range(len(hyps)):
                self.forword([refs[i]], hyps[i])

    def forword(self, refs: str, hyp: str, chinese=False):  # TODO: only applicable to English
        if not chinese:
            self.refs.append([nltk.word_tokenize(e.lower()) for e in refs])
            self.hyps.append(nltk.word_tokenize(hyp.lower()))
        else:
            self.refs.append([self.toker.tokenize(e) for e in refs])
            self.hyps.append(self.toker.tokenize(hyp))

    def calc_bleu_k(self, k):
        weights = [1. / k] * k + (4 - k) * [0.]
        try:
            bleu = corpus_bleu(self.refs, self.hyps, weights=weights,
                               smoothing_function=SmoothingFunction().method3)
        except ZeroDivisionError as _:
            warnings.warn('the bleu is invalid')
            bleu = 0.
        return bleu

    def calc_distinct_k(self, k):
        d = {}
        tot = 0
        for sen in self.hyps:
            for i in range(0, len(sen) - k):
                key = tuple(sen[i:i + k])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            warnings.warn('the distinct is invalid')
            dist = 0.
        return dist

    def calc_unigram_f1(self):
        f1_scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            scores = []
            for ref in refs:
                cross = Counter(hyp) & Counter(ref)
                cross = sum(cross.values())
                p = cross / max(len(hyp), 1e-10)
                r = cross / max(len(ref), 1e-10)
                f1 = 2 * p * r / max(p + r, 1e-10)
                scores.append(f1)
            f1_scores.append(max(scores))
        return np.mean(f1_scores), f1_scores

    def calc_rouge_l(self, beta=1.2):
        scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            prec = []
            rec = []
            for ref in refs:
                lcs = my_lcs(ref, hyp)
                prec.append(lcs / max(len(hyp), 1e-10))
                rec.append(lcs / max(len(ref), 1e-10))
            prec_max = max(prec)
            rec_max = max(rec)
            if prec_max != 0 and rec_max != 0:
                score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
            else:
                score = 0.0
            scores.append(score)
        return np.mean(scores), scores

    def calc_bert(self):
        r_scores, p_scores, f1_scores = bert_score.score(
            [" ".join(x) for x in self.hyps],
            [" ".join(x[0]) for x in self.refs],
            lang='en',
            verbose=False
        )
        r_scores, p_scores, f1_scores = r_scores.tolist(), p_scores.tolist(), f1_scores.tolist()
        return (np.mean(r_scores), r_scores), (np.mean(p_scores), p_scores), (np.mean(f1_scores), f1_scores)

    def close(self):
        result = {
            'length': float(np.mean(list(map(len, self.hyps)))),
            **{f"dist-{k}": 100 * self.calc_distinct_k(k) for k in range(1, 4)},
            **{f"bleu-{k}": 100 * self.calc_bleu_k(k) for k in range(1, 5)}
        }

        f1, scores = self.calc_unigram_f1()
        result['f1'] = 100 * f1
        result_list = {
            'f1': scores
        }

        rl, scores = self.calc_rouge_l()
        result['rouge-l'] = 100 * rl
        result_list.update({
            'rouge-l': scores
        })

        # (bert_r, r_scores), (bert_p, p_scores), (bert_f1, f1_scores) = self.calc_bert()
        # result["bert-r"] = bert_r
        # result["bert-p"] = bert_p
        # result["bert-f1"] = bert_f1
        # result_list.update({
        #     "bert-r": r_scores,
        #     "bert-p": p_scores,
        #     "bert-f1": f1_scores,
        # })

        return result, result_list
