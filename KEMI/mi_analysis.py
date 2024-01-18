import os
import json
import nltk
import sys

sys.path.append("/")
from metric.metric_utils import mi_metric


def test_mi():
    infer_input_file = "./_reformat/esconv/none/test.txt"
    save_dir = "./DATA/vanilla.vanilla/2022-05-25142211.3e-05.16.1gpu/res_epoch-1.bin_test_k.0_p.0.9_b.1_t.0.7_lp.1.0_rp.1.0_ng.3"

    mi_res = mi_metric(infer_input_file, os.path.join(save_dir, f'gen.json'), os.path.join(save_dir, f'mi_gen.json'),
                       False, True)


def find_case():
    save_dir = "./DATA/strat.strat.esconv.sbert/2022-07-01105239.3e-05.16.1gpu/res_epoch-2.bin_test_k.30_p.0.3_b.1_t.0.7_lp.1.0_rp.1.03_ng.0/mi_gen.json"

    highest_bleu = 0
    with open(save_dir, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        for d in data:
            dial = d['dialog']
            for utt in dial:
                if utt['speaker'] != 'sys':
                    continue
                if utt['strategy'] != 'Providing Suggestions':
                    continue
                reference = nltk.word_tokenize(utt['text'])
                references = [reference]
                try:
                    hypothesis = nltk.word_tokenize(utt['generated_respons'])
                except Exception as e:
                    continue
                bleu = nltk.translate.bleu_score.sentence_bleu(references, hypothesis)
                if bleu > highest_bleu:
                    highest_bleu = bleu
                    case = utt['text']
                    print(bleu, references, hypothesis)
    print(case)


find_case()
