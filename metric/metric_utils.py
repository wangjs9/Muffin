import click
import json
import os


class InvalidDataDirException(Exception):
    pass


def get_data_dir():
    if os.environ.get('NLGEVAL_DATA'):
        if not os.path.exists(os.environ.get('NLGEVAL_DATA')):
            click.secho("NLGEVAL_DATA variable is set but points to non-existent path.", fg='red', err=True)
            raise InvalidDataDirException()
        return os.environ.get('NLGEVAL_DATA')
    else:
        try:
            from xdg import XDG_CONFIG_HOME
            cfg_file = os.path.join(XDG_CONFIG_HOME, 'nlgeval', 'rc.json')
            with open(cfg_file, 'rt') as f:
                rc = json.load(f)
                if not os.path.exists(rc['data_path']):
                    click.secho("Data path found in {} does not exist: {} " % (cfg_file, rc['data_path']), fg='red',
                                err=True)
                    click.secho(
                        "Run `nlg-eval --setup DATA_DIR' to download or set $NLGEVAL_DATA to an existing location",
                        fg='red', err=True)
                    raise InvalidDataDirException()
                return rc['data_path']
        except:
            click.secho("Could not determine location of data.", fg='red', err=True)
            click.secho("Run `nlg-eval --setup DATA_DIR' to download or set $NLGEVAL_DATA to an existing location",
                        fg='red',
                        err=True)
            raise InvalidDataDirException()


import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig, BlenderbotSmallTokenizer
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import argparse


class BERT_(nn.Module):
    def __init__(self, model_name_or_path, cache_dir, config):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path),
                                                    config=config, cache_dir=cache_dir)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1024, 2)
        self.config = config

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return F.softmax(logits, dim=-1)


class BERT(nn.Module):
    def __init__(self, model_name_or_path, cache_dir, config):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path),
                                                    config=config, cache_dir=cache_dir)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1024, 5)
        self.config = config

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 5), labels.view(-1))
            return loss
        else:
            return F.softmax(logits, dim=-1)


class InitModel(object):
    def __init__(self):
        model_name_or_path = 'roberta-large'
        cache_dir = None
        output_dir = '/home/jiashuo/codes/Muffin/metric/model/initiative/best_checkpoint'

        config = RobertaConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=cache_dir)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[goal]', '[user]', '[system]', '[table]', '[paragraph]', '[scale]']})

        model = BERT_(model_name_or_path, cache_dir, config)
        model.encoder.resize_token_embeddings(len(self.tokenizer))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model.to(self.device)

        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))

        self.model_to_eval = model.module if hasattr(model, 'module') else model

    def predict(self, user_text, system_text):
        user_id = self.tokenizer.encode('[user]' + user_text.encode('utf-8', 'replace').decode('utf-8'))[1:]
        system_id = self.tokenizer.encode('[system]' + system_text.encode('utf-8', 'replace').decode('utf-8'))[1:]
        input_ids = torch.tensor([user_id + system_id]).long()
        attention_mask = input_ids.ne(0)
        pred = self.model_to_eval(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )
        pred = pred.argmax(dim=-1).cpu().tolist()
        return pred[0]


class EmoModel(object):
    def __init__(self):
        model_name_or_path = 'roberta-large'
        cache_dir = None
        output_dir = '/home/jiashuo/codes/Muffin/metric/model/emotion/best_checkpoint'

        config = RobertaConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=cache_dir)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[goal]', '[user]', '[system]', '[table]', '[paragraph]', '[scale]']})

        model = BERT(model_name_or_path, cache_dir, config)
        model.encoder.resize_token_embeddings(len(self.tokenizer))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model.to(self.device)

        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))

        self.model_to_eval = model.module if hasattr(model, 'module') else model

    def predict(self, utterance):
        input_id = self.tokenizer.encode(utterance.encode('utf-8', 'replace').decode('utf-8'))
        input_ids = torch.tensor([input_id]).long()
        attention_mask = input_ids.ne(0)
        pred = self.model_to_eval(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )
        pred = pred[0].cpu().tolist()
        score = sum([(j + 1) * pred[j] for j in range(len(pred))])
        return score


class USi(object):
    def __init__(self):
        model_name_or_path = 'facebook/blenderbot_small-90M'
        cache_dir = None
        output_dir = '/home/jiashuo/codes/Muffin/metric/model/usi/best_checkpoint'

        config = BlenderbotSmallConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name_or_path, do_lower_case=True,
                                                                  cache_dir=cache_dir)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[user]', '[system]', '[situation]']})

        model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                        from_tf=bool('.ckpt' in model_name_or_path),
                                                                        config=config, cache_dir=cache_dir)
        model.resize_token_embeddings(len(self.tokenizer))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model.to(self.device)

        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))

        self.max_seq_length = 160
        self.model_to_eval = model.module if hasattr(model, 'module') else model

    def predict(self, context):
        process = lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
        context_id = process('[situation]') + process(context['situation'])
        for utt in context['dialog']:
            if utt['speaker'] == 'sys':
                context_id += process('[system]') + process(utt['text'])
            else:
                context_id += process('[user]') + process(utt['text'])
        input_ids = torch.tensor([context_id[-self.max_seq_length:]]).long()
        attention_mask = input_ids.ne(0)
        generated_ids = self.model_to_eval.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            num_beams=1,
            max_length=40,
            min_length=5,
            early_stopping=True,
            temperature=0.7,
            top_k=30,
            top_p=0.3,
            repetition_penalty=1.03,
        )
        pred = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return pred


def mi_metric(data_file, output_file, res_file, use_ucm=False, use_ref=False):
    if use_ucm:
        init_model = InitModel()
    USi_model = USi()
    emo_model = EmoModel()
    snowball = SnowballStemmer(language='english')

    def ext_freq_term(sentence):
        words = nltk.word_tokenize(sentence)
        new_words = [word.lower() for word in words if word.isalnum()]
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in new_words if not w in stop_words]
        tagged = nltk.pos_tag(filtered_words)
        final_words = [snowball.stem(w) for w, p in tagged if p.startswith('N') or p.startswith('V')]
        return final_words

    with open(output_file, 'r') as infile:
        output = json.load(infile)

    print(len(output))

    metrics = {'pro': [], 'inf': {'I': [], 'N': []}, 'rep': {'I': [], 'N': []}, 'rel': {'I': [], 'N': []}}

    idx = 0
    new_data = []
    with open(data_file, 'r', encoding="utf-8") as infile, \
            open(res_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)

            all_words = set()
            usr_words = set()
            dial = data['dialog']
            context = {'situation': data['situation'], 'dialog': []}
            for i in range(len(dial)):
                turn = dial[i]
                if turn['speaker'] == 'usr':
                    context['dialog'].append(turn)
                    user_text = turn['text']
                    pred_emo = emo_model.predict(user_text)
                    turn['emotion_intensity'] = pred_emo
                    words = set(ext_freq_term(user_text))
                    usr_words = usr_words | words
                    all_words = all_words | words
                    continue

                if i > 0:
                    res = output[idx]
                    if use_ref:
                        system_text = res['response']
                    else:
                        system_text = res['generation']
                    if not use_ucm and not use_ref:
                        turn['pred_strat_id'] = res['pred_strat_id']

                    ## user simulation
                    new_context = context
                    new_context['dialog'].append({'text': system_text, 'speaker': 'sys'})
                    simulated_feedback = USi_model.predict(new_context)
                    current_emo = emo_model.predict(simulated_feedback)
                    turn['usi_emotion_intensity'] = current_emo
                    turn['usi_feedback'] = simulated_feedback
                    turn['generated_respons'] = system_text

                    words = set(ext_freq_term(system_text))

                    inf = len(words - all_words)
                    rep = len(words & usr_words)

                    all_words = all_words | words

                    ## initiative classification
                    if use_ucm:
                        init = init_model.predict(user_text, system_text)
                        if init == 1:
                            init = 'I'
                            metrics['pro'].append(1)
                        else:
                            init = 'N'
                            metrics['pro'].append(0)
                    else:
                        if use_ref:
                            if turn['strategy'] in ['Question', 'Restatement or Paraphrasing', 'Providing Suggestions',
                                                    'Information']:
                                init = 'I'
                                metrics['pro'].append(1)
                            else:
                                init = 'N'
                                metrics['pro'].append(0)
                        else:
                            if res['pred_strat_id'] in [0, 1, 5, 6]:
                                init = 'I'
                                metrics['pro'].append(1)
                            else:
                                init = 'N'
                                metrics['pro'].append(0)

                    turn['initiative'] = init
                    metrics['inf'][init].append(inf)
                    metrics['rep'][init].append(rep)
                    metrics['rel'][init].append(pred_emo - current_emo)
                    idx += 1

                context['dialog'].append(turn)

            new_data.append(data)
        json.dump(new_data, outfile, indent=2)

    mi_res = {}
    mi_res['pro'] = float(sum(metrics['pro'])) / len(metrics['pro'])
    mi_res['inf_i'] = float(sum(metrics['inf']['I'])) / len(metrics['inf']['I'])
    mi_res['inf_n'] = float(sum(metrics['inf']['N'])) / len(metrics['inf']['N'])
    mi_res['inf_all'] = float(sum(metrics['inf']['I']) + sum(metrics['inf']['N'])) / len(
        metrics['inf']['I'] + metrics['inf']['N'])
    mi_res['rep_i'] = float(sum(metrics['rep']['I'])) / len(metrics['rep']['I'])
    mi_res['rep_n'] = float(sum(metrics['rep']['N'])) / len(metrics['rep']['N'])
    mi_res['rep_all'] = float(sum(metrics['rep']['I']) + sum(metrics['rep']['N'])) / len(
        metrics['rep']['I'] + metrics['rep']['N'])
    mi_res['rel_i'] = float(sum(metrics['rel']['I'])) / len(metrics['rel']['I'])
    mi_res['rel_n'] = float(sum(metrics['rel']['N'])) / len(metrics['rel']['N'])
    mi_res['rel_all'] = float(sum(metrics['rel']['I']) + sum(metrics['rel']['N'])) / len(
        metrics['rel']['I'] + metrics['rel']['N'])
    print(mi_res)
    return mi_res


if __name__ == '__main__':
    # init_model = InitModel()
    # print(init_model.predict("i'm sad.","i'm sorry to hear that."))
    # print(init_model.predict("i'm sad.","i'm sorry to hear that. What is going on?"))
    USi_model = USi()
    emo_model = EmoModel()
    context = {"situation": "I am always depressed and the upcoming holidays are making it a lot worse.",
               "dialog": [{"text": "Hello. How are you today?", "speaker": "sys", "strategy": "Others"},
                          {"text": "hi i am okay, a little bit sad though", "speaker": "usr"},
                          {"text": "Okay. I am very sorry to hear that! Do you want to tell me more about that?",
                           "speaker": "sys", "strategy": "Question"}, {
                              "text": "Well with the holidays coming up i have been very stressed and nervous about what i am going to do",
                              "speaker": "usr"},
                          {"text": "i am very sorry to hear that. have you been able to go out for a walk?",
                           "speaker": "sys", "strategy": "Affirmation and Reassurance"}]}
    feedback = USi_model.predict(context)
    print(feedback)
    print(
        "Yes. I am a college student and I live on campus. The day before I moved in my dad said some unforgivable things to me and we haven't spoke since")
    print(emo_model.predict(
        "Well with the holidays coming up i have been very stressed and nervous about what i am going to do"))
    print(emo_model.predict(feedback))
    print(emo_model.predict(
        "Yes. I am a college student and I live on campus. The day before I moved in my dad said some unforgivable things to me and we haven't spoke since"))
