## Running Scripts for *Muffin (ESConv)*

### Requirements

### Data Download

Download [ESConv.json](https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json)
and [strategy.json](https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/strategy.json) and
put them in
the folder `DATA`.

### Data Preprocessing

Enter `DATA` and run ``python process.py``.

To preprocess the training data (for Blender-Vanilla and Blender-Joint, respectively), run:

```console
❱❱❱ python prepare.py --config_name vanilla --inputter_name vanilla --train_input_file DATA/train.txt --max_input_length 160 --max_decoder_input_length 40
```
```console
❱❱❱ python prepare.py  --config_name strat --inputter_name strat --train_input_file DATA/train.txt --max_input_length 160  --max_decoder_input_length 40
```

### Base Model Training

Run:

```console
❱❱❱ CUDA_VISIBLE_DEVICES=0 nohup python train_ESConv.py --config_name vanilla --inputter_name vanilla --eval_input_file DATA/valid.txt --seed 13 --max_input_length 160 --max_decoder_input_length 40 --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16 --learning_rate 3e-5 --epoch_num 3  --warmup_steps 100 --fp16 false --loss_scale 0.0 --pbar true > vanilla_base.log 2>&1 &
```
```console
❱❱❱ CUDA_VISIBLE_DEVICES=1 nohup python train_ESConv.py --config_name strat --inputter_name strat --eval_input_file DATA/valid.txt --seed 13 --max_input_length 160 --max_decoder_input_length 40 --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16 --learning_rate 3e-5 --epoch_num 3  --warmup_steps 100 --fp16 false --loss_scale 0.0 --pbar true > strat_base.log 2>&1 &
```

### Generating Samples

Run:

```console
❱❱❱ CUDA_VISIBLE_DEVICES=0 nohup python sample_ESConv.py --config_name vanilla --inputter_name vanilla --seed 0 --load_checkpoint ./DATA/vanilla.vanilla/2023-07-23185716.3e-05.16.1gpu/best_model.bin --fp16 false --max_input_length 160 --max_decoder_input_length 40 --max_length 40 --min_length 10 --infer_batch_size 8 --infer_input_file ./DATA/train.txt --temperature 0.7 --top_k 0 --top_p 0.9 --num_beams 10 --num_beam_groups 10 --repetition_penalty 1 --no_repeat_ngram_size 3 --num_return_sequences 10 > vanilla_sample.log 2>&1 &
```

### Getting AI Feedback
First, process the raw by enter ``reward_model/Llama`` and run:
```console
python process_raw.py --input_file_dir
/home/jiashuo/codes/Muffin/ESConv/DATA/vanilla.vanilla/2023-06-20204748.3e-05.16.1gpu/candidates_10_best_model.bin_train
python process_raw.py --input_file_dir
/home/jiashuo/codes/Muffin/ESConv/DATA/strat.strat/2023-06-20204057.3e-05.16.1gpu/candidates_10_best_model.bin_train
```

### Optimize Model

Run:

```console
❱❱❱ CUDA_VISIBLE_DEVICES=0 nohup python mitigation_ESConv.py --config_name vanilla --inputter_name vanilla --eval_input_file DATA/valid.txt --infer_input_file DATA/test.txt --checkpoint_dir DATA/vanilla.vanilla/2023-06-20204748.3e-05.16.1gpu --warmup_step 2400 --max_lr 3e-5 --train_batch_size 8 --accumulate_step 1
```
```console
❱❱❱ CUDA_VISIBLE_DEVICES=0 nohup python mitigation_ESConv.py --config_name strat --inputter_name strat --eval_input_file DATA/valid.txt --infer_input_file DATA/test.txt --checkpoint_dir DATA/strat.strat/2023-06-20204057.3e-05.16.1gpu --warmup_step 2400 --max_lr 3e-5 --train_batch_size 8 --accumulate_step 1 > strat_muffin.log 2>&1 &
```

Change ``--preference_model_dir`` and ``--base_model_dir`` to your own model directories.

### Model Inference

Run:

```console
❱❱❱ CUDA_VISIBLE_DEVICES=0 nohup python infer_ESConv.py --config_name config_name --inputter_name inputter_name --add_nlg_eval --seed 13 --load_checkpoint model_checkpoint_path --fp16 false --max_length 40 --infer_batch_size 16 --infer_input_file ./DATA/test.txt --temperature 0.7 --top_k 0 --top_p 0.9 --num_beams 1 --repetition_penalty 1 --no_repeat_ngram_size 3 > test.log 2>&1 &
```

Change ``--config_name``, ``--inputter_name``, and ``--load_checkpoint``.