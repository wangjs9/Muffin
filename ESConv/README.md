## Running Scripts for *Muffin (ESConv)*

### Downloading the Dataset

Download [ESConv.json](https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json) and [strategy.json](https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/strategy.json) and put them in the folder `DATA`.

To preprocess the training data (for Blender-Vanilla and Blender-Joint, respectively), enter `DATA` and run ``python process.py``.

```console
cd DATA
python process.py
python prepare.py --config_name vanilla --inputter_name vanilla --train_input_file DATA/train.txt --max_input_length 160 --max_decoder_input_length 40
python prepare.py --config_name strat --inputter_name strat --train_input_file DATA/train.txt --max_input_length 160  --max_decoder_input_length 40
```

### Training Base Model
Run:

```console
python train_ESConv.py --config_name vanilla --inputter_name vanilla --eval_input_file DATA/valid.txt --seed 13 --max_input_length 160 --max_decoder_input_length 40 --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16 --learning_rate 3e-5 --epoch_num 3  --warmup_steps 100 --fp16 false --loss_scale 0.0 --pbar true
python train_ESConv.py --config_name strat --inputter_name strat --eval_input_file DATA/valid.txt --seed 13 --max_input_length 160 --max_decoder_input_length 40 --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16 --learning_rate 3e-5 --epoch_num 3  --warmup_steps 100 --fp16 false --loss_scale 0.0 --pbar true
```
For more details, refer the original codes from [here](https://github.com/thu-coai/Emotional-Support-Conversation/tree/main/codes_zcj) to download the dataset and train the base model.

### Generating Samples

Run:

```console
python sample_ESConv.py --config_name vanilla --inputter_name vanilla --seed 0 --load_checkpoint ./DATA/vanilla.vanilla/best_model.bin --fp16 false --max_input_length 160 --max_decoder_input_length 40 --max_length 40 --min_length 10 --infer_batch_size 8 --infer_input_file ./DATA/train.txt --temperature 0.7 --top_k 0 --top_p 0.9 --num_beams 10 --num_beam_groups 10 --repetition_penalty 1 --no_repeat_ngram_size 3 --num_return_sequences 10
```
`./DATA/vanilla.vanilla/best_model.bin` is the path with the checkpoint of the base model. Change all * vanilla * as * strat * to generate samples for the Joint model.

### Getting AI Feedback
First, process the raw by enter ``reward_model/Llama`` and run:
```console
python process_raw.py --input_file_dir /home/jiashuo/codes/Muffin/ESConv/DATA/vanilla.vanilla/candidates_10_best_model.bin_train
python process_raw.py --input_file_dir /home/jiashuo/codes/Muffin/ESConv/DATA/strat.strat/candidates_10_best_model.bin_train
```

### Training Muffin Model

Run:

```console
python mitigation_ESConv.py --config_name vanilla --inputter_name vanilla --eval_input_file DATA/valid.txt --infer_input_file DATA/test.txt --checkpoint_dir DATA/vanilla.vanilla/ --warmup_step 2400 --max_lr 3e-5 --train_batch_size 8 --accumulate_step 1
python mitigation_ESConv.py --config_name strat --inputter_name strat --eval_input_file DATA/valid.txt --infer_input_file DATA/test.txt --checkpoint_dir DATA/strat.strat/ --warmup_step 2400 --max_lr 3e-5 --train_batch_size 8 --accumulate_step 1
```

Change ``--preference_model_dir`` and ``--base_model_dir`` to your own model directories.

### Inference with a Model

Run:

```console
python infer_ESConv.py --config_name config_name --inputter_name inputter_name --add_nlg_eval --seed 13 --load_checkpoint model_checkpoint_path --fp16 false --max_length 40 --infer_batch_size 16 --infer_input_file ./DATA/test.txt --temperature 0.7 --top_k 0 --top_p 0.9 --num_beams 1 --repetition_penalty 1 --no_repeat_ngram_size 3
```

Change ``--config_name``, ``--inputter_name``, and ``--load_checkpoint``.