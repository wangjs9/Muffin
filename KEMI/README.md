# Running Scripts for *Muffin (KEMI)*

### Training Base Model

Run:

```console
python train_KEMI.py --config_name strat --inputter_name strat --data_name esconv --knowledge_name sbert --eval_input_file ./DATA --seed 13 --max_input_length 256 --max_decoder_input_length 40 --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16 --learning_rate 3e-5 --num_epochs 5 --warmup_steps 100 --fp16 false --loss_scale 0.0 --pbar true
```

For more details, refers to the original codes from [here](https://github.com/dengyang17/KEMI).

### Generating Samples

Run:

```console
python sample_ESConv.py --config_name vanilla --inputter_name vanilla --seed 0 --load_checkpoint ./DATA/vanilla.vanilla/best_model.bin --fp16 false --max_input_length 160 --max_decoder_input_length 40 --max_length 40 --min_length 10 --infer_batch_size 8 --infer_input_file ./DATA/train.txt --temperature 0.7 --top_k 0 --top_p 0.9 --num_beams 10 --num_beam_groups 10 --repetition_penalty 1 --no_repeat_ngram_size 3 --num_return_sequences 10
```

`./DATA/vanilla.vanilla/best_model.bin` is the path with the checkpoint of the base model. Change all * vanilla * as *
strat * to generate samples for the Joint model.

### Getting AI Feedback

First, process the raw by enter ``reward_model/Llama`` and run:

```console
python process_raw.py --input_file_dir /home/jiashuo/codes/Muffin/KEMI/DATA/strat.strat.esconv.sbert/candidates_10_best_model.bin_train
```

### Training Muffin Model

Run `bash RUN/train_vanilla.sh` to train your model.

