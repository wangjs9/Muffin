# Running Scripts for *Muffin (MultiESC)*

## Preprocessing Training Data

First, enter `_reformat` and run `python process.py`.

Then, run `bash RUN/prepare_strat.sh` to preprocess the training data.

## Training Your Model

Run `bash RUN/train_strat.sh` to train your model.

## Getting AI Feedback
First, process the raw by enter ``reward_model/Llama`` and run:
```console
python process_raw.py --input_file_dir ./final_output/whlookahead_generate_candidate_10
```

## Optimize Model

Run:
```console
python mitigation_MultiESC.py --data_type=8 --model_name_or_path=./final_output/lwg_whlookahead_generate --learning_rate=3e-5 --lr2=1e-4 --with_cause --with_strategy --lookahead --model_type=1 --candidate_num=10 --per_device_train_batch_size=8 --gradient_accumulation_steps=2 --warmup_steps=1200
```

## Inference with Your Model

Every time of model training will create a new folder in `DATA/{inputter_name}.{config_name}`, which is named after the time when the training starts. You should select a checkpoint (it may be based on the PPL of validation), and replace the checkpoint path in `RUN/infer_strat.sh --load_checkpoint` with the path of your selected checkpoint.

Then, run `bash RUN/infer_strat.sh` to do the inference.

**Note**: When you run `infer_strat.sh`, you can change `GOLDEN_TRUTH` in  `inputters/PARAMS.py` to control whether use the golden strategy during inference.
