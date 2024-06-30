# Running Scripts for *Muffin (KEMI)*

## Preparing Enviroment

```bash
conda env create -f env.yml -n cuda
conda activate cuda
```

## Getting AI Feedback
First, process the raw by enter ``reward_model/Llama`` and run:
```console
python process_raw.py --input_file_dir
/home/jiashuo/codes/Muffin/KEMI/DATA/strat.strat.esconv.sbert/2023-06-30223758.3e-05.16.1gpu/candidates_10_epoch-4.bin_train
```

## Preprocessing Training Data

First, enter `_reformat` and run `python process.py`.

Then, run `bash RUN/prepare_vanilla.sh` to preprocess the training data.

## Training Your Model

Run `bash RUN/train_vanilla.sh` to train your model.

## Inference with Your Model

Every time of model training will create a new folder in `DATA/{inputter_name}.{config_name}`, which is named after the time when the training starts. You should select a checkpoint (it may be based on the PPL of validation), and replace the checkpoint path in `RUN/infer_vanilla.sh --load_checkpoint` with the path of your selected checkpoint.

Then, run `bash RUN/infer_vanilla.sh` to do the inference.

**Note**: When you run `infer_strat.sh`, you can change `GOLDEN_TRUTH` in  `inputters/PARAMS.py` to control whether use the golden strategy during inference.

## Interacting with Your Model

Similar to inference, after designating the checkpoint in `RUN/interact_vanilla.sh --load_checkpoint`, run `bash RUN/interact_vanilla.sh`.
