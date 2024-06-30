## Running Scripts for the AI Feedback Module

### Instruction Tuning
```console
CUDA_VISIBLE_DEVICES=6,7 nohup python finetune.py --base_model decapoda-research/llama-7b-hf --output_dir ./lora-alpaca_option > finetune.log 2>&1 &
```

### Test Model
```console
python infer.py --base_model yahma/llama-7b-hf --lora_weight ./lora-alpaca
```