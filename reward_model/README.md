## Running Scripts for the AI Feedback Module

### Download The Model
Download the llama-7b-hf model from [Hugging Face](https://huggingface.co/decapoda-research/llama-7B-hf)

### Download The Dataset


### Instruction Tuning
```console
python finetune.py --base_model decapoda-research/llama-7b-hf --output_dir ./lora-alpaca_option
```

### Test Model
```console
python infer.py --base_model decapoda-research/llama-7b-hf --lora_weight ./lora-alpaca_option
```