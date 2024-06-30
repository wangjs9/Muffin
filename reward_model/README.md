## Running Scripts for the AI Feedback Module

### Download The Model

Download the llama-7b-hf model from [Hugging Face](https://huggingface.co/decapoda-research/llama-7B-hf)

### Download The Dataset

```console
cd dataset
```

Download the dataset for empathetic response classification
from [behavioral-data/Empathy-Mental-Health](https://github.com/behavioral-data/Empathy-Mental-Health/tree/master/dataset).

Download the dataset for strategy classification
from [Motivational-Interviewing-Dataset](https://github.com/anuradha1992/Motivational-Interviewing-Dataset).

Process the dataset (including obtain the dataset for coherence classification) using the following command:

```console
python data_process.py
cd ../
```

### Instruction Tuning

```console
python finetune.py --base_model decapoda-research/llama-7b-hf --output_dir ./lora-alpaca 
```

### Test the Multifaceted AI Feedback Model

```console
python test.py --base_model decapoda-research/llama-7b-hf --lora_weight ./lora-alpaca
```