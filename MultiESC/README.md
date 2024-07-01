# Running Scripts for *Muffin (MultiESC)*

### Training Base Model

Refer to the [original codes](https://github.com/lwgkzl/MultiESC) for more details.

[//]: # (### Generating Samples)

[//]: # (Run:)

[//]: # (```console)
[//]: # (python mitigation_MultiESC.py --data_type=8 --model_name_or_path=./final_output/whlookahead_generate --learning_rate=3e-5 --lr2=1e-4 --with_cause --with_strategy --lookahead --model_type=1 --candidate_num=10 --per_device_train_batch_size=8 --gradient_accumulation_steps=2 --warmup_steps=1200)
[//]: # (```)

[//]: # (### Getting AI Feedback)

[//]: # (First, process the raw by enter ``reward_model/Llama`` and run:)

[//]: # (```console)
[//]: # (python process_raw.py --input_file_dir ./final_output/whlookahead_generate_candidate_10)

[//]: # (```)

### Training Muffin Model

Run:

```console
python mitigation_MultiESC.py --data_type=8 --model_name_or_path=./final_output/whlookahead_generate --learning_rate=3e-5 --lr2=1e-4 --with_cause --with_strategy --lookahead --model_type=1 --candidate_num=10 --per_device_train_batch_size=8 --gradient_accumulation_steps=2 --warmup_steps=1200
```
