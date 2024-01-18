CUDA_VISIBLE_DEVICES=4 python sample_ESC.py \
  --config_name vanilla \
  --inputter_name vanilla \
  --seed 0 \
  --load_checkpoint ./DATA/vanilla.vanilla/2023-06-20204748.3e-05.16.1gpu/best_model.bin \
  --fp16 false \
  --max_input_length 160 \
  --max_decoder_input_length 40 \
  --max_length 40 \
  --min_length 10 \
  --infer_batch_size 8 \
  --infer_input_file ./DATA/train.txt \
  --temperature 0.7 \
  --top_k 0 \
  --top_p 0.9 \
  --num_beams 10 \
  --num_beam_groups 10 \
  --repetition_penalty 1 \
  --no_repeat_ngram_size 3 \
  --num_return_sequences 10