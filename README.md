# huggingface_playaround
for playing around with huggingface code: https://github.com/huggingface/transformers

### XNLI run
```shell
export XNLI_DIR=/home/jqu/Documents/data/XNLI
python run_xnli.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --language en \
  --train_language en \
  --do_train \
  --do_eval \
  --data_dir $XNLI_DIR \
  --per_gpu_train_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --max_seq_length 128 \
  --output_dir /home/jqu/Documents/tmp/xnli-2/ \
  --save_steps 5000 \
  --do_eval \
  --local_rank -1 \
  --overwrite_output_dir \
  --logging_steps 500 \
  --runs_dir runs/xnli-2
```

### XNLI evaluation 
```shell
python evaluate_xnli.py --model_type=bert --model_dir="/home/jqu/Documents/tmp/debug_xnli/"
```
