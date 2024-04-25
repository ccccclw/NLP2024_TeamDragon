export OUTPUT_DIR=./results/

python3 marian_trans_priming.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=marian \
    --model_checkpoint=Helsinki-NLP/opus-mt-zh-en \
    --train_file=trans_train.json \
    --dev_file=trans_train.json \
    --test_file=trans_valid.json \
    --max_input_length=128 \
    --max_target_length=128 \
    --learning_rate=1e-5 \
    --num_train_epochs=3 \
    --batch_size=32 \
    --do_pred \
    --warmup_proportion=0. \
    --seed=42