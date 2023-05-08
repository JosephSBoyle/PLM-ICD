python run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/PLM_MIMIC_HIV_TEST.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 0 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type roberta \
    --model_mode laat

python run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/test_full.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 0 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type roberta \
    --model_mode laat

python run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/test_full.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 1 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type roberta \
    --model_mode laat


python run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/test_full.csv \
    --max_length 256 \
    --chunk_size 128 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 1 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type bert \
    --model_mode cls-max

# CAML

NOTE:
for some reason when you run with more than ~250 max length and chunk size
you get some weird pytorch error in the backend!


python run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/test_full.csv \
    --max_length 250 \
    --chunk_size 250 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --num_train_epochs 1 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type caml \
    --model_mode laat

python run_icd.py \
    --train_file ../data/mimic3/train_50.csv \
    --validation_file ../data/mimic3/test_50.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --num_train_epochs 1 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type caml \
    --model_mode laat


python run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/test_full.csv \
    --max_length 3072 \
    --chunk_size 3072 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --num_train_epochs 10 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 64
    