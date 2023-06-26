"Per-label attention models need per-label regularization"

Mullenbach paper: 
> CAML, with a tuned regularization coefficient of 𝜆 = 0.01.

Train, early stop on eval set. Scan possible weight_decay hparams for optimal dev F1 score, and **save** that model.
- for top  50 codes
- for rare 50 codes

First try on the CAML models, since those are cheaper, then try the longformer ones.

# TOP-50 ; CAML ; WEIGHT DECAY 0

python run_icd.py \
    --train_file ../data/mimic3/train_50.csv \
    --validation_file ../data/mimic3/test_50.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50_weightdecay_0 \
    --num_train_epochs 20 \
    --output_dir ../models/caml_50_weightdecay_0 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50 \
    --weight_decay 0

# TOP-50 ; CAML ; WEIGHT DECAY 0.001

python run_icd.py \
    --train_file ../data/mimic3/train_50.csv \
    --validation_file ../data/mimic3/test_50.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50_weightdecay_0_001 \
    --num_train_epochs 20 \
    --output_dir ../models/caml_50_weightdecay_0_001 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50 \
    --weight_decay 0.001

# TOP-50 ; CAML ; WEIGHT DECAY 0.00001

python run_icd.py \
    --train_file ../data/mimic3/train_50.csv \
    --validation_file ../data/mimic3/test_50.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50_weightdecay_0_00001 \
    --num_train_epochs 20 \
    --output_dir ../models/caml_50_weightdecay_0_00001 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50 \
    --weight_decay 0.00001

# top-50 10k training steps (instead of n epochs)

python run_icd.py \
    --train_file ../data/mimic3/train_50.csv \
    --validation_file ../data/mimic3/test_50.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50_weightdecay_0 \
    --max_train_steps 10000 \
    --output_dir ../models/caml_50_weightdecay_0 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50 \
    --weight_decay 0 && \
python run_icd.py \
    --train_file ../data/mimic3/train_50.csv \
    --validation_file ../data/mimic3/test_50.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50_weightdecay_0_001 \
    --max_train_steps 10000 \
    --output_dir ../models/caml_50_weightdecay_0_001 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50 \
    --weight_decay 0.001 && \
python run_icd.py \
    --train_file ../data/mimic3/train_50.csv \
    --validation_file ../data/mimic3/test_50.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50_weightdecay_0_00001 \
    --max_train_steps 10000 \
    --output_dir ../models/caml_50_weightdecay_0_00001 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50 \
    --weight_decay 0.00001

# BOT-50; WEIGHT DECAY 0
python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50l_weightdecay_0 \
    --num_train_epochs 20 \
    --output_dir ../models/caml_50l_weightdecay_0 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50l \
    --weight_decay 0

# BOT-50; WEIGHT DECAY 0.001
python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50l_weightdecay_0_001 \
    --num_train_epochs 20 \
    --output_dir ../models/caml_50l_weightdecay_0_001 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50l \
    --weight_decay 0.001

# BOT-50; WEIGHT DECAY 0.00001

python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50l_weightdecay_0_00001 \
    --num_train_epochs 20 \
    --output_dir ../models/caml_50l_weightdecay_0_00001 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50l \
    --weight_decay 0.00001

# bot-50 all runs:
python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50l_weightdecay_0 \
    --num_train_epochs 20 \
    --output_dir ../models/caml_50l_weightdecay_0 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50l \
    --weight_decay 0 && \
python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50l_weightdecay_0_001 \
    --num_train_epochs 20 \
    --output_dir ../models/caml_50l_weightdecay_0_001 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50l \
    --weight_decay 0.001 && \
python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50l_weightdecay_0_00001 \
    --num_train_epochs 20 \
    --output_dir ../models/caml_50l_weightdecay_0_00001 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50l \
    --weight_decay 0.00001

# bot-50 10k training steps (instead of n epochs)
python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50l_weightdecay_0 \
    --max_train_steps 10000 \
    --output_dir ../models/caml_50l_weightdecay_0 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50l \
    --weight_decay 0 && \
python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50l_weightdecay_0_001 \
    --max_train_steps 10000 \
    --output_dir ../models/caml_50l_weightdecay_0_001 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50l \
    --weight_decay 0.001 && \
python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path caml_50l_weightdecay_0_00001 \
    --max_train_steps 10000 \
    --output_dir ../models/caml_50l_weightdecay_0_00001 \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --code_50l \
    --weight_decay 0.00001
