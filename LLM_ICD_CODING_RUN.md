python run_icd.py \
    --train_file ../data/codiesp/codiesp_gpt_english_caml_format_train.csv \
    --validation_file ../data/codiesp/codiesp_gpt_english_caml_format_dev.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/RoBERTa-base-PM-M3-Voc-distill-align-hf \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 20 \
    --num_warmup_steps 2000 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type roberta \
    --model_mode laat \
    --code_file ../data/ICD10_CM_leaf_codes_only.tsv \
    --only_labels_in_train_set