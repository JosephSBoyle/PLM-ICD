python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/roberta_weight_decay_0 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 0 \
    --output_dir ../models/roberta_weight_decay_0 \
    --model_type roberta \
    --model_mode laat

06/09/2023 10:55:43 - INFO - __main__ -   metrics: {'acc_macro': 0.0, 'prec_macro': 0.0, 'rec_macro': 0.0, 'f1_macro': 0.0, 'acc_micro': 0.0, 'prec_micro': nan, 'rec_micro': 0.0, 'f1_micro': nan, 'rec_at_8': 0.23943661971830985, 'prec_at_8': 0.02992957746478873, 'f1_at_8': 0.05320813771517997, 'auc_macro': 0.5471084381677866, 'auc_micro': 0.9976217637987227}

==================

python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/roberta_weight_decay_0001 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 0 \
    --output_dir ../models/roberta_weight_decay_0001 \
    --model_type roberta \
    --model_mode laat

06/09/2023 11:00:09 - INFO - __main__ -   metrics: {'acc_macro': 0.0, 'prec_macro': 0.0, 'rec_macro': 0.0, 'f1_macro': 0.0, 'acc_micro': 0.0, 'prec_micro': nan, 'rec_micro': 0.0, 'f1_micro': nan, 'rec_at_8': 0.23943661971830985, 'prec_at_8': 0.02992957746478873, 'f1_at_8': 0.05320813771517997, 'auc_macro': 0.528209437306817, 'auc_micro': 0.9976795941461943}

===================
python run_icd.py \
    --train_file ../data/mimic3/train_50l.csv \
    --validation_file ../data/mimic3/test_50l.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/roberta_weight_decay_000001 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 0 \
    --output_dir ../models/roberta_weight_decay_000001 \
    --model_type roberta \
    --model_mode laat

06/09/2023 11:01:48 - INFO - __main__ -   metrics: {'acc_macro': 0.0, 'prec_macro': 0.0, 'rec_macro': 0.0, 'f1_macro': 0.0, 'acc_micro': 0.0, 'prec_micro': nan, 'rec_micro': 0.0, 'f1_micro': nan, 'rec_at_8': 0.25, 'prec_at_8': 0.03169014084507042, 'f1_at_8': 0.056249999999999994, 'auc_macro': 0.5115244210307957, 'auc_micro': 0.9976873160346593}