roberta, LAAT, from paper.
AUC macro 0.518, AUC micro: 0.910

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

05/01/2023 16:27:40 - INFO - __main__ -  
metrics: {'acc_macro': 0.0, 'prec_macro': 0.0, 'rec_macro': 0.0, 'f1_macro': 0.0, 'acc_micro': 0.0, 'prec_micro': 0.0, 'rec_micro': 0.0, 'f1_micro': 0.0, 'rec_at_8': 0.0, 'prec_at_8': 0.0, 'f1_at_8': nan, 'auc_macro': 0.5175950504075504, 'auc_micro': 0.9053594099697847}


NOTE: THE CAML RUNS ARE LIMITED TO 250 TOKENS!

CAML with hacky roberta embeddings instead of w2v ones.
AUC macro 0.483, AUC micro, 0.999

05/01/2023 16:54:41 - INFO - __main__ -   metrics: {'acc_macro': 0.0, 'prec_macro': 0.0, 'rec_macro': 0.0, 'f1_macro': 0.0, 'acc_micro': 0.0, 'prec_micro': nan, 'rec_micro': 0.0, 'f1_micro': nan, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.002001779359430605, 'f1_at_8': 0.0035587188612099647, 'auc_macro': 0.482986893143143, 'auc_micro': 0.9998533049779265}


Same caml model, but with the logits of each label conditioned on each other.
auc macro 0.519, auc micro 0.645
05/02/2023 10:45:25 - INFO - __main__ -   metrics: {'acc_macro': 0.006227758007117253, 'prec_macro': 0.006227758007117253, 'rec_macro': 0.5, 'f1_macro': 0.012302284710017214, 'acc_micro': 0.012411347517730497, 'prec_micro': 0.012455516014234875, 'rec_micro': 0.7777777777777778, 'f1_micro': 0.024518388791593692, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.519449024917775, 'auc_micro': 0.6350121795936444}

### 3072 TOKENS ###
Bigger batch size for faster training.
Early stopping if eval loss increases over 1 epoch.

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
    --per_device_train_batch_size 64 \
    --learning_rate 0.0001

05/02/2023 13:31:46 - INFO - __main__ -   metrics: {'acc_macro': 0.0, 'prec_macro': 0.0, 'rec_macro': 0.0, 'f1_macro': 0.0, 'acc_micro': 0.0, 'prec_micro': nan, 'rec_micro': 0.0, 'f1_micro': nan, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.46172064922064926, 'auc_micro': 0.5969080440679844}


#####
CAML
max length          : 2500
early stop tolerance: 100
LR                  : 0.0001
Batch size          : 16
num epochs          : ? (set a value of 100, this value affects the LR annealing!)
#####

### RUN COMMAND ####
Same for both runs:

python run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/test_full.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --num_train_epochs 5 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --learning_rate 0.0001

### RESULTS ###
No conditioning (independent)

05/02/2023 15:07:59 - INFO - __main__ -   metrics: {'acc_macro': 0.04166666666631944, 'prec_macro': 0.49999999995, 'rec_macro': 0.041666666666666664, 'f1_macro': 0.0769230769224852, 'acc_micro': 0.018518518518518517, 'prec_micro': 1.0, 'rec_micro': 0.018518518518518517, 'f1_micro': 0.03636363636363636, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.8365006747819247, 'auc_micro': 0.8543459004595028}

Conditioning    (dependent)
05/02/2023 15:29:37 - INFO - __main__ -   metrics: {'acc_macro': 0.0, 'prec_macro': 0.0, 'rec_macro': 0.0, 'f1_macro': 0.0, 'acc_micro': 0.0, 'prec_micro': nan, 'rec_micro': 0.0, 'f1_micro': nan, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.8390695382882882, 'auc_micro': 0.8393428555610918}

Conclusion; adding conditioning doesn't seem to help???

## TODO BALANCE THE DATASET! MODEL IS SIMPLY PREDICTING ALL SAMPLES as 0!

### 5 EPOCHS ####
conditioned (dependent)
05/03/2023 10:43:06 - INFO - __main__ -   metrics: {'acc_macro': 0.0, 'prec_macro': 0.0, 'rec_macro': 0.0, 'f1_macro': 0.0, 'acc_micro': 0.0, 'prec_micro': nan, 'rec_micro': 0.0, 'f1_micro': nan, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.8699531004218504, 'auc_micro': 0.8859270331617118}

### MORE EPOCHS AND SUBSAMPLED CLASS (balanced)###
python run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/test_full.csv \
    --max_length 2500 \
    --chunk_size 2500 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --num_train_epochs 100 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type caml \
    --model_mode laat \
    --per_device_train_batch_size 16 \
    --learning_rate 0.0001

## Early stopped after 10 epochs.

No conditioning (independent)
05/03/2023 16:30:13 - INFO - __main__ -   metrics: {'acc_macro': 0.045598006644432656, 'prec_macro': 0.055384451996463205, 'rec_macro': 0.5297619047619048, 'f1_macro': 0.10028456110154542, 'acc_micro': 0.020998864926220204, 'prec_micro': 0.021203438395415473, 'rec_micro': 0.6851851851851852, 'f1_micro': 0.04113396331295164, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.7965370504433005, 'auc_micro': 0.7991031390134529}

Conditioning    (dependent)
after 10 epochs training:
05/03/2023 15:50:41 - INFO - __main__ -   metrics: {'acc_macro': 0.01360544217686959, 'prec_macro': 0.013722126929672921, 'rec_macro': 0.38095238095238093, 'f1_macro': 0.02649006622516337, 'acc_micro': 0.026936026936026935, 'prec_micro': 0.0274442538593482, 'rec_micro': 0.5925925925925926, 'f1_micro': 0.05245901639344262, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 
0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.6428227334477334, 'auc_micro': 0.7720644411227371}





CAML 10 EPOCHS
05/05/2023 10:28:45 - INFO - __main__ -   metrics: {'acc_macro': 0.04328837508005973, 'prec_macro': 0.1350258732180904, 'rec_macro': 0.41071428571428575, 'f1_macro': 0.2032361158108854, 'acc_micro': 0.020356234096692113, 'prec_micro': 0.02064516129032258, 'rec_micro': 0.5925925925925926, 'f1_micro': 0.0399002493765586, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.6161204731517231, 'auc_micro': 0.7211509715994021}

CAML 10 EPOCHS loaded + FC-LAYER

05/05/2023 10:49:43 - INFO - __main__ -   metrics: {'acc_macro': 0.010036973544388473, 'prec_macro': 0.010066684747549043, 'rec_macro': 0.8333333333333334, 'f1_macro': 0.019893060887950727, 'acc_micro': 0.008594346829640947, 'prec_micro': 0.008609144824947389, 'rec_micro': 0.8333333333333334, 'f1_micro': 
0.017042226850975197, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.5710442585442586, 'auc_micro': 0.6773404196423629}


CAML 50 EPOCHS:
05/05/2023 11:24:45 - INFO - __main__ -   metrics: {'acc_macro': 0.04229562519825983, 'prec_macro': 0.04449680650363222, 'rec_macro': 0.6309523809523809, 'f1_macro': 0.08313094909177746, 'acc_micro': 0.02392947103274559, 'prec_micro': 0.024173027989821884, 'rec_micro': 0.7037037037037037, 'f1_micro': 0.046740467404674045, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.8216773693336192, 'auc_micro': 0.8346011183081438}

CAML 50 EPOCHS loaded + FC-layer
05/05/2023 11:46:50 - INFO - __main__ -   metrics: {'acc_macro': 0.01844051798899742, 'prec_macro': 0.018948818246921725, 'rec_macro': 0.6309523809523809, 'f1_macro': 0.03679267557548465, 'acc_micro': 0.015473191795609931, 'prec_micro': 0.015534682080924855, 'rec_micro': 0.7962962962962963, 'f1_micro': 0.030474840538625085, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.7375995415057914, 'auc_micro': 0.778395061728395}


.6488095238095238, 'f1_macro': 0.12291584354371705, 'acc_micro': 0.02458471760797342, 'prec_micro': 0.02486559139784946, 'rec_micro': 0.6851851851851852, 'f1_micro': 0.04798962386511024, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.8029900436150436, 'auc_micro': 0.8241238996844378}

05/05/2023 14:40:18 - INFO - __main__ -   metrics: {'acc_macro': 0.07166161928813422, 'prec_macro': 0.07736501561790637, 'rec_macro': 0.5892857142857143, 'f1_macro': 0.13677356505921615, 'acc_micro': 0.029850746268656716, 'prec_micro': 0.030476190476190476, 'rec_micro': 0.5925925925925926, 'f1_micro': 0.057971014492753624, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.80920015998141, 'auc_micro': 0.832184576205503}