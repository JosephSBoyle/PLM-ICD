# Base CAML metrics
Early stopping - tolerance of 3 epochs @ 4 epochs

05/10/2023 10:48:48 - INFO - __main__ -   metrics: {'acc_macro': 0.020218339814138994, 'prec_macro': 0.02255925703458216, 'rec_macro': 0.3035714285714286, 'f1_macro': 0.0419975561193984, 'acc_micro': 0.022593320235756387, 'prec_micro': 0.02330293819655522, 'rec_micro': 0.42592592592592593, 'f1_micro': 0.04418828049951969, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.6281358367295868, 'auc_micro': 0.7180631678015833}

# Frozen CAML + LL - initialized to identity matrix
Early stopping - tolerance of 3 epochs @ 29 epochs
05/10/2023 11:25:21 - INFO - __main__ -   metrics: {'acc_macro': 0.022143268468069917, 'prec_macro': 0.029726554852079437, 'rec_macro': 0.5059523809523809, 'f1_macro': 0.056153864561929674, 'acc_micro': 0.014174344436569808, 'prec_micro': 0.014245014245014245, 'rec_micro': 0.7407407407407407, 'f1_micro': 0.027952480782669462, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.6378960433647933, 'auc_micro': 0.7305154182583181}

(Pdb) model._conditioning.weight 
Parameter containing:
tensor([[ 1.0585, -0.0417],
        [-0.0394,  1.0744]], device='cuda:0', requires_grad=True)

# un-frozen CAML + LL
Early stopping - tolerance of 10 epochs (increased due to under-fitting)
05/10/2023 16:32:46 - INFO - __main__ -   metrics: {'acc_macro': 0.017434014511546542, 'prec_macro': 0.017573688585433803, 'rec_macro': 0.7261904761904762, 'f1_macro': 0.03431691357736857, 'acc_micro': 0.014621968616262483, 'prec_micro': 0.0146900752418488, 'rec_micro': 0.7592592592592593, 'f1_micro': 0.028822495606326888, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.7050753878878878, 'auc_micro': 0.7591762165753198}

# unfrozen CAML + LL initialized to Identity



# base CAML
early stopping tolerance=3 epochs dev f1 decrease
Early stopped after 4 epochs
TRAIN metrics: {'acc_macro': 0.4212677231024825, 'prec_macro': 0.7435080404999295, 'rec_macro': 0.5027829313543599, 'f1_macro': 0.5998970714389422, 'acc_micro': 0.4367396593673966, 'prec_micro': 0.7311608961303462, 'rec_micro': 0.5202898550724637, 'f1_micro': 0.6079593564775613, 'rec_at_8': 0.7012195121951219, 'prec_at_8': 0.35060975609756095, 'f1_at_8': 0.46747967479674796, 'auc_macro': 0.8135011756129769, 'auc_micro': 0.8105883286838584}

DEV metrics: {'acc_macro': 0.03570901033960833, 'prec_macro': 0.06636838180400532, 'rec_macro': 0.3869047619047619, 'f1_macro': 0.11330140916701455, 'acc_micro': 0.021834061135371178, 'prec_micro': 0.022222222222222223, 'rec_micro': 0.5555555555555556, 'f1_micro': 0.042735042735042736, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.6133268312955813, 'auc_micro': 0.720118474229087}

# base CAML frozen + linear layer initialized to Identity
early stopping tolerance=3 epochs dev f1 decrease
Early stopped after 4 additional epochs

TRAIN metrics: {'acc_macro': 0.4434029899862605, 'prec_macro': 0.715135814099255, 'rec_macro': 0.5359204287775716, 'f1_macro': 0.6126917064015709, 'acc_micro': 0.45443645083932854, 'prec_micro': 0.7246653919694073, 'rec_micro': 0.5492753623188406, 'f1_micro': 0.6248969497114591, 'rec_at_8': 0.7012195121951219, 'prec_at_8': 0.35060975609756095, 'f1_at_8': 0.46747967479674796, 'auc_macro': 0.8219155470708266, 'auc_micro': 0.8194075888503323}

DEV   metrics: {'acc_macro': 0.035961810466635186, 'prec_macro': 0.06660331036605652, 'rec_macro': 0.41071428571428575, 'f1_macro': 0.1146194117620478, 'acc_micro': 0.022315202231520222, 'prec_micro': 0.0226628895184136, 'rec_micro': 0.5925925925925926, 'f1_micro': 0.043656207366985, 'rec_at_8': 0.01601423487544484, 'prec_at_8': 0.00800711743772242, 'f1_at_8': 0.010676156583629895, 'auc_macro': 0.6139315654940655, 'auc_micro': 0.7204174278912695}

Learnt matrix:
tensor([[ 1.0131, -0.0122],
        [-0.0150,  1.0195]], device='cuda:0', requires_grad=True)