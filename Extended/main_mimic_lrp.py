import os
import torch.optim as optim
import torch.nn as nn
import datetime
import argparse
import warnings
import random
import pickle
from Upload.help_physionet import  *
from Upload.models_phy_lrp import *
import torch
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


warnings.filterwarnings('ignore')
os.environ["GEVENT_SUPPORT"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.backends.cudnn.enabled = False
JOBLIB_MULTIPROCESSING=1


# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="which dataset to use", type=str, default='physionet')  # physionet or mimic
parser.add_argument('--fold_num', type=int, default=0)
parser.add_argument('--l1', type=float, default=5e-4)
parser.add_argument('--w_decay', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--lr_decay', type=int, default=15)
parser.add_argument('--lr_ratio', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gpu_id', type=int, default=4)

print('dropout zero, relu')
args = parser.parse_args()
dataset = args.dataset
fold_num = args.fold_num
l1 = args.l1
w_decay = args.w_decay
batch_size = args.batch_size
lr = args.lr
lr_decay = args.lr_decay
lr_ratio = args.lr_ratio

# GPU Configuration
# gpu_id = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
print('gpu ID is ', str(args.gpu_id))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Kfold dataset (PhysioNEt)
# data_dir = '/home/yrlee/irregular_data/Var35/'#'./Data/'
# kfold_data = np.load(open(data_dir + 'kfold_data_35.p', 'rb'), mmap_mode='r', allow_pickle=True)
# kfold_mask = np.load(open(data_dir + 'kfold_mask_35.p', 'rb'), mmap_mode='r', allow_pickle=True)
# kfold_label = np.load(open(data_dir + 'kfold_label_35.p', 'rb'), mmap_mode='r', allow_pickle=True)
# kfold_label2 = np.load(open('/home/yrlee/irregular_data/kfold_los_label.p', 'rb'), mmap_mode='r', allow_pickle=True)
# kfold_times = np.load(open(data_dir + 'kfold_times_35.p', 'rb'), mmap_mode='r', allow_pickle=True)

# Load Kfold dataset (MIMIC)
data_dir ='/home/yrlee/mimic/irregular/' #'./Data/'
kfold_data = np.load(open(data_dir + 'data.p', 'rb'), mmap_mode='r', allow_pickle=True)
kfold_mask = np.load(open(data_dir + 'mask.p', 'rb'), mmap_mode='r', allow_pickle=True)
kfold_label = np.load(open(data_dir + 'label.p', 'rb'), mmap_mode='r', allow_pickle=True)
kfold_label_los = np.load(open(data_dir + 'label_los.p', 'rb'), mmap_mode='r', allow_pickle=True)
kfold_label2 = np.load(open(data_dir + 'label_phenotyping.p', 'rb'), mmap_mode='r', allow_pickle=True)
kfold_times = np.load(open(data_dir + 'kfold_time.p', 'rb'), mmap_mode='r', allow_pickle=True)



# Training Parameters
n_epochs = 60
alpha = 9
gamma = 0.15
# Loss rates
lambda_1 = 0.1
lambda_2 = 1
lambda_3 = 10
print('focal(y):', str(lambda_1), ', mse(x):', str(lambda_2))
KFold = len(kfold_data) #4

# Network architecture
max_length = kfold_data[0][0].shape[1]
input_dim = kfold_data[0][0].shape[2]

d_model = 128
d_ff = 128
num_stacks = 1
num_heads = 8

# Seed
manualSeed = 128
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
random.seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

# kfold performance
kfold_mse = []
kfold_mae = []
kfold_acc = []
kfold_balacc = []
kfold_auc = []
kfold_auprc = []
kfold_sen = []
kfold_spec = []
kfold_precision = []
kfold_recall = []
kfold_f1_score_pr = []
kfold_f2_score_pr = []


def switch(fold_num):
    return {0: range(0, 1),
            1: range(1, 2),
            2: range(2, 3),
            3: range(3, 4),
            4: range(4, 5)}[fold_num]


# Create Directories
log_dir = './log/' + str(datetime.datetime.now().strftime('%y%m%d')) + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    os.chmod(log_dir, mode=0o777)
dir = log_dir + 'observation_mask_multi_encoder_' + str(datetime.datetime.now().strftime('%H.%M.%S')) + '/'


if not os.path.exists(dir):
    os.makedirs(dir)
    os.makedirs(dir + 'model/')
    os.makedirs(dir + 'tflog/')
    for k in range(KFold):
        os.makedirs(dir + 'model/' + str(k) + '/')


# Text Logging
f = open(dir + 'log.txt', 'a')
writelog(f, '---------------')
# writelog(f, 'Observation Single Encoder')
writelog(f, 'Observation + Masking Multi-Pipeline Attention (residual)')
writelog(f, 'Mortality')
writelog(f, 'Dataset :' + str(data_dir))
writelog(f, '---------------')
writelog(f, 'TRAINING PARAMETER')
writelog(f, 'Learning Rate : ' + str(lr))
writelog(f, 'LR decay : '+ str(lr_ratio))
writelog(f, 'Batch Size : ' + str(batch_size))
writelog(f, 'lambda1 : ' + str(l1))
writelog(f, '---------------')
writelog(f, 'Transformer Setup')
writelog(f, 'hidden_dim : ' + str(d_model))
writelog(f, 'FFN_dim : ' + str(d_ff))
writelog(f, 'num_heads : ' + str(num_heads))
writelog(f, 'num_stacks : ' + str(num_stacks))
writelog(f, '---------------')
writelog(f, 'Loss Setup')
writelog(f, 'cls:'+ str(lambda_1) + ', reg:' + str(lambda_2) +', imp:'+ str(lambda_3))
writelog(f, '---------------')


def test(phase, epoch, test_loader):
    model.eval()
    test_lrp_X = []
    test_lrp_M = []
    test_lrp_D = []
    test_loss = 0.0
    n_batches = 0.0

    y_gts = np.array([]).reshape(0)
    y_gt = np.array([]).reshape(0)
    y_preds = np.array([]).reshape(0)
    y_scores = np.array([]).reshape(0)
    y_prd = np.array([]).reshape(0)
    y_scs = np.array([]).reshape(0)

    for batch_idx, data in enumerate(test_loader):
        x = data['values'].to(device)  # Batch x Time x Variable
        m = data['masks'].to(device)  # Batch x Time x Variable
        deltas = data['deltas'].to(device)  # Batch x Time x Variable
        times = data['times'].to(device)  # Batch x Time x Variable
        y = data['labels'].to(device)
        y1 = data['los_labels'].to(device)

        attn_mask = deltas.data.eq(0)[:, :, 0]
        attn_mask[:, 0] = 0

        y_gts = np.hstack([y_gts, y.to('cpu').detach().numpy().flatten()]) #mimic
        y_gt = np.hstack([y_gt, y.to('cpu').detach().numpy().flatten()])

        # model
        los_out, y_hat, out = model(x, m, times, deltas, attn_mask)

        # LRP score
        model_dict = model.state_dict()
        lrp_score_X, lrp_score_M, lrp_score_D = model.backward_lrp(y_hat, model_dict)
        test_lrp_X.append(lrp_score_X)
        test_lrp_M.append(lrp_score_M)
        test_lrp_D.append(lrp_score_D)


        # Calculate and store the loss
        loss_los = criterion_mse(los_out, y1[:,-1])
        loss_cls = criterion_focal(model, y_hat, y[:,-1])
        loss_imp = criterion_mse(out, x)
        loss = lambda_1*loss_cls + lambda_2*loss_los + lambda_3*loss_imp

        test_loss += loss.item()
        n_batches += 1

        y_score = y_hat
        y_pred = np.round(y_score.to('cpu').detach().numpy())
        y_score = y_score.to('cpu').detach().numpy()
        y_preds = np.hstack([y_preds, y_pred])
        y_scores = np.hstack([y_scores, y_score])

        y_sc = los_out
        y_pr = np.round(y_sc.to('cpu').detach().numpy())
        y_sc = y_sc.to('cpu').detach().numpy()
        y_prd = np.hstack([y_prd, y_pr])
        y_scs = np.hstack([y_scs, y_sc])

        n_batches += 1

    # Averaging the loss
    test_loss /= n_batches
    writelog(f, 'Test loss : ' + str(test_loss))

    rmse = np.sqrt(mean_squared_error(y_gt, y_scs))
    mae = mean_absolute_error(y_gt, y_scs)
    auc, auprc, acc, balacc, sen, spec, prec, recall = calculate_performance(y_gts, y_scores, y_preds)

    attribution_scores_state_dict = dict()
    attribution_scores_state_dict['lrpX'] = test_lrp_X
    attribution_scores_state_dict['lrpM'] = test_lrp_M
    attribution_scores_state_dict['lrpD'] = test_lrp_D
    with open('LRPs.pickle', 'wb') as fw:
        pickle.dump(attribution_scores_state_dict, fw)


    writelog(f, 'AUC : ' + str(auc))
    writelog(f, 'AUC PRC : ' + str(auprc))
    writelog(f, 'Accuracy : ' + str(acc))
    writelog(f, 'BalACC : ' + str(balacc))
    writelog(f, 'Sensitivity : ' + str(sen))
    writelog(f, 'Specificity : ' + str(spec))
    writelog(f, 'Precision : ' + str(prec))
    writelog(f, 'Recall : ' + str(recall))
    writelog(f, 'RMSE : ' + str(rmse))
    writelog(f, 'MAE : ' + str(mae))


    # Tensorboard Logging
    info = {'loss': test_loss,
            'rmse' : rmse,
            'mae' : mae,
            'balacc': balacc,
            'auc': auc,
            'auc_prc': auprc,
            'sens': sen,
            'spec': spec,
            'precision': prec,
            'recall': recall
            }

    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        if phase == 'valid':
            tfw_valid.add_summary(summary, epoch)
        else:
            tfw_test.add_summary(summary, epoch)

    return rmse, mae, auc, auprc, acc, balacc, sen, spec, prec, recall

# Loop for kfold
for k in range(KFold):
    writelog(f, 'FOLD ' + str(k))

    # Tensorboard Logging
    tfw_train = tf.compat.v1.summary.FileWriter(dir + 'tflog/kfold_' + str(k) + '/train_')
    tfw_valid = tf.compat.v1.summary.FileWriter(dir + 'tflog/kfold_' + str(k) + '/valid_')
    tfw_test = tf.compat.v1.summary.FileWriter(dir + 'tflog/kfold_' + str(k) + '/test_')

# Get dataset
    train_data = kfold_data[k][0]
    train_mask = kfold_mask[k][0]
    tr_miss_idx = np.where(train_mask == 0)
    train_data[tr_miss_idx] = 0#np.nan
    train_label = kfold_label[k][0]
    train_label2 = kfold_label2[k][0]
    train_time = kfold_times[k][0]

    test_data = kfold_data[k][2]
    test_mask = kfold_mask[k][2]
    ts_miss_idx = np.where(test_mask == 0)
    test_data[ts_miss_idx] = 0#np.nan
    test_label = kfold_label[k][2]
    test_label2 = kfold_label2[k][2]
    test_time = kfold_times[k][2]

    # Winsorization (2nd-98th percentile)
    writelog(f, 'Winsorization')
    test_data = Winsorize(test_data)
    #
    # Normalization
    writelog(f, 'Normalization')
    train_data, mean_set, std_set = normalize(train_data, train_mask, [], [])
    test_data, m, s = normalize(test_data, test_mask, mean_set, std_set)

    # Define Loaders
    test_loader = msk_sample_loader('test', k, test_data, test_mask, test_label, test_label2, test_time, batch_size,
                                 ZeroImpute=True)

    # Define Model & Optimizer
    criterion_focal = FocalLoss(l1, device, gamma=gamma, alpha=alpha, logits=False).to(device)
    criterion_mse = nn.MSELoss()
    model = Multi_Duration_Pipeline_Residual(input_dim, d_model, d_ff, num_stacks, num_heads, max_length, n_iter=num_stacks).to(device)


    # optimizer = RAdam(list(model.parameters()), lr=lr, weight_decay=w_decay)
    optimizer = optim.Adam(list(model.parameters()), lr=lr, betas=(0.9, 0.98), weight_decay=w_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=lr_ratio)

    # Reset Best AUC
    bestRMse = 530
    bestValidAUC = 0
    best_epoch = 0

    # Training, Validation, Test Loop
    for i in enumerate([45, 7, 13, 4, 12]):
        # Load Best Validation
        best_dir = './log/observation_mask_multi_encoder_22.37.57/model/'
        vrnn_best_model = torch.load(best_dir + str(i[0]) + '/' + str(i[-1]) + '_self_attention.pt')
        writelog(f, 'Final Test')
        rmse, mae, auc, auprc, acc, balacc, sen, spec, prec, recall = test('test', i[-1], test_loader)

    kfold_mse.append(rmse)
    kfold_mae.append(mae)
    kfold_auc.append(auc)
    kfold_auprc.append(auprc)
    kfold_acc.append(acc)
    kfold_balacc.append(balacc)
    kfold_sen.append(sen)
    kfold_spec.append(spec)
    kfold_precision.append(prec)
    kfold_recall.append(recall)
writelog(f, '---------------')
writelog(f, 'SUMMARY OF ALL KFOLD')

mean_rmse = round(np.mean(kfold_mse), 5)
std_rmse = round(np.std(kfold_mse), 5)

mean_mae = round(np.mean(kfold_mae), 5)
std_mae = round(np.std(kfold_mae), 5)

mean_acc = round(np.mean(kfold_acc), 5)
std_acc = round(np.std(kfold_acc), 5)

mean_auc = round(np.mean(kfold_auc), 5)
std_auc = round(np.std(kfold_auc), 5)

mean_auc_prc = round(np.mean(kfold_auprc), 5)
std_auc_prc = round(np.std(kfold_auprc), 5)

mean_sen = round(np.mean(kfold_sen), 5)
std_sen = round(np.std(kfold_sen), 5)

mean_spec = round(np.mean(kfold_spec), 5)
std_spec = round(np.std(kfold_spec), 5)

mean_precision = round(np.mean(kfold_precision), 5)
std_precision = round(np.std(kfold_precision), 5)

mean_recall = round(np.mean(kfold_recall), 5)
std_recall = round(np.std(kfold_recall), 5)

mean_balacc = round(np.mean(kfold_balacc), 5)
std_balacc = round(np.std(kfold_balacc), 5)


writelog(f, 'RMSE : ' + str(mean_rmse) + ' + ' + str(std_rmse))
writelog(f, 'MAE : ' + str(mean_mae) + ' + ' + str(std_mae))
writelog(f, 'AUC : ' + str(mean_auc) + ' + ' + str(std_auc))
writelog(f, 'AUC PRC : ' + str(mean_auc_prc) + ' + ' + str(std_auc_prc))
writelog(f, 'Accuracy : ' + str(mean_acc) + ' + ' + str(std_acc))
writelog(f, 'BalACC : ' + str(mean_balacc) + ' + ' + str(std_balacc))
writelog(f, 'Sensitivity : ' + str(mean_sen) + ' + ' + str(std_sen))
writelog(f, 'Specificity : ' + str(mean_spec) + ' + ' + str(std_spec))
writelog(f, 'Precision : ' + str(mean_precision) + ' + ' + str(std_precision))
writelog(f, 'Recall : ' + str(mean_recall) + ' + ' + str(std_recall))
writelog(f, '---------------------')
writelog(f, 'END OF CROSS VALIDATION TRAINING')
f.close()
torch.cuda.empty_cache()
