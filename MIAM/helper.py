import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.optim.optimizer import Optimizer

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn import metrics
from scipy.stats.mstats import winsorize
import random

# Define the function to print and write log
def writelog(file, line):
    file.write(line + '\n')
    print(line)

def Winsorize(data):
    n_patients = data.shape[0]
    n_hours = data.shape[1]
    n_variables = data.shape[2]
    mask = ~np.isnan(data) * 1
    mask = mask.reshape(n_patients * n_hours, n_variables)
    measure = data.copy().reshape(n_patients * n_hours, n_variables)
    measure_orig = data.copy().reshape(n_patients * n_hours, n_variables)

    for v in range(n_variables):
       idx = np.where(mask[:,v] == 1)[0]
       if len(idx) > 0:
           limit = 0.02
           measure[:, v][idx] = winsorize(measure[:, v][idx], limits=limit)

    normalized_data = measure.reshape(n_patients, n_hours, n_variables)
    return normalized_data


# Normalization
def normalize(data, data_mask, mean, std):
    n_patients = data.shape[0]
    n_hours = data.shape[1]
    n_variables = data.shape[2]

    mask = data_mask.copy().reshape(n_patients * n_hours, n_variables)
    measure = data.copy().reshape(n_patients * n_hours, n_variables)

    isnew = 0
    if len(mean) == 0 or len(std) == 0:
        isnew = 1
        mean_set = np.zeros([n_variables])
        std_set = np.zeros([n_variables])
    else:
        mean_set = mean
        std_set = std
    for v in range(n_variables):
        idx = np.where(mask[:, v] == 1)[0]

        if idx.sum() == 0:
            continue

        if isnew:
            measure_mean = np.mean(measure[:, v][idx])
            measure_std = np.std(measure[:, v][idx])

            # Save the Mean & STD Set
            mean_set[v] = measure_mean
            std_set[v] = measure_std
        else:
            measure_mean = mean[v]
            measure_std = std[v]

        for ix in idx:
            if measure_std != 0:
                measure[:, v][ix] = (measure[:, v][ix] - measure_mean) / measure_std
            else:
                measure[:, v][ix] = measure[:, v][ix] - measure_mean

    normalized_data = measure.reshape(n_patients, n_hours, n_variables)

    return normalized_data, mean_set, std_set


# Positive Label Boosting
def boost_positive_labels(data, mask, label):

    # Get Positive Data
    pos_idx = np.where(label == 1)
    pos_data = data[pos_idx]
    pos_mask = mask[pos_idx]
    pos_label = label[pos_idx]
    pos_n = pos_data.shape[0]

    # Get Negative Data
    neg_idx = np.where(label == 0)
    neg_data = data[neg_idx]
    neg_mask = mask[neg_idx]
    neg_label = label[neg_idx]
    neg_n = neg_data.shape[0]

    # Boosting
    ridx = np.random.randint(pos_n, size=neg_n)
    data = np.vstack([neg_data, pos_data[ridx]])
    mask = np.vstack([neg_mask, pos_mask[ridx]])
    label = np.hstack([neg_label, pos_label[ridx]])

    return data, mask, label


# Define Weighted Sample Loader
def weighted_sample_loader(data, data_mask, label, batch_size):
    # Calculate weights
    class_sample_count = np.array([len(np.where(label == t)[0]) for t in np.unique(label)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in label])

    # Define sampler using WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    # Concatenate data and its mask
    d = np.expand_dims(data, axis=0)
    dm = np.expand_dims(data_mask, axis=0)
    alldata = np.vstack([d, dm]).transpose(1, 0, 2, 3)

    # Convert to tensor
    data = torch.tensor(alldata).float()
    label = torch.tensor(label).float()

    # Define the loader
    dataset = torch.utils.data.TensorDataset(data, label)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=1,
                        sampler=sampler,
                        pin_memory=True)
    return loader

def parse_delta(masks):

    deltas = []

    for h in range(48):
        if h == 0:
            deltas.append(np.ones(35))
        else:
            deltas.append(np.ones(35) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)

def collate_fn(recs):
    rec_dict = {'values': torch.FloatTensor(np.array([r['values'] for r in recs])),
                'masks': torch.FloatTensor(np.array([r['masks'] for r in recs])),
                'deltas': torch.FloatTensor(np.array([r['deltas'] for r in recs])),
                'times': torch.FloatTensor(np.array([r['times'] for r in recs])),
                'labels': torch.FloatTensor(np.array([r['labels'] for r in recs]))
                }
    return rec_dict

# Define Sample Loader
def sample_loader(phase, k, data, mask, label, time, batch_size, ZeroImpute):
    # Random seed
    manualSeed = 128
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    random.seed(manualSeed)

    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    [N, T, D] = data.shape
    ori_time = time / 60

    recs = []
    times = np.zeros((N, T, D))
    if D ==35:
        times = np.load(open('./Data/kfold_delta_35_' + str(k) + '_' + phase + '.npz', 'rb'), mmap_mode='r', allow_pickle=True)['delta'] #physionet #/home/yrlee/MultiMHA/Data/kfold_delta_35_
    else:
        times = np.load(open('./Data/kfold_delta_mimic_' + str(k) + '_' + phase + '.npz', 'rb'), mmap_mode='r', allow_pickle=True)['delta'] #mimic
    # times = np.load(open('./Data/kfold_delta_mimic_' + str(k) + '_' + phase + '.npz', 'rb'), mmap_mode='r', allow_pickle=True)['delta'] #mimic


    for i in range(N):
        # print(str(i)+'th subject')
        values = data[i]
        masks = mask[i]

        ##############################################################
        # for d in range(D):
        #     app_time = []
        #     app_time.append(0)
        #
        #     for t in range(1,np.where(ori_time[i]==max(ori_time[i]))[0][0]+1):
        #         if masks[t-1,d] == True:
        #             times[i, t, d] = ori_time[i][t] - ori_time[i][t - 1]
        #         elif masks[t-1,d] == False:
        #             times[i, t, d] = ori_time[i][t] - ori_time[i][t - 1] + app_time[-1]
        #         app_time.append(times[i, t, d])
        # ##############################################################


        rec = {}
        rec['labels'] = label[i]
        if ZeroImpute:
            values[np.isnan(values)] = 0
            rec['values'] = np.nan_to_num(values).tolist()
            rec['masks'] = masks.astype('int32').tolist()
            rec['deltas'] = times[i].tolist()
            rec['times'] = ori_time[i].tolist()
        else:
            rec['values'] = values.tolist()
            rec['masks'] = masks.astype('int32').tolist()
            rec['deltas'] = times[i].tolist()
            rec['times'] = ori_time[i].tolist()

        recs.append(rec)

    # print(phase, 'save!!!!!')
    # np.savez('./Data/kfold_delta_35_' + str(k) + '_' + phase + '.npz', delta=times, allow_pickle=True)

    loader = torch.utils.data.DataLoader(recs,
                                         batch_size=batch_size,
                                         # num_workers=1,
                                         shuffle=True,
                                         pin_memory=True,
                                         collate_fn=collate_fn)
    return loader



def msk_sample_loader(phase, k, data, mask, msk_data, mmsk, label, time, batch_size, ZeroImpute):#data, data_mask, org, label, batch_size):
    # Random seed
    manualSeed = 128
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    random.seed(manualSeed)

    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    [N, T, D] = data.shape
    ori_time = time / 60

    recs = []
    # times = np.zeros((N, T, D))
    if D ==35:
        times = np.load(open('./Data/kfold_delta_35_' + str(k) + '_' + phase + '.npz', 'rb'), mmap_mode='r', allow_pickle=True)['delta'] #physionet
    else:
        times = np.load(open('./Data/kfold_delta_mimic_' + str(k) + '_' + phase + '.npz', 'rb'), mmap_mode='r', allow_pickle=True)['delta'] #mimic


    for i in range(N):
        # print(str(i)+'th subject')
        vals = np.expand_dims(data[i], axis=0)
        mskval = np.expand_dims(msk_data[i], axis=0)
        values = np.vstack([vals, mskval]).transpose(1,2,0)
        msk = np.expand_dims(mask[i], axis=0)
        mmask = np.expand_dims(np.tile(np.expand_dims(mmsk[i], axis=1), 35), axis=0)#mmsk[i]
        masks = np.vstack([msk, mmask]).transpose(1,2,0)

        rec = {}
        rec['labels'] = label[i]
        if ZeroImpute:
            values[np.isnan(values)] = 0
            rec['values'] = np.nan_to_num(values).tolist()
            # rec['maskedval'] = np.nan_to_num(mskval).tolist()
            rec['masks'] = masks.astype('int32').tolist()
            # rec['randmask'] = mmask.astype('int32').tolist()
            rec['deltas'] = times[i].tolist()
            rec['times'] = ori_time[i].tolist()
        else:
            rec['values'] = values.tolist()
            # rec['maskedval'] = mskval.tolist()
            rec['masks'] = masks.astype('int32').tolist()
            # rec['randmask'] = mmask.astype('int32').tolist()
            rec['deltas'] = times[i].tolist()
            rec['times'] = ori_time[i].tolist()

        recs.append(rec)

    # print(phase, 'save!!!!!')
    # np.savez('./Data/kfold_delta_35_' + str(k) + '_' + phase + '.npz', delta=times, allow_pickle=True)

    loader = torch.utils.data.DataLoader(recs,
                                         batch_size=batch_size,
                                         num_workers=1,
                                         shuffle=True,
                                         pin_memory=True,
                                         collate_fn=collate_fn)

    return loader


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")



def calculate_performance(y, y_score, y_pred):
    # Calculate Evaluation Metrics
    acc = accuracy_score(y_pred, y)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    # total = tn + fp + fn + tp
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        p, r, t = precision_recall_curve(y, y_score)
        auprc = np.nan_to_num(metrics.auc(r, p))
    spec = np.nan_to_num(tn / (tn + fp))
    # acc = ((tn + tp) / total) * 100
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    try:
        auc = roc_auc_score(y, y_score)
    except ValueError:
        auc = 0

    return auc, auprc, acc, balacc, sen, spec, prec, recall


class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose
    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('Training process stopped early')
                return True
        else:
            self._step = 0
            self._loss = loss
        return False


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

class FocalLoss(nn.Module):
    def __init__(self,  lambda1, device, alpha=1, gamma=0, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.device = device
        self.lambda1 = torch.tensor(lambda1).to(device)

    def forward(self, model, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-1*BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        # Regularization
        l1_regularization = torch.tensor(0).float().to(self.device)
        for param in model.parameters():
            l1_regularization += torch.norm(param.to(self.device), 1)

        # Take the average
        loss = torch.mean(F_loss) + (self.lambda1 * l1_regularization)

        return loss


class WeightedBCE(nn.Module):
    def __init__(self,  device):
        super(WeightedBCE, self).__init__()
        self.device = device

    def forward(self, model, inputs, targets):
        inputs = inputs.detach().cpu()
        targets = targets.detach().cpu()

        pos_num = len(np.where(targets == 1)[0])
        neg_num = len(np.where(targets == 0)[0])
        if pos_num == 0:
            pos_weight = 1.0
        else:
            pos_weight = neg_num / pos_num
        weights = torch.zeros(len(targets))

        for i in range(len(targets)):
            if i == 1:
                weights[i] = pos_weight
            else:
                weights[i] = 1.0

        loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=weights)

        return loss.to(self.device)

def random_mask(data):
    data = torch.tensor(data)
    ground_truth_dataset = []
    masked_dataset = []
    masking = np.zeros([data.size(0), data.size(1)]) # mask vector; 0 if masking, 1 otherwise

    for i in range(len(data)):
        prob = 0.1
        idx = torch.randperm(data.size(1))[:round(data.size(1) * prob)]
        masked_data = data[i].clone()
        masked_data[idx, :] = 0

        masked_dataset.append(masked_data.tolist())
        ground_truth_dataset.append(data[i].tolist())
        masking[i, idx] = 1

    masked_dataset = np.asarray(masked_dataset, dtype='double')
    ground_truth_dataset = np.asarray(ground_truth_dataset)  # ground truth
    # mask_dt = (inputs_dt != 0)*1
    return masked_dataset, ground_truth_dataset, masking

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out
