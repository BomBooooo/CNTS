import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from exp.exp_main_impdet import Exp_Main
import random
import numpy as np
import time
import logging

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model', type=str, default='Combined',
                    help='model name, options: [Imputation, Detection, DetectionSupervised, Combined]')

# supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Fourier',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=32, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

# data loader
parser.add_argument('--data', type=str, default='NASA-MSL', help='dataset type, options:[NASA-MSL, NASA-SMAP, NAB]')
parser.add_argument('--root_path', type=str, default='./data/Yahoo/', help='root path of the data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                        'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                        'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--i_out', type=int, default=1, help='imputation output size')
parser.add_argument('--d_out', type=int, default=1, help='detection output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', 
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=10, help='experiments times')
parser.add_argument('--train_epochs_det', type=int, default=50, help='train epochs')
parser.add_argument('--train_epochs_imp', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate_det', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--learning_rate_imp', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--det_loss_type', type=str, default='random', help='det loss, options: [random, det]')
parser.add_argument('--missing_rate', type=float, default=0.1, help='missing rate')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')
parser.add_argument('--start_time', type=str, default=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
                    help='start training time')
parser.add_argument('--des', type=str, default='test1', help='exp description')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
# print(args)

# setting record of experiments
setting = '{}_itr{}_{}_{}_bz{}_lrd{}_lri{}_mis{}_{}_sl{}_dm{}_lrad{}_{}'.format(
    args.data,
    args.itr,
    args.train_epochs_det,
    args.train_epochs_imp,
    args.batch_size,
    args.learning_rate_det, 
    args.learning_rate_imp, 
    args.missing_rate, 
    args.model, 
    args.seq_len, 
    args.d_model, 
    args.lradj, 
    args.des, 
    )

# create logger
logger = logging.getLogger("train_log")
logger.setLevel(logging.INFO)
# create file handler
# logger_path = "./logs/%s" % setting
logger_path = os.path.join('./figs', '%s_%s'%(args.start_time, setting))
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
fh = logging.FileHandler(logger_path+'/train_log.log')
fh.setLevel(logging.INFO)
# create formatter
fmt = "%(asctime)-15s %(message)s"
formatter = logging.Formatter(fmt)
# add handler and formatter to logger
fh.setFormatter(formatter)
logger.addHandler(fh)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

logger.info(setting)

args.data_name = os.listdir(f'./process/{args.data}')
args.data_name.sort()
data_num = len(args.data_name)
for args.idx in range(0, data_num):
    logger.info('>>>>>>>start idx : {}>>>>>>>\n'.format(args.data_name[args.idx]))
    # for args.idx in range(21, 22):
    # args.idx = 0
    start = time.clock()
    Exp = Exp_Main
    exp = Exp(args, logger, logger_path)  # set experiments

    for ii in range(args.itr):
        if args.model == 'Combined':
            logger.info('[>>>>>>>start itr : {}>>>>>>>]'.format(ii))
            exp.train_combined(ii, setting, 'imp')    # 0.2182 0.2850
            exp.train_combined(ii, setting, 'det')    # 0.6266 0.5410
            # exp.train_combined(ii, setting, 'imp')  # 0.2182 0.2850
        elif args.model == 'DetectionSupervised':
            exp.train_supervised(setting)
        elif args.model == 'Detection':
            exp.train_alone(setting, 'det')
            # [DET: 0.9820, 0.5387, 0.5025, 0.5805]
            # [DET: 0.9752, 0.4465, 0.4068, 0.4948]
        elif args.model == 'Imputation':
            exp.train_alone(setting, 'imp') # 0.0550  0.0630
            # Vali Normal Loss: 0.0057 Test Normal Loss: 0.0063
            # Vali Abnormal Loss: 0.1417 Test Abnormal Loss: 0.1220
        torch.cuda.empty_cache()

    end = time.clock()
    total_time = time.strftime("%H:%M:%S", time.gmtime(end-start))
    best_results = "det f1: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}] \nimp mse: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
        *exp.metrics_det[:-1], *exp.metrics_imp)
    logger.info(best_results)
    logger.info('_______Finish________\n')

