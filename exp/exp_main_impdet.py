
import os
import time
from tkinter.messagebox import NO
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from data_provider.data_factory_own import data_provider
from exp.exp_basic import Exp_Basic
from models import Imputation, Detection
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.loss_func import LossFunc
from utils.plot import PlotResult
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args, logger, logger_path):
        super(Exp_Main, self).__init__(args)
        self.logger = logger
        self.logger_path = logger_path
        self.imp_loss = np.inf
        self.det_loss = np.inf
        self.stage_r = []
        self.stage_d = []
        self.metrics_imp = (np.inf, np.inf, np.inf, np.inf)
        self.metrics_det = (0, 0, 0, 0, 0, 0)
        self.criterion = self._select_criterion()
        self.plot = PlotResult(logger_path, args)
        
    def _build_model(self):
        assert self.args.model in ['Imputation', 'Detection', 'DetectionSupervised', 'Combined']
        if self.args.model == 'Imputation':
            model_imp = Imputation.Model(self.args).float()
            return model_imp.to(self.device)
        elif self.args.model == 'Detection' or self.args.model == 'DetectionSupervised':
            model_det = Detection.Model(self.args).float()
            return model_det.to(self.device)
        elif self.args.model == 'Combined':
            model_det = Detection.Model(self.args).float()
            model_imp = Imputation.Model(self.args).float()
            self.load_data()
            return model_det.to(self.device), model_imp.to(self.device)
        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
    
    def load_data(self):
        self.train_data, self.train_loader = self._get_data(flag='train', task='det')
        # self.vali_data, self.vali_loader = self._get_data(flag='vali', task='det')
        self.test_data, self.test_loader = self._get_data(flag='test', task='det')

    def _get_data(self, flag, task='imp'):
        data_set, data_loader = data_provider(self.args, flag, task)
        return data_set, data_loader

    def _select_optimizer_det(self):
        model_optim = optim.Adam(self.model_det.parameters(), lr=self.args.learning_rate_det)
        return model_optim

    def _select_optimizer_imp(self):
        model_optim = optim.Adam(self.model_imp.parameters(), lr=self.args.learning_rate_imp)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        return criterion

    def det_metrics(self, dets, batch_y=None, threshold_list=None):
        threshold_list = torch.Tensor(threshold_list)
        det = dets.reshape(-1)
        batch_y = batch_y.reshape(-1)
        batch_y[batch_y==-1] = 0
        best_f1, best_acc, best_p, best_r, best_threshold = 0, 0, 0, 0, 0
        for threshold in threshold_list:
            pred = (det > threshold).float()
            tn = torch.sum((1 - batch_y) * (1 - pred))
            fp = torch.sum((1 - batch_y) * pred)
            tp = torch.sum(batch_y * pred)
            fn = torch.sum(batch_y * (1 - pred))

            acc = ((tn + tp) / (tn + tp + fn + fp)).cpu().numpy()
            p = (tp / (tp + fp + 1e-5)).cpu().numpy()
            r = (tp / (tp + fn + 1e-5)).cpu().numpy()
            f1 = ((2 * p * r) / (p + r + 1e-5))
            if f1 > best_f1 or best_f1 == 0:
                fpr, tpr, _ = roc_curve(batch_y, det + 1, pos_label=1)
                best_auc = auc(fpr, tpr)
                best_f1 = f1
                best_acc = acc
                best_p = p
                best_r = r
                best_threshold = threshold
        if self.args.model == 'Combined':
            return np.array((best_acc, best_f1, best_p, best_r, best_auc))
        else:
            return np.array((best_acc, best_f1, best_p, best_r, best_auc, best_threshold))

    def pre_threshold(self, ab_score, batch_y):
        ab_score = np.array(ab_score)
        batch_y = np.array(batch_y)
        n = ab_score.shape[0]
        batch_y[batch_y==-1] = 0
        positive = np.sum(batch_y)
        if positive == 0:
            return np.array((1, 1, 1, 1, 1, np.max(ab_score)))
        else:
            negative = n - positive
            best_f1, best_acc = 0, 0
            fpr, tpr, thresholds = roc_curve(batch_y, ab_score)
            for fpr_, tpr_, threshold in zip(fpr, tpr, thresholds):
                tp = tpr_ * positive
                fp = fpr_ * negative
                fn = positive - tp
                tn = negative - fp
                precision = tp / (tp + fp + 1e-5)
                recall = tp / (tp + fn + 1e-5)
                acc = (tp + tn) / n
                f1 = (2 * precision * recall) / (precision + recall + 1e-5)
                f1 = 0 if np.isnan(f1) else f1
                acc = 0 if np.isnan(acc) else acc
                precision = 0 if np.isnan(precision) else precision
                # recall = 0 if np.isnan(recall) else recall
                # cond = (f1 >= best_f1 and acc >= best_acc)
                cond = (acc >= best_acc)
                if f1 > best_f1 or cond:
                    best_auc = auc(fpr, tpr)
                    best_acc = acc
                    best_f1 = f1
                    best_p = precision
                    best_r = recall
                    best_threshold = threshold

            return np.array((best_acc, best_f1, best_p, best_r, best_auc, best_threshold))

    def test_combined(self, vali_data, vali_loader, name, task):
        self.model_det.eval()
        self.model_imp.eval()

        if task == 'det':
            loss_func = self.loss_func.det_vali
        elif task == 'imp':
            loss_func = self.loss_func.imp_test

        plot_dict = {'x': [], 'score': [], 'label': [], 
                    'y': [], 'imp': [], 'loss_mask': [],
                    'imp_input': [], 'abnormal_loss': [], 
                    'imp_index': [], 'normal_loss': [],
                    'loss': []}
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                loss, dets, imps = loss_func(batch_x, batch_y)
                total_loss.append(loss.item())
                if task == 'imp':
                    plot_dict['imp_index'] += imps['imp_index'].reshape(-1).tolist()
                    plot_dict['abnormal_loss'] += imps['abnormal_loss'].reshape(-1).tolist()
                    plot_dict['normal_loss'] += imps['normal_loss'].reshape(-1).tolist()
                elif task == 'det':
                    plot_dict['score'] += dets['score'].reshape(-1).tolist()
                    plot_dict['label'] += dets['label'].reshape(-1).tolist()
                    plot_dict['loss'] += dets['loss'].reshape(-1).tolist()
                    plot_dict['loss_mask'] += imps['loss_mask'].reshape(-1).tolist()
                plot_dict['imp_input'] += imps['imp_input'].reshape(-1).tolist()
                plot_dict['imp'] += imps['imp'].reshape(-1).tolist()
                plot_dict['x'] += batch_x.reshape(-1).tolist()
                plot_dict['y'] += batch_y.reshape(-1).tolist()
                
        if task == 'det':
            total_loss = np.nanmean(plot_dict['loss'])
            self.model_det.train()
            self.model_imp.eval()
            metrics = self.pre_threshold(torch.Tensor(plot_dict['score']), torch.Tensor(plot_dict['y']))
            plot_dict['threshold'] = metrics[-1]
            self.plot.plot_solo_d(self.itr, self.epoch, plot_dict, total_loss, name, task)
        elif task == 'imp':
            total_std = np.nanstd(plot_dict['normal_loss'])
            total_loss = np.nanmean(plot_dict['normal_loss'])
            self.model_det.eval()
            self.model_imp.train()
            metrics_std = np.nanstd(plot_dict['abnormal_loss'])
            metrics = np.nanmean(plot_dict['abnormal_loss'])
            metrics = (total_loss, total_std, metrics, metrics_std)
            total_loss = np.nanmean(loss.cpu().numpy())
            self.plot.plot_solo_i(self.itr, self.epoch, plot_dict, total_loss, name, task)

        return total_loss, metrics

    def train_combined(self, itr, setting, task):
        self.setting = setting
        self.itr = itr
        self.loss_func = LossFunc(self.args, self.model_det, self.model_imp, device=self.device, criterion=self.criterion)
        self.logger.info('\n>>>>>>>start training : {}>>>>>>>'.format(task))
        path = os.path.join(self.args.checkpoints, '%s_%s'%(self.args.start_time, setting))
        if not os.path.exists(path):
            os.makedirs(path)

        best_model_path = path + '/' + '%s_checkpoint.pth'%task
        if itr != 0:
            if task == 'det':
                self.model_det.load_state_dict(torch.load(best_model_path))
            elif task == 'imp':
                self.model_imp.load_state_dict(torch.load(best_model_path))
        # else:
        #     if task == 'det':
        #         self.model_imp.load_state_dict(torch.load(
        #             './checkpoints/2022-10-09 15:40:40_NeurIPS-TS_itr10_20_20_bz256_lrd0.0001_lri0.0001_mis(0.2, 0.8)_Combined_sl100_dm512_lradtype1_init I/imp_checkpoint.pth'
        #         ))
        
        if task == 'det':
            early_stopping = EarlyStopping(val_loss_min=self.metrics_det[1], patience=self.args.patience, verbose=True, 
                                            task=task, logger=self.logger)
            train_epochs = self.args.train_epochs_det
            model_optim = self.optimizer_det
            loss_func = self.loss_func.det_loss
            self.model_det.train()
            self.model_imp.eval()

        elif task == 'imp':
            early_stopping = EarlyStopping(val_loss_min=self.metrics_imp[0], patience=self.args.patience, verbose=True, 
                                            task=task, logger=self.logger)
            train_epochs = self.args.train_epochs_imp
            model_optim = self.optimizer_imp
            loss_func = self.loss_func.imp_loss
            self.model_det.eval()
            self.model_imp.train()

        train_data, train_loader = self.train_data, self.train_loader
        test_data, test_loader = self.test_data, self.test_loader
        train_steps = len(train_loader)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for self.epoch in range(train_epochs):
            iter_count = 0
            train_loss = []
            # epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                loss, dets, imps = loss_func(batch_x)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)

            test_loss, test_metrics = self.test_combined(test_data, test_loader, name='vali', task=task)

            if task == 'imp':
                self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.4f}".format(
                                self.epoch + 1, train_steps, train_loss))
                self.logger.info("Test Loss: {:.4f} std: {:.4f} | Test Abnormal Loss: {:.4f} std: {:.4f}\n".format(
                            test_metrics[0], test_metrics[1], test_metrics[-2], test_metrics[-1]))
                
                if test_metrics[0] < self.metrics_imp[0]:
                    self.loss_imp = test_loss
                    self.metrics_imp= test_metrics
                    self.args.metrics_mse = test_metrics
                    self.stage_r.append(self.metrics_imp[0])
                early_stopping(self.metrics_imp[0], self.model_imp, path)
                adjust_learning_rate(model_optim, self.epoch + 1, self.args, self.args.learning_rate_imp)

            elif task == 'det':
                self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Test Loss: {3:.4f}".format(
                                self.epoch + 1, train_steps, train_loss, test_loss))
                self.logger.info('             acc      f1       p       r     auc' )
                self.logger.info('test: [DET: %.4f, %.4f, %.4f, %.4f, %.4f] ||  [%.4f]\n' % (*test_metrics, ))
                cond = self.metrics_det[1] == test_metrics[1] and test_metrics[-2] > self.metrics_det[-2]
                if test_metrics[1] > self.metrics_det[1] or cond:
                    self.loss_det = test_loss
                    self.metrics_det = test_metrics
                    self.stage_d.append(self.metrics_det[1])
                early_stopping((test_metrics[1], test_metrics[-2]), self.model_det, path)
                adjust_learning_rate(model_optim, self.epoch + 1, self.args, self.args.learning_rate_det)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        if task == 'det':
            self.model_det.load_state_dict(torch.load(best_model_path))
        elif task == 'imp':
            self.model_imp.load_state_dict(torch.load(best_model_path))

        return self.model_det, self.model_imp

    def test_alone(self, vali_data, vali_loader, criterion, name, task):
        if task == 'det':
            self.model_det.eval()
            loss_func = self.loss_func.det_loss_alone
        elif task == 'det_supervised':
            self.model_det.eval()
            loss_func = self.loss_func.det_supervised_loss
        elif task == 'imp':
            self.model_imp.eval()
            loss_func = self.loss_func.imp_test_alone
        total_loss = []
        plot_dict = {'x': [], 'output': [], 'y': [],  'abnormal_loss': [], 
                    'normal_loss': [], 'reconstruction': [], 'loss': []
                    }

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y = batch
                batch_x = batch_x.float().to(self.device)

                if task == 'det':
                    loss, outputs, ab_score = loss_func(batch_x, batch_y)
                    plot_dict['reconstruction'] += outputs.reshape(-1).tolist()
                    plot_dict['output'] += ab_score.reshape(-1).tolist()
                    
                elif task == 'det_supervised':
                    batch_y = batch_y.float().to(self.device)
                    loss, ab_score = loss_func(batch_x, batch_y)
                    plot_dict['loss'] += loss.reshape(-1).tolist()
                    plot_dict['output'] += ab_score.reshape(-1).tolist()

                elif task == 'imp':
                    batch_y = batch_y.float().to(self.device)
                    loss, outputs, abnormal_loss = loss_func(batch_x, batch_y)
                    plot_dict['abnormal_loss'] += abnormal_loss.reshape(-1).tolist()
                    plot_dict['normal_loss'] += loss.reshape(-1).tolist()
                    plot_dict['output'] += outputs.reshape(-1).tolist()

                plot_dict['x'] += batch_x.reshape(-1).tolist()
                plot_dict['y'] += batch_y.reshape(-1).tolist()

        if task == 'det':
            total_loss = np.average(plot_dict['reconstruction'])
            self.model_det.train()
            metrics = self.pre_threshold(plot_dict['output'], plot_dict['y'])

        elif task == 'imp':
            total_loss = np.average(plot_dict['normal_loss'])
            normal_std = np.std(plot_dict['normal_loss'])
            self.model_imp.train()
            abnormal = np.nanmean(plot_dict['abnormal_loss'])
            abnormal_std = np.std(plot_dict['abnormal_loss'])
            metrics = (total_loss, normal_std, abnormal, abnormal_std)
        
        return total_loss, metrics
        
    def train_alone(self, setting, task):
        
        if task == 'imp':
            self.loss_func = LossFunc(self.args, model_imp=self.model_imp, device=self.device, criterion=nn.MSELoss())
            self.model_imp.train()
            early_stopping = EarlyStopping(val_loss_min=self.metrics_imp[0], patience=self.args.patience, 
                                            verbose=True, task=task, logger=self.logger)
            train_epochs = self.args.train_epochs_imp
            model_optim = self.optimizer_imp
            loss_func = self.loss_func.imp_loss_alone
        elif task == 'det':
            self.loss_func = LossFunc(self.args, model_det=self.model_det, device=self.device, criterion=nn.MSELoss())
            self.model_det.train()
            early_stopping = EarlyStopping(val_loss_min=self.metrics_det[1], patience=self.args.patience, 
                                            verbose=True, task=task, logger=self.logger)
            train_epochs = self.args.train_epochs_det
            model_optim = self.optimizer_det
            loss_func = self.loss_func.det_loss_alone

        self.setting = setting
        train_data, train_loader = self._get_data(flag='train', task=task)
        test_data, test_loader = self._get_data(flag='test', task=task)

        path = os.path.join(self.args.checkpoints, '%s_%s'%(self.args.start_time, setting))
        if not os.path.exists(path):
            os.makedirs(path)
        best_model_path = path + '/' + '%s_checkpoint.pth' % task

        train_steps = len(train_loader)
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for self.epoch in range(train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch.float().to(self.device)
                if task == 'det':
                    loss, outputs, ab_score = loss_func(batch_x)
                elif task == 'imp':
                    loss, outputs = loss_func(batch_x)
                
                train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            train_loss = np.average(train_loss)
            test_loss, test_metrics = self.test_alone(test_data, test_loader, self.criterion, name='test', task=task)

            if task == 'imp':
                self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.4f}".format(
                                self.epoch + 1, train_steps, train_loss))
                self.logger.info("Test Loss: {:.4f} std: {:.4f} | Test Abnormal Loss: {:.4f} std: {:.4f}\n".format(
                            test_metrics[0], test_metrics[1], test_metrics[-2], test_metrics[-1]))
                if test_loss < self.metrics_imp[0]:
                    self.metrics_imp = test_metrics
                early_stopping(self.metrics_imp[0], self.model_imp, path)
            elif task == 'det':
                self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                                self.epoch + 1, train_steps, train_loss, test_loss))
                self.logger.info('             acc      f1       p       r     auc  ||   threshold' )
                self.logger.info('test: [DET: %.4f, %.4f, %.4f, %.4f, %.4f] ||  [%.4f]\n' % (*test_metrics, ))
                cond = self.metrics_det[1] == 0 and test_metrics[-2] > self.metrics_det[-2]
                if test_metrics[1] > self.metrics_det[1] or cond:
                    self.metrics_det = test_metrics
                early_stopping((test_metrics[1], test_metrics[-2]), self.model_det, path)

            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break
        
        if task == 'det':
            self.model_det.load_state_dict(torch.load(best_model_path))
            return self.model_det
        elif task == 'imp':
            self.model_imp.load_state_dict(torch.load(best_model_path))
            return self.model_imp

