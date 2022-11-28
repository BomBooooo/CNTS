from ast import arg
from unicodedata import decimal
import torch 
import numpy as np
import copy
import torch.nn as nn


class LossFunc():
    def __init__(self, args, model_det=None, model_imp=None, criterion=None, device=None):
        self.args = args
        self.model_det = model_det
        self.model_imp = model_imp
        self.device = device
        self.criterion = criterion
        self.SM = torch.nn.Softmax(dim=1)
        self.criterion_imp = nn.MSELoss()

    def cal_labels(self, det_input, batch_x):
        labels, _ = torch.max(torch.square(det_input - batch_x), dim=-1, keepdim=True)
        # if not np.isnan(self.args.metrics_mse[2]):
        #     thre = self.args.metrics_mse[2] + self.args.metrics_mse[3]
        #     # thre = 20
        #     labels[labels>thre] = thre
        # max_value = torch.max(labels, dim=1, keepdim=True)[0]
        # labels /= max_value
        labels = labels.cpu()
        # zero_point = (self.args.metrics_mse[2] - self.args.metrics_mse[0]) / 2 + self.args.metrics_mse[0]
        # zero_point = self.args.alpha_low
        # labels = 2 * torch.tanh(labels * 0.5494 / zero_point) - 1
        # ab_num = torch.sum((labels > 0).int())
        # ab_num = 1 if ab_num==0 else ab_num
        # labels_ = labels[:, 1:]
        # _labels = labels[:, :-1]
        # _labels[_labels<labels_] = labels_[_labels<labels_]
        # labels[:, :-1] = _labels
        # labels_ = labels[:, 1:]
        # labels_[labels_<_labels] = _labels[labels_<_labels]
        # labels[:, 1:] = labels_

        loss_mask = torch.zeros_like(labels)
        _, top_ind = torch.topk(labels, int(self.args.seq_len * 0.2), dim=1, largest=True)
        # _, low_ind = torch.topk(labels, int(self.args.seq_len * 0.2), dim=1, largest=False)
        loss_mask.scatter_(1, torch.LongTensor(top_ind), 1)
        # loss_mask.scatter_(1, torch.LongTensor(low_ind), 1)
        # loss_mask[top_ind] = 1
        # loss_mask[low_ind] = 1

        
        # if self.args.alpha_high < self.args.alpha_low:
        #     labels = 2 * (labels - self.args.alpha_high) / (self.args.alpha_low - self.args.alpha_high) - 1
        #     labels = -labels
        # else:
        #     labels = 2 * (labels - self.args.alpha_low) / (self.args.alpha_high - self.args.alpha_low) - 1
        # labels[labels>1] = 1
        # labels[labels<-1] = -1
        
        # loss_mask[labels<1] = 1
        # if self.acc > 0.9:
        # loss_mask[labels>-1] = 1
        # else:
        # loss_mask[labels>-1] = 1
        ### version 2
        # se = se.cpu()
        # labels = copy.deepcopy(se)
        # loss_mask = torch.zeros_like(labels)
        # seq_lens = torch.sum(1-loss_mask, dim=1, keepdim=True)
        # low_rate = self.args.missing_rate[0]
        # high_rate = self.args.missing_rate[1]
        # ab_rate = np.random.randint(low_rate*100, high_rate*100)/100
        # ab_lens = (seq_lens * ab_rate).int()
        # axis1_ = [[i,] * int(ab_lens[i,0]) for i in range(self.args.batch_size) if ab_lens[i] > 0]
        # axis2_max = [labels[i,:].topk(k=ab_lens[i,0])[1] for i in range(self.args.batch_size) if ab_lens[i,0]>0]
        # labels = -labels
        # axis2_min = [labels[i,:].topk(k=ab_lens[i,0])[1] for i in range(self.args.batch_size) if ab_lens[i,0]>0]
        # label1_axis1, label1_axis2 = [], []
        # label2_axis1, label2_axis2 = [], []
        # label3_axis1, label3_axis2 = [], []
        # for ax1, ax2_max, ax2_min in zip(axis1_, axis2_max, axis2_min):
        #     label1_axis1 += ax1[:int(len(ax1)/2)]
        #     label1_axis2 += ax2_max[:int(len(ax2_max)/2)]
        #     label2_axis1 += ax1[int(len(ax1)/2):]
        #     label2_axis2 += ax2_max[int(len(ax2_max)/2):]
        #     label3_axis1 += ax1
        #     label3_axis2 += ax2_min
            
        # ab_ind1 = [torch.Tensor(label1_axis1).long().to(self.device), 
        #             torch.Tensor(label1_axis2).long().to(self.device)]
        # ab_ind2 = [torch.Tensor(label2_axis1).long().to(self.device), 
        #             torch.Tensor(label2_axis2).long().to(self.device)]
        # ab_ind3 = [torch.Tensor(label3_axis1).long().to(self.device), 
        #             torch.Tensor(label3_axis2).long().to(self.device)]

        # se = 2 * torch.tanh(se * 0.5494 / self.args.alpha) - 1
        # labels = torch.zeros_like(se) - 1
        # labels[ab_ind1] = 1
        # labels[ab_ind2] = se[ab_ind2]

        # change_mask = torch.zeros_like(se) 
        # change_mask[ab_ind1] = 1

        # loss_mask[ab_ind1] = 1
        # loss_mask[ab_ind2] = 1
        # loss_mask[ab_ind3] = 1
        ### version 1
        # labels = 2 * torch.tanh(torch.square(det_input - batch_x) * 0.5494 / self.args.alpha) - 1
        # det_input = torch.where(labels > 0.9, batch_x, det_input)
        return labels.to(self.device), loss_mask.to(self.device)

    def det_test(self, batch_x, batch_y):
        
        det = self.model_det(batch_x)
        loss = self.criterion(det, batch_y)

        return loss, [det, batch_y], 0

    def det_vali(self, batch_x, batch_y):

        # imp_index = self.generate_missing(det=(torch.zeros((batch_x.shape[0], batch_x.shape[1], 1))-1))
        # imp_index = self.generate_missing(det=copy.deepcopy(batch_y))
        imp_input = copy.deepcopy(batch_x)
        # imp_input *= 1 - imp_index

        imp = self.model_imp(imp_input)
        imp = imp.detach()

        # imp_labels = torch.where(imp_index > 0, imp_input, imp)
        # labels, loss_mask = self.cal_labels(imp, imp_input)
        labels, loss_mask = self.cal_labels(imp, batch_x)

        # det_output = self.model_det(imp)
        det_output = self.model_det(batch_x)
        batch_y = self.SM(batch_y.squeeze(-1))
        loss_output = self.SM(det_output.squeeze(-1))
        loss = self.criterion(loss_output, batch_y)
        dets = {'score': det_output, 'label': labels, 'loss': loss}
        imps = {'imp': imp, 'imp_input': imp_input, 'loss_mask': loss_mask}

        return loss, dets, imps
    
    def det_loss(self, batch_x):

        # imp_index = self.generate_missing(det=(torch.zeros((batch_x.shape[0], batch_x.shape[1], 1))-1))
        imp_input = copy.deepcopy(batch_x)
        # imp_input *= 1 - imp_index

        imp = self.model_imp(imp_input)
        imp = imp.detach()

        # imp_labels = torch.where(imp_index > 0, imp_input, imp)
        # labels, loss_mask = self.cal_labels(imp, imp_input)
        labels, loss_mask = self.cal_labels(imp, batch_x)

        # det_output = self.model_det(imp)
        det_output = self.model_det(batch_x)
        labels = self.SM(labels.squeeze(-1))
        loss_output = self.SM(det_output.squeeze(-1))
        loss = self.criterion(loss_output, labels)
        # loss = self.criterion(det_output[loss_mask>0], labels[loss_mask>0])
        dets = {'det_output': det_output, 'labels': labels}
        imps = {'batch_x': batch_x, 'imp_input': imp_input, 'loss_mask': loss_mask}

        return loss, dets, imps

    def generate_missing(self, det):
        imp_index = torch.zeros_like(det).to(self.device)
        det = det.cpu().numpy()
        low_rate = self.args.missing_rate
        # high_rate = self.args.missing_rate[1]
        missing_type = 'point' if np.random.rand() < 0.5 else 'seq'
        if missing_type == 'point':
            imp_ind = np.where(det < 0)
            imp_lens = len(imp_ind[0])
            # missing_lens = np.random.randint(imp_lens*low_rate, imp_lens*high_rate)
            missing_lens = int(imp_lens*low_rate)
            imp_ind_ = np.random.permutation(imp_lens)[:missing_lens]
            imp_ind = [torch.Tensor(ind[imp_ind_]).long().to(self.device) for ind in imp_ind]
        elif missing_type == 'seq':
            imp_mask = np.where(det < 0, np.ones_like(det), np.zeros_like(det))
            imp_ind = np.where(det < 0)
            imp_lens = np.sum(imp_mask[:, :, 0], axis=1)
            # missing_rate = np.random.randint(low_rate*100, high_rate*100)/100
            missing_rate = low_rate
            seq_lens = (imp_lens*missing_rate).astype('int')
            minus = imp_lens - seq_lens
            seq_lens = [i for i in seq_lens if i > 0]
            loc_list = [int(np.random.randint(i)) for i in minus if i > 0]
            axis1_ = [[i,] * int(seq_lens[i]) for i in range(len(seq_lens)) if seq_lens[i] > 0]
            axis2_ = [range(loc, loc+seq_lens_) for loc, seq_lens_ in zip(loc_list, seq_lens)]
            axis3_ = [[0,] * int(seq_lens[i]) for i in range(len(seq_lens)) if seq_lens[i] > 0]
            axis1, axis2, axis3 = [], [], []
            num = 0
            for i, (ax1, ax2, ax3) in enumerate(zip(axis1_, axis2_, axis3_)):
                axis1 += ax1
                ind = int(num) + np.array(ax2)
                axis2 += list(imp_ind[1][ind])
                axis3 += ax3
                num += imp_lens[i]
            imp_ind = [torch.Tensor(axis1).long().to(self.device), 
                        torch.Tensor(axis2).long().to(self.device), 
                        torch.Tensor(axis3).long().to(self.device)]
        imp_index[imp_ind] = 1
        imp_index = imp_index.repeat(1, 1, self.args.enc_in)
        return imp_index

    def imp_loss_mask(self, det):
        det = det.cpu()
        loss_mask_normal = torch.zeros_like(det)
        loss_mask_abnormal = torch.zeros_like(det)
        _, low_ind = torch.topk(det, int(det.shape[1] * 0.90), dim=1, largest=False)
        _, top_ind = torch.topk(det, int(det.shape[1] * 0.05), dim=1, largest=True)
        loss_mask_normal.scatter_(1, torch.LongTensor(low_ind), 1)
        loss_mask_abnormal.scatter_(1, torch.LongTensor(top_ind), 1)
        return loss_mask_normal.cuda(), loss_mask_abnormal.cuda()

    def imp_loss(self, batch_x):

        # imp_index = self.generate_missing(det=(torch.zeros((batch_x.shape[0], batch_x.shape[1], 1))-1))
        imp_input = copy.deepcopy(batch_x)
        # imp_input *= 1 - imp_index

        imp = self.model_imp(imp_input)
        # det = self.model_det(batch_x)
        det = self.model_det(imp)
        det = det.detach()
        # loss_mask = self.imp_loss_mask(det)
        # loss_mask = loss_mask.repeat(1, 1, batch_x.shape[-1])
        # loss = self.criterion_imp(imp[loss_mask > 0], batch_x[loss_mask > 0])
        loss_mask_normal, loss_mask_abnormal = self.imp_loss_mask(det)
        # mask = loss_mask_normal * (1 - imp_index)
        # loss_normal = torch.square(imp[mask > 0] -  imp_input[mask > 0])
        loss_normal = torch.square(imp[loss_mask_normal > 0] - batch_x[loss_mask_normal > 0])
        # loss_abnormal = -torch.square(imp[loss_mask_abnormal > 0] - batch_x[loss_mask_abnormal > 0])
        # loss = torch.mean(torch.concat([loss_abnormal, loss_normal], dim=0))
        loss = torch.mean(loss_normal)

        # return loss, det, [imp, imp_index]
        return loss, det, [imp, 0]

    def imp_test(self, batch_x, batch_y):
    
        # imp_index = self.generate_missing(det=copy.deepcopy(batch_y))
        imp_index = self.generate_missing(det=(torch.zeros((batch_x.shape[0], batch_x.shape[1], 1))-1))
        imp_input = copy.deepcopy(batch_x)
        # imp_input *= 1 - imp_index

        imp = self.model_imp(imp_input)
        # det = self.model_det(batch_x)
        det = self.model_det(imp)
        det = det.detach()
        # loss_mask = self.imp_loss_mask(det)
        # loss_mask = loss_mask.repeat(1, 1, batch_x.shape[-1])
        # loss = self.criterion_imp(imp[loss_mask > 0], batch_x[loss_mask > 0])
        loss_mask_normal, loss_mask_abnormal = self.imp_loss_mask(det)
        # mask = loss_mask_normal * (1 - imp_index)
        # loss_normal = torch.square(imp[mask > 0] -  imp_input[mask > 0])
        loss_normal = torch.square(imp[loss_mask_normal > 0] - batch_x[loss_mask_normal > 0])
        # loss_abnormal = -torch.square(imp[loss_mask_abnormal > 0] - batch_x[loss_mask_abnormal > 0])
        # loss = torch.mean(torch.concat([loss_abnormal, loss_normal], dim=0))
        loss = torch.mean(loss_normal)

        batch_y = batch_y.repeat(1, 1, batch_x.shape[-1])
        normal_loss = torch.square((imp - batch_x)[batch_y<0])
        abnormal_loss = torch.square((imp - batch_x)[batch_y>0])
        # imp = torch.where(imp_index > 0, imp, batch_x)
        imps = {'imp': imp, 'normal_loss': normal_loss, 'abnormal_loss': abnormal_loss, 
                'imp_index': imp_index, 'imp_input': imp_input}
        dets = {}

        return loss, dets, imps

    def imp_test_alone(self, batch_x, batch_y):

        # imp_index = self.generate_missing(det=batch_y)
        imp_input = copy.deepcopy(batch_x)
        # imp_input *= 1 - imp_index
        outputs = self.model_imp(imp_input)
        batch_y = batch_y.repeat(1, 1, batch_x.shape[-1])
        loss = torch.square((outputs - batch_x)[batch_y<0.5])
        abnormal_loss = torch.square((outputs - batch_x)[batch_y>0.5])
        
        return loss, outputs, abnormal_loss

    def imp_loss_alone(self, batch_x, batch_y=None):

        # imp_index = self.generate_missing(det=(torch.zeros((batch_x.shape[0], batch_x.shape[1], 1))-1))
        imp_input = copy.deepcopy(batch_x)
        # imp_input *= 1 - imp_index

        outputs = self.model_imp(imp_input)
        # loss = self.criterion(outputs[imp_index<1], imp_input[imp_index<1])
        loss = self.criterion_imp(outputs, batch_x)

        return loss, outputs

    def det_loss_alone(self, batch_x, batch_y=None):
        # if batch_y is not None:
        #     det = batch_y
        # else:
        #     det = (torch.zeros((batch_x.shape[0], batch_x.shape[1], 1))-1)
        # imp_index = self.generate_missing(det=det)
        imp_input = copy.deepcopy(batch_x)
        # imp_input *= 1 - imp_index

        outputs = self.model_det(imp_input)
        # loss = self.criterion(outputs[imp_index<1], imp_input[imp_index<1])
        loss = self.criterion(outputs, imp_input)
        # outputs = torch.where(imp_index > 0, imp_input, outputs)
        ab_score = torch.mean(torch.square(outputs - imp_input), dim=-1)
        return loss, outputs, ab_score
        

    def det_supervised_loss(self, batch_x, batch_y):

        det_output = self.model_det(batch_x)
        # loss_mask = self.balance_abnormal(batch_y)
        SM = torch.nn.Softmax(dim=1)
        batch_y = SM(batch_y.squeeze(-1))
        loss_output = SM(det_output.squeeze(-1))
        loss = self.criterion(loss_output, batch_y)

        return loss, det_output

    def balance_abnormal(self, batch_y):
        loss_mask = copy.deepcopy(batch_y)
        batch_y = batch_y.cpu().numpy()[:,:,0]
        total_lens = np.sum(np.ones_like(batch_y)).astype('int32')
        normal_ind = np.where(batch_y < 0)
        normal_lens = len(normal_ind[0])
        ab_lens = total_lens - normal_lens
        if ab_lens == 0:
            ab_lens = int(normal_lens * 0.1)
        missing_lens = ab_lens if ab_lens < total_lens/2 else normal_lens
        ab_ind_ = np.random.permutation(ab_lens)[:missing_lens]
        ab_ind = [torch.Tensor(ind[ab_ind_]).long().to(self.device) for ind in normal_ind]
        loss_mask[ab_ind] = 1
        return loss_mask

