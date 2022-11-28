import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, learning_rate):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.9 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj =='type3':
        lr_adjust = {epoch: learning_rate}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: learning_rate * (0.9 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {:.7f}'.format(lr))


class EarlyStopping:
    def __init__(self, val_loss_min=np.Inf, patience=7, verbose=False, task='imp', logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.task = task
        self.best_auc = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = val_loss_min
        self.logger = logger

    def __call__(self, val_loss, model, path):
        if self.task[:3] == 'det':
            f1, auc = val_loss
            # cond = (self.val_loss_min - f1) < (auc - self.best_auc)
            cond = self.val_loss_min == f1 and (auc > self.best_auc)
            if f1 > self.val_loss_min or cond:
            # if f1 > self.val_loss_min:
                self.best_auc = auc
                self.save_checkpoint(f1, model, path)
                self.val_loss_min = f1
                self.counter = 0
            else:
                self.counter += 1
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} || best: {self.val_loss_min:.4f}')
                if self.counter >= self.patience:
                    self.early_stop = True
        
        elif self.task == 'imp':
            if val_loss < self.val_loss_min:
                self.save_checkpoint(val_loss, model, path)
                self.val_loss_min = val_loss
                self.counter = 0
            else:
                self.counter += 1
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} || best: {self.val_loss_min:.4f}')
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.logger.info(f'Metrics ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + '%s_checkpoint.pth'%self.task)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
