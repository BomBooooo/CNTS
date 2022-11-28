import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        assert args.model in ['Imputation', 'Detection', 'DetectionSupervised', 'Combined']
        if args.model == 'Imputation':
            self.model_imp = self._build_model()
            self.optimizer_imp = self._select_optimizer_imp()
        elif args.model == 'Detection' or args.model == 'DetectionSupervised':
            self.model_det = self._build_model()
            self.optimizer_det = self._select_optimizer_det()
        elif args.model == 'Combined':
            self.model_det, self.model_imp = self._build_model()
            self.optimizer_det = self._select_optimizer_det()
            self.optimizer_imp = self._select_optimizer_imp()

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, *args, **kwargs):
        pass

    def vali(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass
